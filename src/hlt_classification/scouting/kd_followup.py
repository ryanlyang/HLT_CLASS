"""Paired CE/self-KD/T100 controls with per-epoch validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import math
from pathlib import Path, PurePosixPath
import re

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, validate_content_hash,
    with_content_hash, write_immutable_json,
)
from hlt_classification.provenance import (
    validate_source_snapshot_payload,
)
from .campaign import PMARD_PILOT_ROWS, PMARD_SITE
from .engine import validate_pmard_training_report
from .kd_sweep import (
    T100_SWEEP_ARM, target_cache_paths,
    validate_t100_sweep_inputs,
    validate_t100_sweep_report, validate_t100_sweep_spec,
    validate_t100_sweep_target_artifact,
)
from .training import LossConfiguration

KD_FOLLOWUP_SPEC_CONTRACT = "hlt_classification_pmard_kd_followup_spec_v1"
KD_FOLLOWUP_SPEC_VERSION = 1
KD_FOLLOWUP_REPORT_CONTRACT = "hlt_classification_pmard_kd_followup_report_v1"
KD_FOLLOWUP_REPORT_VERSION = 1
KD_FOLLOWUP_LEDGER_CONTRACT = "hlt_classification_pmard_kd_followup_ledger_v1"
KD_FOLLOWUP_LEDGER_VERSION = 1
KD_FOLLOWUP_TASKS = ("grid", "aggregate")
KD_FOLLOWUP_PASSES = (10, 20, 40, 60)
KD_FOLLOWUP_STUDY = "PMARD_PAIRED_KD_SCHEDULE_FOLLOWUP/v1"


def _artifact_reference(path: Path) -> dict[str, str]:
    payload = load_json(path)
    contract = payload.get("contract")
    version = payload.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError("follow-up parent artifact lacks a versioned contract")
    digest = validate_content_hash(
        payload, expected_contract=contract, expected_schema_version=version,
    )
    return {"path": str(path.resolve()), "content_hash": digest}


def _load_reference(
    spec: Mapping[str, object], name: str,
) -> tuple[Path, dict[str, object]]:
    reference = spec.get("artifacts", {}).get(name)
    if not isinstance(reference, Mapping):
        raise ValueError(f"KD follow-up lacks artifact {name!r}")
    path = Path(str(reference.get("path")))
    payload = load_json(path)
    digest = validate_content_hash(
        payload, expected_contract=str(payload.get("contract")),
        expected_schema_version=int(payload.get("schema_version", -1)),
    )
    if digest != reference.get("content_hash"):
        raise ValueError(f"KD follow-up artifact {name!r} content hash differs")
    return path, payload


def _utility_key(candidate: Mapping[str, object]) -> tuple[float, float, float, float, str]:
    metrics = candidate["validation"]
    return (
        -float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
        -float(metrics["macro_ovr_auc"]), float(metrics["cross_entropy"]),
        float(metrics["top_label_ece_15_bin"]), str(candidate["experiment_id"]),
    )


def _ce_key(candidate: Mapping[str, object]) -> tuple[float, float, str]:
    metrics = candidate["validation"]
    return (
        float(metrics["cross_entropy"]), -float(metrics["accuracy"]),
        str(candidate["experiment_id"]),
    )


def selected_kd_recipes(
    aggregate: Mapping[str, object],
) -> dict[int, tuple[dict[str, object], ...]]:
    candidates = aggregate.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("KD follow-up requires parent sweep candidates")
    selected: dict[int, tuple[dict[str, object], ...]] = {}
    for passes in KD_FOLLOWUP_PASSES:
        # Sixty passes is an extrapolation beyond the parent grid, so its
        # frozen recipes are selected from the longest available parent rows.
        parent_passes = min(passes, 40)
        group = [
            row for row in candidates
            if row.get("training_passes") == parent_passes
        ]
        if len(group) != 12:
            raise ValueError(
                f"KD follow-up requires 12 parent rows at {parent_passes} passes"
            )
        winners: dict[str, dict[str, object]] = {}
        roles: dict[str, list[str]] = {}
        for role, winner in (("best_ce", min(group, key=_ce_key)),
                             ("best_utility", min(group, key=_utility_key))):
            experiment = str(winner["experiment_id"])
            winners[experiment] = dict(winner)
            roles.setdefault(experiment, []).append(role)
        rows = []
        for experiment in sorted(winners):
            winner = winners[experiment]
            rows.append({
                "parent_experiment_id": experiment,
                "parent_training_passes": parent_passes,
                "parent_training_report_sha256": winner["training_report_sha256"],
                "selection_roles": roles[experiment],
                "ce_weight": float(winner["ce_weight"]),
                "hlt_kd_weight": float(winner["hlt_kd_weight"]),
                "privileged_kd_weight": float(winner["privileged_kd_weight"]),
                "hlt_temperature": float(winner["hlt_temperature"]),
                "privileged_temperature": float(winner["privileged_temperature"]),
            })
        selected[passes] = tuple(rows)
    return selected


def followup_registry(
    aggregate: Mapping[str, object], *, base_learning_rate: float,
    batch_size: int, train_rows: int = PMARD_PILOT_ROWS["train"],
) -> tuple[dict[str, object], ...]:
    if base_learning_rate <= 0 or batch_size <= 0 or train_rows <= 0:
        raise ValueError("KD follow-up optimizer/population values must be positive")
    recipes = selected_kd_recipes(aggregate)
    rows: list[dict[str, object]] = []
    for passes in KD_FOLLOWUP_PASSES:
        schedules = ("scaled_lr",) if passes == 10 else ("scaled_lr", "fixed_lr")
        for schedule in schedules:
            learning_rate = (
                base_learning_rate * math.sqrt(10.0 / passes)
                if schedule == "scaled_lr" else base_learning_rate
            )
            shared = {
                "training_passes": passes,
                "schedule": schedule,
                "peak_learning_rate": learning_rate,
                "peak_learning_rate_hex": learning_rate.hex(),
                "total_updates": math.ceil(train_rows / batch_size) * passes,
                "validation_interval_updates": math.ceil(train_rows / batch_size),
                "validation_cadence": "once_per_complete_train_role_pass_v1",
            }
            for control in ("K0", "K1"):
                rows.append({
                    "index": len(rows),
                    "experiment_id": f"p{passes}_{schedule}_{control.lower()}",
                    "model_role": "ce_only" if control == "K0" else "hlt_self_kd",
                    "loss_arm": control, "parent_kd_recipe": None, **shared,
                })
            for recipe in recipes[passes]:
                safe_parent = re.sub(r"[^a-zA-Z0-9]+", "_", recipe["parent_experiment_id"])
                rows.append({
                    "index": len(rows),
                    "experiment_id": f"p{passes}_{schedule}_kd_{safe_parent}",
                    "model_role": "t100_dual_kd", "loss_arm": T100_SWEEP_ARM,
                    "parent_kd_recipe": recipe, **shared,
                })
    return tuple(rows)


def create_kd_followup_spec(
    *, parent_sweep_root: str | Path, output_root: str | Path,
    source_snapshot: Mapping[str, object], project_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise ValueError("KD follow-up requires a clean source snapshot")
    parent_root = Path(parent_sweep_root).resolve()
    destination = Path(output_root).resolve()
    if destination == parent_root:
        raise ValueError("KD follow-up must use a separate output root")
    parent_spec_path = parent_root / "sweep_spec.json"
    parent_spec = load_json(parent_spec_path)
    validate_t100_sweep_spec(parent_spec)
    parent_inputs = validate_t100_sweep_inputs(parent_spec)
    target_data, target_manifest_path = target_cache_paths(parent_spec)
    target_manifest = validate_t100_sweep_target_artifact(parent_spec)
    aggregate_path = parent_root / "aggregate_report.json"
    aggregate = load_json(aggregate_path)
    validate_t100_sweep_report(parent_spec, aggregate)
    locked = parent_inputs["payloads"]["training_lock"]["payload"]
    registry = followup_registry(
        aggregate, base_learning_rate=float(locked["peak_learning_rate"]),
        batch_size=int(locked["batch_size"]),
    )
    artifacts = {
        "parent_sweep_spec": _artifact_reference(parent_spec_path),
        "parent_sweep_report": _artifact_reference(aggregate_path),
        "teacher_target_manifest": _artifact_reference(target_manifest_path),
    }
    site = dict(PMARD_SITE)
    if project_dir is not None:
        site["project_dir"] = str(Path(project_dir).resolve())
    identity = canonical_sha256({
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "artifacts": artifacts, "teacher_target_file_sha256": target_manifest["target_file_sha256"],
        "registry": list(registry), "site": site,
    })
    return with_content_hash({
        "contract": KD_FOLLOWUP_SPEC_CONTRACT,
        "schema_version": KD_FOLLOWUP_SPEC_VERSION,
        "followup_id": f"pmard_kd_followup_{identity[:16]}",
        "source_snapshot": dict(source_snapshot),
        "parent_sweep_root": str(parent_root), "output_root": str(destination),
        "site": site, "artifacts": artifacts,
        "teacher_target_file": str(target_data),
        "teacher_target_file_sha256": target_manifest["target_file_sha256"],
        "registry": list(registry), "tasks": list(KD_FOLLOWUP_TASKS),
        "checkpoint_selector": "minimum_ce_then_maximum_accuracy_then_earliest_update_v1",
        "final_test_access": False,
    })


def validate_kd_followup_spec(spec: Mapping[str, object]) -> str:
    digest = validate_content_hash(
        spec, expected_contract=KD_FOLLOWUP_SPEC_CONTRACT,
        expected_schema_version=KD_FOLLOWUP_SPEC_VERSION,
    )
    validate_source_snapshot_payload(spec.get("source_snapshot", {}))
    if spec["source_snapshot"].get("worktree_clean") is not True:
        raise ValueError("KD follow-up source is not clean")
    site = spec.get("site")
    if not isinstance(site, Mapping) or set(site) != set(PMARD_SITE):
        raise ValueError("KD follow-up site inventory differs")
    for name, value in PMARD_SITE.items():
        if name != "project_dir" and site.get(name) != value:
            raise ValueError("KD follow-up site differs")
    project = str(site.get("project_dir"))
    if not (Path(project).is_absolute() or PurePosixPath(project).is_absolute()):
        raise ValueError("KD follow-up project directory must be absolute")
    references = spec.get("artifacts")
    expected_artifacts = {
        "parent_sweep_spec", "parent_sweep_report", "teacher_target_manifest",
    }
    if not isinstance(references, Mapping) or set(references) != expected_artifacts:
        raise ValueError("KD follow-up artifact inventory differs")
    for name, reference in references.items():
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ValueError(f"KD follow-up reference {name!r} is invalid")
        require_sha256(reference.get("content_hash"), name=f"artifacts[{name}]")
    require_sha256(spec.get("teacher_target_file_sha256"), name="teacher_target_file_sha256")
    target_file = str(spec.get("teacher_target_file", ""))
    if not (Path(target_file).is_absolute() or PurePosixPath(target_file).is_absolute()):
        raise ValueError("KD follow-up target file must be absolute")
    if Path(str(spec.get("parent_sweep_root"))).resolve() == Path(str(spec.get("output_root"))).resolve():
        raise ValueError("KD follow-up output must differ from its parent sweep")
    registry = spec.get("registry")
    if not isinstance(registry, list) or not registry:
        raise ValueError("KD follow-up registry is empty")
    if [row.get("index") for row in registry] != list(range(len(registry))):
        raise ValueError("KD follow-up registry indices differ")
    if len({row.get("experiment_id") for row in registry}) != len(registry):
        raise ValueError("KD follow-up experiment identities collide")
    if spec.get("tasks") != list(KD_FOLLOWUP_TASKS):
        raise ValueError("KD follow-up tasks differ")
    if spec.get("checkpoint_selector") != "minimum_ce_then_maximum_accuracy_then_earliest_update_v1":
        raise ValueError("KD follow-up checkpoint selector differs")
    if spec.get("final_test_access") is not False:
        raise PermissionError("KD follow-up may not access final test")
    expected_identity = canonical_sha256({
        "source_snapshot_sha256": spec["source_snapshot"]["source_snapshot_sha256"],
        "artifacts": references,
        "teacher_target_file_sha256": spec["teacher_target_file_sha256"],
        "registry": registry, "site": dict(site),
    })
    if spec.get("followup_id") != f"pmard_kd_followup_{expected_identity[:16]}":
        raise ValueError("KD follow-up identity differs")
    return digest


def validate_kd_followup_inputs(spec: Mapping[str, object]) -> dict[str, object]:
    validate_kd_followup_spec(spec)
    _, parent_spec = _load_reference(spec, "parent_sweep_spec")
    _, aggregate = _load_reference(spec, "parent_sweep_report")
    target_path, target_manifest = _load_reference(spec, "teacher_target_manifest")
    validate_t100_sweep_spec(parent_spec)
    parent_inputs = validate_t100_sweep_inputs(parent_spec)
    validate_t100_sweep_report(parent_spec, aggregate)
    validated_target = validate_t100_sweep_target_artifact(parent_spec)
    if validated_target["content_hash"] != target_manifest["content_hash"]:
        raise ValueError("KD follow-up target manifest differs from parent sweep")
    data_path, expected_manifest_path = target_cache_paths(parent_spec)
    if target_path.resolve() != expected_manifest_path.resolve():
        raise ValueError("KD follow-up target manifest path differs")
    if (
        str(data_path.resolve()) != str(Path(str(spec["teacher_target_file"])).resolve())
        or validated_target["target_file_sha256"] != spec["teacher_target_file_sha256"]
    ):
        raise ValueError("KD follow-up target file differs")
    locked = parent_inputs["payloads"]["training_lock"]["payload"]
    expected_registry = list(followup_registry(
        aggregate, base_learning_rate=float(locked["peak_learning_rate"]),
        batch_size=int(locked["batch_size"]),
    ))
    if spec["registry"] != expected_registry:
        raise ValueError("KD follow-up registry differs from parent winners")
    return {
        "parent_sweep_spec": parent_spec, "parent_sweep_report": aggregate,
        "target_manifest": target_manifest, "parent_inputs": parent_inputs,
    }


def aggregate_kd_followup(spec: Mapping[str, object]) -> dict[str, object]:
    inputs = validate_kd_followup_inputs(spec)
    rows = []
    reports: dict[str, dict[str, object]] = {}
    for registered in spec["registry"]:
        path = (
            Path(str(spec["output_root"])) / "training"
            / str(registered["experiment_id"]) / "training_report.json"
        )
        report = load_json(path)
        validate_pmard_training_report(report)
        config = report.get("config", {})
        scientific = report.get("scientific_config", {})
        if registered["loss_arm"] in {"K0", "K1"}:
            expected_loss = asdict(LossConfiguration.for_arm(
                str(registered["loss_arm"]), temperature=1.0,
            ))
        else:
            recipe = registered["parent_kd_recipe"]
            expected_loss = asdict(LossConfiguration.for_mixture(
                arm=T100_SWEEP_ARM,
                ce=float(recipe["ce_weight"]),
                hlt_kd=float(recipe["hlt_kd_weight"]),
                privileged_kd=float(recipe["privileged_kd_weight"]),
                hlt_temperature=float(recipe["hlt_temperature"]),
                privileged_temperature=float(recipe["privileged_temperature"]),
            ))
        expected_history_updates = [
            int(registered["validation_interval_updates"]) * index
            for index in range(1, int(registered["training_passes"]) + 1)
        ]
        if (
            report.get("experiment_id") != registered["experiment_id"]
            or config.get("total_updates") != registered["total_updates"]
            or config.get("validation_interval")
            != registered["validation_interval_updates"]
            or config.get("effective_batch_size")
            != inputs["parent_inputs"]["payloads"]["training_lock"]["payload"]["batch_size"]
            or config.get("loss") != expected_loss
            or float(config.get("peak_learning_rate", -1))
            != float(registered["peak_learning_rate"])
            or scientific.get("registry_index") != registered["index"]
            or scientific.get("study") != KD_FOLLOWUP_STUDY
            or scientific.get("registered_row") != registered
            or scientific.get("teacher_sources") != (
                {"hlt": "none", "privileged": "none"}
                if registered["model_role"] == "ce_only" else
                {"hlt": "T0", "privileged": "none"}
                if registered["model_role"] == "hlt_self_kd" else
                {"hlt": "T0", "privileged": "T100"}
            )
            or [row["update"] for row in report.get("validation_history", [])]
            != expected_history_updates
        ):
            raise ValueError("KD follow-up training report configuration differs")
        parents = report.get("parents", {})
        if (
            parents.get("followup_spec_sha256") != spec["content_hash"]
            or parents.get("teacher_target_manifest_sha256")
            != inputs["target_manifest"]["content_hash"]
        ):
            raise ValueError("KD follow-up training report lineage differs")
        reports[str(registered["experiment_id"])] = report

    group_summaries = []
    for passes in KD_FOLLOWUP_PASSES:
        schedules = ("scaled_lr",) if passes == 10 else ("scaled_lr", "fixed_lr")
        for schedule in schedules:
            registered_group = [
                row for row in spec["registry"]
                if row["training_passes"] == passes and row["schedule"] == schedule
            ]
            k0_row = next(row for row in registered_group if row["model_role"] == "ce_only")
            k1_row = next(row for row in registered_group if row["model_role"] == "hlt_self_kd")
            k0 = reports[k0_row["experiment_id"]]["validation"]
            k1 = reports[k1_row["experiment_id"]]["validation"]
            candidates = []
            for registered in registered_group:
                report = reports[registered["experiment_id"]]
                metrics = report["validation"]
                summary = {
                    **registered,
                    "training_report_path": str(
                        Path(str(spec["output_root"])) / "training"
                        / str(registered["experiment_id"]) / "training_report.json"
                    ),
                    "training_report_sha256": report["content_hash"],
                    "validation": metrics,
                    "delta_vs_ce_only": {
                        name: float(metrics[name]) - float(k0[name])
                        for name in (
                            "cross_entropy", "accuracy", "macro_ovr_auc",
                            "macro_mean_log_qcd_rejection_at_50pct_signal",
                            "top_label_ece_15_bin",
                        )
                    },
                    "delta_vs_hlt_self_kd": {
                        name: float(metrics[name]) - float(k1[name])
                        for name in (
                            "cross_entropy", "accuracy", "macro_ovr_auc",
                            "macro_mean_log_qcd_rejection_at_50pct_signal",
                            "top_label_ece_15_bin",
                        )
                    },
                }
                candidates.append(summary); rows.append(summary)
            kd_candidates = [row for row in candidates if row["model_role"] == "t100_dual_kd"]
            group_summaries.append({
                "training_passes": passes, "schedule": schedule,
                "peak_learning_rate": registered_group[0]["peak_learning_rate"],
                "ce_only_experiment_id": k0_row["experiment_id"],
                "hlt_self_kd_experiment_id": k1_row["experiment_id"],
                "best_kd_by_ce": min(kd_candidates, key=_ce_key)["experiment_id"],
                "best_kd_by_utility": min(kd_candidates, key=_utility_key)["experiment_id"],
                "candidate_experiment_ids": [row["experiment_id"] for row in candidates],
            })
    report = with_content_hash({
        "contract": KD_FOLLOWUP_REPORT_CONTRACT,
        "schema_version": KD_FOLLOWUP_REPORT_VERSION,
        "followup_spec_sha256": spec["content_hash"],
        "parent_sweep_report_sha256": spec["artifacts"]["parent_sweep_report"]["content_hash"],
        "teacher_target_manifest_sha256": inputs["target_manifest"]["content_hash"],
        "candidate_count": len(rows), "groups": group_summaries,
        "candidates": rows, "final_test_access": False,
    })
    write_immutable_json(Path(str(spec["output_root"])) / "aggregate_report.json", report)
    return report


def _sbatch_base(spec: Mapping[str, object], *, name: str, gpu: bool) -> list[str]:
    site = spec["site"]
    command = [
        "sbatch", "--parsable", f"--account={site['account']}",
        f"--partition={site['partition']}", "--cpus-per-task=8",
        "--mem=192G" if gpu else "--mem=32G",
        "--time=48:00:00" if gpu else "--time=02:00:00",
        f"--job-name={spec['followup_id']}_{name}",
    ]
    if gpu:
        command.extend((f"--gres={site['gpu_gres']}", "--signal=B:USR1@120"))
    return command


def submit_kd_followup(
    spec: Mapping[str, object], *, spec_path: str, dry_run: bool,
    runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, object]:
    validate_kd_followup_inputs(spec)
    if not dry_run and runner is None:
        raise ValueError("executing KD follow-up submission requires a runner")
    worker = str(Path(str(spec["site"]["project_dir"])) / "sbatch/run_pmard_kd_followup.sh")
    jobs = {}; commands = []
    for index, task in enumerate(KD_FOLLOWUP_TASKS, start=1):
        gpu = task == "grid"
        command = _sbatch_base(spec, name=task, gpu=gpu)
        if task == "grid":
            command.append(f"--array=0-{len(spec['registry']) - 1}")
        else:
            command.append(f"--dependency=afterok:{jobs['grid']}")
        command.extend((
            "--export=ALL,"
            f"PROJECT_DIR={spec['site']['project_dir']},"
            f"PMARD_KD_FOLLOWUP_SPEC={Path(spec_path).resolve()},"
            f"PMARD_KD_FOLLOWUP_TASK={task}",
            worker,
        ))
        commands.append(command)
        if dry_run:
            jobs[task] = str(92_000 + index)
        else:
            output = runner(command).strip().split(";")[0]
            if re.fullmatch(r"[1-9][0-9]*", output) is None:
                raise RuntimeError("KD follow-up sbatch returned an invalid job ID")
            jobs[task] = output
    return with_content_hash({
        "contract": KD_FOLLOWUP_LEDGER_CONTRACT,
        "schema_version": KD_FOLLOWUP_LEDGER_VERSION,
        "followup_spec_sha256": spec["content_hash"],
        "dry_run": dry_run, "mutated": not dry_run,
        "jobs": jobs, "commands": commands,
    })


__all__ = [
    "KD_FOLLOWUP_LEDGER_CONTRACT", "KD_FOLLOWUP_REPORT_CONTRACT",
    "KD_FOLLOWUP_SPEC_CONTRACT", "KD_FOLLOWUP_STUDY", "KD_FOLLOWUP_TASKS",
    "aggregate_kd_followup", "create_kd_followup_spec", "followup_registry",
    "selected_kd_recipes", "submit_kd_followup", "validate_kd_followup_inputs",
    "validate_kd_followup_spec",
]
