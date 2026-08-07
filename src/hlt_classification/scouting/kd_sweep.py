"""Authenticated supplemental T100 weight x privileged-temperature sweep."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, canonical_sha256,
    deterministic_npz_bytes, identity_key_array, load_json, load_npz_arrays,
    require_sha256, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.provenance import (
    validate_source_snapshot, validate_source_snapshot_payload,
)
from .campaign import PMARD_PILOT_ROWS, PMARD_SITE, validate_pmard_campaign_spec
from .engine import validate_pmard_training_report
from .locks import validate_lock, validate_selective_assignment_authorization
from .selection import utility_key
from .selective_assignment import (
    validate_assignment_manifest, validate_row_selection,
)
from .splits import SCOUTING_SPLIT_CONTRACT, SCOUTING_SPLIT_VERSION
from .targets import EphemeralTeacherTargets
from .training import TEMPERATURE_GRID

T100_SWEEP_SPEC_CONTRACT = "hlt_classification_pmard_t100_kd_sweep_spec_v2"
T100_SWEEP_SPEC_VERSION = 2
T100_SWEEP_TARGET_CONTRACT = "hlt_classification_pmard_t100_kd_targets_v1"
T100_SWEEP_TARGET_VERSION = 1
T100_SWEEP_REPORT_CONTRACT = "hlt_classification_pmard_t100_kd_sweep_report_v2"
T100_SWEEP_REPORT_VERSION = 2
T100_SWEEP_LEDGER_CONTRACT = "hlt_classification_pmard_t100_kd_sweep_ledger_v2"
T100_SWEEP_LEDGER_VERSION = 2
T100_SWEEP_ARM = "T100_SWEEP"
T100_SWEEP_PRIVILEGED_WEIGHTS = (0.15, 0.25, 0.35, 0.50)
T100_SWEEP_PRIVILEGED_TEMPERATURES = (1.0, 2.0, 4.0)
T100_SWEEP_TRAINING_PASSES = (10, 20, 40)
T100_SWEEP_CE_WEIGHT = 0.25
T100_SWEEP_HLT_TEMPERATURE = 1.0
T100_SWEEP_TASKS = ("teacher_targets", "grid", "aggregate")


def t100_sweep_grid() -> tuple[dict[str, object], ...]:
    rows = []
    for privileged_weight in T100_SWEEP_PRIVILEGED_WEIGHTS:
        for privileged_temperature in T100_SWEEP_PRIVILEGED_TEMPERATURES:
            for training_passes in T100_SWEEP_TRAINING_PASSES:
                hlt_weight = 1.0 - T100_SWEEP_CE_WEIGHT - privileged_weight
                rows.append({
                    "index": len(rows),
                    "experiment_id": (
                        f"t100_w{round(privileged_weight * 100):02d}_"
                        f"tau{round(privileged_temperature)}_p{training_passes}"
                    ),
                    "ce_weight": T100_SWEEP_CE_WEIGHT,
                    "hlt_kd_weight": hlt_weight,
                    "privileged_kd_weight": privileged_weight,
                    "hlt_temperature": T100_SWEEP_HLT_TEMPERATURE,
                    "privileged_temperature": privileged_temperature,
                    "training_passes": training_passes,
                })
    return tuple(rows)


def updates_for_training_passes(
    *, train_rows: int, batch_size: int, training_passes: int,
) -> int:
    if train_rows <= 0 or batch_size <= 0 or training_passes not in T100_SWEEP_TRAINING_PASSES:
        raise ValueError("invalid T100 sweep training-exposure request")
    return math.ceil(train_rows / batch_size) * training_passes


def _validate_untyped_content(payload: Mapping[str, object]) -> str:
    contract = payload.get("contract")
    version = payload.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError("artifact has no versioned contract identity")
    return validate_content_hash(
        payload, expected_contract=contract, expected_schema_version=version,
    )


def _artifact_reference(path: Path) -> dict[str, str]:
    payload = load_json(path)
    digest = _validate_untyped_content(payload)
    return {"path": str(path.resolve()), "content_hash": digest}


def _load_reference(spec: Mapping[str, object], name: str) -> tuple[Path, dict[str, object]]:
    references = spec.get("artifacts")
    if not isinstance(references, Mapping) or not isinstance(references.get(name), Mapping):
        raise ValueError(f"sweep specification lacks artifact {name!r}")
    reference = references[name]
    path = Path(str(reference.get("path")))
    payload = load_json(path)
    digest = _validate_untyped_content(payload)
    if digest != reference.get("content_hash"):
        raise ValueError(f"sweep artifact {name!r} content hash differs")
    return path, payload


def create_t100_sweep_spec(
    *, parent_campaign_root: str | Path, output_root: str | Path,
    source_snapshot: Mapping[str, object], project_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise ValueError("T100 sweep requires a clean source snapshot")
    parent_root = Path(parent_campaign_root).resolve()
    destination = Path(output_root).resolve()
    if destination == parent_root:
        raise ValueError("supplemental sweep must use a separate output root")
    parent_spec_path = parent_root / "campaign_spec.json"
    parent_spec = load_json(parent_spec_path)
    parent_hash = validate_pmard_campaign_spec(parent_spec)
    if parent_spec.get("mode") != "pilot":
        raise ValueError("T100 sweep parent must be a pilot campaign")

    t0_reference = load_json(parent_root / "training/teachers/T0_reference.json")
    _validate_untyped_content(t0_reference)
    artifacts = {
        "parent_campaign_spec": {
            "path": str(parent_spec_path), "content_hash": parent_hash,
        },
        "split_manifest": _artifact_reference(
            parent_root / "data/splits/split_manifest.json"
        ),
        "feature_audit": _artifact_reference(parent_root / "data/feature_audit.json"),
        "row_selection": _artifact_reference(parent_root / "data/row_selection.json"),
        "assignment_manifest": _artifact_reference(
            parent_root / "matcher/assignment_manifest.json"
        ),
        "full_endpoint_lock": _artifact_reference(
            parent_root / "locks/04_full_endpoint_authorized.json"
        ),
        "training_lock": _artifact_reference(parent_root / "locks/05_training.json"),
        "t0_training_report": _artifact_reference(
            Path(str(t0_reference["training_report_path"]))
        ),
        "t100_training_report": _artifact_reference(
            parent_root / "training/teachers/T100/training_report.json"
        ),
        "k1_training_report": _artifact_reference(
            parent_root / "training/kd_controls/K1/training_report.json"
        ),
        "k2_alpha1_training_report": _artifact_reference(
            parent_root / "training/k2_alpha/1/training_report.json"
        ),
    }
    site = dict(PMARD_SITE)
    if project_dir is not None:
        site["project_dir"] = str(Path(project_dir).resolve())
    identity = canonical_sha256({
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "parent_campaign_spec_sha256": parent_hash,
        "artifacts": artifacts, "site": site,
        "grid": list(t100_sweep_grid()),
    })
    return with_content_hash({
        "contract": T100_SWEEP_SPEC_CONTRACT,
        "schema_version": T100_SWEEP_SPEC_VERSION,
        "sweep_id": f"pmard_t100_kd_sweep_{identity[:16]}",
        "source_snapshot": dict(source_snapshot),
        "parent_campaign_root": str(parent_root),
        "output_root": str(destination),
        "site": site,
        "artifacts": artifacts,
        "grid": list(t100_sweep_grid()),
        "tasks": list(T100_SWEEP_TASKS),
        "selection_rule": (
            "max_macro_mean_log_qcd_rejection_then_auc_then_ce_then_ece_v1"
        ),
        "final_test_access": False,
    })


def validate_t100_sweep_spec(spec: Mapping[str, object]) -> str:
    digest = validate_content_hash(
        spec, expected_contract=T100_SWEEP_SPEC_CONTRACT,
        expected_schema_version=T100_SWEEP_SPEC_VERSION,
    )
    validate_source_snapshot_payload(spec.get("source_snapshot", {}))
    if spec["source_snapshot"].get("worktree_clean") is not True:
        raise ValueError("T100 sweep source is not clean")
    site = spec.get("site")
    if not isinstance(site, Mapping) or set(site) != set(PMARD_SITE):
        raise ValueError("T100 sweep site inventory differs")
    for name, value in PMARD_SITE.items():
        if name != "project_dir" and site.get(name) != value:
            raise ValueError("T100 sweep site differs")
    project_dir_text = str(site.get("project_dir"))
    if not (Path(project_dir_text).is_absolute() or PurePosixPath(project_dir_text).is_absolute()):
        raise ValueError("T100 sweep project directory must be absolute")
    if spec.get("grid") != list(t100_sweep_grid()):
        raise ValueError("T100 sweep grid differs")
    if spec.get("tasks") != list(T100_SWEEP_TASKS):
        raise ValueError("T100 sweep tasks differ")
    if spec.get("final_test_access") is not False:
        raise PermissionError("supplemental T100 sweep may not access final test")
    if spec.get("selection_rule") != (
        "max_macro_mean_log_qcd_rejection_then_auc_then_ce_then_ece_v1"
    ):
        raise ValueError("T100 sweep selector differs")
    parent = Path(str(spec.get("parent_campaign_root"))).resolve()
    output = Path(str(spec.get("output_root"))).resolve()
    if parent == output:
        raise ValueError("T100 sweep output aliases its parent campaign")
    references = spec.get("artifacts")
    expected_names = {
        "parent_campaign_spec", "split_manifest", "feature_audit",
        "row_selection", "assignment_manifest", "full_endpoint_lock",
        "training_lock", "t0_training_report", "t100_training_report",
        "k1_training_report", "k2_alpha1_training_report",
    }
    if not isinstance(references, Mapping) or set(references) != expected_names:
        raise ValueError("T100 sweep artifact inventory differs")
    for name, reference in references.items():
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ValueError(f"T100 sweep artifact reference {name!r} is invalid")
        require_sha256(reference.get("content_hash"), name=f"artifacts[{name}]")
    expected_identity = canonical_sha256({
        "source_snapshot_sha256": spec["source_snapshot"]["source_snapshot_sha256"],
        "parent_campaign_spec_sha256": references["parent_campaign_spec"]["content_hash"],
        "artifacts": references, "site": dict(site),
        "grid": list(t100_sweep_grid()),
    })
    if spec.get("sweep_id") != f"pmard_t100_kd_sweep_{expected_identity[:16]}":
        raise ValueError("T100 sweep identity differs")
    return digest


def validate_t100_sweep_inputs(spec: Mapping[str, object]) -> dict[str, object]:
    validate_t100_sweep_spec(spec)
    paths = {}; payloads = {}
    for name in spec["artifacts"]:
        path, payload = _load_reference(spec, name)
        paths[name] = path; payloads[name] = payload

    parent_hash = validate_pmard_campaign_spec(payloads["parent_campaign_spec"])
    if parent_hash != spec["artifacts"]["parent_campaign_spec"]["content_hash"]:
        raise ValueError("T100 sweep parent campaign differs")
    split_hash = validate_content_hash(
        payloads["split_manifest"], expected_contract=SCOUTING_SPLIT_CONTRACT,
        expected_schema_version=SCOUTING_SPLIT_VERSION,
    )
    if split_hash != payloads["parent_campaign_spec"]["split_manifest_sha256"]:
        raise ValueError("T100 sweep split differs from parent campaign")
    selection_hash = validate_row_selection(
        payloads["row_selection"], split_manifest_sha256=split_hash,
    )
    roles = payloads["row_selection"]["roles"]
    if roles.get("train", {}).get("rows") != PMARD_PILOT_ROWS["train"]:
        raise ValueError("T100 sweep requires the exact 300k pilot train selection")
    if roles.get("validation", {}).get("rows") != PMARD_PILOT_ROWS["validation"]:
        raise ValueError("T100 sweep requires the exact 100k pilot validation selection")
    assignment_hash = validate_assignment_manifest(
        payloads["assignment_manifest"], split_manifest_sha256=split_hash,
        selection_manifest_sha256=selection_hash,
    )
    authorization_hash = validate_selective_assignment_authorization(
        payloads["full_endpoint_lock"],
        assignment_manifest=payloads["assignment_manifest"],
        row_selection=payloads["row_selection"], split_manifest_sha256=split_hash,
    )
    validate_lock(payloads["training_lock"], expected_level="training")
    locked = payloads["training_lock"]["payload"]
    if float(locked.get("temperature")) != T100_SWEEP_HLT_TEMPERATURE:
        raise ValueError("T100 sweep requires the pilot's selected T0 temperature 1")
    expected_parent_updates = updates_for_training_passes(
        train_rows=PMARD_PILOT_ROWS["train"],
        batch_size=int(locked.get("batch_size", 0)), training_passes=10,
    )
    if int(locked.get("total_updates", 0)) != expected_parent_updates:
        raise ValueError("T100 sweep parent must use the selected 10-pass pilot budget")
    for name in (
        "t0_training_report", "t100_training_report", "k1_training_report",
        "k2_alpha1_training_report",
    ):
        validate_pmard_training_report(payloads[name])
    if payloads["t0_training_report"].get("config", {}).get("model_input", "hlt") != "hlt":
        raise ValueError("T100 sweep ordinary teacher is not HLT-only")
    t100 = payloads["t100_training_report"]
    if (
        t100.get("config", {}).get("model_input") != "privileged"
        or float(t100.get("scientific_config", {}).get("alpha", -1)) != 1.0
    ):
        raise ValueError("T100 sweep privileged teacher is not the alpha-one endpoint")
    if payloads["k1_training_report"].get("scientific_config", {}).get("arm") != "K1":
        raise ValueError("T100 sweep reference is not K1")
    k2_config = payloads["k2_alpha1_training_report"].get("scientific_config", {})
    if k2_config.get("arm") != "K2" or float(k2_config.get("alpha", -1)) != 1.0:
        raise ValueError("T100 sweep reference is not the alpha-one K2 row")
    return {
        "paths": paths, "payloads": payloads,
        "split_manifest_sha256": split_hash,
        "row_selection_sha256": selection_hash,
        "assignment_manifest_sha256": assignment_hash,
        "full_endpoint_authorization_sha256": authorization_hash,
    }


def target_cache_paths(spec: Mapping[str, object]) -> tuple[Path, Path]:
    root = Path(str(spec["output_root"])) / "teacher_targets"
    return root / "targets.npz", root / "manifest.json"


def publish_t100_sweep_targets(
    spec: Mapping[str, object], *, hlt_targets: EphemeralTeacherTargets,
    privileged_targets: EphemeralTeacherTargets,
) -> dict[str, object]:
    validate_t100_sweep_spec(spec)
    if hlt_targets.identities != privileged_targets.identities:
        raise ValueError("T0 and T100 target identity orders differ")
    expected_rows = PMARD_PILOT_ROWS["train"]
    if len(hlt_targets.identities) != expected_rows:
        raise ValueError("T100 sweep target cache does not cover exactly 300k train jets")
    arrays = {
        "identity_keys": identity_key_array(hlt_targets.identities),
        "hlt_logits": np.ascontiguousarray(hlt_targets.logits, dtype=np.float32),
        "privileged_logits": np.ascontiguousarray(
            privileged_targets.logits, dtype=np.float32,
        ),
    }
    data_path, manifest_path = target_cache_paths(spec)
    serialized = deterministic_npz_bytes(arrays)
    atomic_publish_bytes(data_path, serialized)
    references = spec["artifacts"]
    manifest = with_content_hash({
        "contract": T100_SWEEP_TARGET_CONTRACT,
        "schema_version": T100_SWEEP_TARGET_VERSION,
        "sweep_spec_sha256": spec["content_hash"],
        "rows": expected_rows,
        "dtype": "float32",
        "identity_order_sha256": array_sha256("identity_keys", arrays["identity_keys"]),
        "hlt_logits_sha256": array_sha256("hlt_logits", arrays["hlt_logits"]),
        "privileged_logits_sha256": array_sha256(
            "privileged_logits", arrays["privileged_logits"],
        ),
        "target_file": data_path.name,
        "target_file_sha256": sha256_file(data_path),
        "split_manifest_sha256": references["split_manifest"]["content_hash"],
        "row_selection_sha256": references["row_selection"]["content_hash"],
        "assignment_manifest_sha256": references["assignment_manifest"]["content_hash"],
        "full_endpoint_lock_sha256": references["full_endpoint_lock"]["content_hash"],
        "hlt_teacher_report_sha256": references["t0_training_report"]["content_hash"],
        "privileged_teacher_report_sha256": references["t100_training_report"]["content_hash"],
        "model_inputs": {"hlt": "hlt", "privileged": "selective_alpha1_endpoint"},
        "final_test_access": False,
    })
    write_immutable_json(manifest_path, manifest)
    return manifest


def load_t100_sweep_targets(
    spec: Mapping[str, object],
) -> tuple[EphemeralTeacherTargets, EphemeralTeacherTargets, dict[str, object]]:
    validate_t100_sweep_spec(spec)
    data_path, manifest_path = target_cache_paths(spec)
    manifest = load_json(manifest_path)
    validate_content_hash(
        manifest, expected_contract=T100_SWEEP_TARGET_CONTRACT,
        expected_schema_version=T100_SWEEP_TARGET_VERSION,
    )
    if manifest.get("sweep_spec_sha256") != spec["content_hash"]:
        raise ValueError("T100 sweep target cache belongs to another sweep")
    if manifest.get("rows") != PMARD_PILOT_ROWS["train"]:
        raise ValueError("T100 sweep target row count differs")
    if sha256_file(data_path) != manifest.get("target_file_sha256"):
        raise ValueError("T100 sweep target file hash differs")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_keys", "hlt_logits", "privileged_logits"}:
        raise ValueError("T100 sweep target arrays differ")
    if (
        array_sha256("identity_keys", arrays["identity_keys"])
        != manifest.get("identity_order_sha256")
        or array_sha256("hlt_logits", arrays["hlt_logits"])
        != manifest.get("hlt_logits_sha256")
        or array_sha256("privileged_logits", arrays["privileged_logits"])
        != manifest.get("privileged_logits_sha256")
    ):
        raise ValueError("T100 sweep target array hash differs")
    identities = tuple(map(str, arrays["identity_keys"].tolist()))
    split_hash = spec["artifacts"]["split_manifest"]["content_hash"]
    hlt = EphemeralTeacherTargets.create(
        identities, np.asarray(arrays["hlt_logits"], np.float32),
        teacher_report_sha256=spec["artifacts"]["t0_training_report"]["content_hash"],
        split_manifest_sha256=split_hash,
    )
    privileged = EphemeralTeacherTargets.create(
        identities, np.asarray(arrays["privileged_logits"], np.float32),
        teacher_report_sha256=spec["artifacts"]["t100_training_report"]["content_hash"],
        split_manifest_sha256=split_hash,
    )
    return hlt, privileged, manifest


def aggregate_t100_sweep(spec: Mapping[str, object]) -> dict[str, object]:
    validate_t100_sweep_spec(spec)
    _, _, target_manifest = load_t100_sweep_targets(spec)
    _, k1 = _load_reference(spec, "k1_training_report")
    _, k2_alpha1 = _load_reference(spec, "k2_alpha1_training_report")
    _, training_lock = _load_reference(spec, "training_lock")
    locked = training_lock["payload"]
    candidates = []
    for registered in t100_sweep_grid():
        report_path = (
            Path(str(spec["output_root"])) / "training"
            / str(registered["experiment_id"]) / "training_report.json"
        )
        report = load_json(report_path)
        validate_pmard_training_report(report)
        loss = report.get("config", {}).get("loss", {})
        expected = {
            "arm": T100_SWEEP_ARM,
            "ce": registered["ce_weight"],
            "hlt_kd": registered["hlt_kd_weight"],
            "privileged_kd": registered["privileged_kd_weight"],
            "temperature": registered["hlt_temperature"],
            "privileged_temperature": registered["privileged_temperature"],
        }
        if loss != expected or report.get("experiment_id") != registered["experiment_id"]:
            raise ValueError("T100 sweep training report configuration differs")
        config = report.get("config", {})
        expected_updates = updates_for_training_passes(
            train_rows=PMARD_PILOT_ROWS["train"],
            batch_size=int(locked["batch_size"]),
            training_passes=int(registered["training_passes"]),
        )
        if (
            config.get("total_updates") != expected_updates
            or config.get("effective_batch_size") != int(locked["batch_size"])
            or float(config.get("peak_learning_rate", -1))
            != float(locked["peak_learning_rate"])
        ):
            raise ValueError("T100 sweep training exposure or optimizer budget differs")
        scientific = report.get("scientific_config", {})
        if (
            scientific.get("study")
            != "T100_KD_WEIGHT_X_PRIVILEGED_TEMPERATURE_X_EXPOSURE/v2"
            or scientific.get("training_passes") != registered["training_passes"]
            or float(scientific.get("alpha", -1)) != 1.0
            or scientific.get("teacher_sources")
            != {"hlt": "T0", "privileged": "T100"}
        ):
            raise ValueError("T100 sweep scientific configuration differs")
        parents = report.get("parents", {})
        if (
            parents.get("sweep_spec_sha256") != spec["content_hash"]
            or parents.get("teacher_target_manifest_sha256")
            != target_manifest["content_hash"]
        ):
            raise ValueError("T100 sweep training report lineage differs")
        candidates.append((registered, report_path, report))
    selected = min((row[2] for row in candidates), key=utility_key)
    baseline = k1["validation"]
    prior_t100 = k2_alpha1["validation"]
    summaries = []
    for registered, path, report in candidates:
        metrics = report["validation"]
        summaries.append({
            **registered,
            "total_updates": report["config"]["total_updates"],
            "training_report_path": str(path),
            "training_report_sha256": report["content_hash"],
            "validation": metrics,
            "delta_vs_k1": {
                "cross_entropy": metrics["cross_entropy"] - baseline["cross_entropy"],
                "accuracy": metrics["accuracy"] - baseline["accuracy"],
                "macro_ovr_auc": metrics["macro_ovr_auc"] - baseline["macro_ovr_auc"],
                "macro_mean_log_qcd_rejection_at_50pct_signal": (
                    metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
                    - baseline["macro_mean_log_qcd_rejection_at_50pct_signal"]
                ),
                "top_label_ece_15_bin": (
                    metrics["top_label_ece_15_bin"]
                    - baseline["top_label_ece_15_bin"]
                ),
            },
            "delta_vs_prior_k2_alpha1": {
                "cross_entropy": metrics["cross_entropy"] - prior_t100["cross_entropy"],
                "accuracy": metrics["accuracy"] - prior_t100["accuracy"],
                "macro_ovr_auc": metrics["macro_ovr_auc"] - prior_t100["macro_ovr_auc"],
                "macro_mean_log_qcd_rejection_at_50pct_signal": (
                    metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
                    - prior_t100["macro_mean_log_qcd_rejection_at_50pct_signal"]
                ),
                "top_label_ece_15_bin": (
                    metrics["top_label_ece_15_bin"]
                    - prior_t100["top_label_ece_15_bin"]
                ),
            },
        })
    report = with_content_hash({
        "contract": T100_SWEEP_REPORT_CONTRACT,
        "schema_version": T100_SWEEP_REPORT_VERSION,
        "sweep_spec_sha256": spec["content_hash"],
        "teacher_target_manifest_sha256": target_manifest["content_hash"],
        "k1_training_report_sha256": spec["artifacts"]["k1_training_report"]["content_hash"],
        "prior_k2_alpha1_training_report_sha256": (
            spec["artifacts"]["k2_alpha1_training_report"]["content_hash"]
        ),
        "candidate_count": len(candidates),
        "selection_rule": spec["selection_rule"],
        "selected_experiment_id": selected["experiment_id"],
        "selected_training_report_sha256": selected["content_hash"],
        "candidates": summaries,
        "final_test_access": False,
    })
    output = Path(str(spec["output_root"])) / "aggregate_report.json"
    write_immutable_json(output, report)
    return report


def _sbatch_base(spec: Mapping[str, object], *, name: str, gpu: bool) -> list[str]:
    site = spec["site"]
    command = [
        "sbatch", "--parsable", f"--account={site['account']}",
        f"--partition={site['partition']}", "--cpus-per-task=8",
        "--mem=192G" if gpu else "--mem=32G",
        "--time=48:00:00" if gpu else "--time=02:00:00",
        f"--job-name={spec['sweep_id']}_{name}",
    ]
    if gpu:
        command.extend((f"--gres={site['gpu_gres']}", "--signal=B:USR1@120"))
    return command


def submit_t100_sweep(
    spec: Mapping[str, object], *, spec_path: str,
    dry_run: bool, runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, object]:
    validate_t100_sweep_spec(spec)
    validate_t100_sweep_inputs(spec)
    if not dry_run and runner is None:
        raise ValueError("executing T100 sweep submission requires a runner")
    worker = str(Path(str(spec["site"]["project_dir"])) / "sbatch/run_pmard_t100_kd_sweep.sh")
    jobs = {}; commands = []
    for index, task in enumerate(T100_SWEEP_TASKS, start=1):
        gpu = task != "aggregate"
        command = _sbatch_base(spec, name=task, gpu=gpu)
        if task == "grid":
            command.append(f"--array=0-{len(t100_sweep_grid()) - 1}")
        if task != "teacher_targets":
            command.append(f"--dependency=afterok:{jobs[T100_SWEEP_TASKS[index - 2]]}")
        command.extend((
            "--export=ALL,"
            f"PROJECT_DIR={spec['site']['project_dir']},"
            f"PMARD_T100_SWEEP_SPEC={Path(spec_path).resolve()},"
            f"PMARD_T100_SWEEP_TASK={task}",
            worker,
        ))
        commands.append(command)
        if dry_run:
            jobs[task] = str(91_000 + index)
        else:
            output = runner(command).strip().split(";")[0]
            if re.fullmatch(r"[1-9][0-9]*", output) is None:
                raise RuntimeError("T100 sweep sbatch returned an invalid job ID")
            jobs[task] = output
    return with_content_hash({
        "contract": T100_SWEEP_LEDGER_CONTRACT,
        "schema_version": T100_SWEEP_LEDGER_VERSION,
        "sweep_spec_sha256": spec["content_hash"],
        "dry_run": dry_run, "mutated": not dry_run,
        "jobs": jobs, "commands": commands,
    })


__all__ = [
    "T100_SWEEP_ARM", "T100_SWEEP_LEDGER_CONTRACT",
    "T100_SWEEP_REPORT_CONTRACT", "T100_SWEEP_SPEC_CONTRACT",
    "T100_SWEEP_TARGET_CONTRACT", "aggregate_t100_sweep",
    "create_t100_sweep_spec", "load_t100_sweep_targets",
    "publish_t100_sweep_targets", "submit_t100_sweep",
    "t100_sweep_grid", "target_cache_paths", "updates_for_training_passes",
    "validate_t100_sweep_inputs", "validate_t100_sweep_spec",
]
