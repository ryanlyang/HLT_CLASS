#!/usr/bin/env python3
"""Execute one immutable PRAD campaign task inside its Tigris worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.contracts import (  # noqa: E402
    build_final_test_execution_lock,
    build_finalist_lock,
)
from hlt_classification.data import discover_file_records  # noqa: E402
from hlt_classification.data.cache_contracts import (  # noqa: E402
    array_sha256,
    atomic_publish_bytes,
    deterministic_npz_bytes,
    load_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.prad.audit import run_prad_data_audit  # noqa: E402
from hlt_classification.prad.cache import PradCacheDataset  # noqa: E402
from hlt_classification.prad.evaluation import prad_classification_metrics  # noqa: E402
from hlt_classification.prad.campaign import (  # noqa: E402
    PRAD_VARIANTS,
    build_prad_task_attestation,
    validate_prad_campaign_spec,
)
from hlt_classification.prad.experiments import (  # noqa: E402
    CORE_EXPERIMENTS,
    experiment_requires_teacher,
    experiment_variant,
)
from hlt_classification.prad.splits import (  # noqa: E402
    build_prad_split_manifest,
    load_prad_split_manifest,
)
from hlt_classification.prad.streaming import (  # noqa: E402
    PRAD_MINIMUM_STORAGE_PROFILE,
    ephemeral_paired_manifest,
)
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def _paths(spec: dict[str, Any]) -> dict[str, Path]:
    root = Path(spec["site"]["campaign_root"])
    ephemeral_root = root / ".unused_ephemeral"
    if _minimum_storage(spec):
        base = Path(os.environ.get("SLURM_TMPDIR", "/dev/shm")).resolve()
        project = Path(spec["site"]["project_dir"]).resolve()
        durable_root = root.resolve()
        if (
            base == Path("/")
            or base == project
            or base.is_relative_to(project)
            or project.is_relative_to(base)
            or base == durable_root
            or base.is_relative_to(durable_root)
            or durable_root.is_relative_to(base)
        ):
            raise RuntimeError("unsafe PRAD ephemeral base")
        job = os.environ.get("SLURM_JOB_ID", f"pid_{os.getpid()}")
        array = os.environ.get("SLURM_ARRAY_TASK_ID", "none")
        ephemeral_root = (
            base
            / f"prad_{spec['content_hash'][:12]}_{job}_{array}"
        ).resolve()
        if ephemeral_root.parent != base:
            raise RuntimeError("PRAD ephemeral root escaped its base")
    return {
        "root": root,
        "split": root / "splits" / "split_manifest.json",
        "paired": (
            ephemeral_root / "paired"
            if _minimum_storage(spec)
            else root / "cache" / "paired"
        ),
        "targets": (
            ephemeral_root / "targets"
            if _minimum_storage(spec)
            else root / "cache" / "targets"
        ),
        "teacher_outputs": (
            ephemeral_root / "teacher_outputs"
            if _minimum_storage(spec)
            else root / "cache" / "teacher_outputs"
        ),
        "weights": root / "artifacts" / "semantic_weights.json",
        "runs": root / "runs",
        "reports": root / "reports",
        "results": root / "results",
        "locks": root / "locks",
        "ephemeral_root": ephemeral_root,
    }


def _minimum_storage(spec: dict[str, Any]) -> bool:
    return (
        spec.get("storage_policy", {}).get("profile")
        == PRAD_MINIMUM_STORAGE_PROFILE
    )


def _python(spec: dict[str, Any], script: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        "-s",
        str(Path(spec["site"]["project_dir"]) / "scripts" / script),
        *(str(value) for value in arguments),
    ]


def _train_options(spec: dict[str, Any]) -> list[str]:
    return (
        ["--batch-size", "8", "--amp-dtype", "none"]
        if spec["mode"] == "smoke"
        else ["--batch-size", "64", "--amp-dtype", "bfloat16"]
    )


def _cache_arguments(paths: dict[str, Path]) -> list[str]:
    result = []
    for replica in range(4):
        result.extend(
            [
                "--train-cache",
                f"{replica}={paths['paired'] / 'train' / f'replica_{replica}'}",
            ]
        )
    return result


def _target_arguments(paths: dict[str, Path]) -> list[str]:
    result = []
    for replica in range(4):
        result.extend(
            [
                "--train-target-cache",
                f"{replica}={paths['targets'] / 'train' / f'replica_{replica}'}",
            ]
        )
    return result


def _training_data_arguments(
    spec: dict[str, Any], paths: dict[str, Path]
) -> list[object]:
    del spec
    return [
        *_cache_arguments(paths),
        "--validation-cache",
        paths["paired"] / "val" / "replica_0",
    ]


def _student_data_arguments(
    spec: dict[str, Any], paths: dict[str, Path]
) -> list[object]:
    del spec
    return [
        *_cache_arguments(paths),
        *_target_arguments(paths),
        "--validation-cache",
        paths["paired"] / "val" / "replica_0",
        "--validation-target-cache",
        paths["targets"] / "val",
    ]


def _prepare_ephemeral_inputs(
    task: str,
    spec: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """Build task-scoped cache-compatible arrays outside durable storage."""

    if not _minimum_storage(spec):
        return
    requirements: dict[str, tuple[tuple[int, ...], bool, bool]] = {
        "train_statistics": ((0, 1, 2, 3), True, False),
        "E0_baseline": ((0, 1, 2, 3), False, True),
        "E1_teacher": ((0,), True, True),
        "E2_oracle": ((0, 1, 2, 3), True, True),
        "core_screen": ((0, 1, 2, 3), True, True),
        "variant_screen": ((0, 1, 2, 3), True, True),
        "confirmation": ((0, 1, 2, 3), True, True),
    }
    if task not in requirements:
        return
    train_replicas, train_targets_required, val_targets_required = requirements[task]
    source_hash = spec["source_snapshot"]["source_snapshot_sha256"]

    def paired(role: str, replica: int) -> None:
        _run(
            _python(
                spec,
                "build_prad_paired_cache.py",
                "--split-manifest",
                paths["split"],
                "--role",
                role,
                "--replica-id",
                replica,
                "--output-dir",
                paths["paired"] / role / f"replica_{replica}",
                "--source-snapshot-sha256",
                source_hash,
            )
        )

    def targets(role: str, replica: int) -> None:
        output = (
            paths["targets"] / role / f"replica_{replica}"
            if role == "train"
            else paths["targets"] / role
        )
        _run(
            _python(
                spec,
                "build_prad_targets.py",
                "--split-manifest",
                paths["split"],
                "--paired-cache",
                paths["paired"] / role / f"replica_{replica}",
                "--role",
                role,
                "--output-dir",
                output,
                "--source-snapshot-sha256",
                source_hash,
            )
        )

    for replica in train_replicas:
        paired("train", replica)
        if train_targets_required:
            targets("train", replica)
    if task != "train_statistics":
        paired("val", 0)
        if val_targets_required:
            targets("val", 0)


def _experiment_report(paths: dict[str, Path], experiment_id: str, *, seed: int | None = None) -> Path:
    return (
        paths["runs"] / "screen" / experiment_id / "training_report.json"
        if seed is None
        else paths["runs"] / "confirmation" / experiment_id / f"seed_{seed}" / "training_report.json"
    )


def _run_graph(
    spec: dict[str, Any],
    paths: dict[str, Path],
    *,
    experiment_id: str,
    seed: int,
    output_dir: Path,
    baseline_report: Path,
    include_teacher_validation: bool = True,
) -> None:
    source_hash = spec["source_snapshot"]["source_snapshot_sha256"]
    common = [
        *_training_data_arguments(spec, paths),
        "--source-snapshot-sha256",
        source_hash,
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--output-dir",
        str(output_dir),
        *_train_options(spec),
    ]
    if experiment_id == "E0":
        _run(_python(spec, "train_prad_reference.py", "--experiment", "E0", *common))
        _evaluate_graph(
            spec,
            paths,
            output_dir,
            include_teacher_validation=include_teacher_validation,
        )
        return
    if experiment_id == "E4":
        _run(
            _python(
                spec,
                "train_prad_reference.py",
                "--experiment",
                "E4",
                "--baseline-report",
                baseline_report,
                "--teacher-report",
                _experiment_report(paths, "E1"),
                *common,
            )
        )
        _evaluate_graph(
            spec,
            paths,
            output_dir,
            include_teacher_validation=include_teacher_validation,
        )
        return
    experiment = (
        CORE_EXPERIMENTS[experiment_id]
        if experiment_id in CORE_EXPERIMENTS
        else experiment_variant(*experiment_id.split("_", 1))
    )
    teacher_report = _experiment_report(paths, "E1")
    # V7 dimensions use their matching-dimension teacher, trained by the
    # variant worker before the student.
    if experiment_id in {"E9_V7_8", "E9_V7_32"}:
        teacher_report = paths["runs"] / "variant_teachers" / experiment_id / "training_report.json"
    teacher_options: list[object] = []
    if experiment_requires_teacher(experiment):
        teacher_options = ["--teacher-report", teacher_report]
    _run(
        _python(
            spec,
            "train_prad_student.py",
            "--experiment",
            experiment_id,
            *_student_data_arguments(spec, paths),
            "--semantic-weights",
            paths["weights"],
            "--baseline-report",
            baseline_report,
            *teacher_options,
            "--source-snapshot-sha256",
            source_hash,
            "--seed",
            seed,
            "--device",
            "cuda",
            "--output-dir",
            output_dir,
            *_train_options(spec),
        )
    )
    if experiment_id != "E2":
        _evaluate_graph(
            spec,
            paths,
            output_dir,
            include_teacher_validation=include_teacher_validation,
        )


def _evaluate_graph(
    spec: dict[str, Any],
    paths: dict[str, Path],
    output_dir: Path,
    *,
    include_teacher_validation: bool,
) -> None:
    validation = output_dir / "validation"
    input_arguments: list[object] = [
        "--cache",
        paths["paired"] / "val" / "replica_0",
    ]
    diagnostics: list[object] = ["--target-cache", paths["targets"] / "val"]
    if include_teacher_validation:
        if _minimum_storage(spec):
            teacher_root = paths["teacher_outputs"] / "val"
            if not (teacher_root / "manifest.json").is_file():
                _run(
                    _python(
                        spec,
                        "cache_prad_teacher_outputs.py",
                        "--split-manifest",
                        paths["split"],
                        "--paired-cache",
                        paths["paired"] / "val" / "replica_0",
                        "--teacher-report",
                        _experiment_report(paths, "E1"),
                        "--role",
                        "val",
                        "--output-dir",
                        teacher_root,
                        "--source-snapshot-sha256",
                        spec["source_snapshot"]["source_snapshot_sha256"],
                        "--device",
                        "cuda",
                    )
                )
        diagnostics.extend(
            [
                "--teacher-output-cache",
                str(paths["teacher_outputs"] / "val"),
            ]
        )
    _run(
        _python(
            spec,
            "evaluate_prad.py",
            "--training-report",
            output_dir / "training_report.json",
            *input_arguments,
            "--prediction-dir",
            validation / "predictions",
            "--metrics-output",
            validation / "metrics.json",
            "--benchmark-output",
            validation / "benchmark.json",
            "--source-snapshot-sha256",
            spec["source_snapshot"]["source_snapshot_sha256"],
            "--device",
            "cuda",
            *diagnostics,
            *_train_options(spec)[2:],
        )
    )


def _selection(paths: dict[str, Path], spec: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for experiment_id in [f"E{index}" for index in range(11) if index != 1] + [
        f"E9_{variant}" for variant in PRAD_VARIANTS
    ]:
        report_path = _experiment_report(paths, experiment_id)
        report = load_json(report_path)
        validate_content_hash(report, expected_contract=report["contract"])
        if experiment_id == "E2":
            continue
        candidates.append(
            {
                "graph_id": experiment_id,
                "validation_score": float(
                    report["best_selection"]["macro_log_rejection"]
                ),
                "training_report": str(report_path),
                "training_report_file_sha256": sha256_file(report_path),
            }
        )
    best = max(candidates, key=lambda item: item["validation_score"])
    variants = [item for item in candidates if item["graph_id"].startswith("E9_V")]
    best_variant = max(variants, key=lambda item: item["validation_score"])
    tolerance = 0.01 * max(abs(best["validation_score"]), 1.0e-12)
    within = [
        item["graph_id"]
        for item in candidates
        if best["validation_score"] - item["validation_score"] <= tolerance
    ]
    confirmation = sorted(
        set(("E0", "E8", "E9", best_variant["graph_id"], *within))
    )
    result = with_content_hash(
        {
            "contract": "hlt_classification_prad_selection_v1",
            "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"],
            "primary_metric": "macro_log_rejection",
            "one_percent_rule": "absolute_score_gap_le_0p01_times_abs_best",
            "best_graph": best["graph_id"],
            "best_variant": best_variant["graph_id"],
            "confirmation_graphs": confirmation,
            "candidates": candidates,
            "test_metrics_read": False,
        }
    )
    write_immutable_json(paths["results"] / "selection.json", result)
    return result


def _evaluate_teacher_test_cache(
    paths: dict[str, Path],
    teacher_outputs: Path,
    source_hash: str,
) -> dict[str, Any]:
    teacher = PradCacheDataset(teacher_outputs)
    split = load_prad_split_manifest(paths["split"])
    identities = split.identities("test")
    expected_keys = [item.key for item in identities]
    if teacher.manifest["identity_order_sha256"] != split.payload["roles"]["test"][
        "identity_order_sha256"
    ]:
        raise ValueError("PRAD teacher test population differs")
    logits = []
    labels = np.asarray([item.label for item in identities], dtype=np.int64)
    for start in range(0, len(teacher), 4096):
        stop = min(start + 4096, len(teacher))
        teacher_arrays = teacher.read_range(start, stop)
        if [str(value) for value in teacher_arrays["identity_keys"]] != expected_keys[
            start:stop
        ]:
            raise ValueError("PRAD teacher test identity order differs")
        logits.append(teacher_arrays["teacher_logits"])
    compact_logits = np.concatenate(logits).astype(np.float32)
    prediction_root = paths["results"] / "test" / "E1" / "predictions"
    prediction_path = prediction_root / "teacher_logits.npz"
    atomic_publish_bytes(
        prediction_path,
        deterministic_npz_bytes({"logits": compact_logits}),
    )
    prediction_manifest = with_content_hash(
        {
            "contract": "hlt_classification_prad_teacher_prediction_manifest_v1",
            "schema_version": 1,
            "source_snapshot_sha256": source_hash,
            "split_manifest_sha256": split.content_hash,
            "teacher_checkpoint_sha256": teacher.manifest["parents"][
                "teacher_checkpoint_sha256"
            ],
            "teacher_output_cache_manifest_sha256": teacher.manifest_sha256,
            "logical_role": "test",
            "rows": len(identities),
            "identity_order_sha256": split.payload["roles"]["test"][
                "identity_order_sha256"
            ],
            "identity_encoding": "implicit_authenticated_split_row",
            "labels_in_prediction_artifact": False,
            "filename": prediction_path.relative_to(prediction_root).as_posix(),
            "file_sha256": sha256_file(prediction_path),
            "logits_sha256": array_sha256("logits", compact_logits),
        }
    )
    write_immutable_json(prediction_root / "manifest.json", prediction_manifest)
    report = with_content_hash(
        {
            "contract": "hlt_classification_prad_teacher_test_report_v1",
            "schema_version": 1,
            "paired_cache_manifest_sha256": ephemeral_paired_manifest(
                split,
                logical_role="test",
                replica_id=0,
                source_snapshot_sha256=source_hash,
            )["content_hash"],
            "teacher_output_cache_manifest_sha256": teacher.manifest_sha256,
            "teacher_prediction_manifest_sha256": prediction_manifest[
                "content_hash"
            ],
            "source_snapshot_sha256": source_hash,
            "metrics": prad_classification_metrics(
                compact_logits,
                labels,
            ),
        }
    )
    write_immutable_json(paths["results"] / "test" / "E1" / "metrics.json", report)
    return report


def _final_selection(paths: dict[str, Path], spec: dict[str, Any]) -> dict[str, Any]:
    screen_selection = load_json(paths["results"] / "selection.json")
    rows = []
    for graph in screen_selection["confirmation_graphs"]:
        scores = []
        for seed in spec["run_registry"]["confirmation_seeds"]:
            report = load_json(_experiment_report(paths, graph, seed=seed))
            validate_content_hash(report, expected_contract=report["contract"])
            scores.append(float(report["best_selection"]["macro_log_rejection"]))
        import numpy as np

        rows.append(
            {
                "graph_id": graph,
                "validation_scores": scores,
                "mean_validation_score": float(np.mean(scores)),
                "std_validation_score": float(np.std(scores, ddof=1)),
            }
        )
    primary = max(rows, key=lambda row: (row["mean_validation_score"], row["graph_id"]))
    result = with_content_hash(
        {
            "contract": "hlt_classification_prad_final_selection_v1",
            "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"],
            "screen_selection_sha256": screen_selection["content_hash"],
            "selection_role": "validation_only_five_seed_mean",
            "primary_graph": primary["graph_id"],
            "locked_comparison_graphs": screen_selection["confirmation_graphs"],
            "rows": rows,
            "test_metrics_read": False,
        }
    )
    write_immutable_json(paths["results"] / "final_selection.json", result)
    return result


def _dispatch(task: str, spec: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    source_hash = spec["source_snapshot"]["source_snapshot_sha256"]
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task == "split":
        records = discover_file_records(spec["site"]["data_dir"])
        manifest = build_prad_split_manifest(
            records,
            data_root=spec["site"]["data_dir"],
            output_dir=paths["split"].parent,
            split_sizes=spec["split"]["sizes"],
        )
        return manifest.audit()
    if task == "weaver_parity":
        _run(_python(spec, "validate_weaver_parity.py", "--device", "cpu"))
        return {"validated": True}
    if task == "prad_runtime":
        output = paths["reports"] / "prad_runtime_validation.json"
        _run(
            _python(
                spec,
                "validate_prad_runtime.py",
                "--device",
                "cuda",
                "--output",
                output,
            )
        )
        report = load_json(output)
        validate_content_hash(
            report,
            expected_contract="hlt_classification_prad_runtime_validation_v2",
        )
        if report.get("passed") is not True:
            raise RuntimeError("installed-Weaver PRAD runtime validation failed")
        return {"report_sha256": report["content_hash"]}
    manifest = load_prad_split_manifest(paths["split"])
    if task == "data_audit":
        from hlt_classification.data.identity import FileRecord

        report = run_prad_data_audit(
            files=tuple(FileRecord.from_dict(item) for item in manifest.payload["files"]),
            identities=manifest.identities("train")[: min(1000, len(manifest.identities("train")))],
            data_root=manifest.payload["data_root"],
            split_manifest_sha256=manifest.content_hash,
            output_markdown=paths["reports"] / "data_audit.md",
            output_json=paths["reports"] / "data_audit.json",
        )
        return {"report_sha256": report["content_hash"]}
    if task.startswith("paired_"):
        role = {"paired_train": "train", "paired_val": "val", "paired_test_inputs": "test"}[task]
        replica = array_id if role == "train" else 0
        _run(
            _python(
                spec,
                "build_prad_paired_cache.py",
                "--split-manifest",
                paths["split"],
                "--role",
                role,
                "--replica-id",
                replica,
                "--output-dir",
                paths["paired"] / role / f"replica_{replica}",
                "--source-snapshot-sha256",
                source_hash,
            )
        )
        return {"role": role, "replica": replica}
    if task.startswith("targets_"):
        role = {"targets_train": "train", "targets_val": "val", "targets_test_inputs": "test"}[task]
        replica = array_id if role == "train" else 0
        output = (
            paths["targets"] / role / f"replica_{replica}"
            if role == "train"
            else paths["targets"] / role
        )
        _run(
            _python(
                spec,
                "build_prad_targets.py",
                "--split-manifest",
                paths["split"],
                "--paired-cache",
                paths["paired"] / role / f"replica_{replica}",
                "--role",
                role,
                "--output-dir",
                output,
                "--source-snapshot-sha256",
                source_hash,
            )
        )
        return {"role": role, "replica": replica}
    if task == "train_statistics":
        arguments: list[object] = [
            "--train-paired-cache",
            paths["paired"] / "train" / "replica_0",
            *_target_arguments(paths),
        ]
        _run(_python(spec, "fit_prad_statistics.py", *arguments, "--output", paths["weights"]))
        return {"statistics": str(paths["weights"])}
    if task == "E0_baseline":
        _run_graph(
            spec,
            paths,
            experiment_id="E0",
            seed=11,
            output_dir=paths["runs"] / "screen" / "E0",
            baseline_report=Path("unused"),
            include_teacher_validation=False,
        )
        return {"experiment": "E0"}
    if task == "E1_teacher":
        teacher_data_arguments: list[object] = [
            "--train-paired-cache",
            paths["paired"] / "train" / "replica_0",
            "--train-target-cache",
            paths["targets"] / "train" / "replica_0",
            "--validation-paired-cache",
            paths["paired"] / "val" / "replica_0",
            "--validation-target-cache",
            paths["targets"] / "val",
        ]
        _run(
            _python(
                spec,
                "train_prad_teacher.py",
                *teacher_data_arguments,
                "--semantic-weights",
                paths["weights"],
                "--output-dir",
                paths["runs"] / "screen" / "E1",
                "--source-snapshot-sha256",
                source_hash,
                "--seed",
                11,
                "--device",
                "cuda",
                *_train_options(spec),
            )
        )
        return {"experiment": "E1"}
    if task == "teacher_val_outputs":
        output = paths["teacher_outputs"] / "val"
        _run(
            _python(
                spec,
                "cache_prad_teacher_outputs.py",
                "--split-manifest",
                paths["split"],
                "--paired-cache",
                paths["paired"] / "val" / "replica_0",
                "--teacher-report",
                _experiment_report(paths, "E1"),
                "--role",
                "val",
                "--output-dir",
                output,
                "--source-snapshot-sha256",
                source_hash,
                "--device",
                "cuda",
            )
        )
        dataset = PradCacheDataset(
            output,
            expected_kind="teacher_outputs",
            expected_role="val",
        )
        return {"cache_manifest_sha256": dataset.manifest_sha256}
    if task == "E2_oracle":
        _run_graph(
            spec,
            paths,
            experiment_id="E2",
            seed=11,
            output_dir=paths["runs"] / "screen" / "E2",
            baseline_report=_experiment_report(paths, "E0"),
        )
        return {"experiment": "E2"}
    if task == "core_screen":
        experiment_id = f"E{array_id}"
        _run_graph(
            spec,
            paths,
            experiment_id=experiment_id,
            seed=11,
            output_dir=paths["runs"] / "screen" / experiment_id,
            baseline_report=_experiment_report(paths, "E0"),
        )
        return {"experiment": experiment_id}
    if task == "variant_screen":
        variant = PRAD_VARIANTS[array_id]
        experiment_id = f"E9_{variant}"
        if variant in {"V7_8", "V7_32"}:
            relation_dim = 8 if variant == "V7_8" else 32
            variant_teacher_data: list[object] = [
                "--train-paired-cache",
                paths["paired"] / "train" / "replica_0",
                "--train-target-cache",
                paths["targets"] / "train" / "replica_0",
                "--validation-paired-cache",
                paths["paired"] / "val" / "replica_0",
                "--validation-target-cache",
                paths["targets"] / "val",
            ]
            _run(
                _python(
                    spec,
                    "train_prad_teacher.py",
                    *variant_teacher_data,
                    "--semantic-weights",
                    paths["weights"],
                    "--output-dir",
                    paths["runs"] / "variant_teachers" / experiment_id,
                    "--source-snapshot-sha256",
                    source_hash,
                    "--relation-dim",
                    relation_dim,
                    "--seed",
                    11,
                    "--device",
                    "cuda",
                    *_train_options(spec),
                )
            )
        _run_graph(
            spec,
            paths,
            experiment_id=experiment_id,
            seed=11,
            output_dir=paths["runs"] / "screen" / experiment_id,
            baseline_report=_experiment_report(paths, "E0"),
        )
        return {"experiment": experiment_id}
    if task == "selection":
        return _selection(paths, spec)
    if task == "confirmation":
        selection = load_json(paths["results"] / "selection.json")
        seed = spec["run_registry"]["confirmation_seeds"][array_id]
        baseline_output = paths["runs"] / "confirmation" / "E0" / f"seed_{seed}"
        _run_graph(
            spec,
            paths,
            experiment_id="E0",
            seed=seed,
            output_dir=baseline_output,
            baseline_report=Path("unused"),
        )
        baseline_report = baseline_output / "training_report.json"
        for graph in selection["confirmation_graphs"]:
            if graph == "E0":
                continue
            _run_graph(
                spec,
                paths,
                experiment_id=graph,
                seed=seed,
                output_dir=paths["runs"] / "confirmation" / graph / f"seed_{seed}",
                baseline_report=baseline_report,
            )
        return {"seed": seed, "graphs": selection["confirmation_graphs"]}
    if task == "finalist_lock":
        selection_path = paths["results"] / "selection.json"
        selection = load_json(selection_path)
        final_selection = _final_selection(paths, spec)
        finalists = []
        for graph in selection["confirmation_graphs"]:
            for seed in spec["run_registry"]["confirmation_seeds"]:
                report_path = _experiment_report(paths, graph, seed=seed)
                report = load_json(report_path)
                finalists.append(
                    {
                        "graph_id": f"{graph}_seed_{seed}",
                        "checkpoint_sha256": report["selected_checkpoint"]["sha256"],
                        "training_report_sha256": report["content_hash"],
                    }
                )
        lock = build_finalist_lock(
            campaign_spec_sha256=spec["content_hash"],
            finalists=finalists,
            selection_artifacts={
                "screen_selection": sha256_file(selection_path),
                "five_seed_final_selection": sha256_file(
                    paths["results"] / "final_selection.json"
                ),
            },
            source_snapshot_sha256=source_hash,
        )
        write_immutable_json(paths["locks"] / "finalists.json", lock)
        test_input_manifest = ephemeral_paired_manifest(
            load_prad_split_manifest(paths["split"]),
            logical_role="test",
            replica_id=0,
            source_snapshot_sha256=source_hash,
        )
        execution = build_final_test_execution_lock(
            campaign_spec_sha256=spec["content_hash"],
            finalist_lock_sha256=lock["content_hash"],
            final_test_cache_manifest_sha256=test_input_manifest["content_hash"],
            source_snapshot_sha256=source_hash,
        )
        write_immutable_json(paths["locks"] / "final_test_execution.json", execution)
        return {
            "primary_graph": final_selection["primary_graph"],
            "finalist_lock_sha256": lock["content_hash"],
            "execution_lock_sha256": execution["content_hash"],
        }
    if task == "final_test":
        selection = load_json(paths["results"] / "selection.json")
        teacher_outputs = paths["teacher_outputs"] / "test"
        _run(
            _python(
                spec,
                "cache_prad_teacher_outputs.py",
                "--split-manifest",
                paths["split"],
                "--streaming-inputs",
                "--teacher-report",
                _experiment_report(paths, "E1"),
                "--role",
                "test",
                "--output-dir",
                teacher_outputs,
                "--source-snapshot-sha256",
                source_hash,
                "--final-evaluation-lock",
                paths["locks"] / "final_test_execution.json",
                "--device",
                "cuda",
            )
        )
        teacher_test = _evaluate_teacher_test_cache(
            paths, teacher_outputs, source_hash
        )
        for graph in selection["confirmation_graphs"]:
            for seed in spec["run_registry"]["confirmation_seeds"]:
                output = paths["results"] / "test" / graph / f"seed_{seed}"
                _run(
                    _python(
                        spec,
                        "evaluate_prad.py",
                        "--training-report",
                        _experiment_report(paths, graph, seed=seed),
                        "--streaming-split-manifest",
                        paths["split"],
                        "--streaming-targets",
                        "--prediction-dir",
                        output / "predictions",
                        "--metrics-output",
                        output / "metrics.json",
                        "--source-snapshot-sha256",
                        source_hash,
                        "--teacher-output-cache",
                        teacher_outputs,
                        "--device",
                        "cuda",
                        "--final-evaluation",
                        "--finalist-lock",
                        paths["locks"] / "finalists.json",
                        "--execution-lock",
                        paths["locks"] / "final_test_execution.json",
                        "--execution-claim",
                        paths["locks"] / "claims" / f"{graph}_seed_{seed}.json",
                        "--campaign-spec-sha256",
                        spec["content_hash"],
                    )
                )
        return {"evaluated": True, "teacher_test_report_sha256": teacher_test["content_hash"]}
    if task == "aggregate_report":
        _run(
            _python(
                spec,
                "report_prad.py",
                "--campaign-root",
                paths["root"],
            )
        )
        report_name = "final_report.md" if spec["mode"] == "production" else "smoke_report.md"
        return {"report": str(paths["reports"] / report_name)}
    raise ValueError(f"unsupported PRAD campaign task {task!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_prad_campaign_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    task_names = {row["name"] for row in spec["tasks"]}
    if args.task not in task_names:
        raise ValueError("requested task is absent from PRAD campaign graph")
    paths = _paths(spec)
    try:
        _prepare_ephemeral_inputs(args.task, spec, paths)
        result = _dispatch(args.task, spec, paths)
        attestation = build_prad_task_attestation(
            campaign_spec_sha256=spec["content_hash"],
            task=args.task,
            array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            result=result,
        )
        suffix = (
            f"_{os.environ['SLURM_ARRAY_TASK_ID']}"
            if "SLURM_ARRAY_TASK_ID" in os.environ
            else ""
        )
        write_immutable_json(
            paths["root"] / "task_attestations" / f"{args.task}{suffix}.json",
            attestation,
        )
        print(json.dumps(attestation, indent=2, sort_keys=True))
        return 0
    finally:
        if _minimum_storage(spec):
            ephemeral = paths["ephemeral_root"].resolve()
            base = ephemeral.parent
            if (
                ephemeral.name.startswith(f"prad_{spec['content_hash'][:12]}_")
                and base != Path("/")
                and ephemeral.exists()
            ):
                shutil.rmtree(ephemeral)


if __name__ == "__main__":
    raise SystemExit(main())
