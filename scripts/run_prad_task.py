#!/usr/bin/env python3
"""Execute one immutable PRAD campaign task inside its Tigris worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

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
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def _paths(spec: dict[str, Any]) -> dict[str, Path]:
    root = Path(spec["site"]["campaign_root"])
    return {
        "root": root,
        "split": root / "splits" / "split_manifest.json",
        "paired": root / "cache" / "paired",
        "targets": root / "cache" / "targets",
        "teacher_outputs": root / "cache" / "teacher_outputs",
        "weights": root / "artifacts" / "semantic_weights.json",
        "runs": root / "runs",
        "reports": root / "reports",
        "results": root / "results",
        "locks": root / "locks",
    }


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
        *_cache_arguments(paths),
        "--validation-cache",
        str(paths["paired"] / "val" / "replica_0"),
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
            *_cache_arguments(paths),
            *_target_arguments(paths),
            "--validation-cache",
            paths["paired"] / "val" / "replica_0",
            "--validation-target-cache",
            paths["targets"] / "val",
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
    diagnostics = [
        "--target-cache",
        str(paths["targets"] / "val"),
    ]
    if include_teacher_validation:
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
            "--cache",
            paths["paired"] / "val" / "replica_0",
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
    paths: dict[str, Path], teacher_outputs: Path, source_hash: str
) -> dict[str, Any]:
    paired = PradCacheDataset(paths["paired"] / "test" / "replica_0")
    teacher = PradCacheDataset(teacher_outputs)
    if (
        len(paired) != len(teacher)
        or paired.manifest["identity_order_sha256"]
        != teacher.manifest["identity_order_sha256"]
    ):
        raise ValueError("PRAD teacher test population differs")
    logits = []
    labels = []
    for start in range(0, len(paired), 4096):
        stop = min(start + 4096, len(paired))
        paired_arrays = paired.read_range(start, stop)
        teacher_arrays = teacher.read_range(start, stop)
        if not (paired_arrays["identity_keys"] == teacher_arrays["identity_keys"]).all():
            raise ValueError("PRAD teacher test identity order differs")
        logits.append(teacher_arrays["teacher_logits"])
        labels.append(paired_arrays["labels"])
    import numpy as np

    report = with_content_hash(
        {
            "contract": "hlt_classification_prad_teacher_test_report_v1",
            "schema_version": 1,
            "paired_cache_manifest_sha256": paired.manifest_sha256,
            "teacher_output_cache_manifest_sha256": teacher.manifest_sha256,
            "source_snapshot_sha256": source_hash,
            "metrics": prad_classification_metrics(
                np.concatenate(logits).astype(np.float32),
                np.concatenate(labels).astype(np.int64),
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
            expected_contract="hlt_classification_prad_runtime_validation_v1",
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
        _run(
            _python(
                spec,
                "fit_prad_statistics.py",
                "--train-paired-cache",
                paths["paired"] / "train" / "replica_0",
                *_target_arguments(paths),
                "--output",
                paths["weights"],
            )
        )
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
        _run(
            _python(
                spec,
                "train_prad_teacher.py",
                "--train-paired-cache",
                paths["paired"] / "train" / "replica_0",
                "--train-target-cache",
                paths["targets"] / "train" / "replica_0",
                "--validation-paired-cache",
                paths["paired"] / "val" / "replica_0",
                "--validation-target-cache",
                paths["targets"] / "val",
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
            _run(
                _python(
                    spec,
                    "train_prad_teacher.py",
                    "--train-paired-cache",
                    paths["paired"] / "train" / "replica_0",
                    "--train-target-cache",
                    paths["targets"] / "train" / "replica_0",
                    "--validation-paired-cache",
                    paths["paired"] / "val" / "replica_0",
                    "--validation-target-cache",
                    paths["targets"] / "val",
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
        test_cache = PradCacheDataset(paths["paired"] / "test" / "replica_0")
        execution = build_final_test_execution_lock(
            campaign_spec_sha256=spec["content_hash"],
            finalist_lock_sha256=lock["content_hash"],
            final_test_cache_manifest_sha256=test_cache.manifest_sha256,
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
                "--paired-cache",
                paths["paired"] / "test" / "replica_0",
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
                        "--cache",
                        paths["paired"] / "test" / "replica_0",
                        "--prediction-dir",
                        output / "predictions",
                        "--metrics-output",
                        output / "metrics.json",
                        "--source-snapshot-sha256",
                        source_hash,
                        "--target-cache",
                        paths["targets"] / "test",
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


if __name__ == "__main__":
    raise SystemExit(main())
