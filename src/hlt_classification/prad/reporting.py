"""Aggregate completed PRAD runs into CSV, plots, and the final report."""

from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.evaluation.metrics import paired_class_balanced_bootstrap

from .plotting import (
    save_gate_heatmap,
    save_loss_curves,
    save_matching_diagnostics,
    save_relation_curves,
    save_roc_curves,
    save_score_comparison,
    save_stratified_performance,
)
from .evaluation import prad_classification_metrics


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_publish_bytes(path, stream.getvalue().encode("utf-8"))


def _score(report: Mapping[str, Any]) -> float:
    return float(report["best_selection"]["macro_log_rejection"])


def _test_metrics(root: Path, graph: str, seed: int) -> dict[str, Any]:
    report = load_json(root / "results" / "test" / graph / f"seed_{seed}" / "metrics.json")
    validate_content_hash(report, expected_contract=report["contract"])
    return report["metrics"]


def _prediction_logits(root: Path, graph: str, seed: int) -> np.ndarray:
    prediction_root = root / "results" / "test" / graph / f"seed_{seed}" / "predictions"
    manifest = load_json(prediction_root / "manifest.json")
    return np.concatenate(
        [
            np.load(prediction_root / record["filename"], allow_pickle=False)["logits"]
            for record in manifest["shards"]
        ]
    ).astype(np.float32)


def _test_labels(root: Path) -> np.ndarray:
    from .cache import PradCacheDataset

    dataset = PradCacheDataset(root / "cache" / "paired" / "test" / "replica_0")
    return np.concatenate([arrays["labels"] for arrays in dataset.iter_shards()]).astype(np.int64)


def _benchmark(root: Path, graph: str, seed: int) -> dict[str, Any]:
    report = load_json(
        root
        / "runs"
        / "confirmation"
        / graph
        / f"seed_{seed}"
        / "validation"
        / "benchmark.json"
    )
    validate_content_hash(report, expected_contract=report["contract"])
    return report


def build_paired_bootstrap_evidence(
    *,
    graphs: Sequence[str],
    seeds: Sequence[int],
    labels: np.ndarray,
    logits_loader: Callable[[str, int], np.ndarray],
    samples: int,
) -> list[dict[str, Any]]:
    """Build paired finite-test intervals for every five-seed graph."""

    if "E0" not in graphs or len(set(graphs)) != len(graphs):
        raise ValueError("PRAD bootstrap graphs require one E0 baseline")
    if len(set(int(seed) for seed in seeds)) != len(seeds) or not seeds:
        raise ValueError("PRAD bootstrap seeds differ")
    if samples <= 0:
        raise ValueError("PRAD bootstrap sample count must be positive")
    targets = np.asarray(labels, dtype=np.int64)

    def statistic(logits: np.ndarray, sample_labels: np.ndarray) -> float:
        return float(
            prad_classification_metrics(logits.astype(np.float32), sample_labels)[
                "macro_log_rejection"
            ]
        )

    baseline = {
        int(seed): np.asarray(logits_loader("E0", int(seed)), dtype=np.float32)
        for seed in seeds
    }
    rows: list[dict[str, Any]] = []
    for graph_index, graph in enumerate(graphs):
        intervals = []
        for seed in seeds:
            numeric_seed = int(seed)
            if graph == "E0":
                interval = {
                    "sampling_unit": "paired_jet_within_class",
                    "balanced_class_handling": "exact_self_comparison",
                    "seed": 80_410 + numeric_seed + graph_index * 100_003,
                    "samples": samples,
                    "quantiles": [0.025, 0.975],
                    "quantile_method": "linear",
                    "observed_difference": 0.0,
                    "confidence_interval": [0.0, 0.0],
                }
            else:
                interval = paired_class_balanced_bootstrap(
                    np.asarray(logits_loader(graph, numeric_seed), dtype=np.float32),
                    baseline[numeric_seed],
                    targets,
                    statistic=statistic,
                    samples=samples,
                    seed=80_410 + numeric_seed + graph_index * 100_003,
                )
            intervals.append({"seed": numeric_seed, **interval})
        rows.append(
            {
                "graph_id": graph,
                "baseline_graph": "E0",
                "paired_test_bootstrap": intervals,
                "all_seed_intervals_exclude_zero": bool(
                    graph != "E0"
                    and all(
                        row["confidence_interval"][0] > 0.0
                        or row["confidence_interval"][1] < 0.0
                        for row in intervals
                    )
                ),
            }
        )
    return rows


def aggregate_prad_campaign(
    campaign_root: str | Path, *, bootstrap_samples: int = 1000
) -> dict[str, Any]:
    root = Path(campaign_root)
    spec = load_json(root / "campaign_spec.json")
    selection = load_json(root / "results" / "selection.json")
    final_selection = load_json(root / "results" / "final_selection.json")
    validate_content_hash(
        final_selection, expected_contract="hlt_classification_prad_final_selection_v1"
    )
    seeds = [int(value) for value in spec["run_registry"]["confirmation_seeds"]]
    graphs = list(selection["confirmation_graphs"])
    if (
        final_selection.get("campaign_spec_sha256") != spec["content_hash"]
        or final_selection.get("test_metrics_read") is not False
        or final_selection.get("locked_comparison_graphs") != graphs
        or final_selection.get("primary_graph") not in graphs
    ):
        raise ValueError("PRAD final selection lineage or test isolation differs")
    all_rows: list[dict[str, Any]] = []
    for report_path in sorted((root / "runs").glob("**/training_report.json")):
        report = load_json(report_path)
        validate_content_hash(report, expected_contract=report["contract"])
        config = report["config"]
        experiment_id = (
            config.get("experiment_id")
            or config.get("experiment", {}).get("experiment_id")
            or ("E1" if "teacher" in report["contract"] else "unknown")
        )
        seed = int(config["seed"])
        relative_run = report_path.parent.relative_to(root / "runs").as_posix()
        phase = relative_run.split("/", 1)[0]
        validation_metrics = next(
            (row["metrics"] for row in reversed(report["history"]) if row.get("kind") == "validation"),
            None,
        )
        all_rows.append(
            {
                "run_id": relative_run.replace("/", "__"),
                "phase": phase,
                "experiment_id": experiment_id,
                "git_commit": spec["source_snapshot"]["git_commit"],
                "config_sha256": report["parents"]["config_sha256"],
                "split_manifest_sha256": load_json(root / "splits" / "split_manifest.json")["content_hash"],
                "random_seed": seed,
                "parameter_count": report.get("parameter_count"),
                "training_time_seconds": report.get("training_time_seconds"),
                "best_epoch": report["best_selection"]["epoch"],
                "best_validation_score": _score(report),
                "checkpoint_path": report["selected_checkpoint"]["path"],
                "checkpoint_sha256": report["selected_checkpoint"]["sha256"],
                "validation_accuracy": None if validation_metrics is None else validation_metrics["secondary"]["accuracy"],
                "test_score": (
                    _test_metrics(root, experiment_id, seed)["macro_log_rejection"]
                    if phase == "confirmation" and experiment_id in graphs and seed in seeds
                    else None
                ),
                "resolved_config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
            }
        )
    all_fields = list(all_rows[0])
    _write_csv(root / "results" / "all_runs.csv", all_rows, all_fields)

    comparison = []
    baseline_scores = {
        seed: _test_metrics(root, "E0", seed)["macro_log_rejection"] for seed in seeds
    }
    for graph in graphs:
        scores = np.asarray(
            [_test_metrics(root, graph, seed)["macro_log_rejection"] for seed in seeds],
            dtype=np.float64,
        )
        baseline = np.asarray([baseline_scores[seed] for seed in seeds])
        delta = scores - baseline
        benchmarks = [_benchmark(root, graph, seed) for seed in seeds]
        comparison.append(
            {
                "graph_id": graph,
                "seeds": len(seeds),
                "mean_test_score": float(scores.mean()),
                "std_test_score": float(scores.std(ddof=1)),
                "mean_paired_delta_vs_E0": float(delta.mean()),
                "std_paired_delta_vs_E0": float(delta.std(ddof=1)),
                "relative_percent_vs_E0": float(100.0 * delta.mean() / abs(baseline.mean())),
                "mean_parameter_count": float(
                    np.mean([row["parameter_count"] for row in benchmarks])
                ),
                "mean_latency_seconds_per_jet": float(
                    np.mean([row["latency_seconds_per_jet"] for row in benchmarks])
                ),
                "mean_throughput_jets_per_second": float(
                    np.mean([row["throughput_jets_per_second"] for row in benchmarks])
                ),
                "mean_peak_memory_bytes": (
                    None
                    if any(row["peak_memory_bytes"] is None for row in benchmarks)
                    else float(np.mean([row["peak_memory_bytes"] for row in benchmarks]))
                ),
            }
        )
    best = next(
        row
        for row in comparison
        if row["graph_id"] == final_selection["primary_graph"]
    )
    baseline = next(row for row in comparison if row["graph_id"] == "E0")
    teacher_test_report = load_json(root / "results" / "test" / "E1" / "metrics.json")
    validate_content_hash(
        teacher_test_report, expected_contract="hlt_classification_prad_teacher_test_report_v1"
    )
    teacher_test_score = float(teacher_test_report["metrics"]["macro_log_rejection"])
    labels = _test_labels(root)
    bootstrap_graphs = build_paired_bootstrap_evidence(
        graphs=graphs,
        seeds=seeds,
        labels=labels,
        logits_loader=lambda graph, seed: _prediction_logits(root, graph, seed),
        samples=bootstrap_samples,
    )
    bootstrap_by_graph = {row["graph_id"]: row for row in bootstrap_graphs}
    for row in comparison:
        bootstrap = bootstrap_by_graph[row["graph_id"]]
        intervals = bootstrap["paired_test_bootstrap"]
        row["minimum_bootstrap_ci_lower"] = float(
            min(item["confidence_interval"][0] for item in intervals)
        )
        row["maximum_bootstrap_ci_upper"] = float(
            max(item["confidence_interval"][1] for item in intervals)
        )
        row["all_seed_bootstrap_intervals_exclude_zero"] = bootstrap[
            "all_seed_intervals_exclude_zero"
        ]
    _write_csv(
        root / "results" / "final_comparison.csv",
        comparison,
        list(comparison[0]),
    )
    statistical = with_content_hash(
        {
            "contract": "hlt_classification_prad_statistical_report_v2",
            "schema_version": 2,
            "best_graph": best["graph_id"],
            "selection_basis": "locked_five_seed_validation_mean_before_test",
            "baseline_graph": "E0",
            "bootstrap_samples_per_seed": bootstrap_samples,
            "graphs": bootstrap_graphs,
        }
    )
    write_immutable_json(root / "results" / "statistical_report.json", statistical)
    best_bootstrap_rows = bootstrap_by_graph[best["graph_id"]][
        "paired_test_bootstrap"
    ]
    finite_sample_supported = all(
        row["confidence_interval"][0] > 0.0 for row in best_bootstrap_rows
    )
    significant_seed_scale = (
        best["mean_paired_delta_vs_E0"] > best["std_paired_delta_vs_E0"]
        and finite_sample_supported
    )
    latency_ratio = (
        best["mean_latency_seconds_per_jet"]
        / baseline["mean_latency_seconds_per_jet"]
    )
    parameter_ratio = best["mean_parameter_count"] / baseline["mean_parameter_count"]
    inference_cost_acceptable = latency_ratio <= 2.0 and parameter_ratio <= 2.0
    if not inference_cost_acceptable and best["graph_id"] != "E0":
        recommendation = "abandon this relation target"
    elif best["graph_id"] == "E9" and significant_seed_scale:
        recommendation = "adopt full PRAD"
    elif best["graph_id"] == "E4" and significant_seed_scale:
        recommendation = "use ordinary KD"
    elif best["graph_id"] in {"E5", "E6"} and significant_seed_scale:
        recommendation = "use offline supervision only as an auxiliary loss"
    elif best["graph_id"] != "E0" and significant_seed_scale:
        recommendation = "adopt a simpler variant"
    else:
        recommendation = "abandon this relation target"

    screen = {
        row["experiment_id"]: float(row["best_validation_score"])
        for row in all_rows
        if row["phase"] == "screen"
    }
    plots = root / "plots"
    save_score_comparison(comparison, plots / "rejection_comparison.png")
    save_roc_curves(
        {
            "E0": _prediction_logits(root, "E0", seeds[0]),
            best["graph_id"]: _prediction_logits(root, best["graph_id"], seeds[0]),
        },
        labels,
        plots / "roc_curves.png",
    )
    chosen_report = load_json(
        root / "runs" / "confirmation" / best["graph_id"] / f"seed_{seeds[0]}" / "training_report.json"
    )
    baseline_report = load_json(
        root / "runs" / "confirmation" / "E0" / f"seed_{seeds[0]}" / "training_report.json"
    )
    save_loss_curves(
        {
            "E0": baseline_report["history"],
            best["graph_id"]: chosen_report["history"],
        },
        plots / "loss_curves.png",
    )
    relation_diagnostics = chosen_report.get("relation_diagnostics", {})
    relation_answer = (
        "unavailable for the selected graph"
        if not relation_diagnostics.get("available")
        else "no teacher relation target for this semantic-only graph"
        if not relation_diagnostics.get("teacher_relation_available")
        else (
            f"relation Smooth-L1 {relation_diagnostics['normalized_relation_smooth_l1']:.6f}, "
            f"bias Smooth-L1 {relation_diagnostics['normalized_bias_smooth_l1']:.6f}, "
            f"bias Pearson {relation_diagnostics['teacher_student_bias_pearson']}"
        )
    )
    if "gate_values" in chosen_report:
        save_gate_heatmap(np.asarray(chosen_report["gate_values"]), plots / "gate_heatmap.png")
    if any("relation" in row for row in chosen_report["history"]):
        save_relation_curves(chosen_report["history"], plots / "relation_quality.png")
    audit = load_json(root / "reports" / "data_audit.json")
    save_matching_diagnostics(audit, plots / "matching_diagnostics.png")
    best_metrics = _test_metrics(root, best["graph_id"], seeds[0])
    stratified_plot_available = True
    try:
        save_stratified_performance(best_metrics, plots / "stratified_performance.png")
    except ValueError:
        if spec["mode"] != "smoke":
            raise
        stratified_plot_available = False

    relation_target_candidates = [
        name for name in ("E9_V1", "E9_V2", "E9_V5", "E9_V6") if name in screen
    ]
    best_relation_target = max(
        relation_target_candidates, key=screen.__getitem__, default="unavailable"
    )
    depth_candidates = [
        name
        for name in ("E9_V8_ALL", "E9_V8_AFTER2", "E9_V8_FINAL")
        if name in screen
    ]
    best_injection_depth = max(
        depth_candidates, key=screen.__getitem__, default="unavailable"
    )

    def difference(left: str, right: str) -> str:
        if left not in screen or right not in screen:
            return "not registered in the screening summary"
        return f"{screen[left] - screen[right]:+.6f} macro-log-rejection"

    report_lines = [
        (
            "# Privileged Relational Attention Distillation Final Report"
            if spec["mode"] == "production"
            else "# Privileged Relational Attention Distillation Smoke Report"
        ),
        "",
        f"Campaign mode: `{spec['mode']}`. Campaign: `{spec['content_hash']}`. Test execution occurred only after the finalist and execution locks.",
        "",
        "## Direct answers",
        "",
        f"1. Offline teacher versus HLT baseline: validation {difference('E1', 'E0')}; locked test delta {teacher_test_score - baseline['mean_test_score']:+.6f} against the five-seed baseline mean.",
        f"2. Oracle relation headroom (E2 versus E0): {difference('E2', 'E0')}.",
        f"3. HLT relation predictability for the selected graph: {relation_answer}; semantic AUCs are authenticated in its training report.",
        f"4. Auxiliary relation supervision (E6 versus E0): {difference('E6', 'E0')}.",
        f"5. Attention injection beyond auxiliary supervision (E8 versus E6): {difference('E8', 'E6')}.",
        f"6. KD explanation: E9-E8 is {difference('E9', 'E8')}; E4-E0 is {difference('E4', 'E0')}.",
        f"7. Equal-capacity control (E3 versus E0): {difference('E3', 'E0')}.",
        f"8. Shuffled relation control (E10 versus E8): {difference('E10', 'E8')}.",
        f"9. Best pre-test relation target variant: {best_relation_target}; V1/V2/V5/V6 are preserved in `results/all_runs.csv`.",
        f"10. Best pre-test attention-injection depth variant: {best_injection_depth}.",
        f"11. Deployable recovery of oracle gain: E9-E0 {difference('E9', 'E0')} versus E2-E0 {difference('E2', 'E0')}.",
        f"12. Class and kinematic behavior is recorded in the test metric `stratified` block; stratified plot available: `{stratified_plot_available}`.",
        f"13. The best graph uses {parameter_ratio:.4f}x the baseline parameters and {latency_ratio:.4f}x validation latency; throughput and peak memory are in `results/final_comparison.csv`.",
        f"14. The pre-test five-seed validation selection chose {best['graph_id']}; its locked test mean was {best['mean_test_score']:.6f}, standard deviation {best['std_test_score']:.6f}, and paired delta {best['mean_paired_delta_vs_E0']:+.6f}. Paired class-balanced test bootstrap support across every seed: `{finite_sample_supported}`.",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}.** This rule requires the paired mean gain to exceed seed-to-seed variation, every per-seed paired bootstrap interval to exclude zero, and deployable latency/parameter ratios no greater than 2x (cost accepted: `{inference_cost_acceptable}`).",
        "",
        "Poor performance was retained as a successful scientific result and did not cancel registered experiments.",
        "",
    ]
    report_name = "final_report.md" if spec["mode"] == "production" else "smoke_report.md"
    atomic_publish_bytes(
        root / "reports" / report_name,
        "\n".join(report_lines).encode("utf-8"),
    )
    return {
        "runs": len(all_rows),
        "comparison_rows": len(comparison),
        "best_graph": best["graph_id"],
        "recommendation": recommendation,
        "baseline_mean": baseline["mean_test_score"],
        "statistical_report_sha256": statistical["content_hash"],
        "report": str(root / "reports" / report_name),
    }


__all__ = ["aggregate_prad_campaign", "build_paired_bootstrap_evidence"]
