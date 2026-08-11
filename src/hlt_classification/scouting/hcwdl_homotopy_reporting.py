"""Authenticated validation-only reporting for the HCWDL U/J graph."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash,
)

from .engine import validate_pmard_training_report
from .hcwdl_homotopy_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETION_CONTRACT, NODE_RUNTIME_CONTRACT,
    TRAINING_REPORT_CONTRACT,
)
from .hcwdl_homotopy_graph import DOMAINS, GRAPH_SHA256, NODE_REGISTRY
from .hcwdl_homotopy_runner import node_output_dir
from .schema import CLASS_NAMES


REQUIRED_METRICS = (
    "cross_entropy", "accuracy", "macro_ovr_auc",
    "macro_mean_log_qcd_rejection_at_50pct_signal", "top_label_ece_15_bin",
    "balanced_accuracy", "multiclass_brier_score",
)


def _metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    value = report.get("validation")
    if not isinstance(value, Mapping):
        raise ValueError("HCWDL-UJ report lacks validation metrics")
    for name in REQUIRED_METRICS:
        if not math.isfinite(float(value[name])):
            raise FloatingPointError(f"HCWDL-UJ metric {name} is nonfinite")
    required_structured = ("per_class", "confusion_matrix")
    if any(name not in value for name in required_structured):
        raise ValueError("HCWDL-UJ report lacks required class diagnostics")
    result = dict(value)
    per_class = result["per_class"]
    if not isinstance(per_class, Mapping) or set(per_class) != set(CLASS_NAMES):
        raise ValueError("HCWDL-UJ per-class metric registry differs")
    for name in CLASS_NAMES:
        class_row = per_class[name]
        if not isinstance(class_row, Mapping) or any(
            not math.isfinite(float(class_row[field]))
            for field in ("ovr_auc", "recall", "precision")
        ):
            raise FloatingPointError(f"HCWDL-UJ per-class metrics are nonfinite for {name}")
    matrix = result["confusion_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 15 or any(len(row) != 15 for row in matrix):
        raise ValueError("HCWDL-UJ confusion matrix differs")
    natural_counts = [sum(map(int, row)) for row in matrix]
    if "class_counts" in result and list(map(int, result["class_counts"])) != natural_counts:
        raise ValueError("HCWDL-UJ natural class counts differ from confusion matrix")
    result["class_counts"] = natural_counts
    qcd_total = natural_counts[0]
    if qcd_total <= 0:
        raise ValueError("HCWDL-UJ validation population contains no QCD jets")
    operating_points = {}
    for signal in ("Xbb", "Xcc"):
        row = per_class.get(signal)
        rejection = row.get("qcd_rejection", {}).get("50pct") if isinstance(row, Mapping) else None
        if not isinstance(rejection, Mapping) or "qcd_pass" not in rejection or "rejection" not in rejection:
            raise ValueError(f"HCWDL-UJ {signal}/QCD 50% operating point is absent")
        qcd_pass = int(rejection["qcd_pass"])
        recorded_rejection = float(rejection["rejection"])
        if qcd_pass < 0 or qcd_pass > qcd_total or not math.isfinite(recorded_rejection):
            raise ValueError(f"HCWDL-UJ {signal}/QCD operating point differs")
        fpr = qcd_pass / qcd_total
        operating_points[signal] = {
            "signal_efficiency_target": 0.5,
            "achieved_signal_efficiency": rejection.get("achieved_signal_efficiency"),
            "qcd_pass": qcd_pass, "qcd_total": qcd_total,
            "false_positive_rate": fpr,
            "background_rejection": recorded_rejection,
        }
    result["signal_qcd_operating_points"] = operating_points
    return result


def _fraction(value: float, lower: float, upper: float, *, larger: bool = True) -> float | None:
    numerator = value - lower if larger else lower - value
    denominator = upper - lower if larger else lower - upper
    return None if denominator == 0 else numerator / denominator


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build the frozen 80-fit comparison table; finite poor science completes."""

    imported: dict[str, dict[str, Any]] = {}
    for node_id, record in spec["imported_controls"].items():
        report = load_json(record["report_path"]); validate_pmard_training_report(report)
        if report["content_hash"] != record["report_sha256"]:
            raise ValueError(f"HCWDL-UJ imported {node_id} report drifted")
        imported[node_id] = report
    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = dict(imported)
    runtimes: dict[str, dict[str, Any]] = {}
    for node_id, node in NODE_REGISTRY.items():
        output_dir = node_output_dir(spec["campaign_root"], node_id)
        path = output_dir / "training_report.json"
        report = load_json(path); validate_pmard_training_report(report)
        scientific = report.get("scientific_config")
        payload = scientific.get("node") if isinstance(scientific, Mapping) else None
        if (
            not isinstance(payload, Mapping) or payload.get("node_id") != node_id
            or scientific.get("graph_sha256") != GRAPH_SHA256
            or scientific.get("recipe_overlay_sha256") != spec["recipe_overlay_sha256"]
        ):
            raise ValueError(f"HCWDL-UJ training lineage differs for {node_id}")
        reports[node_id] = report
        wrapper = load_json(output_dir / "hcwdl_training_report.json")
        wrapper_hash = validate_content_hash(
            wrapper, expected_contract=TRAINING_REPORT_CONTRACT,
            expected_schema_version=1,
        )
        if (
            wrapper.get("node_id") != node_id
            or wrapper.get("pmard_engine_report_sha256") != report["content_hash"]
            or wrapper.get("complete") is not True
        ):
            raise ValueError(f"HCWDL-UJ wrapper lineage differs for {node_id}")
        runtime = load_json(output_dir / "runtime.json")
        validate_content_hash(
            runtime, expected_contract=NODE_RUNTIME_CONTRACT,
            expected_schema_version=1,
        )
        measured = float(runtime.get("measured_gpu_hours", float("nan")))
        if (
            runtime.get("campaign_spec_sha256") != spec["content_hash"]
            or runtime.get("node_id") != node_id
            or runtime.get("training_report_sha256") != wrapper_hash
            or runtime.get("pmard_engine_report_sha256") != report["content_hash"]
            or runtime.get("final_test_accessed") is not False
            or not math.isfinite(measured) or measured < 0
        ):
            raise ValueError(f"HCWDL-UJ runtime lineage differs for {node_id}")
        runtimes[node_id] = runtime

    m0 = _metrics(imported["M0"]); toff = _metrics(imported["TOFF"])
    imported_order = [name for name in ("M0", "D100", "TOFF", "D0c") if name in imported]
    for node_id in (*imported_order, *NODE_REGISTRY):
        report = reports[node_id]; metrics = _metrics(report)
        node = NODE_REGISTRY.get(node_id)
        teacher_id = None if node is None or not node.teachers else node.teachers[0].node_id
        teacher_metrics = None if teacher_id is None else _metrics(reports[teacher_id])
        domain = None if node is None else DOMAINS[node.student_domain]
        row = {
            "node_id": node_id, "imported": node_id in imported,
            "track": "imported" if node is None else node.track,
            "student_domain": None if node is None else node.student_domain,
            "coordinate": None if domain is None else {"s": domain["s"], "f": domain["f"]},
            "teacher_id": teacher_id, "metrics": metrics,
            "report_sha256": report["content_hash"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "selected_update": report.get("selected_update"),
            "selected_pass": (
                None if node_id in imported
                or report["scientific_config"].get("training_passes") is None else
                float(report["selected_update"])
                * float(report["scientific_config"]["training_passes"])
                / float(report["config"]["total_updates"])
            ),
            "validation_history": report.get("validation_history"),
            "runtime": None if node_id in imported else runtimes[node_id],
            "recovery_m0_to_toff": {
                "macro_ovr_auc": _fraction(float(metrics["macro_ovr_auc"]), float(m0["macro_ovr_auc"]), float(toff["macro_ovr_auc"])),
                "cross_entropy": _fraction(float(metrics["cross_entropy"]), float(m0["cross_entropy"]), float(toff["cross_entropy"]), larger=False),
                "log_qcd_rejection": _fraction(
                    float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
                    float(m0["macro_mean_log_qcd_rejection_at_50pct_signal"]),
                    float(toff["macro_mean_log_qcd_rejection_at_50pct_signal"]),
                ),
            },
            "retention_vs_teacher": None if teacher_metrics is None else {
                "delta_macro_ovr_auc": float(metrics["macro_ovr_auc"]) - float(teacher_metrics["macro_ovr_auc"]),
                "delta_cross_entropy": float(metrics["cross_entropy"]) - float(teacher_metrics["cross_entropy"]),
                "delta_log_qcd_rejection": float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]) - float(teacher_metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
            },
        }
        rows.append(row)
    by_id = {row["node_id"]: row for row in rows}
    comparisons = []
    frozen_pairs = (
        ("P0CE", "TOFF"), ("P0KD", "P0CE"),
        ("U100", "D100direct"), ("U100", "S100_01"),
        ("U100", "S100_10"), ("D100direct", "S100_01"),
        ("D0F", "D0c"), ("D0F", "D0direct"), ("D0F", "S0_20"),
        ("J100", "D0direct"), ("J100", "S0_20"),
        ("D0direct", "S0_01"), ("D0F", "J100"),
        ("M1F", "M1J"), ("M1F", "S0_21"), ("M1J", "S0_21"),
        ("M1F", "M0self"), ("M1J", "M0self"),
        ("S0_21", "M0self"),
        ("U010P0KD", "U010"),
    )
    for left, right in frozen_pairs:
        if left not in by_id or right not in by_id:
            comparisons.append({
                "left": left, "right": right, "available": False,
                "unavailable_reason": "control_not_registered_for_mode",
                "delta_macro_ovr_auc": None, "delta_cross_entropy": None,
                "identical_input": None,
            })
            continue
        comparisons.append({
            "left": left, "right": right, "available": True,
            "delta_macro_ovr_auc": float(by_id[left]["metrics"]["macro_ovr_auc"]) - float(by_id[right]["metrics"]["macro_ovr_auc"]),
            "delta_cross_entropy": float(by_id[left]["metrics"]["cross_entropy"]) - float(by_id[right]["metrics"]["cross_entropy"]),
            "identical_input": left in {"D0F", "J100", "M1F", "M1J"} and right in {"D0F", "J100", "D0direct", "S0_20", "M1F", "M1J", "S0_21"},
        })
    trajectory_comparisons = []
    for transition in range(1, 21):
        if transition <= 10:
            factorized = f"U{transition * 10:03d}"
            stationary = f"S100_{transition:02d}"
        else:
            factorized = f"D{100 - (transition - 10) * 10}F"
            stationary = f"S0_{transition:02d}"
        joint = f"J{transition * 5:03d}"
        for left, right, comparison in (
            (factorized, joint, "factorized_minus_joint"),
            (factorized, stationary, "factorized_minus_stationary"),
            (joint, stationary, "joint_minus_stationary"),
        ):
            trajectory_comparisons.append({
                "transition_index": transition, "comparison": comparison,
                "left": left, "right": right,
                "delta_macro_ovr_auc": (
                    float(by_id[left]["metrics"]["macro_ovr_auc"])
                    - float(by_id[right]["metrics"]["macro_ovr_auc"])
                ),
                "same_input": transition == 20,
                "trajectory_descriptive_only": transition != 20,
            })
    audit = load_json(Path(spec["campaign_root"]) / "coupling/full_role_audit.json")
    displacement = audit.get("sampled_realized_view_displacement", {})
    transition_summaries = audit.get("transition_summaries", {})
    for row in rows:
        node_id = row["node_id"]
        node = NODE_REGISTRY.get(node_id)
        row["sampled_realized_view_displacement"] = None
        row["structural_transition_summary"] = None
        if (
            node is not None and node.track in {"factorized", "joint"}
            and node.transition_index is not None
            and node.transition_index <= 20
        ):
            row["sampled_realized_view_displacement"] = {
                role: displacement.get(role, {}).get(node.track, {}).get(node_id)
                for role in ("train", "validation")
            }
            domain = DOMAINS[node.student_domain]
            if domain.get("s") is not None:
                level = int(round(float(domain["s"]) * 100))
                row["structural_transition_summary"] = {
                    role: transition_summaries.get(role, {}).get(f"s{level:03d}")
                    for role in ("train", "validation")
                }
    contextual_rows = []
    for record in spec.get("contextual_dense_reports", ()):
        report = load_json(record["report_path"]); validate_pmard_training_report(report)
        contextual_rows.append({
            "campaign": record["campaign"], "node_id": record["experiment_id"],
            "metrics": _metrics(report), "report_sha256": report["content_hash"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "paired_with_screening": False,
        })
    return with_content_hash({
        "contract": AGGREGATE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "graph_sha256": GRAPH_SHA256, "fit_count": 80,
        "primary_metric": "validation_macro_ovr_auc",
        "rows": rows, "ordered_comparisons": comparisons,
        "trajectory_comparisons": trajectory_comparisons,
        "contextual_controls": contextual_rows,
        "measured_gpu_hours": sum(
            float(runtime["measured_gpu_hours"]) for runtime in runtimes.values()
        ),
        "coupling_audit_sha256": audit["content_hash"],
        "transition_summaries": transition_summaries,
        "partition_role_summaries": audit.get("partition_role_summaries", {}),
        "sampled_realized_view_displacement": displacement,
        "screening_seed_only": True, "final_test_accessed": False,
        "scientific_result_does_not_control_completion": True,
    })


def build_validation_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility name used by the filesystem workflow."""

    return build_aggregate(spec)


def build_campaign_completion(
    spec: Mapping[str, Any], *, aggregate_sha256: str,
    resource_measurement_sha256: str | None,
) -> dict[str, Any]:
    """Publish validation-only completion without implying a test claim."""

    from hlt_classification.data.cache_contracts import require_sha256

    return with_content_hash({
        "contract": CAMPAIGN_COMPLETION_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            spec["content_hash"], name="campaign specification",
        ),
        "aggregate_sha256": require_sha256(
            aggregate_sha256, name="validation aggregate",
        ),
        "resource_measurement_sha256": (
            None if resource_measurement_sha256 is None else require_sha256(
                resource_measurement_sha256, name="resource measurement",
            )
        ),
        "fit_count": 80,
        "mode": spec["mode"],
        "validation_only": True,
        "screening_seed_only": True,
        "final_test_accessed": False,
        "scientific_result_does_not_control_completion": True,
    })


__all__ = [
    "REQUIRED_METRICS", "build_aggregate", "build_campaign_completion",
    "build_validation_aggregate",
]
