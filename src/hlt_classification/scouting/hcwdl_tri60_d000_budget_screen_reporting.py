"""Validation-only aggregation for the TRI60 D000 budget screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, sha256_file

from .hcwdl_tri60_d000_budget_screen_campaign import validate_campaign
from .hcwdl_tri60_d000_budget_screen_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri60_d000_budget_screen_graph import (
    CONDITION_REGISTRY, FIT_ORDER, IMPORTED_CONTROL_ID, SOURCE_NODE_ID,
    TEACHER_ID,
)


METRICS = (
    "macro_ovr_auc", "accuracy", "cross_entropy",
    "macro_mean_log_qcd_rejection_at_50pct_signal",
)


def training_report(spec: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    condition = CONDITION_REGISTRY[node_id]
    if (
        report.get("node_id") != node_id
        or report.get("node_spec") != condition.node.payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != spec["parents"]["graph"]
        or report.get("loss_schedule") != dict(condition.loss_schedule)
        or report.get("learning_rate_schedule")
        != dict(condition.learning_rate_schedule)
        or report.get("passes") != condition.passes
        or report.get("validations") != condition.passes
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError(f"TRI60 D000 budget-screen report differs: {node_id}")
    if float(report.get("peak_learning_rate", 0)) != condition.peak_learning_rate:
        raise ValueError(f"TRI60 D000 budget-screen LR differs: {node_id}")
    for name, digest_name in (
        ("selected_checkpoint", "selected_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        checkpoint = path.parent / str(report.get(name, ""))
        if not checkpoint.is_file() or sha256_file(checkpoint) != report.get(digest_name):
            raise ValueError(f"TRI60 D000 budget-screen checkpoint differs: {node_id}")
    return report


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(left[name]) - float(right[name]) for name in METRICS}


def _rank_key(row: Mapping[str, Any]):
    metrics = row["metrics"]
    return (
        -float(metrics["macro_ovr_auc"]), float(metrics["cross_entropy"]),
        -float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
        str(row["row_id"]),
    )


def _landmarks(report: Mapping[str, Any], passes: int) -> list[dict[str, Any]]:
    wanted = (20, 40, 60) if passes == 60 else (20, 40, 60, 75, 90)
    rows = {int(row["pass"]): row for row in report["validation_history"]}
    if not all(value in rows for value in wanted):
        raise ValueError("TRI60 D000 budget-screen landmark is absent")
    return [dict(rows[value]) for value in wanted]


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    reports = {node_id: training_report(spec, node_id) for node_id in FIT_ORDER}
    source_lock = load_json(spec["artifact_paths"]["source_lock"])
    source_metrics = source_lock["source_validation"]
    teacher_stage = load_json(spec["artifact_paths"]["teacher_stage_report"])
    teacher_metrics = teacher_stage["ensemble_metrics"]
    rows = [{
        "row_id": IMPORTED_CONTROL_ID, "kind": "imported_control",
        "axis": "original_60pass_reference", "passes": 60,
        "peak_learning_rate": 3e-4,
        "learning_rate_schedule": {
            "kind": "fractional_warmup_cosine_v1",
            "warmup_fraction": .05, "minimum_lr_fraction": .05,
        },
        "loss_schedule": {
            "kind": "constant_v1", "ce_weight": .25, "kd_weight": .75,
        },
        "metrics": source_metrics, "selected_pass": 60,
        "delta_from_source": {name: 0.0 for name in METRICS},
        "delta_from_teacher": _delta(source_metrics, teacher_metrics),
    }]
    for node_id in FIT_ORDER:
        condition = CONDITION_REGISTRY[node_id]
        report = reports[node_id]
        metrics = report["validation"]
        rows.append({
            "row_id": node_id, "kind": "fresh_screen_fit",
            "axis": condition.axis, "passes": condition.passes,
            "peak_learning_rate": condition.peak_learning_rate,
            "learning_rate_schedule": dict(condition.learning_rate_schedule),
            "loss_schedule": dict(condition.loss_schedule),
            "metrics": metrics, "selected_pass": report["selected_pass"],
            "selected_update": report["selected_update"],
            "validation_landmarks": _landmarks(report, condition.passes),
            "delta_from_source": _delta(metrics, source_metrics),
            "delta_from_teacher": _delta(metrics, teacher_metrics),
            "runtime_seconds": float(report["runtime_seconds"]),
            "preparation_seconds": dict(report.get("preparation_seconds", {})),
            "peak_rss_bytes": int(report["peak_rss_bytes"]),
            "peak_cuda_bytes": int(report["peak_cuda_bytes"]),
        })
    ranked = sorted(rows, key=_rank_key)
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_lock": spec["parents"]["source_lock"],
            "graph": spec["parents"]["graph"],
            "teacher_stage": teacher_stage["content_hash"],
            "source_training_report": source_lock["parents"]["source_training_report"],
        },
        "primary_question": "match_long_horizon_gain_with_60_or_90_pass_protocol",
        "primary_metric": "validation_macro_ovr_auc",
        "source": {"row_id": SOURCE_NODE_ID, "metrics": source_metrics},
        "teacher": {"row_id": TEACHER_ID, "metrics": teacher_metrics},
        "rows": rows, "ranked_condition_ids": [row["row_id"] for row in ranked],
        "top_five_diagnostic_ids": [row["row_id"] for row in ranked[:5]],
        "top_five_do_not_trigger_jobs": True,
        "imported_condition_count": 1, "fresh_fit_count": 17,
        "condition_count": 18,
        "source_probability_bank_copied": False,
        "scientific_result_does_not_control_completion": True,
        "automatic_finalist_selection": False,
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


def build_campaign_complete(
    spec: Mapping[str, Any], aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
    if aggregate.get("parents", {}).get("campaign_spec") != spec["content_hash"]:
        raise ValueError("TRI60 D000 budget-screen aggregate belongs elsewhere")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "aggregate": aggregate["content_hash"],
        },
        "fresh_fit_count": 17, "imported_condition_count": 1,
        "condition_count": 18,
        "ordinary_access_roles": ["train", "validation"],
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    }, contract=CAMPAIGN_COMPLETE_CONTRACT)


__all__ = ["build_aggregate", "build_campaign_complete", "training_report"]
