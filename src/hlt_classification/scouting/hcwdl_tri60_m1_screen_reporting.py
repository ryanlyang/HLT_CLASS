"""Validation-only aggregation for the TRI60 M1 compression screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, sha256_file

from .hcwdl_tri60_m1_screen_campaign import validate_campaign
from .hcwdl_tri60_m1_screen_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri60_m1_screen_graph import (
    CONDITION_REGISTRY, FIT_ORDER, IMPORTED_CONTROL_ID, TEACHER_ID,
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
        or report.get("passes") != 60 or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError(f"TRI60 M1 screen report differs: {node_id}")
    if float(report.get("peak_learning_rate", 0)) != condition.peak_learning_rate:
        raise ValueError(f"TRI60 M1 screen learning rate differs: {node_id}")
    for name, digest_name in (
        ("selected_checkpoint", "selected_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        checkpoint = path.parent / str(report.get(name, ""))
        if not checkpoint.is_file() or sha256_file(checkpoint) != report.get(digest_name):
            raise ValueError(f"TRI60 M1 screen checkpoint differs: {node_id}/{name}")
    return report


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(left[name]) - float(right[name]) for name in METRICS}


def _rank_key(row: Mapping[str, Any]):
    metrics = row["metrics"]
    return (
        -float(metrics["macro_ovr_auc"]),
        float(metrics["cross_entropy"]),
        -float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
        str(row["row_id"]),
    )


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    reports = {node_id: training_report(spec, node_id) for node_id in FIT_ORDER}
    source_m1 = load_json(spec["artifact_paths"]["source_m1_report"])
    teacher_stage = load_json(spec["artifact_paths"]["teacher_stage_report"])
    source_metrics = source_m1["validation"]
    teacher_metrics = teacher_stage["ensemble_metrics"]
    rows = [{
        "row_id": IMPORTED_CONTROL_ID, "kind": "imported_control",
        "initialization": "fresh", "initialization_source": None,
        "ce_weight": .10, "kd_weight": .90, "temperature": 1.0,
        "peak_learning_rate": 3e-4,
        "metrics": source_metrics,
        "selected_pass": source_m1["selected_pass"],
        "selected_update": source_m1["selected_update"],
        "delta_from_source_m1": {name: 0.0 for name in METRICS},
        "delta_from_teacher": _delta(source_metrics, teacher_metrics),
    }]
    for node_id in FIT_ORDER:
        condition = CONDITION_REGISTRY[node_id]
        report = reports[node_id]
        metrics = report["validation"]
        rows.append({
            "row_id": node_id, "kind": "fresh_screen_fit",
            "initialization": condition.initialization,
            "initialization_source": condition.initialization_source,
            "ce_weight": condition.ce_weight, "kd_weight": condition.kd_weight,
            "temperature": condition.temperature,
            "peak_learning_rate": condition.peak_learning_rate,
            "loss_schedule": dict(condition.loss_schedule),
            "metrics": metrics, "selected_pass": report["selected_pass"],
            "selected_update": report["selected_update"],
            "delta_from_source_m1": _delta(metrics, source_metrics),
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
            "source_m1_report": source_m1["content_hash"],
        },
        "primary_question": "recover_LOGIT_D000E_with_one_exact_HLT_model",
        "primary_metric": "validation_macro_ovr_auc",
        "teacher": {"row_id": TEACHER_ID, "metrics": teacher_metrics},
        "rows": rows, "ranked_condition_ids": [row["row_id"] for row in ranked],
        "top_three_diagnostic_ids": [row["row_id"] for row in ranked[:3]],
        "top_three_do_not_trigger_jobs": True,
        "imported_condition_count": 1, "fresh_fit_count": 19,
        "condition_count": 20,
        "temperature_two_definition": "softmax(log(p_LOGIT_D000E)/2)",
        "source_probability_bank_copied": False,
        "temperature_two_bank_persisted": False,
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
        raise ValueError("TRI60 M1 screen aggregate belongs to another campaign")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "aggregate": aggregate["content_hash"],
        },
        "fresh_fit_count": 19, "imported_condition_count": 1,
        "condition_count": 20,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
        "scientific_result_does_not_control_completion": True,
    }, contract=CAMPAIGN_COMPLETE_CONTRACT)


__all__ = ["build_aggregate", "build_campaign_complete", "training_report"]
