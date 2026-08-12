"""Authenticated validation reporting for the direct offline-KD ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash,
)

from .hcwdl_direct_offline_kd_graph import NODE_ORDER
from .hcwdl_direct_offline_kd_runner import (
    BASE_REPORT_CONTRACT, REPRESENTATION_REPORT_CONTRACT, node_output_dir,
    validate_base_wrapper, validate_representation_wrapper,
)


AGGREGATE_CONTRACT = "HCWDL_DIRECT_OFFLINE_KD_AGGREGATE/v1"
COMPLETION_CONTRACT = "HCWDL_DIRECT_OFFLINE_KD_COMPLETION/v1"


def _report(spec: Mapping[str, Any], node_id: str) -> tuple[dict[str, Any], str]:
    path = node_output_dir(spec["campaign_root"], node_id) / "direct_report.json"
    value = load_json(path)
    contract = (
        REPRESENTATION_REPORT_CONTRACT
        if node_id in {"HLT_RSET", "HLT_RREL"} else BASE_REPORT_CONTRACT
    )
    digest = validate_content_hash(value, expected_contract=contract, expected_schema_version=1)
    if node_id in {"HLT_RSET", "HLT_RREL"}:
        validate_representation_wrapper(spec, node_id=node_id, value=value)
    else:
        validate_base_wrapper(spec, node_id=node_id, value=value)
    if value.get("node_id") != node_id or value.get("final_test_accessed") is not False:
        raise ValueError("direct KD report identity differs")
    return value, digest


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    rows = []; parents = {}
    for node_id in NODE_ORDER:
        report, digest = _report(spec, node_id); parents[f"report_{node_id}"] = digest
        metrics = report["validation"]
        rows.append({
            "node_id": node_id,
            "input_domain": "toff" if node_id == "TOFF_CE" else "hlt",
            "deployable_hlt_only": node_id != "TOFF_CE",
            "loss": {
                "HLT_CE": "CE", "TOFF_CE": "CE",
                "HLT_LOGIT": "CE+logitKD", "HLT_RSET": "CE+logitKD+RSET",
                "HLT_RREL": "CE+logitKD+RREL",
            }[node_id],
            "cross_entropy": float(metrics["cross_entropy"]),
            "accuracy": float(metrics["accuracy"]),
            "macro_ovr_auc": float(metrics["macro_ovr_auc"]),
            "macro_mean_log_qcd_rejection_at_50pct_signal": float(
                metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
            ),
        })
    baseline = next(row for row in rows if row["node_id"] == "HLT_CE")
    logit = next(row for row in rows if row["node_id"] == "HLT_LOGIT")
    comparisons = []
    for row in rows:
        if row["node_id"] == "TOFF_CE":
            continue
        comparisons.append({
            "node_id": row["node_id"],
            "delta_auc_vs_hlt_ce": row["macro_ovr_auc"] - baseline["macro_ovr_auc"],
            "delta_logr50_vs_hlt_ce": (
                row["macro_mean_log_qcd_rejection_at_50pct_signal"]
                - baseline["macro_mean_log_qcd_rejection_at_50pct_signal"]
            ),
            "incremental_auc_vs_logit": row["macro_ovr_auc"] - logit["macro_ovr_auc"],
        })
    return with_content_hash({
        "contract": AGGREGATE_CONTRACT, "schema_version": 1,
        "parents": {"campaign_spec": spec["content_hash"], **parents},
        "rows": rows, "paired_hlt_comparisons": comparisons,
        "primary_question": "does direct offline supervision improve exact-HLT students",
        "representation_question": "does RSET or RREL add value beyond paired logit KD",
        "finite_poor_results_retained": True,
        "selection_role": "validation", "final_test_accessed": False,
    })


def build_completion(
    spec: Mapping[str, Any], *, aggregate: Mapping[str, Any],
    cleanup: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate_hash = validate_content_hash(
        aggregate, expected_contract=AGGREGATE_CONTRACT, expected_schema_version=1,
    )
    cleanup_hash = validate_content_hash(
        cleanup, expected_contract="HCWDL_DIRECT_OFFLINE_KD_TARGET_CLEANUP/v1",
        expected_schema_version=1,
    )
    return with_content_hash({
        "contract": COMPLETION_CONTRACT, "schema_version": 1,
        "parents": {"campaign_spec": spec["content_hash"],
                    "aggregate": aggregate_hash, "target_cleanup": cleanup_hash},
        "fit_count": 5, "mode": "pilot300k",
        "scientific_result_does_not_control_completion": True,
        "deployable_models": ["HLT_CE", "HLT_LOGIT", "HLT_RSET", "HLT_RREL"],
        "offline_teacher_deployable": False, "final_test_accessed": False,
    })


__all__ = [
    "AGGREGATE_CONTRACT", "COMPLETION_CONTRACT", "build_aggregate",
    "build_completion",
]
