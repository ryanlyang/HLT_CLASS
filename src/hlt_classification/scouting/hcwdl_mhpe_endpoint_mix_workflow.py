"""Workflow and reporting for the MHPE endpoint-mixture add-on."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, validate_content_hash, with_content_hash, write_immutable_json
from .engine import validate_pmard_training_report
from .hcwdl_mhpe_endpoint_mix import (
    AGGREGATE_CONTRACT, COMPLETION_CONTRACT, NODES, campaign_tasks,
    validate_campaign,
)
from .hcwdl_mhpe_contracts import stage_report_contract
from .hcwdl_mhpe_endpoint_mix_runner import build_targets, train_node


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("validation")
    if not isinstance(value, Mapping):
        raise ValueError("endpoint-mixture training report lacks validation metrics")
    return value


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, verify_source_tree=False)
    root = Path(spec["campaign_root"]); rows = []
    for node_id in NODES:
        report = load_json(root / "training" / node_id / "training_report.json")
        report_hash = validate_pmard_training_report(report)
        rows.append({
            "node_id": node_id, "teacher_mixture": NODES[node_id].payload(),
            "report_sha256": report_hash,
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "metrics": dict(_metrics(report)),
        })
    baseline = rows[0]["metrics"]
    comparisons = []
    for row in rows:
        metrics = row["metrics"]
        comparisons.append({
            "node_id": row["node_id"], "reference": "M1_D0only",
            "delta_macro_ovr_auc": float(metrics["macro_ovr_auc"]) - float(baseline["macro_ovr_auc"]),
            "delta_cross_entropy": float(metrics["cross_entropy"]) - float(baseline["cross_entropy"]),
            "delta_macro_mean_log_qcd_rejection_at_50pct_signal": (
                float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
                - float(baseline["macro_mean_log_qcd_rejection_at_50pct_signal"])
            ),
        })
    source_root = Path(spec["source"]["source_root"])
    source_context = []
    for node_id, path in (
        ("source_M1", source_root / "training/M1/training_report.json"),
        ("M0paired", Path(spec["source"]["foundation_root"]) / "training/M0paired/training_report.json"),
    ):
        report = load_json(path); report_hash = validate_pmard_training_report(report)
        source_context.append({"node_id": node_id, "report_sha256": report_hash, "metrics": dict(_metrics(report))})
    stage = load_json(source_root / "reports/D0E_stage.json")
    validate_content_hash(
        stage, expected_contract=stage_report_contract(spec["source"]["source_profile"]),
        expected_schema_version=1,
    )
    source_context.append({"node_id": "source_D0E", "report_sha256": stage["content_hash"], "metrics": dict(stage["ensemble_metrics"])})
    payload = with_content_hash({
        "contract": AGGREGATE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "rows": rows,
        "comparisons": comparisons, "source_context": source_context,
        "primary_comparison": "M1_mix90_minus_M1_D0only",
        "fresh_fit_count": 4, "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    })
    return payload


class EndpointMixWorkflow:
    def __init__(self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None) -> None:
        validate_campaign(spec, verify_source_tree=recovery_spec_sha256 is None)
        self.spec = spec; self.root = Path(spec["campaign_root"]); self.recovery_spec_sha256 = recovery_spec_sha256

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        task = next((row for row in campaign_tasks() if row["task_id"] == task_id), None)
        if task is None:
            raise ValueError("unknown endpoint-mixture task")
        if task["kind"] == "targets":
            return build_targets(spec=self.spec, device=device)
        if task["kind"] == "train":
            started = time.monotonic()
            result = train_node(
                spec=self.spec, node_id=task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
            )
            runtime = with_content_hash({
                "contract": "HCWDL_MHPE_ENDPOINT_MIX_RUNTIME/v1", "schema_version": 1,
                "node_id": task["node_id"], "elapsed_seconds": time.monotonic() - started,
                "cache_array_bytes": result.pop("_cache_array_bytes"),
                "final_test_accessed": False,
            })
            write_immutable_json(self.root / "reports/runtime" / f"{task['node_id']}.json", runtime)
            return result
        if task["kind"] == "aggregate":
            result = build_aggregate(self.spec)
            write_immutable_json(self.root / "reports/validation_aggregate.json", result)
            return result
        aggregate = load_json(self.root / "reports/validation_aggregate.json")
        aggregate_hash = validate_content_hash(aggregate, expected_contract=AGGREGATE_CONTRACT, expected_schema_version=1)
        if aggregate.get("campaign_spec_sha256") != self.spec["content_hash"] or len(aggregate.get("rows", ())) != 4:
            raise ValueError("endpoint-mixture aggregate differs")
        result = with_content_hash({
            "contract": COMPLETION_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": self.spec["content_hash"],
            "aggregate_sha256": aggregate_hash, "fresh_fit_count": 4,
            "scientific_result_does_not_control_completion": True,
            "final_test_accessed": False,
        })
        write_immutable_json(self.root / "reports/campaign_complete.json", result); return result


__all__ = ["EndpointMixWorkflow", "build_aggregate"]
