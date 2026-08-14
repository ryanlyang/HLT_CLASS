"""Paired validation reporting for the two homotopy representation tracks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, validate_content_hash

from .hcwdl_homotopy_representation_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT, TRAINING_REPORT_CONTRACT,
    FIT_COUNT, TARGET_BANK_COUNT, build_artifact, validate_artifact,
)
from .hcwdl_homotopy_representation_graph import (
    CONTROL_SUFFIXES, NODE_REGISTRY, STRATEGIES,
)


METRICS = (
    "cross_entropy", "accuracy", "macro_ovr_auc",
    "macro_mean_log_qcd_rejection_at_50pct_signal",
)


def _metrics(report: Mapping[str, Any]) -> dict[str, float]:
    value = report.get("validation")
    if not isinstance(value, Mapping):
        raise ValueError("HCWDL-U-RKD training report lacks validation metrics")
    result = {}
    for name in METRICS:
        number = float(value[name])
        if not math.isfinite(number):
            raise FloatingPointError(f"HCWDL-U-RKD metric is nonfinite: {name}")
        result[name] = number
    result["R50"] = math.exp(result["macro_mean_log_qcd_rejection_at_50pct_signal"])
    return result


def _recovery(
    value: Mapping[str, float], *, m0: Mapping[str, float],
    toff: Mapping[str, float],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in (*METRICS, "R50"):
        if name == "cross_entropy":
            numerator = m0[name] - value[name]
            denominator = m0[name] - toff[name]
        else:
            numerator = value[name] - m0[name]
            denominator = toff[name] - m0[name]
        result[name] = (
            None if abs(denominator) <= 1e-12 else numerator / denominator
        )
    return result


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(spec["campaign_root"])
    rows = {}
    report_parents = {}
    for node_id, node in NODE_REGISTRY.items():
        path = root / "training" / node.strategy / node_id / "combined_training_report.json"
        report = load_json(path)
        digest = validate_artifact(
            report, contract=TRAINING_REPORT_CONTRACT,
            required_parents=("campaign_spec", "engine_report"),
            required_fields=("node_id", "validation"),
        )
        if report["node_id"] != node_id:
            raise ValueError("HCWDL-U-RKD aggregate node identity differs")
        rows[node_id] = _metrics(report)
        report_parents[f"report_{node_id}"] = digest
    paired = []
    for suffix in CONTROL_SUFFIXES:
        rset = rows[f"F_RSET_{suffix}"]
        rrel = rows[f"F_RREL_{suffix}"]
        reference = spec["logit_control_reports"][suffix]
        logit_path = Path(reference["report_path"])
        if not logit_path.is_file():
            raise FileNotFoundError(
                f"paired U/J logit-only report is not ready: {logit_path}"
            )
        logit = load_json(logit_path)
        logit_hash = validate_content_hash(
            logit, expected_contract=str(logit["contract"]),
            expected_schema_version=int(logit["schema_version"]),
        )
        scientific = logit.get("scientific_config", {})
        node = scientific.get("node") if isinstance(scientific, Mapping) else None
        observed = node.get("node_id") if isinstance(node, Mapping) else logit.get("experiment_id")
        if observed != reference["expected_node_id"]:
            raise ValueError("paired U/J logit-only node identity differs")
        frozen_hash = reference.get("report_sha256")
        if frozen_hash is not None and logit_hash != frozen_hash:
            raise ValueError("paired logit-only report changed")
        logit_metrics = _metrics(logit)
        report_parents[f"logit_{suffix}"] = logit_hash
        paired.append({
            "coordinate": suffix,
            "logit_only": logit_metrics, "RSET": rset, "RREL": rrel,
            "RSET_minus_logit": {name: rset[name] - logit_metrics[name] for name in rset},
            "RREL_minus_logit": {name: rrel[name] - logit_metrics[name] for name in rrel},
            "RREL_minus_RSET": {name: rrel[name] - rset[name] for name in rset},
        })
    imported = spec["imported_controls"]
    baselines = {}
    for node_id in ("M0", "TOFF"):
        report = load_json(imported[node_id]["report_path"])
        digest = validate_content_hash(
            report, expected_contract=str(report["contract"]),
            expected_schema_version=int(report["schema_version"]),
        )
        if digest != imported[node_id]["report_sha256"]:
            raise ValueError("HCWDL-U-RKD imported baseline changed")
        baselines[node_id] = _metrics(report)
        report_parents[f"baseline_{node_id}"] = digest
    recovery = {
        node_id: _recovery(value, m0=baselines["M0"], toff=baselines["TOFF"])
        for node_id, value in rows.items()
    }
    for row in paired:
        row["recovery"] = {
            "logit_only": _recovery(
                row["logit_only"], m0=baselines["M0"], toff=baselines["TOFF"],
            ),
            "RSET": recovery[f"F_RSET_{row['coordinate']}"],
            "RREL": recovery[f"F_RREL_{row['coordinate']}"],
        }
    return build_artifact(
        AGGREGATE_CONTRACT,
        parents={"campaign_spec": spec["content_hash"], **report_parents},
        fit_count=FIT_COUNT, target_bank_count=TARGET_BANK_COUNT,
        terminal_candidates=["F_RSET_M1", "F_RREL_M1"],
        rows=rows, recovery=recovery, paired_comparisons=paired,
        baselines=baselines,
        primary_metric="macro_ovr_auc", scientific_pruning=False,
        validation_only=True,
    )


def build_campaign_complete(
    spec: Mapping[str, Any], aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate_hash = validate_artifact(
        aggregate, contract=AGGREGATE_CONTRACT,
        required_parents=("campaign_spec",),
    )
    return build_artifact(
        CAMPAIGN_COMPLETE_CONTRACT,
        parents={"campaign_spec": spec["content_hash"], "aggregate": aggregate_hash},
        mode=spec["mode"], fit_count=FIT_COUNT,
        target_bank_count=TARGET_BANK_COUNT,
        scientific_result_does_not_control_completion=True,
        final_test_task_registered=False, final_test_capability_issued=False,
        complete=True,
    )


__all__ = ["build_aggregate", "build_campaign_complete"]
