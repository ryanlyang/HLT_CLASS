"""HCWDL validation aggregation, frozen screen selection, and gap accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash

from .engine import validate_pmard_training_report
from .hcwdl_ladder import GRAPH_SHA256, NODE_REGISTRY
from .hcwdl_qualification import recovered_fraction
from .hcwdl_training import TRAINING_REPORT_CONTRACT


SCREEN_REPORT_CONTRACT: Final = "HCWDL_SCREEN_AGGREGATE/v1"
CONFIRMATION_REPORT_CONTRACT: Final = "HCWDL_CONFIRMATION_AGGREGATE/v1"
FINAL_REPORT_CONTRACT: Final = "HCWDL_FINAL_AGGREGATE/v1"


def result_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    metrics = row["validation"]
    auc = float(metrics["macro_ovr_auc"])
    ce = float(metrics["cross_entropy"])
    logr = float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
    if not all(math.isfinite(value) for value in (auc, ce, logr)):
        raise FloatingPointError("HCWDL selector input is nonfinite")
    return (-auc, ce, -logr, str(row["node_id"]))


def select_declared_candidate(
    rows: Sequence[Mapping[str, Any]], *, allowed_nodes: Sequence[str],
) -> dict[str, Any]:
    allowed = set(allowed_nodes)
    candidates = [dict(row) for row in rows if row.get("node_id") in allowed]
    if {row["node_id"] for row in candidates} != allowed:
        raise ValueError("HCWDL declared candidate table is incomplete")
    selected = min(candidates, key=result_key)
    return {
        "selected_node_id": selected["node_id"],
        "ordered_nodes": [row["node_id"] for row in sorted(candidates, key=result_key)],
        "comparison_hex": {
            row["node_id"]: {
                "macro_ovr_auc": float(row["validation"]["macro_ovr_auc"]).hex(),
                "cross_entropy": float(row["validation"]["cross_entropy"]).hex(),
                "logr50": float(row["validation"]["macro_mean_log_qcd_rejection_at_50pct_signal"]).hex(),
            } for row in candidates
        },
    }


def build_screen_aggregate(
    reports: Sequence[Mapping[str, Any]], *,
    node_reports: Sequence[Mapping[str, Any]], campaign_spec_sha256: str,
    recipe_sha256: str, assignment_lock_sha256: str,
) -> dict[str, Any]:
    engine_by_hash: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        digest = validate_pmard_training_report(report)
        if digest in engine_by_hash:
            raise ValueError("HCWDL screen aggregate repeats a PMARD engine report")
        engine_by_hash[digest] = report

    by_node: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for node_report in node_reports:
        validate_content_hash(
            node_report, expected_contract=TRAINING_REPORT_CONTRACT,
            expected_schema_version=1,
        )
        node_id = str(node_report.get("node_id"))
        if node_id in by_node:
            raise ValueError("HCWDL screen aggregate repeats a graph node")
        if node_id not in NODE_REGISTRY:
            raise ValueError("HCWDL screen aggregate contains an unknown graph node")
        if (
            node_report.get("graph_sha256") != GRAPH_SHA256
            or node_report.get("recipe_sha256") != recipe_sha256
            or node_report.get("complete") is not True
        ):
            raise ValueError("HCWDL node report lineage or completion differs")
        engine_hash = require_sha256(
            node_report.get("pmard_engine_report_sha256"),
            name=f"{node_id} PMARD engine report SHA-256",
        )
        engine_report = engine_by_hash.get(engine_hash)
        if engine_report is None:
            raise ValueError("HCWDL node report lacks its authenticated PMARD engine report")
        if (
            node_report.get("selected_checkpoint_sha256")
            != engine_report.get("selected_checkpoint_sha256")
            or node_report.get("final_checkpoint_sha256")
            != engine_report.get("final_checkpoint_sha256")
        ):
            raise ValueError("HCWDL node and PMARD checkpoint lineage differs")
        scientific = engine_report.get("scientific_config")
        if not isinstance(scientific, Mapping):
            raise ValueError("HCWDL PMARD engine report lacks scientific configuration")
        engine_node = scientific.get("node")
        if (
            scientific.get("campaign") != "HCWDL"
            or scientific.get("graph_sha256") != GRAPH_SHA256
            or scientific.get("recipe_sha256") != recipe_sha256
            or not isinstance(engine_node, Mapping)
            or engine_node.get("node_id") != node_id
        ):
            raise ValueError("HCWDL PMARD engine report node lineage differs")
        by_node[node_id] = (node_report, engine_report)

    if (
        set(by_node) != set(NODE_REGISTRY)
        or len(node_reports) != len(NODE_REGISTRY)
        or len(reports) != len(NODE_REGISTRY)
    ):
        raise ValueError("HCWDL screen aggregate requires every primary graph node")
    rows = []
    for node_id in sorted(by_node):
        node_report, engine_report = by_node[node_id]
        metrics = dict(engine_report["validation"])
        result_key({"node_id": node_id, "validation": metrics})
        rows.append({
            "node_id": node_id, "validation": metrics,
            "report_sha256": require_sha256(
                node_report["content_hash"], name=f"{node_id} report SHA-256",
            ),
            "checkpoint_sha256": require_sha256(
                node_report["selected_checkpoint_sha256"],
                name=f"{node_id} checkpoint SHA-256",
            ),
        })
    intermediate_c = select_declared_candidate(rows, allowed_nodes=[f"M{i}c" for i in range(2, 6)])
    intermediate_w = select_declared_candidate(rows, allowed_nodes=[f"M{i}w" for i in range(2, 6)])
    return with_content_hash({
        "contract": SCREEN_REPORT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign SHA-256"),
        "recipe_sha256": require_sha256(recipe_sha256, name="recipe SHA-256"),
        "assignment_lock_sha256": require_sha256(
            assignment_lock_sha256, name="assignment lock SHA-256",
        ),
        "graph_sha256": GRAPH_SHA256, "rows": rows,
        "selected_intermediate_cold": intermediate_c,
        "selected_intermediate_warm": intermediate_w,
        "all_registered_nodes_completed": True,
    })


def build_confirmation_registry(
    screen: Mapping[str, Any], *, seeds: Sequence[int],
    include_label_only_warm_continuation: bool = False,
) -> list[dict[str, Any]]:
    validate_content_hash(screen, expected_contract=SCREEN_REPORT_CONTRACT, expected_schema_version=1)
    if len(seeds) != 5 or len(set(map(int, seeds))) != 5:
        raise ValueError("HCWDL confirmation requires five distinct predeclared seeds")
    cold = str(screen["selected_intermediate_cold"]["selected_node_id"])
    warm = str(screen["selected_intermediate_warm"]["selected_node_id"])
    primary = ("M0", "D0c", "D0w", "M1c", "M1w", "M6c", "M6w", cold, warm)
    rows = []
    for seed in map(int, seeds):
        rows.extend({"node_id": node, "seed": seed, "kind": "primary"} for node in primary)
        rows.extend((
            {"node_id": "NULL_M1_SELF_KD", "seed": seed, "kind": "control"},
            {"node_id": "NULL_M6_PREDECESSOR_ONLY", "seed": seed, "kind": "control"},
        ))
        if include_label_only_warm_continuation:
            rows.append({
                "node_id": "NULL_WARM_LABEL_ONLY", "seed": seed, "kind": "control",
            })
    expected = 60 if include_label_only_warm_continuation else 55
    if len(rows) != expected:
        raise RuntimeError("HCWDL fixed confirmation registry cardinality differs")
    return rows


def metric_recovery_table(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {"M0", "D100", "TOFF", "M1c", "M1w", "M6c", "M6w"}
    if not required.issubset(rows):
        raise ValueError("HCWDL gap-recovery inputs are incomplete")
    output: dict[str, Any] = {}
    for metric, larger in (
        ("cross_entropy", False), ("macro_ovr_auc", True),
        ("macro_mean_log_qcd_rejection_at_50pct_signal", True),
    ):
        m0 = float(rows["M0"][metric]); upper = float(rows["TOFF"][metric])
        output[metric] = {
            name: recovered_fraction(float(rows[name][metric]), m0, upper, larger_is_better=larger)
            for name in ("D100", "M1c", "M1w", "M6c", "M6w")
        }
    return output


def build_final_report(
    *, campaign_spec_sha256: str, execution_lock_sha256: str,
    validation_metrics: Mapping[str, Mapping[str, Any]],
    test_metrics: Mapping[str, Mapping[str, Any]], report_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if not test_metrics or set(test_metrics) - set(validation_metrics):
        raise ValueError("HCWDL final-test table differs from frozen validation finalists")
    return with_content_hash({
        "contract": FINAL_REPORT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign SHA-256"),
        "execution_lock_sha256": require_sha256(execution_lock_sha256, name="execution SHA-256"),
        "graph_sha256": GRAPH_SHA256,
        "validation": {name: dict(value) for name, value in sorted(validation_metrics.items())},
        "final_test": {name: dict(value) for name, value in sorted(test_metrics.items())},
        "gap_recovery": metric_recovery_table(validation_metrics),
        "report_hashes": {
            name: require_sha256(value, name=f"final report parent {name}")
            for name, value in sorted(report_hashes.items())
        },
        "negative_gains_retained": True, "test_used_for_selection": False,
    })


__all__ = [
    "CONFIRMATION_REPORT_CONTRACT", "FINAL_REPORT_CONTRACT", "SCREEN_REPORT_CONTRACT",
    "build_confirmation_registry", "build_final_report", "build_screen_aggregate",
    "metric_recovery_table", "result_key", "select_declared_candidate",
]
