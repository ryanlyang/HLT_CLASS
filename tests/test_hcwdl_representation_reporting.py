from __future__ import annotations

import copy
import math

import pytest

from hlt_classification.data.cache_contracts import canonical_sha256

from hlt_classification.scouting.hcwdl_representation_reporting import (
    CONFIRMATION_METRICS,
    CONFIRMATION_SEEDS,
    build_confirmation_aggregate,
    build_confirmation_registry,
    derive_representation_execution_id,
    build_screen_aggregate,
    gap_recovery,
    select_checkpoint,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    CONTROL_REGISTRY,
    NODE_REGISTRY,
)


def _validation(value: float) -> dict[str, float]:
    return {
        "cross_entropy": 1.0 - value,
        "accuracy": value,
        "balanced_accuracy": value,
        "macro_ovr_auc": value,
        "macro_mean_log_qcd_rejection_at_50pct_signal": value,
        "multiclass_brier": 1.0 - value,
        "top_label_ece_15_bin": 1.0 - value,
    }


def test_checkpoint_selector_uses_macro_auc_then_frozen_tiebreakers() -> None:
    rows = [
        {"checkpoint_id": "b", "update": 2, "validation": _validation(.8)},
        {"checkpoint_id": "a", "update": 1, "validation": _validation(.8)},
        {"checkpoint_id": "c", "update": 3, "validation": _validation(.7)},
    ]
    assert select_checkpoint(rows)["selected_checkpoint_id"] == "a"


def test_confirmation_registry_and_ddof_one_aggregate() -> None:
    registry = build_confirmation_registry(
        screen_sha256="a" * 64,
        campaign_sha256="c" * 64,
        recipe_sha256="d" * 64,
        target_logical_bank_sha256="b" * 64,
        objectives=("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"),
    )
    assert registry["execution_count"] == 20
    assert tuple(registry["seeds"]) == CONFIRMATION_SEEDS == (11, 22, 33, 44, 55)
    assert len({row["execution_id"] for row in registry["rows"]}) == 20
    assert all(
        row["execution_id"] == canonical_sha256(row["execution_identity_payload"])
        for row in registry["rows"]
    )
    reports = []
    for row in registry["rows"]:
        value = 0.7 + row["seed"] / 10_000
        reports.append(
            {
                **row,
                "validation": _validation(value),
            }
        )
    aggregate = build_confirmation_aggregate(
        registry=registry,
        reports=reports,
        metric_names=CONFIRMATION_METRICS,
    )
    summary = aggregate["objectives"]["RSET_M6c"]["macro_ovr_auc"]
    assert len(summary["raw"]) == 5
    assert summary["student_t_df"] == 4
    assert summary["standard_error"] == pytest.approx(
        summary["standard_deviation_ddof1"] / math.sqrt(5.0)
    )
    assert aggregate["used_for_finalist_selection"] is False
    assert tuple(aggregate["paired_seed_contrasts"]) == (
        "RREL_M6c-minus-RSET_M6c",
        "RREL_M6w-minus-RSET_M6w",
        "RSET_M6w-minus-RSET_M6c",
        "RREL_M6w-minus-RREL_M6c",
    )
    paired_summary = aggregate["paired_seed_contrasts"][
        "RREL_M6c-minus-RSET_M6c"
    ]["macro_ovr_auc"]
    assert paired_summary["standard_error"] == pytest.approx(
        paired_summary["standard_deviation_ddof1"] / math.sqrt(5.0)
    )

    forged = copy.deepcopy(reports)
    forged[0]["seed"] = 999
    with pytest.raises(ValueError, match="identity lineage"):
        build_confirmation_aggregate(
            registry=registry, reports=forged,
            metric_names=CONFIRMATION_METRICS,
        )
    with pytest.raises(ValueError, match="metric registry"):
        build_confirmation_aggregate(
            registry=registry, reports=reports,
            metric_names=("macro_ovr_auc",),
        )


def test_execution_identity_binds_every_frozen_scientific_input() -> None:
    execution, payload = derive_representation_execution_id(
        campaign_sha256="1" * 64,
        strategy="HCWDL_REP_SET/v1",
        node_id="RSET_M1c",
        purpose="screen",
        seed=1337,
        initialization_parent=None,
        teacher="D0c",
        logical_target_bank_sha256="2" * 64,
        target_purpose="screen",
        recipe_sha256="3" * 64,
    )
    assert execution == canonical_sha256(payload)
    assert set(payload) == {
        "campaign", "strategy", "node_id", "purpose", "seed",
        "initialization_parent", "teacher", "logical_target_bank",
        "target_purpose", "recipe",
    }
    changed, _ = derive_representation_execution_id(
        campaign_sha256="1" * 64,
        strategy="HCWDL_REP_SET/v1",
        node_id="RSET_M1c",
        purpose="screen",
        seed=1338,
        initialization_parent=None,
        teacher="D0c",
        logical_target_bank_sha256="2" * 64,
        target_purpose="screen",
        recipe_sha256="3" * 64,
    )
    assert changed != execution


def test_screen_aggregate_contains_every_predeclared_ordered_comparison() -> None:
    graph = "4" * 64
    recipe = "5" * 64
    primaries = []
    for index, (node_id, spec) in enumerate(NODE_REGISTRY.items()):
        primaries.append({
            "node_id": node_id,
            "complete": True,
            "graph_sha256": graph,
            "recipe_sha256": recipe,
            "parent_counterpart": spec.parent_counterpart,
            "validation": _validation(0.70 + index / 10_000),
            "content_hash": f"{(index + 1) % 16:x}" * 64,
        })
    controls = []
    for index, (control_id, spec) in enumerate(CONTROL_REGISTRY.items()):
        controls.append({
            "node_id": control_id,
            "complete": True,
            "graph_sha256": graph,
            "recipe_sha256": recipe,
            "control_counterpart": spec.paired_primary_node,
            "validation": _validation(0.69 + index / 10_000),
            "content_hash": f"{(index + 9) % 16:x}" * 64,
        })
    parent_ids = {
        spec.parent_counterpart for spec in NODE_REGISTRY.values()
    } | {"M0", "D0c", "D0w", "D100", "TOFF"}
    parents = {
        node_id: {
            "validation": _validation(0.68),
            "content_hash": f"{(index + 3) % 16:x}" * 64,
        }
        for index, node_id in enumerate(sorted(parent_ids))
    }
    aggregate = build_screen_aggregate(
        primary_reports=primaries,
        control_reports=controls,
        parent_reports=parents,
        graph_sha256=graph,
        recipe_sha256=recipe,
        campaign_spec_sha256="6" * 64,
        expected_primary_ids=tuple(NODE_REGISTRY),
        expected_control_ids=tuple(CONTROL_REGISTRY),
    )
    assert len(aggregate["ordered_comparisons"]) == 80
    kinds = {row["kind"] for row in aggregate["ordered_comparisons"]}
    assert kinds == {
        "representation_minus_logit",
        "relation_package_minus_set_package",
        "warm_minus_cold",
        "registered_m5_control",
        "child_minus_same_branch_predecessor",
        "terminal_minus_first_rung",
        "terminal_minus_hlt_baseline",
    }
    assert {row["node_id"] for row in aggregate["parent_rows"]} == parent_ids
    assert len(aggregate["gap_recovery_tables"]) == 24 * 3
    assert all(
        set(row["metrics"]) == {
            "cross_entropy", "macro_ovr_auc",
            "macro_mean_log_qcd_rejection_at_50pct_signal",
        }
        for row in aggregate["gap_recovery_tables"]
    )
    registered = {
        row["right"]: row
        for row in aggregate["ordered_comparisons"]
        if row["kind"] == "registered_m5_control"
    }
    assert set(registered) == set(CONTROL_REGISTRY)
    for control_id, control in CONTROL_REGISTRY.items():
        row = registered[control_id]
        assert row["left"] == control.paired_primary_node
        assert row["comparison"] == (
            f"{control.paired_primary_node}-minus-{control_id}"
        )
        primary = next(
            item for item in primaries
            if item["node_id"] == control.paired_primary_node
        )
        ablation = next(
            item for item in controls if item["node_id"] == control_id
        )
        assert row["delta"]["macro_ovr_auc"] == pytest.approx(
            primary["validation"]["macro_ovr_auc"]
            - ablation["validation"]["macro_ovr_auc"]
        )
        assert row["delta"]["cross_entropy"] == pytest.approx(
            primary["validation"]["cross_entropy"]
            - ablation["validation"]["cross_entropy"]
        )


def test_gap_recovery_is_unclipped_and_undefined_for_wrong_oracle_direction() -> None:
    above = gap_recovery(
        student=1.2, lower=0.4, upper=1.0, higher_is_better=True,
    )
    assert above["value"] > 1.0
    undefined = gap_recovery(
        student=0.5, lower=0.4, upper=0.3, higher_is_better=True,
    )
    assert undefined["value"] is None
    assert undefined["reason"] == "nonpositive_declared_oracle_denominator"
