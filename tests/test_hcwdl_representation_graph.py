from __future__ import annotations

from dataclasses import replace
import json

import pytest

from hlt_classification.scouting.hcwdl_representation_graph import (
    ASCENT_GRAPH_SHA256,
    CONTROL_REGISTRY,
    CONTROL_REGISTRY_SHA256,
    NODE_REGISTRY,
    RREL_STRATEGY,
    RSET_STRATEGY,
    ascent_graph_artifact,
    control_registry_artifact,
    validate_ascent_graph,
    validate_ascent_graph_artifact,
    validate_control_registry,
    validate_control_registry_artifact,
)


H = "a" * 64
G = "b" * 64


def test_exact_24_node_four_ascent_registry_and_teacher_mapping():
    expected = {
        f"{strategy}_M{rung}{suffix}"
        for strategy in ("RSET", "RREL")
        for suffix in ("c", "w")
        for rung in range(1, 7)
    }
    assert len(NODE_REGISTRY) == 24
    assert set(NODE_REGISTRY) == expected
    assert validate_ascent_graph() == ASCENT_GRAPH_SHA256

    privileged = {
        1: "D0", 2: "D25", 3: "D50", 4: "D75", 5: "D100", 6: "TOFF",
    }
    domains = {1: "hlt", 2: "d25", 3: "d50", 4: "d75", 5: "d100", 6: "toff"}
    for node in NODE_REGISTRY.values():
        prefix = "RSET" if node.strategy == RSET_STRATEGY else "RREL"
        suffix = "c" if node.track == "cold" else "w"
        expected_teacher = privileged[node.rung]
        if node.rung <= 4:
            expected_teacher += suffix
        expected_predecessor = (
            None if node.rung == 1 else f"{prefix}_M{node.rung - 1}{suffix}"
        )
        assert node.student_domain == "hlt"
        assert node.deployable is True
        assert node.parent_counterpart == f"M{node.rung}{suffix}"
        assert node.predecessor_logit_teacher == expected_predecessor
        assert node.representation_logit_teacher == expected_teacher
        assert node.representation_teacher_domain == domains[node.rung]
        assert node.target_bank_identity == expected_teacher
        if node.track == "cold":
            assert node.initialization == "fresh"
            assert node.initialization_parent is None
        elif node.rung == 1:
            assert node.initialization_parent == f"D0{suffix}"
        else:
            assert node.initialization_parent == expected_predecessor


def test_exact_four_controls_are_separate_terminal_validation_rows():
    assert set(CONTROL_REGISTRY) == {
        "RSET_M5c_JET_ONLY_REP",
        "RREL_M5c_NO_REL_REP",
        "RSET_M5c_WITHIN_CLASS_SHUFFLED_REP",
        "RREL_M5c_WITHIN_CLASS_SHUFFLED_REP",
    }
    assert not set(CONTROL_REGISTRY) & set(NODE_REGISTRY)
    assert validate_control_registry() == CONTROL_REGISTRY_SHA256

    expected_allocations = {
        "RSET_M5c_JET_ONLY_REP": {"jet": 1.0, "set": 0.0, "relation": 0.0},
        "RREL_M5c_NO_REL_REP": {"jet": 0.4, "set": 0.6, "relation": 0.0},
        "RSET_M5c_WITHIN_CLASS_SHUFFLED_REP": {
            "jet": 0.4, "set": 0.6, "relation": 0.0,
        },
        "RREL_M5c_WITHIN_CLASS_SHUFFLED_REP": {
            "jet": 0.3, "set": 0.45, "relation": 0.25,
        },
    }
    for control_id, control in CONTROL_REGISTRY.items():
        assert control.rung == 5
        assert control.track == "cold"
        assert control.predecessor_logit_teacher in {"RSET_M4c", "RREL_M4c"}
        assert control.representation_logit_teacher == "D100"
        assert control.target_bank_identity == "D100"
        assert control.parent_counterpart == "M5c"
        assert control.disposition == "validation_only_terminal"
        assert control.descendants == ()
        assert control.deployable is True
        assert control.finalist_eligible is False
        assert control.confirmation_eligible is False
        assert dict(control.component_allocation) == expected_allocations[control_id]
        assert control.shuffled_representation_targets == ("SHUFFLED" in control_id)


def test_target_bank_consumer_multiplicities_are_frozen():
    consumers: dict[str, list[str]] = {}
    for node in NODE_REGISTRY.values():
        consumers.setdefault(node.target_bank_identity, []).append(node.node_id)
    for control in CONTROL_REGISTRY.values():
        consumers.setdefault(control.target_bank_identity, []).append(control.control_id)

    assert {bank: len(rows) for bank, rows in consumers.items()} == {
        "D0c": 2,
        "D0w": 2,
        "D25c": 2,
        "D25w": 2,
        "D50c": 2,
        "D50w": 2,
        "D75c": 2,
        "D75w": 2,
        "D100": 8,
        "TOFF": 4,
    }


def test_graph_and_control_validation_reject_cross_branch_or_absorbed_control():
    graph = dict(NODE_REGISTRY)
    graph["RREL_M3c"] = replace(
        graph["RREL_M3c"], predecessor_logit_teacher="RSET_M2c",
    )
    with pytest.raises(ValueError, match="same-branch predecessor"):
        validate_ascent_graph(graph)

    graph = dict(NODE_REGISTRY)
    graph["RSET_M5c_JET_ONLY_REP"] = graph["RSET_M5c"]
    with pytest.raises(ValueError, match="exactly 24"):
        validate_ascent_graph(graph)

    controls = dict(CONTROL_REGISTRY)
    controls["RREL_M5c_NO_REL_REP"] = replace(
        controls["RREL_M5c_NO_REL_REP"],
        predecessor_logit_teacher="RSET_M4c",
    )
    with pytest.raises(ValueError, match="lineage differs"):
        validate_control_registry(controls)


def test_graph_and_control_artifacts_bind_parent_lineage_and_exact_payloads():
    graph = ascent_graph_artifact(parents={"parent_graph": H, "parent_import": G})
    assert validate_ascent_graph_artifact(
        graph, expected_parents={"parent_graph": H, "parent_import": G},
    ) == graph["content_hash"]
    control = control_registry_artifact(
        ascent_graph_artifact_sha256=graph["content_hash"],
    )
    assert validate_control_registry_artifact(
        control, ascent_graph_artifact_sha256=graph["content_hash"],
    ) == control["content_hash"]
    round_tripped = json.loads(json.dumps(control))
    assert validate_control_registry_artifact(
        round_tripped, ascent_graph_artifact_sha256=graph["content_hash"],
    ) == control["content_hash"]

    with pytest.raises(ValueError, match="parent lineage"):
        validate_ascent_graph_artifact(
            graph, expected_parents={"parent_graph": G, "parent_import": H},
        )
