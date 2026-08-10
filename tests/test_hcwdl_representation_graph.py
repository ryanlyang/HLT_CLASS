from __future__ import annotations

from dataclasses import replace
import json

import pytest

from hlt_classification.scouting.hcwdl_representation_graph import (
    ASCENT_GRAPH_SHA256,
    CONTROL_REGISTRY,
    CONTROL_REGISTRY_SHA256,
    DESCENT_LEVELS,
    NODE_REGISTRY,
    RREL_STRATEGY,
    RSET_STRATEGY,
    TERMINAL_STEP,
    TRACKED_LEVELS,
    ascent_graph_artifact,
    control_registry_artifact,
    validate_ascent_graph,
    validate_ascent_graph_artifact,
    validate_control_registry,
    validate_control_registry_artifact,
)


H = "a" * 64
G = "b" * 64


def test_exact_86_node_four_descent_registry_and_teacher_mapping():
    expected = {f"{strategy}_D100" for strategy in ("RSET", "RREL")}
    expected |= {
        f"{strategy}_D{level}{suffix}"
        for strategy in ("RSET", "RREL")
        for suffix in ("c", "w")
        for level in TRACKED_LEVELS
    }
    expected |= {
        f"{strategy}_M1{suffix}"
        for strategy in ("RSET", "RREL")
        for suffix in ("c", "w")
    }
    assert DESCENT_LEVELS == tuple(range(100, -1, -5))
    assert len(NODE_REGISTRY) == 86
    assert set(NODE_REGISTRY) == expected
    assert validate_ascent_graph() == ASCENT_GRAPH_SHA256

    for strategy, strategy_id in (("RSET", RSET_STRATEGY), ("RREL", RREL_STRATEGY)):
        root = NODE_REGISTRY[f"{strategy}_D100"]
        assert root.strategy == strategy_id
        assert root.track == "shared"
        assert root.rung == 0
        assert root.stage == "offline_to_d100"
        assert root.representation_logit_teacher == "TOFF"
        assert root.target_bank_identity == "TOFF"
        assert root.initialization == "fresh"
        assert root.initialization_parent is None

        for track, suffix in (("cold", "c"), ("warm", "w")):
            predecessor = root.node_id
            predecessor_domain = "d100"
            for step, level in enumerate(TRACKED_LEVELS, start=1):
                node = NODE_REGISTRY[f"{strategy}_D{level}{suffix}"]
                assert node.strategy == strategy_id
                assert node.track == track
                assert node.rung == step
                assert node.stage == "down"
                assert node.privilege_percent == level
                assert node.predecessor_logit_teacher is None
                assert node.representation_logit_teacher == predecessor
                assert node.representation_teacher_domain == predecessor_domain
                assert node.target_bank_identity == predecessor
                assert node.parent_counterpart == f"D{level}{suffix}"
                assert node.deployable is (level == 0)
                assert node.initialization == ("fresh" if track == "cold" else "warm")
                assert node.initialization_parent == (
                    None if track == "cold" else predecessor
                )
                predecessor = node.node_id
                predecessor_domain = "hlt" if level == 0 else f"d{level}"
            terminal = NODE_REGISTRY[f"{strategy}_M1{suffix}"]
            assert terminal.rung == TERMINAL_STEP
            assert terminal.stage == "terminal_m1"
            assert terminal.representation_logit_teacher == predecessor
            assert terminal.target_bank_identity == predecessor
            assert terminal.predecessor_logit_teacher is None
            assert terminal.initialization_parent == (
                None if track == "cold" else predecessor
            )
            assert terminal.deployable is True


def test_dense_descent_does_not_reinterpret_old_m5_controls():
    assert not CONTROL_REGISTRY
    assert validate_control_registry() == CONTROL_REGISTRY_SHA256


def test_target_bank_consumer_multiplicities_are_frozen():
    consumers: dict[str, list[str]] = {}
    for node in NODE_REGISTRY.values():
        consumers.setdefault(node.target_bank_identity, []).append(node.node_id)
    counts = {bank: len(rows) for bank, rows in consumers.items()}
    assert counts["TOFF"] == 2
    assert counts["RSET_D100"] == 2
    assert counts["RREL_D100"] == 2
    for strategy in ("RSET", "RREL"):
        for suffix in ("c", "w"):
            for level in range(95, 0, -5):
                assert counts[f"{strategy}_D{level}{suffix}"] == 1
            assert counts[f"{strategy}_D0{suffix}"] == 1


def test_graph_and_control_validation_reject_cross_branch_or_absorbed_control():
    graph = dict(NODE_REGISTRY)
    graph["RREL_D90c"] = replace(
        graph["RREL_D90c"], representation_logit_teacher="RSET_D95c",
    )
    with pytest.raises(ValueError, match="dense descent routing"):
        validate_ascent_graph(graph)

    graph = dict(NODE_REGISTRY)
    graph["RSET_M5c_JET_ONLY_REP"] = graph["RSET_M1c"]
    with pytest.raises(ValueError, match="exactly 86"):
        validate_ascent_graph(graph)

    with pytest.raises(ValueError, match="must be empty"):
        validate_control_registry({"legacy": object()})


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
