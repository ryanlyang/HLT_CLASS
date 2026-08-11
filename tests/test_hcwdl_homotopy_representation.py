from __future__ import annotations

from dataclasses import asdict

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import (
    SMOKE_RESOURCES, SUBMISSION_PHRASE, _task_registry, build_command_plan,
    materialize_command, submit_command_plan,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import (
    CAMPAIGN_SPEC_CONTRACT, FIT_COUNT, ROLE_COUNTS, SMOKE_ROLE_COUNTS,
    TARGET_BANK_COUNT,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_graph import (
    GRAPH_SHA256, NODE_REGISTRY, STRATEGIES, ordered_nodes, resolved_base_loss,
    target_bank_registry, validate_graph,
)
from hlt_classification.scouting.hcwdl_representation_graph import RREL_STRATEGY


def test_exact_two_track_graph_and_loss_routing():
    assert validate_graph() == GRAPH_SHA256
    assert len(NODE_REGISTRY) == FIT_COUNT == 42
    assert len(target_bank_registry()) == TARGET_BANK_COUNT == 41
    assert target_bank_registry()["TOFF"] == ("F_RREL_U010", "F_RSET_U010")
    for strategy in STRATEGIES:
        rows = ordered_nodes(strategy)
        assert len(rows) == 21
        assert [row.transition_index for row in rows] == list(range(1, 22))
        assert rows[0].teacher.node_id == "TOFF"
        assert rows[-1].student_domain == "hlt"
        assert rows[-1].temperature == 1.0
        assert all(row.initialization == "fresh" for row in rows)
        for parent, child in zip(rows, rows[1:]):
            assert child.teacher.node_id == parent.node_id
        for row in rows:
            loss = resolved_base_loss(row.node_id)
            assert loss.ce == pytest.approx(0.25)
            assert loss.hlt_kd == pytest.approx(0.75)
            assert loss.temperature == pytest.approx(row.temperature)
            assert (row.strategy == RREL_STRATEGY) == row.node_id.startswith("F_RREL_")


def test_exact_87_task_parallel_sequential_dag():
    tasks = _task_registry()
    assert len(tasks) == 87
    by_id = {row["task_id"]: row for row in tasks}
    assert by_id["train_F_RSET_U010"]["dependencies"] == ["target_TOFF"]
    assert by_id["train_F_RREL_U010"]["dependencies"] == ["target_TOFF"]
    assert by_id["target_F_RSET_U010"]["dependencies"] == ["train_F_RSET_U010"]
    assert by_id["train_F_RSET_U020"]["dependencies"] == ["target_F_RSET_U010"]
    assert set(by_id["aggregate"]["dependencies"]) == {
        "train_F_RSET_M1", "train_F_RREL_M1",
    }
    assert not any("final" in row["kind"] for row in tasks)


def test_command_plan_uses_locked_tigris_envelope_and_exact_dependencies(tmp_path):
    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(tmp_path / "project"), "source_commit": "a" * 40,
        "parent_homotopy_spec_sha256": "b" * 64,
        "graph_sha256": GRAPH_SHA256, "combined_recipe_sha256": "c" * 64,
        "resources": SMOKE_RESOURCES, "tasks": _task_registry(),
        "final_test_accessed": False,
    })
    plan = build_command_plan(spec)
    assert len(plan["commands"]) == 87
    training = next(row for row in plan["commands"] if row["task_id"] == "train_F_RSET_U010")
    assert "--cpus-per-task=8" in training["command"]
    assert "--mem=96G" in training["command"]
    assert "--time=06:00:00" in training["command"]
    assert "--gres=gpu:gh200:1" in training["command"]
    assert "--signal=B:USR1@120" in training["command"]
    materialized = materialize_command(training, {"target_TOFF": "12345"})
    assert "--dependency=afterok:12345" in materialized
    assert not any("${JOB_" in token for token in materialized)


def test_role_counts_keep_final_test_sealed():
    assert ROLE_COUNTS == {"train": 300_000, "validation": 100_000, "final_test": 0}
    assert SMOKE_ROLE_COUNTS == {"train": 4096, "validation": 4096, "final_test": 0}


def test_submission_journal_resumes_exact_completed_prefix(monkeypatch, tmp_path):
    import hlt_classification.scouting.hcwdl_homotopy_representation_campaign as module

    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(tmp_path / "project"), "source_commit": "a" * 40,
        "parent_homotopy_spec_sha256": "b" * 64,
        "graph_sha256": GRAPH_SHA256, "combined_recipe_sha256": "c" * 64,
        "resources": SMOKE_RESOURCES, "tasks": _task_registry(),
        "final_test_accessed": False,
    })
    plan = build_command_plan(spec)
    monkeypatch.setattr(
        module, "validate_campaign", lambda *_args, **_kwargs: spec["content_hash"],
    )
    events = []
    calls = []

    def interrupted(command):
        calls.append(command)
        if len(calls) == 3:
            raise RuntimeError("scheduler unavailable")
        return str(9000 + len(calls))

    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        submit_command_plan(
            spec=spec, command_plan=plan, scheduler=interrupted,
            authorization_phrase=SUBMISSION_PHRASE, event_writer=events.append,
        )
    assert [row["task_id"] for row in events] == ["authenticate", "graph_recipe_lock"]

    resumed_calls = []
    ledger = submit_command_plan(
        spec=spec, command_plan=plan,
        scheduler=lambda command: resumed_calls.append(command) or str(9100 + len(resumed_calls)),
        authorization_phrase=SUBMISSION_PHRASE, event_writer=events.append,
        prior_events=events,
    )
    assert len(resumed_calls) == 85
    assert ledger["jobs"]["authenticate"] == "9001"
    assert ledger["jobs"]["graph_recipe_lock"] == "9002"
    assert len(ledger["jobs"]) == 87
