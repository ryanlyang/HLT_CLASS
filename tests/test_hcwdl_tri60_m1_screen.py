from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


SHA = "a" * 64


def test_graph_is_exact_twenty_condition_screen():
    from hlt_classification.scouting.hcwdl_tri60_m1_screen_graph import (
        CONDITIONS, CONDITION_REGISTRY, FIT_ORDER, GRAPH_SHA256,
        IMPORTED_CONTROL_ID, SEED_ALIAS, graph_payload, validate_graph,
    )

    assert validate_graph() == GRAPH_SHA256
    assert graph_payload()["condition_count"] == 20
    assert len(CONDITIONS) == len(FIT_ORDER) == len(CONDITION_REGISTRY) == 19
    assert IMPORTED_CONTROL_ID not in CONDITION_REGISTRY
    assert {row.node.seed_alias for row in CONDITIONS} == {SEED_ALIAS}
    assert {row.node.temperature for row in CONDITIONS} == {1.0, 2.0}
    assert {row.peak_learning_rate for row in CONDITIONS} == {3e-4, 1e-4, 5e-5}
    assert sum(row.initialization == "fresh" for row in CONDITIONS) == 9
    assert sum(row.initialization == "warm_selected_checkpoint" for row in CONDITIONS) == 9
    assert sum(row.initialization == "polish_selected_checkpoint" for row in CONDITIONS) == 1


def test_loss_schedule_is_exact_and_continuous():
    from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
        tri60_loss_schedule, tri60_loss_weights,
    )
    from hlt_classification.scouting.hcwdl_tri60_m1_screen_graph import (
        CONDITION_REGISTRY,
    )

    condition = CONDITION_REGISTRY["COLD_RAMP_C75P25_TO_C10P90_T2_LR1E4"]
    schedule = tri60_loss_schedule(condition.node, condition.loss_schedule)
    assert tri60_loss_weights(schedule, effective_pass=1) == (.75, .25)
    assert tri60_loss_weights(schedule, effective_pass=5) == (.75, .25)
    middle = tri60_loss_weights(schedule, effective_pass=10)
    assert middle == pytest.approx((.425, .575))
    assert tri60_loss_weights(schedule, effective_pass=15) == pytest.approx((.10, .90))
    assert tri60_loss_weights(schedule, effective_pass=60) == pytest.approx((.10, .90))


def test_temperature_two_is_log_probability_softening():
    from hlt_classification.scouting.hcwdl_tri60_m1_screen_targets import _soften

    probabilities = np.asarray([[.81, .09, .10]], dtype=np.float32)
    softened = _soften(probabilities, 2.0)
    expected = np.sqrt(probabilities.astype(np.float64))
    expected /= expected.sum(axis=1, keepdims=True)
    assert softened.dtype == np.float32
    assert softened == pytest.approx(expected.astype(np.float32), abs=1e-7)
    assert _soften(probabilities, 1.0) == pytest.approx(probabilities)


def test_training_authority_allows_only_predeclared_condition_recipe():
    from hlt_classification.scouting.hcwdl_tri60_m1_screen_graph import FIT_ORDER
    from hlt_classification.scouting.hcwdl_tri60_m1_screen_runner import training_authority

    for node_id in FIT_ORDER:
        authority = training_authority(node_id)
        authority.validate()
        assert authority.allowed_initializations == (authority.node.initialization,)
        assert len(authority.allowed_peak_learning_rates) == 1


def test_campaign_shape_is_standalone_low_priority_and_no_smoke(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_tri60_m1_screen_campaign as campaign
    from hlt_classification.scouting.hcwdl_tri60_m1_screen_contracts import (
        SOURCE_LOCK_CONTRACT, artifact,
    )
    from hlt_classification.scouting.hcwdl_tri60_m1_screen_graph import FIT_ORDER

    source = artifact({
        "parents": {
            "source_campaign": "1" * 64, "foundation": "2" * 64,
            "recipe": "3" * 64, "source_graph": "4" * 64,
        },
        "artifact_paths": {
            "source_campaign_spec": str(tmp_path / "source.json"),
            "foundation_spec": str(tmp_path / "foundation.json"),
            "recipe": str(tmp_path / "recipe.json"),
            "endpoint_resource_lock": str(tmp_path / "endpoint.json"),
            "teacher_probability_lock": str(tmp_path / "teacher_lock.json"),
            "teacher_train_manifest": str(tmp_path / "train_manifest.json"),
            "teacher_validation_manifest": str(tmp_path / "validation_manifest.json"),
            "teacher_stage_report": str(tmp_path / "stage.json"),
            "source_m1_report": str(tmp_path / "m1.json"),
            "warm_report": str(tmp_path / "warm.json"),
        },
        "role_counts": {"train": 2_777_855, "validation": 957_541, "final_test": 899_779},
        "replicate_seed": 1337, "source_campaign_completion_required": False,
        "source_scheduler_dependency": False, "source_outputs_read_only": True,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda _path: source)
    monkeypatch.setattr(
        campaign, "validate_source_lock", lambda value: value["content_hash"],
    )
    spec = campaign.create_campaign(
        source_campaign_spec=tmp_path / "source.json",
        campaign_root=tmp_path / "screen", project_dir=tmp_path,
        source_commit="b" * 40, publish=True,
    )
    assert campaign.validate_campaign(spec) == spec["content_hash"]
    assert len(spec["tasks"]) == 23
    assert spec["fresh_fit_count"] == 19
    assert spec["condition_count"] == 20
    assert spec["standalone_smoke_required"] is False
    assert spec["source_campaign_scheduler_dependency"] is False
    assert spec["source_campaign_completion_required"] is False
    assert spec["scheduler_nice"] == 5000
    commands = campaign._command_plan(spec)["commands"]
    train = [row for row in commands if row["task_id"].startswith("train_")]
    assert len(train) == len(FIT_ORDER) == 19
    assert all("--nice=5000" in row["command"] for row in commands)
    assert all(row["dependencies"] == ["preflight"] for row in train)
    assert all(not row.get("subject_dependencies") for row in commands)
    from hlt_classification.scouting.hcwdl_recovery import (
        build_submission_ledger, validate_submission_ledger,
    )
    command_registry = {
        row["task_id"]: list(row["command"]) for row in commands
    }
    dry = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task_id: "1" for task_id in command_registry},
        commands=command_registry, dry_run=True,
    )
    assert validate_submission_ledger(dry) == dry["content_hash"]
    assert len(dry["jobs"]) == 23
