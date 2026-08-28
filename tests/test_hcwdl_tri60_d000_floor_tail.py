from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
    Tri60TrainingRuntime, tri60_early_stopping, tri60_learning_rate,
    tri60_learning_rate_schedule,
)
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_graph import (
    CONDITION_REGISTRY as REFERENCE_CONDITIONS,
)
from hlt_classification.scouting.hcwdl_tri60_d000_floor_tail_contracts import (
    REFERENCE_LOCK_CONTRACT, artifact,
)
from hlt_classification.scouting.hcwdl_tri60_d000_floor_tail_graph import (
    CONDITION_ID, EARLY_STOPPING, LOSS_SCHEDULE, LR_SCHEDULE, NODE,
    REFERENCE_CONDITION_ID, validate_graph,
)


def _reference_lock(tmp_path: Path):
    return artifact({
        "parents": {
            "reference_screen": "1" * 64,
            "reference_graph": "2" * 64,
            "source_lock": "3" * 64,
            "source_campaign": "4" * 64,
            "foundation": "5" * 64,
            "recipe": "6" * 64,
            "teacher_probability_lock": "7" * 64,
            "teacher_train_manifest": "8" * 64,
            "confirmation_graph": validate_graph(),
        },
        "artifact_paths": {
            "reference_screen_spec": str(tmp_path / "reference.json"),
            "reference_training_report": str(tmp_path / "reference-report.json"),
            "source_lock": str(tmp_path / "source-lock.json"),
            "foundation_spec": str(tmp_path / "foundation.json"),
            "recipe": str(tmp_path / "recipe.json"),
            "endpoint_resource_lock": str(tmp_path / "endpoint.json"),
            "teacher_probability_lock": str(tmp_path / "probability-lock.json"),
            "teacher_train_manifest": str(tmp_path / "train-manifest.json"),
            "teacher_validation_manifest": str(tmp_path / "validation-manifest.json"),
            "teacher_stage_report": str(tmp_path / "stage.json"),
            "source_training_report": str(tmp_path / "source-report.json"),
        },
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "reference_condition": REFERENCE_CONDITIONS[
            REFERENCE_CONDITION_ID
        ].payload(),
        "confirmation_condition_id": CONDITION_ID,
        "replicate_seed": 1234,
        "role_counts": {
            "train": 2_777_855,
            "validation": 957_541,
            "final_test": 899_779,
        },
        "reference_report_required_for_training": False,
        "source_outputs_read_only": True,
        "source_scheduler_dependency": False,
        "final_test_accessed": False,
    }, contract=REFERENCE_LOCK_CONTRACT)


def test_exact_match_changes_only_training_protocol():
    assert validate_graph()
    reference = REFERENCE_CONDITIONS[REFERENCE_CONDITION_ID]
    reference_node = reference.node
    scientific_fields = (
        "coordinate_name", "distribution_teacher_id",
        "distribution_teacher_kind", "representation_carrier_id",
        "auxiliary", "ce_weight", "kd_weight", "temperature",
        "seed_alias", "representation_seed_alias", "batch_size",
        "initialization",
    )
    assert all(
        getattr(NODE, name) == getattr(reference_node, name)
        for name in scientific_fields
    )
    assert reference.passes == 90
    assert NODE.training_passes == 100
    assert reference.peak_learning_rate == 3.0e-4
    assert dict(reference.loss_schedule) == dict(LOSS_SCHEDULE)
    assert reference.learning_rate_schedule == {
        "kind": "warmup_hold_cosine_v1", "warmup_passes": 3,
        "hold_through_pass": 45, "minimum_lr_fraction": .05,
    }


def test_floor_tail_schedule_and_early_stopping_are_exact():
    runtime = Tri60TrainingRuntime(passes=100, batch_size=256)
    schedule = tri60_learning_rate_schedule(runtime, dict(LR_SCHEDULE))
    rates = {
        update: tri60_learning_rate(
            runtime, update=update, total_updates=1000,
            updates_per_pass=10, schedule=schedule,
        )
        for update in (0, 29, 30, 449, 450, 599, 600, 999)
    }
    assert rates[0] == pytest.approx(1.0e-5)
    assert rates[29] == pytest.approx(3.0e-4)
    assert rates[30] == pytest.approx(3.0e-4)
    assert rates[449] == pytest.approx(3.0e-4)
    assert rates[450] == pytest.approx(3.0e-4)
    assert rates[599] == pytest.approx(1.5e-5)
    assert rates[600] == pytest.approx(1.5e-5)
    assert rates[999] == pytest.approx(1.5e-5)
    normalized = tri60_early_stopping(runtime, dict(EARLY_STOPPING))
    assert normalized == {
        "kind": "macro_auc_patience_v1", "minimum_passes": 60,
        "patience_passes": 15, "minimum_auc_delta": 5.0e-5,
        "patience_accumulates_before_minimum": True,
        "selected_checkpoint_uses_exact_metrics": True,
    }


def test_campaign_is_one_fit_isolated_and_lower_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import (
        hcwdl_tri60_d000_floor_tail_campaign as campaign,
    )

    reference = _reference_lock(tmp_path)
    monkeypatch.setattr(
        campaign, "build_reference_lock", lambda path: reference,
    )
    monkeypatch.setattr(
        campaign, "validate_reference_lock",
        lambda value: value["content_hash"],
    )
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        reference_screen_spec=tmp_path / "reference.json",
        campaign_root=root, project_dir=tmp_path / "project",
        source_commit="a" * 40, authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert spec["fresh_fit_count"] == 1
    assert spec["reference_report_required_for_training"] is False
    assert spec["source_campaign_scheduler_dependency"] is False
    assert spec["source_outputs_mutated"] is False
    assert spec["standalone_smoke_required"] is False
    assert spec["scheduler_nice"] == 10000
    assert [row["kind"] for row in spec["tasks"]] == [
        "authenticate", "preflight", "train", "campaign_complete",
    ]
    plan = json.loads((root / "command_plan.json").read_text())
    assert len(plan["commands"]) == 4
    train = next(
        row["command"] for row in plan["commands"]
        if row["task_id"] == f"train_{CONDITION_ID}"
    )
    assert "--cpus-per-task=72" in train
    assert "--mem=320G" in train
    assert "--time=3-00:00:00" in train
    assert "--gres=gpu:gh200:1" in train
    assert "--nice=10000" in train
    assert not any(item.startswith("--signal=") for item in train)
    assert not plan["source_scheduler_dependencies"]


def test_worker_is_source_pinned_single_gpu_and_has_no_resume():
    worker = Path(
        "sbatch/run_hcwdl_tri60_d000_floor_tail_task.sh"
    ).read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert '"${PROJECT_DIR}/scripts/run_hcwdl_tri60_d000_floor_tail_task.py"' in worker
    assert "PYTHONNOUSERSITE=1" in worker
    assert "final_test" not in worker
    assert "resume" not in worker.lower()
