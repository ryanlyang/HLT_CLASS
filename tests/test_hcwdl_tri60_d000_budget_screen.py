from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash
from hlt_classification.scouting import hcwdl_tri60_d000_budget_screen_campaign as campaign
from hlt_classification.scouting import hcwdl_tri60_d000_budget_screen_source as source_module
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import NODE_REGISTRY
from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
    Tri60TrainingRuntime, _learning_rate, tri60_learning_rate,
    tri60_learning_rate_schedule, tri60_loss_schedule, tri60_loss_weights,
)
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_contracts import (
    SOURCE_LOCK_CONTRACT, artifact,
)
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_graph import (
    CONDITIONS, CONDITION_REGISTRY, FIT_ORDER, GRAPH_SHA256,
    SOURCE_NODE_ID, TEACHER_ID, validate_graph,
)
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_runner import (
    training_authority,
)


def _source_lock(tmp_path: Path) -> dict:
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text("{}")
    return artifact({
        "parents": {
            "source_campaign": "1" * 64, "source_graph": "2" * 64,
            "foundation": "3" * 64, "recipe": "4" * 64,
            "teacher_probability_lock": "5" * 64,
            "teacher_train_manifest": "6" * 64,
            "teacher_validation_manifest": "7" * 64,
            "teacher_stage": "8" * 64,
            "source_training_report": "9" * 64,
            "source_selected_checkpoint": "a" * 64,
        },
        "artifact_paths": {
            "source_campaign_spec": str(tmp_path / "source.json"),
            "foundation_spec": str(tmp_path / "foundation.json"),
            "recipe": str(tmp_path / "recipe.json"),
            "endpoint_resource_lock": str(endpoint),
            "teacher_probability_lock": str(tmp_path / "lock.json"),
            "teacher_train_manifest": str(tmp_path / "train.json"),
            "teacher_validation_manifest": str(tmp_path / "validation.json"),
            "teacher_stage_report": str(tmp_path / "stage.json"),
            "source_training_report": str(tmp_path / "report.json"),
        },
        "source_node_id": SOURCE_NODE_ID,
        "teacher_distribution_id": TEACHER_ID,
        "source_selected_pass": 60,
        "source_validation": {
            "macro_ovr_auc": .946764, "accuracy": .806579,
            "cross_entropy": .7,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 8.22,
        },
        "replicate_seed": 11,
        "role_counts": {
            "train": 2777855, "validation": 957541, "final_test": 899779,
        },
        "required_source_tasks_only": [
            "train_LOGIT_D000_from_D033E", "reduce_LOGIT_D033E",
        ],
        "source_campaign_completion_required": False,
        "source_scheduler_dependency": False,
        "source_outputs_read_only": True,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def test_graph_is_exact_seventeen_fit_paired_screen():
    assert validate_graph() == GRAPH_SHA256
    assert len(CONDITIONS) == len(FIT_ORDER) == 17
    assert set(CONDITION_REGISTRY) == set(FIT_ORDER)
    assert {row.passes for row in CONDITIONS} == {60, 90}
    assert {row.axis for row in CONDITIONS} == {
        "decay_shape", "decay_floor", "peak_lr", "kd_first",
        "horizon", "horizon_kd_first",
    }
    source = NODE_REGISTRY[SOURCE_NODE_ID]
    for condition in CONDITIONS:
        node = condition.node
        assert node.coordinate == source.coordinate
        assert node.distribution_teacher_id == source.distribution_teacher_id
        assert node.seed_alias == source.seed_alias
        assert (node.ce_weight, node.kd_weight, node.temperature) == (.25, .75, 2.0)
        assert node.batch_size == 256 and node.initialization == "fresh"


def test_every_condition_is_accepted_only_by_its_additive_authority():
    for condition in CONDITIONS:
        authority = training_authority(condition.condition_id)
        runtime = Tri60TrainingRuntime(
            passes=condition.passes, batch_size=256,
            peak_learning_rate=condition.peak_learning_rate,
        )
        runtime.validate(
            execution_mode="scientific",
            allowed_peak_learning_rates=authority.allowed_peak_learning_rates,
            allowed_training_passes=authority.allowed_training_passes,
            allowed_batch_sizes=authority.allowed_batch_sizes,
        )
        assert tri60_loss_schedule(
            condition.node, condition.loss_schedule,
        ) == dict(condition.loss_schedule)
        assert tri60_learning_rate_schedule(
            runtime, condition.learning_rate_schedule,
        ) == dict(condition.learning_rate_schedule)


def test_delayed_decay_reaches_peak_holds_and_finishes_at_floor():
    runtime = Tri60TrainingRuntime(passes=60, peak_learning_rate=3e-4)
    schedule = tri60_learning_rate_schedule(runtime, {
        "kind": "warmup_hold_cosine_v1", "warmup_passes": 3,
        "hold_through_pass": 30, "minimum_lr_fraction": .05,
    })
    kwargs = {"total_updates": 600, "updates_per_pass": 10, "schedule": schedule}
    assert tri60_learning_rate(runtime, update=29, **kwargs) == pytest.approx(3e-4)
    assert tri60_learning_rate(runtime, update=299, **kwargs) == pytest.approx(3e-4)
    assert tri60_learning_rate(runtime, update=599, **kwargs) == pytest.approx(1.5e-5)


def test_default_lr_schedule_is_bitwise_equivalent_to_established_function():
    runtime = Tri60TrainingRuntime(passes=60, peak_learning_rate=3e-4)
    schedule = tri60_learning_rate_schedule(runtime, None)
    total = 600
    for update in (0, 1, 29, 30, 31, 299, 598, 599):
        assert tri60_learning_rate(
            runtime, update=update, total_updates=total,
            updates_per_pass=10, schedule=schedule,
        ) == _learning_rate(runtime, update, total)


def test_kd_first_schedule_switches_to_original_c25p75_endpoint():
    condition = CONDITION_REGISTRY["P60_H30_KD90_P10"]
    schedule = tri60_loss_schedule(condition.node, condition.loss_schedule)
    assert tri60_loss_weights(schedule, effective_pass=10) == pytest.approx((.10, .90))
    assert tri60_loss_weights(schedule, effective_pass=10.001) == pytest.approx((.25, .75))
    assert tri60_loss_weights(schedule, effective_pass=60) == pytest.approx((.25, .75))


def test_piecewise_schedule_rejects_wrong_final_scientific_loss():
    condition = CONDITION_REGISTRY["P60_H30_KD90_P10"]
    broken = {
        "kind": "piecewise_constant_v1",
        "segments": [
            {"through_pass": 10, "ce_weight": .10, "kd_weight": .90},
            {"through_pass": 60, "ce_weight": .10, "kd_weight": .90},
        ],
    }
    with pytest.raises(ValueError, match="endpoint differs"):
        tri60_loss_schedule(condition.node, broken)


def test_campaign_has_isolated_parallel_fits_and_no_source_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    lock = _source_lock(tmp_path)
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: lock)
    monkeypatch.setattr(
        campaign, "validate_source_lock", lambda value: value["content_hash"],
    )
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        source_campaign_spec=tmp_path / "source.json", campaign_root=root,
        project_dir=tmp_path / "project", source_commit="c" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 21
    plan = json.loads((root / "command_plan.json").read_text())
    fits = [row for row in plan["commands"] if row["task_id"].startswith("train_")]
    assert len(fits) == 17
    assert all(row["dependencies"] == ["preflight"] for row in fits)
    joined = " ".join(fits[0]["command"])
    assert "--cpus-per-task=72" in joined
    assert "--mem=320G" in joined
    assert "--time=3-00:00:00" in joined
    assert "--nice=10000" in joined
    assert spec["source_campaign_scheduler_dependency"] is False
    assert spec["source_campaign_outputs_mutated"] is False
    assert spec["standalone_smoke_required"] is False


def test_source_lock_requires_original_selected_pass_60(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "source"
    report_dir = root / "training" / SOURCE_NODE_ID
    report_dir.mkdir(parents=True)
    selected = report_dir / "selected.pt"
    selected.write_bytes(b"selected")
    report = with_content_hash({
        "contract": "SOURCE_REPORT/v1", "schema_version": 1,
        "node_id": SOURCE_NODE_ID,
        "node_spec": NODE_REGISTRY[SOURCE_NODE_ID].payload(),
        "campaign_spec_sha256": "1" * 64, "graph_sha256": "2" * 64,
        "passes": 60, "validations": 60, "selected_pass": 59,
        "selected_checkpoint": selected.name,
        "selected_checkpoint_sha256": sha256_file(selected),
        "validation": {"macro_ovr_auc": .9467}, "complete": True,
        "rolling_resume_published": False, "partial_checkpoint_reuse": False,
        "final_test_accessed": False,
    })
    (report_dir / "training_report.json").write_text(json.dumps(report))
    source = {
        "content_hash": "1" * 64, "campaign_root": str(root),
        "parents": {"graph": "2" * 64},
        "artifact_paths": {
            "foundation_spec": str(tmp_path / "foundation.json"),
            "recipe": str(tmp_path / "recipe.json"),
            "endpoint_resource_lock": str(tmp_path / "endpoint.json"),
        },
        "replicate_seed": 11,
        "role_counts": {"train": 1, "validation": 1, "final_test": 1},
    }
    monkeypatch.setattr(source_module, "load_json", lambda path: (
        source if Path(path) == tmp_path / "source.json"
        else json.loads(Path(path).read_text())
    ))
    monkeypatch.setattr(
        source_module, "validate_source_campaign", lambda *args, **kwargs: "1" * 64,
    )
    monkeypatch.setattr(
        source_module, "validate_source_artifact",
        lambda value, contract: value["content_hash"],
    )
    with pytest.raises(ValueError, match="source report differs"):
        source_module.build_source_lock(tmp_path / "source.json")


def test_worker_is_source_pinned_and_has_no_resume_or_final_test():
    worker = Path("sbatch/run_hcwdl_tri60_d000_budget_screen_task.sh").read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert '"${PROJECT_DIR}/scripts/run_hcwdl_tri60_d000_budget_screen_task.py"' in worker
    assert "NUMEXPR_MAX_THREADS=64" in worker
    assert "resume" not in worker.lower()
    assert "final_test" not in worker
