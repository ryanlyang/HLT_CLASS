from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash
from hlt_classification.scouting import hcwdl_tri60_d000_long180 as study
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import NODE_REGISTRY
from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
    Tri60TrainingRuntime,
)
from hlt_classification.scouting.hcwdl_tri60_d000_long180_contracts import (
    COMPARISON_CONTRACT, SOURCE_LOCK_CONTRACT, SPEC_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)


SHA = "a" * 64


def _source_lock(tmp_path: Path) -> dict:
    return artifact({
        "parents": {
            "source_campaign": "1" * 64, "source_graph": "2" * 64,
            "source_recipe": "3" * 64, "foundation_spec": "4" * 64,
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
            "teacher_probability_lock": str(tmp_path / "lock.json"),
            "teacher_train_manifest": str(tmp_path / "train.json"),
            "teacher_validation_manifest": str(tmp_path / "validation.json"),
            "teacher_stage": str(tmp_path / "stage.json"),
            "source_training_report": str(tmp_path / "source_report.json"),
        },
        "source_node_id": study.SOURCE_NODE_ID,
        "teacher_distribution_id": study.TEACHER_ID,
        "source_selected_pass": 60,
        "source_validation": {
            "macro_ovr_auc": .946764,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 8.22,
        },
        "replicate_seed": 11,
        "role_counts": {
            "train": 2777855, "validation": 957541, "final_test": 899779,
        },
        "source_campaign_completion_required": False,
        "source_scheduler_dependency": False,
        "source_outputs_read_only": True,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def test_node_changes_only_horizon_and_preserves_original_kd_edge():
    source = NODE_REGISTRY[study.SOURCE_NODE_ID]
    node = study.study_node()
    assert node.coordinate == source.coordinate
    assert node.distribution_teacher_id == source.distribution_teacher_id
    assert node.ce_weight == source.ce_weight == .25
    assert node.kd_weight == source.kd_weight == .75
    assert node.temperature == source.temperature == 2.0
    assert node.seed_alias == source.seed_alias
    assert source.training_passes == 60
    assert node.training_passes == 180
    assert node.batch_size == source.batch_size == 256


def test_additive_authority_allows_exact_180_budget_only():
    authority = study.training_authority("b" * 64)
    runtime = Tri60TrainingRuntime(passes=180, batch_size=256)
    runtime.validate(
        execution_mode="scientific",
        allowed_peak_learning_rates=authority.allowed_peak_learning_rates,
        allowed_training_passes=authority.allowed_training_passes,
        allowed_batch_sizes=authority.allowed_batch_sizes,
    )
    with pytest.raises(ValueError, match="pass/batch budget differs"):
        Tri60TrainingRuntime(passes=60, batch_size=256).validate(
            execution_mode="scientific",
            allowed_peak_learning_rates=authority.allowed_peak_learning_rates,
            allowed_training_passes=authority.allowed_training_passes,
            allowed_batch_sizes=authority.allowed_batch_sizes,
        )


def test_source_lock_binds_selected_endpoint_and_teacher_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source_root = tmp_path / "source"
    report_dir = source_root / "training" / study.SOURCE_NODE_ID
    report_dir.mkdir(parents=True)
    selected = report_dir / "selected.pt"
    selected.write_bytes(b"selected")
    report = with_content_hash({
        "contract": "SOURCE_REPORT/v1", "schema_version": 1,
        "node_id": study.SOURCE_NODE_ID,
        "node_spec": NODE_REGISTRY[study.SOURCE_NODE_ID].payload(),
        "campaign_spec_sha256": "1" * 64,
        "passes": 60, "validations": 60, "selected_pass": 60,
        "selected_checkpoint": selected.name,
        "selected_checkpoint_sha256": sha256_file(selected),
        "validation": {"macro_ovr_auc": .946764},
        "complete": True, "final_test_accessed": False,
    })
    (report_dir / "training_report.json").write_text(json.dumps(report))
    stage_path = source_root / "reports" / "stages" / f"{study.TEACHER_ID}.json"
    stage_path.parent.mkdir(parents=True)
    stage = with_content_hash({
        "contract": "SOURCE_STAGE/v1", "schema_version": 1,
        "distribution_id": study.TEACHER_ID,
        "parents": {"probability_lock": "5" * 64},
        "final_test_accessed": False,
    })
    stage_path.write_text(json.dumps(stage))
    foundation = tmp_path / "foundation.json"
    recipe = tmp_path / "recipe.json"
    foundation.write_text("{}")
    recipe.write_text("{}")
    source = {
        "content_hash": "1" * 64, "campaign_root": str(source_root),
        "parents": {"graph": "2" * 64},
        "artifact_paths": {
            "foundation_spec": str(foundation), "recipe": str(recipe),
        },
        "replicate_seed": 11,
        "role_counts": {
            "train": 2777855, "validation": 957541, "final_test": 899779,
        },
    }
    monkeypatch.setattr(study, "_source", lambda path: (source, "1" * 64))
    monkeypatch.setattr(
        study, "validate_source_artifact",
        lambda value, contract: value["content_hash"],
    )
    manifests = {
        role: {
            "content_hash": digit * 64, "temperature": 2.0,
        }
        for role, digit in (("train", "6"), ("validation", "7"))
    }
    monkeypatch.setattr(
        study, "validate_probability_lock",
        lambda path, distribution_id: ({
            "content_hash": "5" * 64,
            "parents": {"campaign_spec": "1" * 64},
        }, manifests),
    )
    monkeypatch.setattr(
        study, "validate_foundation_campaign", lambda *args, **kwargs: "4" * 64,
    )
    monkeypatch.setattr(study, "validate_recipe", lambda value: "3" * 64)
    _, lock = study._source_lock((tmp_path / "source.json").resolve())
    assert lock["source_selected_pass"] == 60
    assert lock["parents"]["teacher_probability_lock"] == "5" * 64
    assert lock["parents"]["teacher_train_manifest"] == "6" * 64
    assert lock["parents"]["source_training_report"] == report["content_hash"]


def test_campaign_is_one_isolated_three_day_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source_lock = _source_lock(tmp_path)
    source = {
        "content_hash": source_lock["parents"]["source_campaign"],
        "replicate_seed": source_lock["replicate_seed"],
        "role_counts": source_lock["role_counts"],
    }
    monkeypatch.setattr(study, "_source_lock", lambda path: (source, source_lock))
    root = tmp_path / "campaign"
    spec = study.create_campaign(
        source_campaign_spec=tmp_path / "source.json", campaign_root=root,
        project_dir=tmp_path / "project", source_commit="c" * 40,
        authorize_live_submission=True,
        authorization_phrase=study.CREATION_PHRASE,
    )
    assert study.validate_campaign(spec, executable=True) == spec["content_hash"]
    plan = json.loads((root / "command_plan.json").read_text())
    assert len(plan["commands"]) == 1
    command = plan["commands"][0]["command"]
    assert "--time=3-00:00:00" in command
    assert "--cpus-per-task=72" in command
    assert "--mem=320G" in command
    assert "--nice=10000" in command
    assert not any(item.startswith("--dependency=") for item in command)
    assert spec["source_campaign_outputs_mutated"] is False
    assert spec["teacher_probability_bank_copied"] is False
    assert spec["rolling_resume"] is False


def test_comparison_keeps_pass_60_120_180_and_global_selection(tmp_path: Path):
    root = tmp_path / "campaign"
    root.mkdir()
    source_lock = _source_lock(tmp_path)
    (root / "source_lock.json").write_text(json.dumps(source_lock))
    history = []
    for pass_index in range(1, 181):
        history.append({
            "update": pass_index * 10,
            "macro_ovr_auc": .94 + pass_index / 1_000_000,
            "cross_entropy": .7,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 8.0,
        })
    report = artifact({
        "campaign_spec_sha256": "d" * 64,
        "graph_sha256": "e" * 64, "recipe_sha256": "3" * 64,
        "node_id": study.NODE_ID, "node_spec": study.study_node().payload(),
        "passes": 180, "validations": 180, "updates": 1800,
        "selected_pass": 173, "selected_update": 1730,
        "validation": dict(history[172]), "validation_history": history,
        "complete": True, "rolling_resume_published": False,
        "partial_checkpoint_reuse": False, "final_test_accessed": False,
    }, contract=TRAINING_REPORT_CONTRACT)
    spec = {
        "content_hash": "d" * 64, "campaign_root": str(root),
        "artifact_paths": {"source_lock": str(root / "source_lock.json")},
    }
    comparison = study.build_comparison(spec, report=report)
    assert validate_artifact(comparison, contract=COMPARISON_CONTRACT)
    assert comparison["long180"]["selected_pass"] == 173
    assert [
        row["pass"] for row in comparison["long180"]["end_of_pass_metrics"]
    ] == [60, 120, 180]
    assert comparison["first_60_pass_schedule_equivalence_claimed"] is False


def test_worker_is_source_pinned_parallel_and_has_no_resume_bank():
    worker = Path("sbatch/run_hcwdl_tri60_d000_long180.sh").read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert '"${PROJECT_DIR}/scripts/run_hcwdl_tri60_d000_long180.py"' in worker
    assert "NUMEXPR_MAX_THREADS=64" in worker
    assert "final_test" not in worker
    assert "resume" not in worker.lower()
