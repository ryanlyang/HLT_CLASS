from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash, write_immutable_json
from hlt_classification.scouting import hcwdl_mhpe_d000_schedule_screen as screen
from hlt_classification.scouting import hcwdl_mhpe_d000_schedule_screen_runner as runner
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen_recovery import failed_downstream_closure
from hlt_classification.scouting.hcwdl_mhpe_graph import PROFILE_C25P75


def _source(tmp_path: Path):
    return {
        "source_spec_path": str(tmp_path / "source/campaign_spec.json"),
        "source_spec_sha256": "1" * 64, "source_root": str(tmp_path / "source"),
        "source_profile": PROFILE_C25P75,
        "source_reuse_lock_sha256": "2" * 64,
        "foundation_root": str(tmp_path / "foundation"),
        "foundation_spec_path": str(tmp_path / "foundation/campaign_spec.json"),
        "foundation_spec_sha256": "3" * 64,
        "foundation_lock_path": str(tmp_path / "foundation/locks/foundation.json"),
        "foundation_lock_sha256": "4" * 64,
        "role_counts": {"train": 2_600_000, "validation": 1_000_000, "final_test": 1_000_000},
        "split_manifest_sha256": "5" * 64,
        "selection_manifest_path": str(tmp_path / "selection.json"),
        "selection_manifest_sha256": "6" * 64,
        "validation_assignment_manifest_path": str(tmp_path / "assignments/manifest.json"),
        "validation_assignment_manifest_sha256": "7" * 64,
        "source_recipe_path": str(tmp_path / "foundation/recipe.json"),
        "source_recipe_sha256": "8" * 64,
        "source_readiness": with_content_hash({
            "contract": screen.SOURCE_READINESS_CONTRACT, "schema_version": 1,
            "source_campaign_spec_sha256": "1" * 64,
            "source_campaign_completion_required": False,
            "required_products_complete": True, "final_test_accessed": False,
        }),
        "source_campaign_completion_required": False,
        "teacher_reports": {
            "U000": {"report_path": "u000.json", "report_sha256": "a" * 64,
                     "checkpoint_sha256": "b" * 64},
        },
        "teacher_targets": {
            "U000": {"path": "u000-target.json", "sha256": "c" * 64},
            "U100E": {"path": "u100-target.json", "sha256": "d" * 64, "lock_sha256": "e" * 64},
            "D066E": {"path": "d066-target.json", "sha256": "f" * 64, "lock_sha256": "0" * 64},
            "D033E": {"path": "d033-target.json", "sha256": "1" * 64, "lock_sha256": "2" * 64},
        },
        "teacher_target_locks": {"U100E": "e" * 64, "D066E": "0" * 64, "D033E": "2" * 64},
        "required_products_complete": True, "final_test_accessed": False,
    }


def _partition():
    return with_content_hash({
        "contract": screen.VALIDATION_PARTITION_CONTRACT, "schema_version": 1,
        "source_selection_manifest_sha256": "6" * 64,
        "source_validation_assignment_manifest_sha256": "7" * 64,
        "split_manifest_sha256": "5" * 64, "source_validation_rows": 10,
        "partition_rule": "test", "partition_seed": screen.VALIDATION_PARTITION_SEED,
        "subsets": {
            "checkpoint": {"role": "validation", "rows": 5, "sources": [], "identity_set_sha256": "3" * 64},
            "scoring": {"role": "validation", "rows": 5, "sources": [], "identity_set_sha256": "4" * 64},
        },
        "disjoint": True, "complete_source_validation_coverage": True,
        "labels_read": False, "final_test_accessed": False,
    })


def test_exact_24_fit_graph_and_horizon_registry():
    assert screen.LR_GRID == (3e-4, 2e-4, 1.5e-4, 1e-4, 7.5e-5, 5e-5)
    assert screen.TEACHERS == ("U000", "U100E", "D066E", "D033E")
    assert screen.HORIZON_PASSES == (20, 40, 60, 80)
    assert len(screen.NODES) == 24
    assert len(screen.campaign_tasks()) == 26
    assert all(node.payload()["student_coordinate"] == "D000" for node in screen.NODES.values())
    assert all(node.payload()["input_domain"] == "hlt" for node in screen.NODES.values())
    assert all(node.payload()["seed_alias"] == screen.SHARED_SEED_ALIAS for node in screen.NODES.values())
    assert screen.graph_payload()["heldout_evaluation_count"] == 96


def test_source_authentication_requires_only_four_ready_products(tmp_path, monkeypatch):
    base = _source(tmp_path)
    base.pop("source_campaign_completion_required"); base.pop("teacher_target_locks")
    base["teacher_targets"] = {
        "U000": base["teacher_targets"]["U000"],
        "U050": {"path": "unused.json", "sha256": "d" * 64},
        "U100E": base["teacher_targets"]["U100E"],
    }
    base["teacher_reports"]["U050"] = {
        "report_path": "unused.json", "report_sha256": "e" * 64,
        "checkpoint_sha256": "f" * 64,
    }
    monkeypatch.setattr(screen, "authenticate_full_source", lambda _path: base)
    calls = []
    def bundle(path, *, ensemble_id, temperature, consumers, profile):
        calls.append((ensemble_id, tuple(consumers), profile, temperature))
        return ensemble_id.lower().ljust(64, "0")[:64], {
            "train": {"content_hash": (ensemble_id + "train").lower().ljust(64, "0")[:64]},
        }
    monkeypatch.setattr(screen, "validate_probability_bundle", bundle)
    value = screen.authenticate_source(tmp_path / "source/campaign_spec.json")
    assert tuple(value["teacher_targets"]) == screen.TEACHERS
    assert [row[0] for row in calls] == ["U100E", "D066E", "D033E"]
    assert value["source_campaign_completion_required"] is False
    assert value["source_readiness"]["required_products_complete"] is True
    assert value["source_readiness"]["required_teacher_targets"] == {
        teacher: value["teacher_targets"][teacher]["sha256"]
        for teacher in screen.TEACHERS
    }
    def missing_d033(path, *, ensemble_id, temperature, consumers, profile):
        if ensemble_id == "D033E":
            raise ValueError("missing D033E target bundle")
        return bundle(
            path, ensemble_id=ensemble_id, temperature=temperature,
            consumers=consumers, profile=profile,
        )
    monkeypatch.setattr(screen, "validate_probability_bundle", missing_d033)
    with pytest.raises(ValueError, match="missing D033E"):
        screen.authenticate_source(tmp_path / "source/campaign_spec.json")


def test_campaign_creation_and_validation_are_24_fit_full_data(tmp_path, monkeypatch):
    source = _source(tmp_path); partition = _partition()
    Path(source["selection_manifest_path"]).write_text("{}", encoding="utf-8")
    Path(source["validation_assignment_manifest_path"]).parent.mkdir(parents=True)
    Path(source["validation_assignment_manifest_path"]).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(screen, "authenticate_source", lambda _path: source)
    monkeypatch.setattr(screen, "validation_partition_payload", lambda *_args, **_kwargs: partition)
    monkeypatch.setattr(screen, "validate_validation_partition", lambda value: value["content_hash"])
    project = Path(__file__).resolve().parents[1]
    spec = screen.create_campaign(
        source_campaign_spec=tmp_path / "source/campaign_spec.json",
        campaign_root=tmp_path / "campaign", project_dir=project,
        source_commit="0" * 40, authorize_live_submission=True,
        authorization_phrase=screen.CREATION_PHRASE,
        authorize_waiver=True, waiver_phrase=screen.WAIVER_PHRASE,
    )
    assert spec["fit_count"] == 24 and spec["heldout_evaluation_count"] == 96
    assert spec["resources"]["gpu"] == {
        "cpus": 8, "memory": "96G", "walltime": "72:00:00", "gpu": "gpu:gh200:1",
    }
    assert screen.validate_campaign(spec, executable=False) == spec["content_hash"]


def test_aggregate_reports_all_96_horizons_and_teacher_contrasts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_campaign", lambda *_args, **_kwargs: "0" * 64)
    monkeypatch.setattr(runner, "validate_pmard_training_report", lambda value: value["content_hash"])
    spec = {
        "campaign_root": str(tmp_path), "content_hash": "1" * 64,
        "graph_sha256": screen.GRAPH_SHA256, "recipe_sha256": "2" * 64,
    }
    metrics = {
        "cross_entropy": .7, "accuracy": .8, "balanced_accuracy": .7,
        "macro_ovr_auc": .94,
        "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0,
    }
    offsets = {"U000": 0.0, "U100E": .001, "D066E": .002, "D033E": .003}
    for node_id, node in screen.NODES.items():
        directory = tmp_path / "training" / node_id
        directory.mkdir(parents=True, exist_ok=True)
        engine_horizons = []
        for horizon in screen.HORIZON_PASSES:
            checkpoint = directory / f"h{horizon}.pt"
            checkpoint.write_bytes(f"{node_id}/{horizon}".encode())
            engine_horizons.append({
                "horizon_update": horizon, "selected_update": horizon,
                "validation": metrics, "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
            })
        engine = with_content_hash({
            "validation": metrics,
            "selection_horizon_checkpoints": engine_horizons,
        })
        manifest = with_content_hash({
            "contract": screen.HORIZON_CHECKPOINTS_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
            "rows": [
                {"horizon_pass": passes, **row}
                for passes, row in zip(screen.HORIZON_PASSES, engine_horizons, strict=True)
            ],
            "one_80pass_trajectory": True,
            "shorter_schedule_equivalence_claimed": False,
            "final_test_accessed": False,
        })
        horizons = []
        for horizon in screen.HORIZON_PASSES:
            scoring = dict(metrics); scoring["macro_ovr_auc"] += offsets[node.teacher_id]
            horizons.append({
                "horizon_pass": horizon, "horizon_update": horizon,
                "selected_update": horizon,
                "checkpoint_sha256": engine_horizons[
                    screen.HORIZON_PASSES.index(horizon)
                ]["checkpoint_sha256"],
                "checkpoint_validation_metrics": metrics,
                "schedule_scoring_metrics": scoring,
            })
        wrapper = with_content_hash({
            "contract": screen.TRAINING_REPORT_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"],
            "graph_sha256": screen.GRAPH_SHA256, "recipe_sha256": spec["recipe_sha256"],
            "node_id": node_id, "schedule_id": node.schedule_id,
            "teacher_id": node.teacher_id, "node": node.payload(),
            "pmard_engine_report_sha256": engine["content_hash"],
            "horizon_checkpoints_sha256": manifest["content_hash"],
            "horizons": horizons, "schedule_scoring_used_for_checkpoint_selection": False,
            "final_test_accessed": False,
        })
        runtime = with_content_hash({
            "contract": screen.RUNTIME_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
            "one_scoring_cache_reused_for_four_horizons": True,
            "final_test_accessed": False,
        })
        write_immutable_json(directory / "training_report.json", engine)
        write_immutable_json(directory / "horizon_checkpoints.json", manifest)
        write_immutable_json(directory / "screen_training_report.json", wrapper)
        write_immutable_json(tmp_path / "reports/runtime" / f"{node_id}.json", runtime)
    aggregate = runner.build_aggregate(spec)
    assert len(aggregate["rows"]) == 96 and len(aggregate["cells"]) == 24
    assert all(cell["contrasts"]["D033E_minus_U000"] == pytest.approx(.003) for cell in aggregate["cells"])
    assert aggregate["schedule_scoring_used_for_checkpoint_selection"] is False
    first = next(iter(screen.NODES))
    (tmp_path / "training" / first / "h20.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint authentication"):
        runner.build_aggregate(spec)


def test_failed_closure_preserves_completed_independent_fits():
    failed = f"train_{next(iter(screen.NODES))}"
    closure = failed_downstream_closure([failed])
    assert closure == (failed, "aggregate", "campaign_complete")


@pytest.mark.parametrize("script", (
    "create_hcwdl_mhpe_d000_schedule_screen.py",
    "run_hcwdl_mhpe_d000_schedule_screen_task.py",
    "submit_hcwdl_mhpe_d000_schedule_screen.py",
    "monitor_hcwdl_mhpe_d000_schedule_screen.py",
    "cancel_hcwdl_mhpe_d000_schedule_screen.py",
    "create_hcwdl_mhpe_d000_schedule_screen_recovery.py",
    "run_hcwdl_mhpe_d000_schedule_screen_recovery_task.py",
    "submit_hcwdl_mhpe_d000_schedule_screen_recovery.py",
))
def test_cli_help(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
