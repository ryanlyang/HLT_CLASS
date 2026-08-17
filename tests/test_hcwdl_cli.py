from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.scouting.hcwdl_campaign import create_campaign_spec
from hlt_classification.scouting.hcwdl_recipe import example_recipe, validate_recipe
from hlt_classification.scouting.selective_assignment import (
    ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "validate_highcov_resources.py", "build_highcov_assignment_shard.py",
    "finalize_highcov_assignments.py", "audit_highcov_assignments.py",
    "build_hcwdl_recipe.py", "build_hcwdl_row_selection.py",
    "create_hcwdl_campaign.py",
    "submit_hcwdl_campaign.py", "continue_hcwdl_campaign.py",
    "monitor_hcwdl_campaign.py",
    "resume_hcwdl_campaign.py", "cancel_hcwdl_campaign.py",
    "train_hcwdl_node.py", "train_hcwdl_qualifier.py", "train_hcwdl_control.py",
    "select_hcwdl_checkpoint.py", "run_hcwdl_cache_miniature.py",
    "evaluate_hcwdl_final.py", "run_hcwdl_task.py",
    "aggregate_hcwdl_campaign.py", "dry_run_hcwdl_campaign.py",
    "run_hcwdl_local_smoke.py",
    "build_hcwdl_resource_profile.py",
    "build_hcwdl_submission_authorization.py",
    "build_hcwdl_endpoint_ack.py",
    "assemble_hcwdl_submission_ledger.py",
    "create_hcwdl_dense_pilot.py", "run_hcwdl_dense_task.py",
    "submit_hcwdl_dense_pilot.py",
    "create_hcwdl_dense_recovery.py", "run_hcwdl_dense_recovery_task.py",
    "submit_hcwdl_dense_recovery.py", "create_hcwdl_dense_reschedule.py",
    "create_hcwdl_campaign_recovery.py",
    "run_hcwdl_campaign_recovery_task.py",
    "submit_hcwdl_campaign_recovery.py",
    "create_hcwdl_final_recovery.py", "run_hcwdl_final_recovery_task.py",
    "submit_hcwdl_final_recovery.py",
    "create_hcwdl_homotopy_pilot.py", "submit_hcwdl_homotopy_pilot.py",
    "run_hcwdl_homotopy_task.py", "monitor_hcwdl_homotopy.py",
    "cancel_hcwdl_homotopy.py", "resume_hcwdl_homotopy.py",
    "run_hcwdl_homotopy_recovery_task.py",
    "build_hcwdl_homotopy_resource_profile.py",
    "build_hcwdl_homotopy_operational_waiver.py",
    "validate_hcwdl_homotopy_weaver.py",
    "run_hcwdl_homotopy_local_smoke.py",
    "build_hcwdl_upper_calibration.py",
    "build_hcwdl_upper_coupling_shard.py",
    "finalize_hcwdl_upper_coupling.py", "build_hcwdl_toff_targets.py",
    "create_hcwdl_architecture_factorial.py",
    "submit_hcwdl_architecture_factorial.py",
    "run_hcwdl_architecture_factorial_task.py",
    "create_hcwdl_mhpe_refined_campaign.py",
    "run_hcwdl_mhpe_refined_task.py",
    "submit_hcwdl_mhpe_refined_campaign.py",
    "monitor_hcwdl_mhpe_refined.py",
    "create_hcwdl_mhpe_refined_recovery.py",
    "run_hcwdl_mhpe_refined_recovery_task.py",
    "submit_hcwdl_mhpe_refined_recovery.py",
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_hcwdl_cli_has_working_help(script: str):
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / script), "--help"],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_recipe_cli_publishes_authenticated_unweighted_ce(tmp_path: Path):
    raw = example_recipe()
    payload = {
        key: value for key, value in raw.items()
        if key not in {
            "contract", "schema_version", "authorized_for_execution", "content_hash",
            "class_weighting", "class_weights",
        }
    }
    payload["recipe_profile"] = "primary_ladder"
    payload["purpose"] = "hcwdl_primary_ladder"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    selection = with_content_hash({
        "contract": ROW_SELECTION_CONTRACT,
        "schema_version": ROW_SELECTION_VERSION,
        "split_manifest_sha256": "a" * 64,
        "seed": 1337,
        "roles": {
            "train": {
                "all_rows": False, "rows": 120,
                "class_counts": list(range(1, 16)),
                "population_class_counts": [2] * 15,
                "sources": [{"path": "fixture.root", "rows": 120,
                             "entries": list(range(120))}],
            },
        },
        "selection_rule": "per_class_smallest_identity_sha256_rank_v1",
        "access_lock_sha256": {},
    })
    selection_path = tmp_path / "selection.json"
    output = tmp_path / "recipe.json"
    write_immutable_json(selection_path, selection)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/build_hcwdl_recipe.py"),
         "--payload", str(payload_path), "--train-row-selection", str(selection_path),
         "--output", str(output), "--authorize"],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    recipe = load_json(output)
    validate_recipe(recipe, expected_profile="primary_ladder")
    assert recipe["class_weighting"]["train_row_selection_sha256"] == selection["content_hash"]
    assert recipe["class_weighting"]["policy"] == "unweighted_per_jet_population_mean_v1"
    assert recipe["class_weights"] == [1.0] * 15


def test_complete_pilot_dry_run_is_nonmutating_and_exact(tmp_path: Path):
    spec = create_campaign_spec(
        mode="pilot", campaign_root="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_pilot_dry",
        source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64,
        source_commit="c" * 40,
        role_source_counts={"train": 42, "validation": 14, "final_test": 14},
        recipe_sha256=None, recipe_path=None, planning_only=True,
        source_manifest_path="/future/source_manifest.json",
        split_manifest_path="/future/split_manifest.json",
        data_root="/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train",
    )
    spec_path = tmp_path / "campaign_spec.json"; ledger_path = tmp_path / "dry_run.json"
    write_immutable_json(spec_path, spec)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/dry_run_hcwdl_campaign.py"),
         "--campaign-spec", str(spec_path), "--output", str(ledger_path)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    ledger = load_json(ledger_path)
    assert ledger["dry_run"] is True
    assert set(ledger["jobs"]) == {row["task_id"] for row in spec["tasks"]}
    assert "--array=0-41" in ledger["commands"]["assign_train"]
    assert "--array=0-13" in ledger["commands"]["assign_validation"]
    assert "--array=0-13" in ledger["commands"]["assign_test"]
    assert all("%" not in argument for command in ledger["commands"].values() for argument in command)
    assert "--hold" not in ledger["commands"]["shell_endpoint_qualification_lock"]
    assert spec["role_counts"] == {"train": 300_000, "validation": 100_000, "final_test": 100_000}
    assert spec["source_manifest_sha256"] == "a" * 64
    assert spec["split_manifest_sha256"] == "b" * 64
    assert "sbatch" in result.stdout

    phase_path = tmp_path / "qualification_phase.json"
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/submit_hcwdl_campaign.py"),
         "--campaign-spec", str(spec_path), "--output", str(phase_path)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    phase = load_json(phase_path)
    assert "endpoint_qualification" in phase["jobs"]
    assert "shell_endpoint_qualification_lock" not in phase["jobs"]
    assert not ({f"train_{node}" for node in (
        "M0", "D100", "TOFF",
    )} & set(phase["jobs"]))


def test_midscale500k_cli_dry_run_locks_fixed_population(tmp_path: Path):
    spec = create_campaign_spec(
        mode="midscale500k",
        campaign_root="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_midscale500k_dry",
        source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64,
        source_commit="c" * 40,
        role_source_counts={"train": 42, "validation": 14, "final_test": 14},
        recipe_sha256=None, recipe_path=None, planning_only=True,
        source_manifest_path="/future/source_manifest.json",
        split_manifest_path="/future/split_manifest.json",
        data_root="/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train",
    )
    spec_path = tmp_path / "midscale_spec.json"
    ledger_path = tmp_path / "midscale_dry_run.json"
    write_immutable_json(spec_path, spec)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/dry_run_hcwdl_campaign.py"),
         "--campaign-spec", str(spec_path), "--output", str(ledger_path)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert spec["role_counts"] == {
        "train": 500_000, "validation": 250_000, "final_test": 250_000,
    }
    ledger = load_json(ledger_path)
    assert ledger["dry_run"] is True
    assert "--array=0-41" in ledger["commands"]["assign_train"]
    assert "--array=0-13" in ledger["commands"]["assign_validation"]
    assert "--array=0-13" in ledger["commands"]["assign_test"]


def test_midscale1m_cli_dry_run_locks_fixed_population(tmp_path: Path):
    spec = create_campaign_spec(
        mode="midscale1m",
        campaign_root="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_midscale1m_dry",
        source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64,
        source_commit="c" * 40,
        role_source_counts={"train": 42, "validation": 14, "final_test": 14},
        recipe_sha256=None, recipe_path=None, planning_only=True,
        source_manifest_path="/future/source_manifest.json",
        split_manifest_path="/future/split_manifest.json",
        data_root="/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train",
    )
    spec_path = tmp_path / "midscale1m_spec.json"
    ledger_path = tmp_path / "midscale1m_dry_run.json"
    write_immutable_json(spec_path, spec)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/dry_run_hcwdl_campaign.py"),
         "--campaign-spec", str(spec_path), "--output", str(ledger_path)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert spec["role_counts"] == {
        "train": 1_000_000, "validation": 400_000, "final_test": 400_000,
    }
    ledger = load_json(ledger_path)
    assert ledger["dry_run"] is True
    assert "--array=0-41" in ledger["commands"]["assign_train"]
    assert "--array=0-13" in ledger["commands"]["assign_validation"]
    assert "--array=0-13" in ledger["commands"]["assign_test"]


def test_midscale2m_cli_dry_run_locks_fixed_population(tmp_path: Path):
    spec = create_campaign_spec(
        mode="midscale2m",
        campaign_root="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_midscale2m_dry",
        source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64,
        source_commit="c" * 40,
        role_source_counts={"train": 42, "validation": 14, "final_test": 14},
        recipe_sha256=None, recipe_path=None, planning_only=True,
        source_manifest_path="/future/source_manifest.json",
        split_manifest_path="/future/split_manifest.json",
        data_root="/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train",
    )
    spec_path = tmp_path / "midscale2m_spec.json"
    ledger_path = tmp_path / "midscale2m_dry_run.json"
    write_immutable_json(spec_path, spec)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/dry_run_hcwdl_campaign.py"),
         "--campaign-spec", str(spec_path), "--output", str(ledger_path)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert spec["role_counts"] == {
        "train": 2_000_000, "validation": 500_000, "final_test": 500_000,
    }
    ledger = load_json(ledger_path)
    assert ledger["dry_run"] is True
    assert "--array=0-41" in ledger["commands"]["assign_train"]
    assert "--array=0-13" in ledger["commands"]["assign_validation"]
    assert "--array=0-13" in ledger["commands"]["assign_test"]


def test_production_worker_rejects_a_planning_spec_before_dispatch(tmp_path: Path):
    spec = create_campaign_spec(
        mode="smoke", campaign_root=tmp_path / "campaign",
        source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64,
        source_commit="c" * 40,
        role_source_counts={"train": 1, "validation": 1, "final_test": 1},
        recipe_sha256=None, recipe_path=None, planning_only=True,
        source_manifest_path=tmp_path / "source.json",
        split_manifest_path=tmp_path / "split.json", data_root=tmp_path / "data",
    )
    path = tmp_path / "planning.json"; write_immutable_json(path, spec)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/run_hcwdl_task.py"),
         "--campaign-spec", str(path), "--task", "source_audit"],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "planning-only" in result.stderr
