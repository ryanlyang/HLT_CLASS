from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_campaign import create_campaign_spec


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "validate_highcov_resources.py", "build_highcov_assignment_shard.py",
    "finalize_highcov_assignments.py", "audit_highcov_assignments.py",
    "build_hcwdl_recipe.py", "build_hcwdl_row_selection.py",
    "create_hcwdl_campaign.py",
    "submit_hcwdl_campaign.py", "monitor_hcwdl_campaign.py",
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
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_hcwdl_cli_has_working_help(script: str):
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / script), "--help"],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


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
    assert "--hold" in ledger["commands"]["shell_endpoint_qualification_lock"]
    assert spec["role_counts"] == {"train": 300_000, "validation": 100_000, "final_test": 100_000}
    assert spec["source_manifest_sha256"] == "a" * 64
    assert spec["split_manifest_sha256"] == "b" * 64
    assert "sbatch" in result.stdout


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
