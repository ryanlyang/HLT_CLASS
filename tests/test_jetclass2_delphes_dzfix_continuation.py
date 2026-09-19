"""Synthetic tests for the deferred dz-fix salience continuation."""
from pathlib import Path

from test_jetclass2_delphes import snapshot  # noqa: F401
from test_jetclass2_delphes_production import campaign  # noqa: F401

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.contracts import artifact
from hlt_classification.jetclass2_delphes import readiness
from hlt_classification.jetclass2_delphes import dzfix_salience_continuation as continuation
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger


def _live_ledger(identity, jobs):
    commands = {task: ["sbatch", task] for task in jobs}
    return build_submission_ledger(
        campaign_spec_sha256=identity, jobs=jobs, commands=commands,
        dry_run=False,
    )


def test_continuation_is_bound_to_recovered_profile_and_stops_before_production(
    campaign, tmp_path, monkeypatch,
):
    monkeypatch.setattr(readiness, "_source", lambda *args: None)
    ready = readiness.create_readiness(
        inventory=campaign["foundation"]["inventory"],
        split_profile=campaign["foundation"]["splits"],
        data_root=Path(campaign["data_root"]), output_root=tmp_path / "ready",
        project=tmp_path / "old_project", source_commit="d" * 40,
        memory_mb=4096, workers=1,
    )
    original = _live_ledger(
        ready["content_hash"],
        {"sample": "101", "assign": "102", "lock": "103", "profile": "104"},
    )
    recovery = artifact(
        "READINESS_RECOVERY",
        source_readiness_sha256=ready["content_hash"],
        source_ledger_sha256=original["content_hash"],
        source_commit=ready["source_commit"],
        project_dir=ready["project_dir"],
        recovery_root=str((tmp_path / "recovery").resolve()),
        foundation_root=ready["foundation_root"],
        evidence_root=ready["evidence_root"], data_root=ready["data_root"],
        completed_assignment_shards_reused=0,
        retry_array_indices=[1], retry_assignment_minutes=480,
        superseded_jobs={"assign": "102", "lock": "103", "profile": "104"},
        scientific_fits=0, final_test_accessed=False,
    )
    recovery_ledger = _live_ledger(
        recovery["content_hash"],
        {"assign": "201", "lock": "202", "profile": "203"},
    )
    recovery_spec_path = tmp_path / "recovery/recovery_spec.json"
    recovery_ledger_path = tmp_path / "recovery/submission_ledger.json"
    write_immutable_json(recovery_spec_path, recovery)
    write_immutable_json(recovery_ledger_path, recovery_ledger)

    monkeypatch.setattr(continuation, "_source", lambda *args: None)
    spec = continuation.create_continuation(
        readiness_spec_path=tmp_path / "ready/readiness_spec.json",
        readiness_recovery_spec_path=recovery_spec_path,
        readiness_recovery_ledger_path=recovery_ledger_path,
        output_root=tmp_path / "continuation", project=tmp_path / "new_project",
        source_commit="e" * 40,
    )
    plan = load_json(
        tmp_path / "continuation/launchers/after_profile/command_plan.json"
    )
    command = plan["commands"][0]["command"]
    assert plan["dependencies"] == ["203"]
    assert "--dependency=afterok:203" in command
    assert spec["live_production"] is False
    assert spec["production_dry_run"] is True
    assert spec["production_branches"] == ["DIRECT", "COARSE", "DENSE"]
    assert spec["ultradense_present"] is False
    assert not (tmp_path / "continuation/production").exists()
    continuation.validate_continuation(spec)


def test_nonlegacy_screen_profile_requires_deep_foundation_validation(
    campaign, tmp_path,
):
    from hlt_classification.jetclass2_delphes.salience_screen import _template

    path = tmp_path / "runtime_profile.json"
    write_immutable_json(path, campaign["runtime_profile"])
    assert _template(path, campaign["foundation"]) == campaign["runtime_profile"]
