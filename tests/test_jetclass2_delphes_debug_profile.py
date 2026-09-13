"""Synthetic profile-only continuation: never real Weaver/A100 evidence."""
from pathlib import Path
import json
import shutil

import pytest

from test_jetclass2_delphes import snapshot
from test_jetclass2_delphes_production import campaign, fake_slurm
from test_jetclass2_delphes_sporc import _allocation_fixture
from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash
from hlt_classification.jetclass2_delphes import execution, production, readiness, submission
from hlt_classification.jetclass2_delphes import profile_attempt as attempt


@pytest.fixture
def ready(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_source", lambda *args: None)
    monkeypatch.setattr(attempt, "_source", lambda *args: None)
    # Freeze the real local semantic hashes before fake_slurm replaces the
    # shared subprocess.run; array/lock/producer comparisons remain real.
    producer = production.assignment_source()
    monkeypatch.setattr(production, "assignment_source", lambda: producer)
    spec = readiness.create_readiness(inventory=campaign["foundation"]["inventory"],
                                     split_profile=campaign["foundation"]["splits"],
                                     data_root=Path(campaign["data_root"]), output_root=tmp_path / "ready",
                                     project=tmp_path / "old-project", source_commit="a" * 40)
    shutil.copytree(campaign["foundation_root"], spec["foundation_root"], dirs_exist_ok=True)
    return spec


def create(ready, tmp_path, **kwargs):
    return attempt.create_profile_attempt(
        readiness_spec=Path(ready["readiness_root"]) / "readiness_spec.json",
        output_root=tmp_path / "debug-attempt", project=tmp_path / "new-project",
        source_commit="d" * 40, **kwargs)


def inventory(root):
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_debug_is_explicit_and_does_not_weaken_tier3_allocation(monkeypatch):
    normal, fields = _allocation_fixture(monkeypatch)
    debug = execution.execution_site("sporc_a100_debug")
    assert execution.production_site(debug) == normal
    assert execution.production_site(normal) == normal
    assert "--partition=debug" in execution.slurm_options(debug)
    fields["Partition"] = "debug"
    assert execution.allocation(debug) == ("123", 8, 98304)
    with pytest.raises(PermissionError, match="scheduler"):
        execution.allocation(normal)
    fields["QOS"] = "qos_interactive"
    with pytest.raises(PermissionError, match="scheduler"):
        execution.allocation(debug)


def test_profile_only_dry_live_idempotence_and_read_only_preparation(ready, tmp_path, monkeypatch):
    old_root = Path(ready["readiness_root"])
    before = inventory(old_root)
    calls, _ = fake_slurm(monkeypatch)
    spec = create(ready, tmp_path)
    root = Path(spec["attempt_root"])
    plan = load_json(root / "command_plan.json")
    assert len(plan["commands"]) == 1
    row = plan["commands"][0]
    assert row["task_id"] == "profile" and row["dependencies"] == []
    assert "--partition=debug" in row["command"]
    assert "--qos=qos_tier3" in row["command"]
    assert "--mem=73728M" in row["command"] and "--cpus-per-task=8" in row["command"]
    assert "--gres=gpu:a100:1" in row["command"] and "--no-requeue" in row["command"]
    assert not any("--array" in x or "--dependency" in x for x in row["command"])
    assert calls == [] and plan["assignment_shards"] == plan["scientific_fits"] == 0
    assert not (root / "foundation").exists()
    assert not (root / "evidence").exists()
    with pytest.raises(PermissionError, match="profile-only"):
        attempt.submit_profile_attempt(spec, execute=True, authorization_phrase=production.AUTHORIZE)
    live = attempt.submit_profile_attempt(spec, execute=True, authorization_phrase=attempt.AUTHORIZE_PROFILE)
    assert list(live["jobs"]) == ["profile"] and len(calls) == 1
    assert calls[0][0] == "sbatch"  # no cancel, hold, scontrol, assignment, or science jobs
    assert attempt.submit_profile_attempt(spec, execute=True, authorization_phrase=attempt.AUTHORIZE_PROFILE) == live
    assert len(calls) == 1
    assert inventory(old_root) == before
    assert not list(root.rglob("*.pt")) and not list(root.rglob("*.npz"))


@pytest.mark.parametrize("failure", ["missing_lock", "corrupt_array", "changed_producer", "changed_lock"])
def test_invalid_preparation_refuses_reuse_before_publication(ready, tmp_path, monkeypatch, failure):
    root = Path(ready["foundation_root"])
    path = root / "foundation_lock.json"
    if failure == "missing_lock":
        path.unlink()
    elif failure == "changed_lock":
        obj = load_json(path)
        path.write_text(json.dumps(with_content_hash({**obj, "shards": []})))
    elif failure == "corrupt_array":
        report = load_json(next((root / "assignments").glob("*.json")))
        (root / report["array_path"]).write_bytes(b"corrupt synthetic fixture")
    else:
        monkeypatch.setattr(production, "assignment_source", lambda: dict(file_sha256={"wrong.py": "a" * 64}))
    with pytest.raises((ValueError, FileNotFoundError)):
        create(ready, tmp_path)
    assert not (tmp_path / "debug-attempt").exists()


def test_attempt_paths_scope_source_and_resources_fail_closed(ready, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="debug/planning"):
        create(ready, tmp_path, profile_minutes=1441)
    spec = create(ready, tmp_path)
    for change in (dict(evidence_root=ready["foundation_root"]), dict(rebuild_preparation=True),
                   dict(scientific_fits=31), dict(source_readiness_sha256="0" * 64),
                   dict(execution_site=execution.execution_site("tigris_gh200"))):
        with pytest.raises(ValueError, match="scope"):
            attempt.validate_profile_attempt(with_content_hash({**spec, **change}))
    with pytest.raises(ValueError, match="fresh disjoint"):
        attempt.create_profile_attempt(readiness_spec=Path(ready["readiness_root"]) / "readiness_spec.json",
                                       output_root=Path(ready["readiness_root"]) / "forbidden",
                                       project=tmp_path / "new-project", source_commit="d" * 40)
    def stale(*args):
        raise ValueError("stale source")
    monkeypatch.setattr(attempt, "_source", stale)
    with pytest.raises(ValueError, match="stale source"):
        attempt.validate_profile_attempt(spec)


def test_attempt_lost_acknowledgement_is_not_resubmitted(ready, tmp_path, monkeypatch):
    calls, _ = fake_slurm(monkeypatch, fail_sbatch=True)
    spec = create(ready, tmp_path)
    with pytest.raises(OSError, match="lost acknowledgement"):
        attempt.submit_profile_attempt(spec, execute=True, authorization_phrase=attempt.AUTHORIZE_PROFILE)
    with pytest.raises(PermissionError, match="Ambiguous"):
        attempt.submit_profile_attempt(spec, execute=True, authorization_phrase=attempt.AUTHORIZE_PROFILE)
    assert len(calls) == 1


def test_submission_requires_dry_ledger_and_rechecks_arrays(ready, tmp_path, monkeypatch):
    spec = create(ready, tmp_path)
    calls, _ = fake_slurm(monkeypatch)
    root = Path(spec["attempt_root"])
    dry = root / "dry_run_submission_ledger.json"
    original_dry = dry.read_bytes()
    dry.unlink()
    with pytest.raises(ValueError, match="Canonical dry"):
        attempt.submit_profile_attempt(spec, execute=True, authorization_phrase=attempt.AUTHORIZE_PROFILE)
    dry.write_bytes(original_dry)
    foundation_root = Path(ready["foundation_root"])
    report = load_json(next((foundation_root / "assignments").glob("*.json")))
    (foundation_root / report["array_path"]).write_bytes(b"corruption after creation")
    with pytest.raises(ValueError, match="checksum"):
        attempt.submit_profile_attempt(spec, execute=True, authorization_phrase=attempt.AUTHORIZE_PROFILE)
    assert calls == []


def test_debug_runtime_produces_only_tier3_science_plans(campaign, tmp_path, monkeypatch):
    old = campaign["runtime_profile"]
    debug = execution.execution_site("sporc_a100_debug")
    profile = with_content_hash({**old, "contract": "JETCLASS2_DELPHES_RUNTIME_PROFILE/v3",
                                 "schema_version": 3, "measurement_site": debug,
                                 "site_transfer_policy": execution.DEBUG_PROFILE_TRANSFER})
    production.validate_profile(profile, campaign["foundation"], campaign["source_commit"])
    spec = production.create_campaign(campaign["foundation"], foundation_root=Path(campaign["foundation_root"]),
                                      data_root=Path(campaign["data_root"]), campaign_root=tmp_path / "science-debug-evidence",
                                      project=Path(campaign["project_dir"]), source_commit=campaign["source_commit"], profile=profile)
    production.validate_campaign(spec)
    plan = submission.command_plan(spec)
    assert len(plan["commands"]) == 59
    for row in plan["commands"]:
        assert "--partition=tier3" in row["command"] and "--partition=debug" not in row["command"]
        assert row["command"][-1] == "sporc_a100"
    for bad in (with_content_hash({**old, "execution_site": debug}),
                with_content_hash({**old, "measurement_site": debug}),
                with_content_hash({**profile, "measurement_site": execution.execution_site("tigris_gh200")}),
                with_content_hash({**profile, "site_transfer_policy": "arbitrary"})):
        with pytest.raises(ValueError):
            production.validate_profile(bad, campaign["foundation"], campaign["source_commit"])
    # The production worker still rejects different GPU capacity/resources/env;
    # existing SPORC tests exercise each of those restrictions.


def test_profile_worker_reuses_foundation_and_checks_exact_allocation(ready, tmp_path, monkeypatch):
    spec = create(ready, tmp_path)
    before = inventory(Path(ready["readiness_root"]))
    monkeypatch.setattr(attempt, "allocation", lambda site: ("456", 4, 73728))
    with pytest.raises(ValueError, match="allocation"):
        attempt.run_profile_attempt(spec)
    seen = {}
    def measure(foundation, **kwargs):
        seen.update(kwargs)
        assert foundation == ready["foundation"]
        return dict(content_hash="f" * 64)
    monkeypatch.setattr(attempt, "allocation", lambda site: ("456", 8, 73728))
    monkeypatch.setattr(attempt, "measure_runtime", measure)
    attempt.run_profile_attempt(spec)
    assert seen["foundation_root"] == Path(ready["foundation_root"])
    assert seen["site"] == execution.execution_site("sporc_a100_debug")
    assert seen["output_root"] == Path(spec["evidence_root"])
    receipt = load_json(Path(spec["attempt_root"]) / "profile_result.json")
    assert receipt["foundation_lock_sha256"] == spec["foundation_lock_sha256"]
    assert inventory(Path(ready["readiness_root"])) == before


def test_debug_worker_has_absolute_helpers_and_no_preparation_launches():
    root = Path(__file__).resolve().parents[1]
    text = (root / "sbatch/run_jetclass2_delphes_profile_attempt.sh").read_text()
    assert "JC2_SITE=sporc_a100_debug" in text
    assert '${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh' in text
    assert "prepare_jetclass2_delphes_debug_profile.py\" run" in text
    assert all(word not in text for word in ("scancel", "sbatch ", "scontrol", "--array"))
