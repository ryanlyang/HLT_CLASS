"""Long-fit tier3 execution without changes to matching, model or training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.jetclass2_delphes import (
    dzfix_fusion_chain as chain, dzfix_fusion_runtime as runtime,
    dzfix_fusion_source as source, dzfix_fusion_submit as scheduler,
)
from hlt_classification.scouting.hcwdl_recovery import build_submission_event, build_submission_ledger
from test_jetclass2_dzfix_fusion_chain import imported_source, rehash, spec_at


def test_runtime_budget_is_request_less_reserve_not_debug_constant():
    spec = chain.registration()
    budget = chain.fit_runtime_budget(spec)
    assert budget == dict(partition="tier3", requested_fit_seconds=72*3600,
                         shutdown_reserve_seconds=3600, projected_fit_limit_seconds=71*3600,
                         projection_margin=1.30)
    assert runtime.require_fit_runtime(spec, 48.50*3600) == budget
    assert runtime.require_fit_runtime(spec, 71*3600) == budget
    # Same helper checks the time before publication and again at science gate.
    for projected in (0, -1, float("nan"), float("inf"), 71*3600+1, 72.91*3600):
        with pytest.raises(RuntimeError, match="acceptance ceiling"):
            runtime.require_fit_runtime(spec, projected)
    spec["resources"]["train"]["minutes"] = 1440
    assert chain.fit_runtime_budget(spec)["projected_fit_limit_seconds"] == 23*3600
    with pytest.raises(RuntimeError, match="23.00h"):
        runtime.require_fit_runtime(spec, 48.50*3600)


@pytest.mark.parametrize("field,value", [("minutes", 0), ("minutes", 72.0),
                                       ("reserve", 0), ("reserve", 72*3600)])
def test_invalid_runtime_budget_rejected(field, value):
    spec = chain.registration()
    if field == "minutes":
        spec["resources"]["train"]["minutes"] = value
    else:
        spec["runtime_shutdown_reserve_seconds"] = value
    with pytest.raises(ValueError, match="runtime budget"):
        chain.fit_runtime_budget(spec)


@pytest.mark.parametrize("kind,version", [("LAUNCH_SPEC", 5), ("CAMPAIGN_SPEC", 5), ("ACCEPTANCE", 3)])
def test_debug_execution_artifacts_cannot_authorize_tier3(kind, version):
    old = rehash(chain.artifact(kind), schema_version=version,
                 contract=f"JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_{kind}/v{version}")
    with pytest.raises(ValueError, match="contract"):
        chain.validate(old, kind)


@pytest.mark.parametrize("fault", ["partition", "walltime", "reserve", "margin"])
def test_execution_changes_require_new_registered_spec(imported_source, monkeypatch, fault):
    launch, _, _ = imported_source
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    spec = chain.create(launch=launch)
    changed = deepcopy(spec)
    if fault == "partition":
        changed["execution_site"] = source.execution_site("sporc_a100_debug")
    elif fault == "walltime":
        changed["resources"]["train"]["minutes"] = 1440
    elif fault == "reserve":
        changed["runtime_shutdown_reserve_seconds"] = 0
    else:
        changed["runtime_projection_margin"] = 1.
    with pytest.raises(ValueError, match="registration"):
        chain.validate_campaign(rehash(changed))


@pytest.mark.parametrize("launch_job", [False, True])
@pytest.mark.parametrize("journal", [False, True])
def test_live_worker_requires_registered_tier3_and_exact_time(tmp_path, monkeypatch, launch_job, journal):
    spec = spec_at(tmp_path)
    if launch_job:
        spec = chain.artifact("LAUNCH_SPEC", registration=chain.registration(),
            launch_root=str(tmp_path / "launch"), project_dir=str(tmp_path / "project"))
    name = "after_matching" if launch_job else "train_FUSION_U050"
    stage = name if launch_job else "science"
    root = Path(spec["launch_root"] if launch_job else spec["campaign_root"])
    directory = root / ("submissions_" + stage)
    resource = chain.RESOURCES["metadata" if launch_job else "train"]
    command = scheduler.command(spec, name, resource, launch=launch_job)
    if journal:
        write_immutable_json(directory / "submission_ledger_journal" / ("0000_" + name + ".json"),
            build_submission_event(campaign_spec_sha256=spec["content_hash"], task_id=name,
                                   job_id="123", command=command, sequence=0))
    else:
        write_immutable_json(directory / "submission_ledger.json",
            build_submission_ledger(campaign_spec_sha256=spec["content_hash"],
                jobs={name: "123"}, commands={name: command}, dry_run=False))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sporc")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(resource["memory_mb"]))
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr(scheduler.sys, "prefix", "/home/ryreu/miniconda3/envs/atlas_kd_sporc")
    fields = dict(Account="reu-aisocial", Partition="tier3", QOS="qos_tier3",
                  NumNodes="1", NumTasks="1", NumCPUs=str(resource["cpus"]),
                  TimeLimit="04:00:00" if launch_job else "3-00:00:00")
    def scontrol(args, **kwargs):
        assert args == ["scontrol", "show", "job", "-o", "123"]
        return SimpleNamespace(stdout=" ".join(f"{k}={v}" for k, v in fields.items()))
    monkeypatch.setattr(scheduler.subprocess, "run", scontrol)
    assert scheduler.authenticate_job(spec, name, launch=launch_job) == "123"
    for key, value in (("Partition", "debug"), ("TimeLimit", "01:00:00"),
                       ("TimeLimit", "UNLIMITED"), ("TimeLimit", "")):
        original = fields[key]
        fields[key] = value
        with pytest.raises(PermissionError):
            scheduler.authenticate_job(spec, name, launch=launch_job)
        fields[key] = original


def test_worker_and_queue_helper_use_tier3_authorization_and_fresh_roots():
    repo = Path(__file__).resolve().parents[1]
    worker = (repo / "sbatch/run_jetclass2_dzfix_fusion_chain.sh").read_text()
    helper = (repo / "scripts/queue_jetclass2_dzfix_fusion_chain.sh").read_text()
    for text in (worker, helper):
        assert "export JC2_SITE=sporc_a100\n" in text
        assert "export JC2_SITE=sporc_a100_debug" not in text
        assert 'source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"' in text
    assert chain.AUTHORIZE in helper
    assert "FUSION CHAIN DEBUG" not in helper
    assert "jc2_dzfix_fusion_tier3_launch_${FUSION_SHORT}_r1" in helper
    assert "jc2_dzfix_fusion_tier3_coarse_${FUSION_SHORT}_r1" in helper
    assert "jc2_dzfix_salience_debug_0d25a4a5_r1/screen_spec.json" in helper
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in worker


def test_old_debug_authorization_does_not_submit(tmp_path):
    spec = spec_at(tmp_path)
    # _submit checks authorization before creating any files or calling sbatch.
    with pytest.raises(PermissionError, match="tier3 authorization"):
        scheduler._submit(spec, {}, tmp_path / "submit", True,
            "AUTHORIZE JETCLASS2 DZFIX FUSION CHAIN DEBUG 500K EXACT SPEC")
    assert not (tmp_path / "submit").exists()
