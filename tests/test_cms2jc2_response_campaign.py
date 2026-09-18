from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from hlt_classification.cms2jc2_response import campaign, submission, orchestration
from hlt_classification.cms2jc2_response.contracts import artifact, publish, with_content_hash, load_json, sha256_file
from hlt_classification.cms2jc2_response.storage import GIB


def fixture_spec(tmp_path):
    source = artifact("SOURCE", commit="a"*40, files={}, dirty=False, executable=True)
    spec = artifact("CAMPAIGN_SPEC", parents={"source": source["content_hash"]}, source=source, project_dir=str(tmp_path/"project"),
                    campaign_root=str(tmp_path), stage="acceptance", attempt="r1", site=campaign.SITE,
                    tasks=campaign.acceptance_graph(), reusable_tasks={}, inputs={},
                    combined_cpu_cap=64, storage_cap_bytes=12*GIB, estimated_remaining_writes=GIB,
                    automatic_followon_submission=False)
    directory = campaign.stage_dir(spec)
    directory.mkdir(parents=True)
    publish(directory/"campaign_spec.json", spec, "CAMPAIGN_SPEC")
    publish(directory/"command_plan.json", campaign.command_plan(spec), "COMMAND_PLAN")
    return spec


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    # Scheduler lifecycle fixtures bypass source/data audits only. Other tests
    # cover actual audits; these receipts never serve as remote acceptance.
    monkeypatch.setattr(submission, "validate_spec", lambda spec: {})
    from hlt_classification.cms2jc2_response import storage
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*GIB))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scontrol":
            assert argv == ["scontrol", "show", "partition", "debug", "-o"]
            return SimpleNamespace(stdout="PartitionName=debug State=UP", stderr="", returncode=0)
        assert argv[0] == "sbatch"
        return SimpleNamespace(stdout=str(1000+len(calls))+";sporc\n", stderr="", returncode=0)
    monkeypatch.setattr(submission.subprocess, "run", run)
    return spec, calls


def test_staged_plans_are_cpu_only_and_do_not_auto_launch_science(scheduler):
    spec, calls = scheduler
    plan = submission.submit(spec)
    assert calls == [] and plan["dry_run"] and len(plan["commands"]) == 15
    assert not plan["automatic_followon_submission"]
    assert plan["allocated_cpu_hour_upper_bound"] > 0
    for row in plan["commands"]:
        assert "--partition=debug" in row["argv"]
        assert "--qos=qos_tier3" in row["argv"]
        assert "--export=NONE" in row["argv"]
        assert "--no-requeue" in row["argv"]
        assert not any("--gpu" in v or "--gres" in v for v in row["argv"])
        assert any(v == f"--comment=c2jr:{spec['content_hash']}:{row['task_id']}" for v in row["argv"])
    with pytest.raises(PermissionError, match="reviewed"):
        submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["science"],
                          reviewed_plan_hash=plan["content_hash"])


def test_exact_receipts_dependency_plan_and_idempotent_submit(scheduler):
    spec, calls = scheduler
    plan = campaign.command_plan(spec)
    ledger = submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["acceptance"],
                               reviewed_plan_hash=plan["content_hash"])
    assert len(ledger["jobs"]) == 15
    jobs = submission.submitted_jobs(spec)
    verify = next(v for v in calls if v[0] == "sbatch" and v[-1] == "miniature_verify")
    assert "--dependency=afterok:"+":".join(jobs[f"miniature_{f}_L"] for f in "ABC") in verify
    before = sum(v[0] == "sbatch" for v in calls)
    again = submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["acceptance"],
                               reviewed_plan_hash=plan["content_hash"])
    assert again == ledger and sum(v[0] == "sbatch" for v in calls) == before


def test_ambiguous_acknowledgement_never_retries(scheduler, monkeypatch):
    spec, calls = scheduler
    original = submission.subprocess.run
    def broken(argv, **kwargs):
        if argv[0] == "sbatch":
            return SimpleNamespace(stdout="", stderr="lost acknowledgement", returncode=1)
        return original(argv, **kwargs)
    monkeypatch.setattr(submission.subprocess, "run", broken)
    kwargs = dict(execute=True, authorization_phrase=campaign.PHRASES["acceptance"],
                  reviewed_plan_hash=campaign.command_plan(spec)["content_hash"])
    with pytest.raises(RuntimeError, match="Ambiguous"):
        submission.submit(spec, **kwargs)
    with pytest.raises(RuntimeError, match="Unacknowledged"):
        submission.submit(spec, **kwargs)


def test_science_resource_gate_and_complete_scope():
    resources = {k: dict(cpus=1, memory_gib=8, time="02:00:00")
                 for k in ("fit_A", "fit_B", "fit_C", "evaluate", "metrics", "report", "visual_select")}
    execution = artifact("EXECUTION_LOCK", ready=False, resource_blockers=["memory"], resources=resources)
    with pytest.raises(PermissionError, match="resources"):
        campaign.science_tasks(execution)
    execution = with_content_hash(dict(execution, ready=True, resource_blockers=[]))
    tasks = campaign.science_tasks(execution)
    assert len(tasks) == 71 and sum(t["action"] == "fit" for t in tasks) == 33
    assert sum(t["action"] == "evaluate" for t in tasks) == 33
    assert tasks[-2]["action"] == "visual_select"
    assert not any(t["action"] == "confirm" for t in tasks)


def test_reference_bytes_not_just_file_existence(tmp_path):
    path = tmp_path/"source.json"
    publish(path, artifact("TEST", name="test"), "TEST")
    ref = campaign.file_ref(path)
    assert campaign.resolve_ref(ref)["name"] == "test"
    path.write_text("{}")
    with pytest.raises(ValueError, match="bytes"):
        campaign.resolve_ref(ref)


def test_monitor_does_not_mutate_or_hide_unknown_jobs(scheduler, monkeypatch):
    spec, _ = scheduler
    submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["acceptance"],
                      reviewed_plan_hash=campaign.command_plan(spec)["content_hash"])
    jobs = submission.submitted_jobs(spec)
    requested = []
    def query(argv, **kwargs):
        requested.append(argv)
        return SimpleNamespace(stdout=jobs["miniature_A_L"]+"|COMPLETED|0:0\n", returncode=0)
    monkeypatch.setattr(submission.subprocess, "run", query)
    report = submission.monitor(spec)
    assert report["read_only"] and requested[0][0] == "sacct"
    assert {r["state"] for r in report["tasks"]} == {"UNKNOWN", "COMPLETED"}
    with pytest.raises(PermissionError, match="active/pending"):
        submission.create_recovery(campaign.stage_dir(spec)/"campaign_spec.json", attempt="r2", approved_job_ids=[])


def test_allocation_does_not_accept_local_development_evidence():
    from hlt_classification.cms2jc2_response.measurement import allocation
    with pytest.raises(PermissionError, match="real"):
        allocation(cpus=16)


def test_worker_allocation_requires_debug_not_tier3(monkeypatch):
    from hlt_classification.cms2jc2_response import measurement
    # Mocked allocation shape only; never remote acceptance evidence.
    monkeypatch.setattr(measurement.platform, "system", lambda: "Linux")
    monkeypatch.setattr(measurement.platform, "machine", lambda: "x86_64")
    for name, value in {
        "SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "debug",
        "SLURM_CPUS_PER_TASK": "16", "SLURM_JOB_NUM_NODES": "1",
        "SLURM_JOB_ACCOUNT": "reu-aisocial", "CONDA_PREFIX": campaign.SITE["conda_prefix"],
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "LD_LIBRARY_PATH": campaign.SITE["conda_prefix"]+"/lib",
        "SLURM_GPUS": "0", "SLURM_GPUS_ON_NODE": "0", "SLURM_JOB_GPUS": "",
    }.items():
        monkeypatch.setenv(name, value)
    assert measurement.allocation(cpus=16)["partition"] == "debug"
    monkeypatch.setenv("SLURM_JOB_PARTITION", "tier3")
    with pytest.raises(PermissionError, match="real"):
        measurement.allocation(cpus=16)


@pytest.mark.parametrize("stage", campaign.STAGES)
def test_every_stage_queues_all_tasks_in_debug(tmp_path, stage):
    spec = fixture_spec(tmp_path)
    resource = dict(cpus=1, memory_gib=8, time="02:00:00")
    keys = ("fit_A", "fit_B", "fit_C", "evaluate", "metrics", "report", "visual_select",
            "confirm", "transfer_train", "transfer_validation", "visual_confirm", "visual_jc2")
    execution = artifact("EXECUTION_LOCK", ready=True, resource_blockers=[],
                         resources={k: resource for k in keys})
    selected = dict(selected_candidate="B_L", family_finalists={"A": "A_L", "B": "B_L", "C": "C_L"})
    tasks = {"acceptance": campaign.acceptance_graph,
             "science": lambda: campaign.science_tasks(execution),
             "confirmation": lambda: campaign.confirmation_tasks(selected, execution)}[stage]()
    spec = with_content_hash(dict(spec, stage=stage, tasks=tasks))
    plan = campaign.command_plan(spec)
    assert len(plan["commands"]) == {"acceptance": 15, "science": 71, "confirmation": 9}[stage]
    for row in plan["commands"]:
        assert "--partition=debug" in row["argv"]
        assert "--partition=tier3" not in row["argv"]
        assert "--qos=qos_tier3" in row["argv"]
        assert "--account=reu-aisocial" in row["argv"]


def test_old_tier3_spec_is_not_silently_rerouted(tmp_path):
    spec = fixture_spec(tmp_path)
    spec = with_content_hash(dict(spec, site=dict(spec["site"], partition="tier3")))
    with pytest.raises(ValueError, match="site differs"):
        campaign.command_plan(spec)


def test_real_worker_synthetic_mechanism_subtest():
    from hlt_classification.cms2jc2_response.synthetic_acceptance import exercise_modules
    from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
    report = exercise_modules(provisional_compatibility(inventory_hash="a"*64), "b"*64)
    assert report["passed"] and not report["real_detector_validation"]
    assert set(report["candidates"]) == {"A_L", "B_L", "C_L"}


def test_terminal_recovery_reuses_authenticated_outputs_and_never_cancels(scheduler, monkeypatch):
    spec, calls = scheduler
    submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["acceptance"],
                      reviewed_plan_hash=campaign.command_plan(spec)["content_hash"])
    jobs = submission.submitted_jobs(spec)
    root = Path(spec["campaign_root"])
    result = artifact("TEST", finished=True)
    path = root/"reports/finished.json"
    publish(path, result, "TEST")
    receipt = artifact("TASK_OUTPUTS", parents={"spec": spec["content_hash"], "source": spec["source"]["content_hash"]},
                       task_id="miniature_A_L", job_id=jobs["miniature_A_L"], outputs={"result": dict(
                           relative=path.relative_to(root).as_posix(), sha256=sha256_file(path), bytes=path.stat().st_size,
                           kind="TEST", content_hash=result["content_hash"])})
    publish(orchestration.receipt_path(spec, "miniature_A_L"), receipt, "TASK_OUTPUTS")
    normal_run = submission.subprocess.run
    def query(argv, **kwargs):
        if argv[0] == "sacct":
            return SimpleNamespace(stdout="".join(f"{job}|{'COMPLETED' if t == 'miniature_A_L' else 'FAILED'}|0:0\n"
                                                  for t, job in jobs.items()), returncode=0)
        return normal_run(argv, **kwargs)
    monkeypatch.setattr(submission.subprocess, "run", query)
    args = dict(attempt="r2", approved_job_ids=[job for t, job in jobs.items() if t != "miniature_A_L"])
    recovered = submission.create_recovery(campaign.stage_dir(spec)/"campaign_spec.json", **args)
    assert set(recovered["reusable_tasks"]) == {"miniature_A_L"}
    assert orchestration.verified_product(recovered, "miniature_A_L", "result") == path
    ledger = submission.submit(recovered, execute=True, authorization_phrase=campaign.PHRASES["acceptance"],
                                reviewed_plan_hash=campaign.command_plan(recovered)["content_hash"])
    assert len(ledger["jobs"]) == 14 and "miniature_A_L" not in ledger["jobs"]
    assert all("--partition=debug" in cmd for cmd in calls if cmd[0] == "sbatch")
    assert not any(cmd[0] == "scancel" for cmd in calls)
    path.write_text("{}")
    with pytest.raises(ValueError, match="payload changed"):
        orchestration.verified_receipt(recovered, "miniature_A_L")


def test_reconcile_checks_exact_comment_owner_source_and_never_resubmits(scheduler, monkeypatch):
    spec, calls = scheduler
    original = submission.subprocess.run
    def broken(argv, **kwargs):
        if argv[0] == "sbatch":
            return SimpleNamespace(stdout="", stderr="lost ack", returncode=1)
        return original(argv, **kwargs)
    monkeypatch.setattr(submission.subprocess, "run", broken)
    with pytest.raises(RuntimeError):
        submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["acceptance"],
                          reviewed_plan_hash=campaign.command_plan(spec)["content_hash"])
    monkeypatch.setenv("USER", "fixture_user")
    command = str(Path(spec["project_dir"])/"sbatch/run_cms2jc2_response_cpu.sh")
    fields = (f"JobId=3001 Comment=c2jr:{spec['content_hash']}:miniature_A_L WorkDir={spec['project_dir']} "
              f"Command={command} UserId=fixture_user(5) Partition=debug Account=reu-aisocial")
    monkeypatch.setattr(submission.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=fields))
    value = submission.reconcile(spec, task_id="miniature_A_L", job_id="3001")
    assert value["reconciled"] and submission.submitted_jobs(spec)["miniature_A_L"] == "3001"
    assert not (campaign.stage_dir(spec)/"submission_ledger.json").exists()


def test_confirmation_has_frozen_one_per_family_and_no_refit_actions():
    resource = dict(cpus=1, memory_gib=8, time="01:00:00")
    execution = {"resources": {k: resource for k in ("confirm", "transfer_train", "transfer_validation", "visual_confirm", "visual_jc2")}}
    selected = {"selected_candidate": "B_L", "family_finalists": {"A": "A_M", "B": "B_H", "C": "C_L"}}
    tasks = campaign.confirmation_tasks(selected, execution)
    assert len(tasks) == 9
    assert {r["params"]["candidate_id"] for r in tasks if r["action"] == "confirm"} == {"A_M", "B_L", "C_L"}
    assert not any("fit" in r["action"] for r in tasks)
