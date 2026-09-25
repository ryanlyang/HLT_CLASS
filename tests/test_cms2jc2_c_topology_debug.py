from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.cms2jc2_response import c_topology_debug as debug
from hlt_classification.cms2jc2_response import dev_campaign as dev, dev_submission as submission
from hlt_classification.cms2jc2_response import c_topology_worker as worker
from hlt_classification.cms2jc2_response.contracts import load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response import c_topology as original, dev_data as data
from test_cms2jc2_frozen_c import frozen_root, free_disk


@pytest.fixture
def ct_root(frozen_root, tmp_path, monkeypatch):
    _, ctx, put, before, donor = frozen_root
    for module in (worker, debug):
        monkeypatch.setattr(module, "COUNTS", data.COUNTS)
    spec = original.create(parent_spec=dev.stage_dir(donor)/"stage_spec.json",
        project_dir=tmp_path/"topo_project", source_commit="e"*40, root=tmp_path/"topo")
    return spec, ctx, put, before, donor


@pytest.mark.parametrize("kind", ["DEV_B_TRACKING", "DEV_B_TRACKING_DEBUG", "DEV_C_DIAGNOSTIC"])
def test_c_migration_cannot_target_b_or_other_c_experiments(kind):
    with pytest.raises(ValueError, match="original tier3 C topology"):
        debug.subject_metadata(dict(contract=f"CMS2JC2_RESPONSE_{kind}/v1"))


def test_debug_uses_original_scientific_workers_and_fixed_resources():
    assert [t["task_id"] for t in debug.tasks()] == list(debug.TARGETS)
    assert [t["cpus"] for t in debug.tasks()] == [36, 36, 36, 36, 1]
    assert [t["hours"] for t in debug.tasks()] == [8, 8, 8, 8, 4]
    assert [t["memory_gib"] for t in debug.tasks()] == [128, 128, 128, 128, 32]
    assert all(t["action"] == "ct_evaluate" and not t["depends_on"] for t in debug.tasks()[:4])
    assert debug.tasks()[-1]["depends_on"] == list(debug.TARGETS[:4])


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETING", "COMPLETED", "UNKNOWN", "SUSPENDED"])
def test_retirement_refuses_active_or_successful_targets(state, monkeypatch):
    jobs = {task: str(i+100) for i, task in enumerate(("ct_acceptance", *debug.TARGETS))}
    status = {j: ("PENDING", "0:0") for j in jobs.values()}
    status[jobs["ct_acceptance"]] = ("COMPLETED", "0:0")
    status[jobs["ct_eval_0"]] = (state, "0:0")
    monkeypatch.setattr(submission, "states", lambda _: status)
    with pytest.raises(PermissionError, match="Preserve original C"):
        debug.old_states(dict(migration=dict(subject_jobs=jobs)))


def test_debug_replacement_lifecycle_and_unchanged_outputs(ct_root, tmp_path, monkeypatch):
    subject, ctx, put, _, donor = ct_root
    calls, pending_race = [], {"mode": None}
    # The B debug jobs are in a different study and must stay untouched.
    unrelated_b = {"99901": ("RUNNING", "0:0"), "99902": ("PENDING", "0:0")}
    statuses = dict(unrelated_b)
    job_counter = [4100]
    monkeypatch.setenv("USER", "fixture")

    def scheduler(argv):
        calls.append(argv)
        out = ""
        if argv[:3] == ["scontrol", "show", "partition"]:
            out = f"PartitionName={argv[3]} State=UP"
        elif argv[0] == "sbatch":
            if "--test-only" not in argv:
                job_counter[0] += 1
                out = str(job_counter[0])+"\n"
                statuses[str(job_counter[0])] = ("PENDING", "0:0")
        elif argv[0] == "sacct":
            out = "\n".join(f"{job}|{state}|{exit_code}" for job, (state, exit_code) in statuses.items())
        elif argv[:3] == ["scontrol", "show", "job"]:
            job = argv[-1]
            task = next(t for t, j in old_ledger["jobs"].items() if j == job)
            study = dev.validate_stage(subject, source=False)
            t = next(t for t in subject["tasks"] if t["task_id"] == task)
            out = (f"JobId={job} JobState={statuses[job][0]} JobName=c2jd_{task} UserId=fixture(123) "
                f"Comment=c2jd:{subject['content_hash']}:{task} WorkDir={study['project_dir']} "
                f"Command={Path(study['project_dir'])/'sbatch/run_cms2jc2_response_dev_cpu.sh'} "
                f"Partition=tier3 Account=reu-aisocial QOS=qos_tier3 NumCPUs={t['cpus']} NumNodes=1-1 "
                "ReqTRES=cpu=36,mem=128G,node=1")
        elif argv[0] == "scancel":
            assert argv[1:5] == ["--ctld", "--state=PENDING", "--partition=tier3", "--account=reu-aisocial"]
            targets = {old_ledger["jobs"][t] for t in debug.TARGETS}
            assert set(argv[5:]) == targets
            assert old_ledger["jobs"]["ct_acceptance"] not in argv
            for job in targets:
                statuses[job] = ("CANCELLED", "0:0")
            if pending_race["mode"]:
                statuses[old_ledger["jobs"]["ct_eval_0"]] = (pending_race["mode"], "0:0")
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(submission, "scheduler", scheduler)
    old_plan = submission.submit(subject)
    old_ledger = submission.submit(subject, execute=True, authorization_phrase=dev.PHRASES["ctopo"],
                                   reviewed_plan_hash=old_plan["content_hash"])
    subject_path = dev.stage_dir(subject)/"stage_spec.json"
    with pytest.raises(FileNotFoundError):
        debug.create(parent_spec=subject_path, project_dir=tmp_path/"debug_project",
                     source_commit="f"*40, root=tmp_path/"not_created")
    assert not (tmp_path/"not_created").exists()
    put(subject, "ct_acceptance", dict(result=worker.acceptance(ctx, subject)))
    statuses[old_ledger["jobs"]["ct_acceptance"]] = ("COMPLETED", "0:0")
    roots = [Path(subject["root"]), Path(donor["root"])]
    before = {p: sha256_file(p) for root in roots for p in root.rglob("*") if p.is_file()}

    spec = debug.create(parent_spec=subject_path, project_dir=tmp_path/"debug_project",
                        source_commit="f"*40, root=tmp_path/"debug_audit")
    plan = submission.submit(spec)
    assert spec["migration"]["acceptance_reused"] and not spec["migration"]["acceptance_rerun"]
    assert spec["parent_spec"] == subject["parent_spec"]
    assert len(plan["commands"]) == 5 and plan["gpus"] == 0
    assert all("--partition=debug" in row["argv"] for row in plan["commands"])
    assert all("--time=08:00:00" in row["argv"] for row in plan["commands"][:4])
    for key, value in (("tasks", subject["tasks"]), ("views", ["FULL"]), ("name", "other"),
                       ("observed_state_privileged", False), ("evaluation_association_recomputed", False)):
        bad = with_content_hash(dict(spec, **{key: value}))
        with pytest.raises(ValueError, match="registration"):
            dev.validate_stage(bad)
    changed = deepcopy(dev.validate_stage(spec))
    changed["source"]["files"]["scientific.py"] = "c"*64
    with pytest.raises(ValueError, match="scientific source"):
        debug.reuse(subject, changed)
    with pytest.raises(PermissionError, match="disjoint"):
        debug.create(parent_spec=subject_path, project_dir="unused", source_commit="f"*40, root=subject["root"])
    prior = len(calls)
    with pytest.raises(FileNotFoundError):
        submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["ctopo"],
                          reviewed_plan_hash=plan["content_hash"])
    assert not any(cmd[0] == "sbatch" for cmd in calls[prior:])
    preview = debug.retire(spec)
    assert preview["read_only"] and set(preview["tasks"]) == set(debug.TARGETS)
    with pytest.raises(PermissionError, match="phrase"):
        debug.retire(spec, execute=True, reviewed_plan_hash=plan["content_hash"])

    kwargs = dict(execute=True, authorization_phrase=debug.RETIRE_PHRASE, reviewed_plan_hash=plan["content_hash"])
    # The pending request range is explicitly allowed only in retirement mode.
    old_study = dev.validate_stage(subject, source=False)
    pending_id = old_ledger["jobs"]["ct_eval_0"]
    with pytest.raises(PermissionError, match="allocation differs"):
        submission.scheduler_identity(subject, old_study, "ct_eval_0", pending_id)
    assert submission.scheduler_identity(subject, old_study, "ct_eval_0", pending_id,
                                         pending=True)["NumNodes"] == "1-1"
    # Debug admission must be proved before any destructive scheduler action.
    def rejected(*a, **kw):
        raise PermissionError("Site rejected debug request")
    with monkeypatch.context() as m:
        m.setattr(submission, "site_checks", rejected)
        with pytest.raises(PermissionError, match="Site rejected"):
            debug.retire(spec, **kwargs)
    assert not any(cmd[0] == "scancel" for cmd in calls)
    # Acceptance fields and acknowledged journals are not trusted on existence.
    real_product = dev.product
    with monkeypatch.context() as m:
        def wrong_acceptance(s, owner, key):
            result = real_product(s, owner, key)
            if s == subject and owner == "ct_acceptance":
                return with_content_hash(dict(result, serial_process_parity=False))
            return result
        m.setattr(dev, "product", wrong_acceptance)
        with pytest.raises(ValueError, match="acceptance differs"):
            debug.validate_stage(spec)
    real_jobs = submission.submitted_jobs
    with monkeypatch.context() as m:
        m.setattr(submission, "submitted_jobs", lambda s, p: {} if s == subject else real_jobs(s, p))
        with pytest.raises(ValueError, match="ledger/journals"):
            debug.validate_stage(spec)
    with monkeypatch.context() as m:
        def wrong_closure(s, owner, key):
            result = real_product(s, owner, key)
            if s == subject and owner == "ct_acceptance":
                return with_content_hash(dict(result, closure=dict(result["closure"], p4_maximum=-1.)))
            return result
        m.setattr(dev, "product", wrong_closure)
        with pytest.raises(ValueError, match="codec closure"):
            debug.validate_stage(spec)
    # Low resolution remains a diagnostic result, not a new quality gate.
    with monkeypatch.context() as m:
        def low_resolution(s, owner, key):
            result = real_product(s, owner, key)
            if s == subject and owner == "ct_acceptance":
                return with_content_hash(dict(result, resolved=0, comparable=0))
            return result
        m.setattr(dev, "product", low_resolution)
        assert debug.subject_metadata(subject)[1]["resolved"] == 0
    real_identity = submission.scheduler_identity
    monkeypatch.setattr(submission, "scheduler_identity", lambda *a, **kw: dict(JobState="RUNNING"))
    with pytest.raises(PermissionError, match="nothing cancelled"):
        debug.retire(spec, **kwargs)
    assert not any(cmd[0] == "scancel" for cmd in calls)
    monkeypatch.setattr(submission, "scheduler_identity", real_identity)
    pending_race["mode"] = "PENDING"
    with pytest.raises(RuntimeError, match="not yet terminal"):
        debug.retire(spec, **kwargs)
    assert not debug.retirement_path(spec).exists()
    # Reset fake accounting so the next test races the same five pending IDs.
    for task in debug.TARGETS:
        statuses[old_ledger["jobs"][task]] = ("PENDING", "0:0")
    pending_race["mode"] = "RUNNING"
    with pytest.raises(PermissionError, match="Preserve original C"):
        debug.retire(spec, **kwargs)
    assert not debug.retirement_path(spec).exists()
    # Emulate operator inspection and the race job becoming terminal; no running cancellation.
    pending_race["mode"] = None
    statuses[old_ledger["jobs"]["ct_eval_0"]] = ("CANCELLED", "0:0")
    evidence = debug.retire(spec, **kwargs)
    assert debug.verify_retirement(spec, live=True) == evidence
    assert debug.retire(spec, **kwargs) == evidence
    prior = len(calls)
    statuses[old_ledger["jobs"]["ct_eval_0"]] = ("PENDING", "0:0")
    with pytest.raises(PermissionError, match="can still run"):
        submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["ctopo"],
                          reviewed_plan_hash=plan["content_hash"])
    assert not any(cmd[0] == "sbatch" for cmd in calls[prior:])
    statuses[old_ledger["jobs"]["ct_eval_0"]] = ("CANCELLED", "0:0")
    prior = len(calls)
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["ctopo"],
                               reviewed_plan_hash=plan["content_hash"])
    live = [cmd for cmd in calls[prior:] if cmd[0] == "sbatch" and "--test-only" not in cmd]
    assert len(live) == 5
    assert not any(a.startswith("--dependency") for cmd in live[:4] for a in cmd)
    assert "--dependency=afterok:"+":".join(ledger["jobs"][t] for t in debug.TARGETS[:4]) in live[-1]
    assert submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["ctopo"],
                             reviewed_plan_hash=plan["content_hash"]) == ledger
    assert before == {p: sha256_file(p) for root in roots for p in root.rglob("*") if p.is_file()}
    assert {job: statuses[job] for job in unrelated_b} == unrelated_b
    assert all(job not in cmd for job in unrelated_b for cmd in calls if cmd[0] == "scancel")

    # Same unchanged worker and results reader handle debug's five-task graph.
    for task in spec["tasks"][:4]:
        put(spec, task["task_id"], dict(result=worker.evaluate(ctx, spec, dict(task, cpus=1))))
    monkeypatch.setattr(worker, "plot_pages", lambda *a, **kw: [])
    put(spec, "ct_report", dict(result=worker.report(ctx, spec)))
    from hlt_classification.cms2jc2_response.c_topology_results import read
    assert read(spec)["jets"] == 12
    assert before == {p: sha256_file(p) for root in roots for p in root.rglob("*") if p.is_file()}
    # Success arriving in the old root must never be silently thrown away.
    put(subject, "ct_eval_0", dict(result=dev.product(spec, "ct_eval_0", "result")))
    with pytest.raises(PermissionError, match="Preserve completed C"):
        debug.validate_stage(spec)
