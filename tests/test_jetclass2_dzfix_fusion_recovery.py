"""Purged-job recovery: exact new IDs, no old ledger rewrite or branch restart."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash, write_immutable_json
from hlt_classification.jetclass2_delphes import dzfix_fusion_recovery as rec
from hlt_classification.scouting.hcwdl_exact_dag_submission import _resolved
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_jetclass2_dzfix_fusion_chain import spec_at


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = Path(rec.__file__).resolve().parents[3]
    source = with_content_hash(dict(spec_at(tmp_path), source_commit=rec.DONOR, project_dir=str(project)))
    root = Path(source["campaign_root"])
    write_immutable_json(root / "campaign_spec.json", source)
    plan = rec.scheduler.plan(source, "science")
    jobs = {r["task_id"]: str(21783363 + i) for i, r in enumerate(plan["commands"])}
    ledger = build_submission_ledger(campaign_spec_sha256=source["content_hash"], jobs=jobs,
        commands={r["task_id"]: _resolved(r, jobs) for r in plan["commands"]}, dry_run=False)
    write_immutable_json(root / "submissions_science/submission_ledger.json", ledger)
    monkeypatch.setattr(rec.chain, "validate_campaign", lambda *a, **k: source["content_hash"])
    monkeypatch.setattr(rec.chain, "_source", lambda *a: None)
    monkeypatch.setattr(rec.runtime, "validate_campaign", lambda *a, **k: source["content_hash"])
    monkeypatch.setattr(rec.runtime, "science_gate", lambda *a: {"passed": True})
    monkeypatch.setattr(rec.os, "getuid", lambda: 1234, raising=False)

    def finish(name):
        directory = root / "outputs" / name
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / "result.json"
        write_immutable_json(out, {"done": name})
        parents = {p: load_json(root / "tasks" / (p + ".json"))["content_hash"]
                   for p in rec.runtime.task(source, name)["dependencies"]}
        report = rec.chain.artifact("TASK_REPORT", campaign_sha256=source["content_hash"],
            source_commit=source["source_commit"], task_id=name, parents=parents, result={"value": name},
            outputs=[dict(path=out.relative_to(root).as_posix(), sha256=sha256_file(out))], final_test_accessed=False)
        write_immutable_json(root / "tasks" / (name + ".json"), report)
        return report

    for row in source["tasks"]:
        if row["task_id"] not in rec.RETRY + rec.KEEP:
            finish(row["task_id"])
    (root / "outputs" / rec.TARGET).mkdir()
    (root / ("slurm-" + jobs[rec.TARGET] + ".out")).write_text("DUE TO NODE FAILURE\n")
    states = {job: {"state": "PENDING", "exit_code": "0:0"} for job in jobs.values()}
    states[jobs[rec.TARGET]]["state"] = "NODE_FAIL"
    states[jobs[rec.KEEP[0]]]["state"] = "RUNNING"
    calls = []
    new = {}

    def controller(args, **kwargs):
        args = list(map(str, args)); calls.append(args)
        stdout, returncode, stderr = "", 0, ""
        if args[0] == "sacct":
            ids = args[args.index("-j") + 1].split(",")
            stdout = "\n".join(f"{j}|{states[j]['state']}|{states[j]['exit_code']}|" for j in ids if j in states)
        elif args[:3] == ["scontrol", "show", "job"]:
            job = args[-1]
            # The failed historical job really has aged out of scontrol.
            if job == jobs[rec.TARGET]:
                return SimpleNamespace(returncode=1, stdout="", stderr="Invalid job id specified")
            name = next((n for n, j in jobs.items() if j == job), new.get(job))
            require = name is not None
            assert require
            resource = source["resources"][rec.runtime.task(source, name)["resource"]]
            fields = dict(JobId=job, JobName=("jc2fcr_" if job in new else "jc2fc_") + name,
                UserId="user(1234)", WorkDir=str(project), JobState=states[job]["state"],
                Account="reu-aisocial", Partition="tier3", QOS="qos_tier3", NumNodes="1", NumTasks="1",
                NumCPUs=str(resource["cpus"]), TimeLimit="3-00:00:00" if name == rec.TARGET else "04:00:00",
                Command=str(project / rec.FILES[1]))
            stdout = " ".join(f"{k}={v}" for k, v in fields.items())
        elif args[0] == "sbatch":
            if "--test-only" in args:
                return SimpleNamespace(returncode=0, stdout="", stderr="test-only passed")
            job = str(9001 + len(new)); new[job] = args[-1]
            states[job] = dict(state="PENDING", exit_code="0:0"); stdout = job
        elif args[0] == "scancel":
            assert args[1] in {jobs["aggregate"], jobs["complete"]}
            states[args[1]]["state"] = "CANCELLED"
        elif args[:2] == ["scontrol", "release"]:
            assert len(new) == 3
            assert all(states[jobs[n]]["state"] == "CANCELLED" for n in ("aggregate", "complete"))
        else:
            raise AssertionError(args)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(rec.subprocess, "run", controller)
    recovery_root = root / "recoveries" / "nodefail_r1"
    def create():
        return rec.create(source_spec=root / "campaign_spec.json", recovery_root=recovery_root,
                          project=project, source_commit="b" * 40)
    return SimpleNamespace(source=source, ledger=ledger, root=root, project=project, jobs=jobs,
        states=states, calls=calls, new=new, create=create, recovery_root=recovery_root, finish=finish)


def test_exact_three_task_dry_run_and_live_purged_id_recovery(case):
    before = snapshot(case.root)
    spec = case.create()
    assert {k: snapshot(case.root)[k] for k in before} == before
    assert not any(c[0] in {"sbatch", "scancel"} for c in case.calls)
    plan = load_json(case.recovery_root / "command_plan.json")
    assert [r["task_id"] for r in plan["commands"]] == list(rec.RETRY)
    first, aggregate, complete = plan["commands"]
    assert "--hold" in first["command"] and first["dependencies"] == []
    assert "--time=4320" in first["command"] and "--partition=tier3" in first["command"]
    assert f"--dependency=afterok:${{JOB_{rec.TARGET}}}:{case.jobs[rec.KEEP[-1]]}" in aggregate["command"]
    assert "--dependency=afterok:${JOB_aggregate}" in complete["command"]
    live = rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    assert set(live["jobs"]) == set(rec.RETRY)
    assert len(case.new) == 3
    assert [c for c in case.calls if c[0] == "scancel"] == [
        ["scancel", case.jobs["complete"]], ["scancel", case.jobs["aggregate"]]]
    assert not any(case.jobs[rec.TARGET] == c[-1] for c in case.calls if c[:2] == ["scontrol", "show"])
    assert load_json(case.root / "submissions_science/submission_ledger.json") == case.ledger
    assert rec.submit(spec, execute=True, authorization=rec.AUTHORIZE) == live
    assert len(case.new) == 3


def test_completed_bridge_needs_no_stale_external_id(case):
    for name in rec.KEEP:
        case.finish(name)
        case.states[case.jobs[name]]["state"] = "COMPLETED"
    spec = case.create()
    assert spec["external_dependencies"] == []
    p = rec.plan(spec, case.source)
    assert next(c for c in p["commands"][1]["command"] if c.startswith("--dependency")) == (
        "--dependency=afterok:${JOB_train_FINAL_DIRECT_D000}")


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETED", "UNKNOWN"])
def test_cannot_replace_non_failed_original(case, state):
    case.states[case.jobs[rec.TARGET]]["state"] = state
    with pytest.raises(ValueError, match="terminal"):
        case.create()
    assert not case.recovery_root.exists()


@pytest.mark.parametrize("fault", ["log", "output", "ledger", "bridge_failed", "missing_accounting"])
def test_creation_rejects_bad_evidence(case, fault):
    if fault == "log":
        (case.root / ("slurm-" + case.jobs[rec.TARGET] + ".out")).write_text("other failure")
    elif fault == "output":
        (case.root / "outputs" / rec.TARGET / "selected.pt").write_bytes(b"unexpected")
    elif fault == "ledger":
        value = deepcopy(case.ledger)
        value["commands"][rec.TARGET][0] = "fake_sbatch"
        (case.root / "submissions_science/submission_ledger.json").write_text(__import__('json').dumps(with_content_hash(value)))
    elif fault == "bridge_failed":
        case.states[case.jobs[rec.KEEP[-1]]]["state"] = "FAILED"
    else:
        del case.states[case.jobs[rec.TARGET]]
    with pytest.raises((ValueError, KeyError)):
        case.create()
    assert not case.recovery_root.exists()


def test_source_import_must_use_original_checkout(case, monkeypatch):
    monkeypatch.setattr(rec.runtime, "__file__", str(case.root / "wrong.py"))
    with pytest.raises(ValueError, match="original pinned checkout"):
        case.create()


@pytest.mark.parametrize("fault", ["scope", "parent", "external", "partial_hash", "helper"])
def test_recovery_rejects_tampering(case, fault):
    spec = case.create()
    if fault == "scope":
        spec["retry_tasks"] += [rec.KEEP[0]]
    elif fault == "parent":
        spec["reused_tasks"].pop("reduce_FUSION_D000")
    elif fault == "external":
        spec["external_dependencies"] = ["999999"]
    elif fault == "helper":
        spec["helper_sha256"][rec.FILES[0]] = "0" * 64
    else:
        (case.root / "outputs/reduce_FUSION_D000/result.json").write_text("changed")
    with pytest.raises(ValueError):
        rec.validate_recovery(with_content_hash(spec))


def test_authorization_and_cross_recovery_exclusion(case):
    spec = case.create()
    with pytest.raises(ValueError, match="authorization"):
        rec.submit(spec, execute=True)
    assert not case.new
    path = case.root / "recovery_reservations" / (rec.TARGET + ".json")
    write_immutable_json(path, {"foreign": True})
    with pytest.raises(FileExistsError):
        rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    assert not case.new


def test_ambiguous_submission_never_duplicates(case, monkeypatch):
    spec = case.create()
    original = rec.subprocess.run
    def lost_ack(args, **kwargs):
        if args[0] == "sbatch" and "--test-only" not in args:
            original(args, **kwargs)
            raise RuntimeError("Lost acknowledgement")
        return original(args, **kwargs)
    monkeypatch.setattr(rec.subprocess, "run", lost_ack)
    with pytest.raises(RuntimeError, match="Lost acknowledgement"):
        rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    assert len(case.new) == 1
    monkeypatch.setattr(rec.subprocess, "run", original)
    with pytest.raises(PermissionError, match="Ambiguous"):
        rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    assert len(case.new) == 1
    assert not any(c[0] == "scancel" for c in case.calls)


def test_adapter_uses_original_dispatch_archives_partial_and_restores_auth(case, monkeypatch):
    spec = case.create()
    ledger = rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    original_auth = rec.scheduler.authenticate_job
    seen = []
    monkeypatch.setattr(rec, "authenticate", lambda s, original, n: ledger["jobs"][n])
    def dispatch(source, name):
        assert source == case.source
        assert rec.scheduler.authenticate_job(source, name) == ledger["jobs"][name]
        with pytest.raises(ValueError):
            rec.scheduler.authenticate_job(source, rec.KEEP[0])
        assert not (case.root / "outputs" / name).exists()
        seen.append(name)
        return case.finish(name)
    monkeypatch.setattr(rec.runtime, "run_task", dispatch)
    before = (case.root / "submissions_science/submission_ledger.json").read_bytes()
    result = rec.run_task(spec, rec.TARGET)
    assert result["restart_zero"] is True and seen == [rec.TARGET]
    assert (case.recovery_root / "archived" / rec.TARGET).is_dir()
    assert rec.scheduler.authenticate_job is original_auth
    assert rec.run_task(spec, rec.TARGET) == result
    assert seen == [rec.TARGET]
    for name in rec.KEEP:
        case.finish(name)
    rec.run_task(spec, "aggregate")
    rec.run_task(spec, "complete")
    assert seen == list(rec.RETRY)
    assert (case.root / "submissions_science/submission_ledger.json").read_bytes() == before


def test_failed_adapter_does_not_resume_or_change_auth(case, monkeypatch):
    spec = case.create()
    ledger = rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    monkeypatch.setattr(rec, "authenticate", lambda *a: ledger["jobs"][rec.TARGET])
    original_auth = rec.scheduler.authenticate_job
    def fail(*a):
        raise RuntimeError("Training error")
    monkeypatch.setattr(rec.runtime, "run_task", fail)
    with pytest.raises(RuntimeError, match="Training error"):
        rec.run_task(spec, rec.TARGET)
    assert rec.scheduler.authenticate_job is original_auth
    with pytest.raises(ValueError, match="interrupted"):
        rec.run_task(spec, rec.TARGET)


def test_worker_authenticates_real_id_with_original_resource_checker(case, monkeypatch):
    spec = case.create()
    ledger = rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    monkeypatch.setenv("SLURM_JOB_ID", ledger["jobs"][rec.TARGET])
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sporc")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "320000")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr(rec.scheduler.sys, "prefix", "/home/ryreu/miniconda3/envs/atlas_kd_sporc")
    assert rec.authenticate(spec, case.source, rec.TARGET) == ledger["jobs"][rec.TARGET]
    monkeypatch.setenv("SLURM_JOB_ID", case.jobs[rec.TARGET])
    with pytest.raises(PermissionError, match="exact submitted job"):
        rec.authenticate(spec, case.source, rec.TARGET)
    monkeypatch.setenv("SLURM_JOB_ID", ledger["jobs"][rec.TARGET])
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "160000")
    with pytest.raises(PermissionError, match="allocation"):
        rec.authenticate(spec, case.source, rec.TARGET)


def test_missing_dry_run_cannot_submit(case):
    spec = case.create()
    (case.recovery_root / "submissions_science/dry_run_submission_ledger.json").unlink()
    with pytest.raises(ValueError, match="Canonical dry run"):
        rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    assert not case.new


def test_partial_acknowledged_prefix_continues_without_duplicate(case, monkeypatch):
    spec = case.create()
    original = rec.subprocess.run
    original_publish = __import__(
        'hlt_classification.jetclass2_delphes.submission', fromlist=['write_immutable_json']
    ).write_immutable_json
    submitted = __import__('hlt_classification.jetclass2_delphes.submission', fromlist=['write_immutable_json'])
    def die_after_receipt(path, value):
        original_publish(path, value)
        if path.name == '0000_' + rec.TARGET + '.json' and path.parent.name == 'submission_ledger_journal':
            raise RuntimeError('Crash after durable receipt')
    monkeypatch.setattr(submitted, 'write_immutable_json', die_after_receipt)
    with pytest.raises(RuntimeError, match='durable receipt'):
        rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    assert len(case.new) == 1
    monkeypatch.setattr(submitted, 'write_immutable_json', original_publish)
    rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    assert len(case.new) == 3
    assert list(case.new.values()).count(rec.TARGET) == 1


def test_original_fit_kernel_and_publication_match_through_adapter(case, monkeypatch, tmp_path):
    import numpy as np
    import torch
    from torch import nn
    from test_jetclass2_dzfix_fusion_chain import make_cache

    spec = case.create()
    ledger = rec.submit(spec, execute=True, authorization=rec.AUTHORIZE)
    monkeypatch.setattr(rec, 'authenticate', lambda *a: ledger['jobs'][rec.TARGET])
    monkeypatch.setattr(rec.runtime, 'execution_gate', lambda *a, **k: None)
    train, val = make_cache('train'), make_cache('validation')
    def model(node):
        torch.manual_seed(node['initialization_seed'])
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(17, 11)
            def forward(self, features, vectors, mask):
                return self.head(features.mean(-1))
        return Tiny()
    monkeypatch.setattr(rec.runtime, 'new_model', model)
    monkeypatch.setattr(rec.runtime, 'caches', lambda *a, **k: dict(train=train, checkpoint=val, report=val))
    monkeypatch.setattr(rec.runtime, 'teacher', lambda *a: (
        np.full((len(train), 11), 1 / 11, np.float32), dict(teacher_node='FUSION_D000')))
    native_kernel = rec.runtime.train_kernel
    monkeypatch.setattr(rec.runtime, 'train_kernel', lambda *a, **k: native_kernel(*a, **k, acceptance_passes=2))
    write_immutable_json(case.root / 'validation_partition.json', {'content_hash': 'v' * 64})
    expected_dir = tmp_path / 'reference_fit'
    expected_dir.mkdir()
    row = rec.runtime.task(case.source, rec.TARGET)
    native_fit = rec.runtime.fit
    torch.set_num_threads(1)
    native_fit(case.source, row, expected_dir, 'cpu')
    monkeypatch.setattr(rec.runtime, 'fit', lambda source, row, directory, device: native_fit(source, row, directory, 'cpu'))
    rec.run_task(spec, rec.TARGET)
    actual_dir = case.root / 'outputs' / rec.TARGET
    expected = torch.load(expected_dir / 'selected.pt', weights_only=True)
    actual = torch.load(actual_dir / 'selected.pt', weights_only=True)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    a = load_json(actual_dir / 'training_report.json')
    b = load_json(expected_dir / 'training_report.json')
    assert a['checkpoint_validation'] == b['checkpoint_validation']
    assert a['report_validation'] == b['report_validation']
    assert a['teacher_lineage'] == b['teacher_lineage']
    assert a['final_test_accessed'] is False
    assert rec.runtime.completed(case.source, rec.TARGET) is not None
