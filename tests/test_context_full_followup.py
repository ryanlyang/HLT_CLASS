"""Bound Oscar continuation: no real scheduler calls or production data."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hlt_classification import context_full_followup as f


def sealed(**value):
    value["content_hash"] = f.common.digest(value)
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    gate = sealed(contract="JETCLASS2_CMS_PROXY_LADDER_GATE_SPEC/v11",
        gate_root=str(tmp_path / "gate"), project_dir=str(tmp_path / "original-code"), source_commit="a" * 40)
    path = tmp_path / "gate/gate_spec.json"
    save(path, gate)
    binding = dict(jobs={"authenticate_release": "101", "build_foundation": "102", "preflight": "103"},
                   ledger_hash="b" * 64)
    monkeypatch.setenv("USER", "rlyang")
    return SimpleNamespace(root=tmp_path, path=path, gate=gate, binding=binding,
        kwargs=dict(gate_hash=gate["content_hash"], source_commit="a" * 40, preflight_job="103",
                    project_dir=str(tmp_path / "executor-code"), executor_commit="c" * 40))


def bind_args(case):
    return {k: case.kwargs[k] for k in ("gate_hash", "source_commit", "preflight_job")}


def accounting(case, state="COMPLETED", code="0:0"):
    return "\n".join(f"{job}|jc2ctxmg_{task}|rlyang|default|{state}|{code}"
                     for task, job in case.binding["jobs"].items())


@contextmanager
def fake_lock(root):
    root.mkdir(exist_ok=True)
    yield


def science(case):
    tasks = [dict(task_id=f"train_{i}", kind="train", dependencies=[]) for i in range(9)]
    tasks += [dict(task_id=f"reduce_{i}", kind="reduce", dependencies=[f"train_{i}"]) for i in range(5)]
    tasks += [dict(task_id="aggregate", kind="aggregate", dependencies=["train_8"]),
              dict(task_id="complete", kind="complete", dependencies=["aggregate"])]
    return sealed(schema_version=8, parents={"gate": case.gate["content_hash"]},
        source_commit=case.gate["source_commit"], project_dir=case.gate["project_dir"],
        gate_root=case.gate["gate_root"], campaign_root=str(case.root / "science"),
        foundation=dict(role_counts=dict(train=1000000, validation=250000)),
        fresh_fit_count=9, reducer_count=5, selected_branches=["DIRECT", "COARSE"],
        final_test_accessed=False, tasks=tasks, runtime_profile=dict(train_minutes=2880, reduce_minutes=1440))


def plan(spec):
    commands = []
    for task in spec["tasks"]:
        gpu = task["kind"] in ("train", "reduce")
        time = "2-00:00:00" if task["kind"] == "train" else "1-00:00:00" if gpu else "01:00:00"
        argv = ["sbatch", "--partition=gpu", "--account=default", "--qos=norm-gpu",
            "--cpus-per-task=" + ("6" if gpu else "4"), "--mem=" + ("180000M" if gpu else "32000M"),
            "--time=" + time] + (["--gres=gpu:l40s:1"] if gpu else [])
        commands.append(dict(task_id=task["task_id"], dependencies=task["dependencies"], command=argv))
    return sealed(parents={"subject": spec["content_hash"]}, mode="science", commands=commands,
                  final_test_accessed=False)


def harness(case, monkeypatch, fail=None):
    calls = []
    monkeypatch.setattr(f, "controller_source", lambda *a: dict(commit="c" * 40, files={"controller": "d" * 64}))
    monkeypatch.setattr(f, "bind", lambda *a: (copy.deepcopy(case.gate), copy.deepcopy(case.binding)))
    monkeypatch.setattr(f.common, "lock", fake_lock)
    monkeypatch.setattr(f, "wait_gate", lambda jobs: calls.append("wait"))
    def command(gate, *args):
        assert gate == case.gate
        action = args[0]
        calls.append(action)
        if fail == action:
            raise RuntimeError("simulated invalid artifact or interrupted submission")
        root = case.root / "science"
        if action == "create-campaign":
            save(root / "campaign_spec.json", science(case))
        elif action == "plan":
            save(root / "command_plan.json", plan(f.common.read(root / "campaign_spec.json")))
        elif action == "submit":
            p = f.common.read(root / "command_plan.json")
            assert args[args.index("--plan-hash") + 1] == p["content_hash"]
            assert args[-1] == f.SCIENCE_PHRASE
            assert (case.root / "context1m_followup/review.json").is_file()
            save(root / "submission_ledger.json", sealed(dry_run=False, jobs={r["task_id"]: str(i+200)
                for i, r in enumerate(p["commands"])}))
    monkeypatch.setattr(f, "command", command)
    def inspect(argv, **kw):
        assert argv[-2] == case.gate["project_dir"] and argv[3] == f.INSPECT_SCIENCE
        calls.append("inspect-ledger")
        ledger = f.common.read(case.root / "science/submission_ledger.json")
        return json.dumps(dict(jobs=ledger["jobs"], ledger_hash=ledger["content_hash"]))
    monkeypatch.setattr(f.common, "capture", inspect)
    return calls


def test_bind_original_gate_and_wrong_preflight(case, monkeypatch):
    calls = []
    def capture(argv, **kw):
        calls.append(argv)
        return "ClusterName = slurmctld\n" if argv[0] == "scontrol" else json.dumps(case.binding)
    monkeypatch.setattr(f.common, "capture", capture)
    assert f.bind(case.path, **bind_args(case)) == (case.gate, case.binding)
    assert calls[1][-2:] == [case.gate["project_dir"], str(case.path)]
    with pytest.raises(ValueError, match="Preflight ID"):
        f.bind(case.path, **{**bind_args(case), "preflight_job": "999"})
    with pytest.raises(ValueError, match="identity differs"):
        f.bind(case.path, **{**bind_args(case), "gate_hash": "e" * 64})


def test_wrong_cluster_refused(case, monkeypatch):
    monkeypatch.setattr(f.common, "capture", lambda *a, **kw: "ClusterName = sporc\n")
    with pytest.raises(PermissionError, match="Oscar"):
        f.bind(case.path, **bind_args(case))


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("CANCELLED by 4", "0:0"),
    ("TIMEOUT", "0:0"), ("OUT_OF_MEMORY", "0:125"), ("COMPLETED", "1:0"), ("NODE_FAIL", "0:0")])
def test_failed_accounting(case, state, code):
    with pytest.raises(RuntimeError, match="no science submitted"):
        f.states(accounting(case, state, code), case.binding["jobs"], "rlyang")


@pytest.mark.parametrize("before,after", [("rlyang", "other"), ("default", "foreign"),
    ("jc2ctxmg_preflight", "other_preflight"), ("103|", "999|")])
def test_wrong_accounting_identity(case, before, after):
    with pytest.raises((ValueError, PermissionError)):
        f.states(accounting(case).replace(before, after), case.binding["jobs"], "rlyang")


def test_wait_all_three_not_just_foundation(case, monkeypatch):
    foundation_only = "\n".join(accounting(case).splitlines()[:2])
    outputs = iter(["", foundation_only, accounting(case, "RUNNING"), accounting(case)])
    sleeps = []
    def capture(argv):
        assert "101,102,103" in argv
        return next(outputs)
    monkeypatch.setattr(f.common, "capture", capture)
    f.wait_gate(case.binding["jobs"], sleep=sleeps.append)
    assert sleeps == [60, 60, 60]
    with pytest.raises(ValueError, match="duplicate"):
        f.states(accounting(case) + "\n" + accounting(case), case.binding["jobs"], "rlyang")


def test_wait_bound_and_accounting_failure(case, monkeypatch):
    monkeypatch.setattr(f.common, "capture", lambda *a: "")
    ticks = iter([0, 0, 14 * 86400])
    with pytest.raises(TimeoutError):
        f.wait_gate(case.binding["jobs"], sleep=lambda n: None, monotonic=lambda: next(ticks))
    def fail(*args):
        raise RuntimeError("scheduler unavailable")
    monkeypatch.setattr(f.common, "capture", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        f.wait_gate(case.binding["jobs"], sleep=lambda n: pytest.fail("must stop"))


def test_review_is_read_only(case, monkeypatch):
    calls = harness(case, monkeypatch)
    f.run(case.path, **case.kwargs)
    assert calls == [] and not (case.root / "context1m_followup").exists()


def test_success_and_no_second_submission(case, monkeypatch, capsys):
    calls = harness(case, monkeypatch)
    before = case.path.read_bytes()
    f.run(case.path, **case.kwargs, execute=True)
    assert calls == ["wait", "create-campaign", "plan", "submit", "inspect-ledger"]
    assert "ARMED:" in capsys.readouterr().out
    done = f.common.read(case.root / "context1m_followup/complete.json")
    assert len(done["jobs"]) == 16 and done["training_complete"] is False
    assert done["contract"] == "JC2_CONTEXT_1M_FOLLOWUP/v1"
    assert case.path.read_bytes() == before
    with pytest.raises(PermissionError, match="already submitted"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls.count("submit") == 1


def test_failed_preflight_never_creates_science(case, monkeypatch):
    calls = harness(case, monkeypatch)
    def fail(jobs):
        raise RuntimeError("preflight failed")
    monkeypatch.setattr(f, "wait_gate", fail)
    with pytest.raises(RuntimeError, match="preflight failed"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls == [] and not (case.root / "science").exists()


@pytest.mark.parametrize("fail", ["create-campaign", "plan", "submit"])
def test_bad_artifact_or_submit_no_retry(case, monkeypatch, fail):
    calls = harness(case, monkeypatch, fail=fail)
    with pytest.raises(RuntimeError, match="simulated"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls.count(fail) == 1 and not (case.root / "context1m_followup/complete.json").exists()


@pytest.mark.parametrize("name", ["live_submission_claim.json", "submission_ledger.json", "submission_ledger_journal/0000.json"])
def test_existing_submission_blocks_arming(case, monkeypatch, name):
    calls = harness(case, monkeypatch)
    save(case.root / "science" / name, {})
    with pytest.raises(PermissionError, match="preserve and inspect"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls == []


def test_gate_or_controller_drift_after_wait(case, monkeypatch):
    calls = harness(case, monkeypatch)
    original = copy.deepcopy(case.gate)
    bindings = iter([(original, case.binding), ({**original, "source_commit": "e" * 40}, case.binding)])
    monkeypatch.setattr(f, "bind", lambda *a: next(bindings))
    with pytest.raises(ValueError, match="changed while waiting"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls == ["wait"]


def test_controller_source_drift_after_wait(case, monkeypatch):
    calls = harness(case, monkeypatch)
    sources = iter([{"commit": "c" * 40}, {"commit": "d" * 40}])
    monkeypatch.setattr(f, "controller_source", lambda *a: next(sources))
    with pytest.raises(ValueError, match="changed while waiting"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls == ["wait"]


def test_real_canonical_slurm_plan_matches_review(case, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission
    from hlt_classification.jetclass2_delphes.execution import execution_site
    spec = science(case)
    spec.pop("content_hash")
    spec["runtime_profile"].update(execution_site=execution_site("oscar_l40s"), cpus=6, memory_mb=180000)
    spec = sealed(**spec)
    monkeypatch.setattr(submission, "validate_campaign", lambda *a, **kw: None)
    canonical = submission.science_plan(spec)
    save(case.root / "science/campaign_spec.json", spec)
    save(case.root / "science/command_plan.json", canonical)
    assert f.review(case.root, case.gate) == canonical


def test_executor_source_is_clean_pushed_and_bound(tmp_path, monkeypatch):
    calls = []
    for path in f.CONTROLLER_FILES:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(path)
    monkeypatch.setattr(f, "__file__", str(tmp_path / "src/hlt_classification/context_full_followup.py"))
    monkeypatch.setattr(f, "validate_source_checkout", lambda project, **kw: calls.append((project, kw)))
    result = f.controller_source(tmp_path, "c" * 40)
    assert calls == [(tmp_path, {"expected_commit": "c" * 40})]
    assert set(result["files"]) == set(f.CONTROLLER_FILES)
    assert all(len(digest) == 64 for digest in result["files"].values())
    with pytest.raises(ValueError, match="exact executor checkout"):
        f.controller_source(tmp_path / "src", "c" * 40)


def test_post_submission_integrity_failure_preserved(case, monkeypatch):
    calls = harness(case, monkeypatch)
    def fail(*a, **kw):
        raise ValueError("journal differs")
    monkeypatch.setattr(f.common, "capture", fail)
    with pytest.raises(ValueError, match="journal differs"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls.count("submit") == 1
    assert (case.root / "science/submission_ledger.json").is_file()
    assert not (case.root / "context1m_followup/complete.json").exists()
    with pytest.raises(PermissionError, match="already submitted"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls.count("submit") == 1


@pytest.mark.parametrize("change", ["population", "branch", "count", "time", "partition", "memory", "gpu", "dependency", "test"])
def test_review_scope(case, change):
    spec = science(case)
    if change == "population":
        spec["foundation"]["role_counts"]["train"] = 100000
    if change == "branch":
        spec["selected_branches"].append("DENSE")
    if change == "count":
        spec["fresh_fit_count"] = 10
    if change == "time":
        spec["runtime_profile"]["train_minutes"] = 2881
    if change == "test":
        spec["final_test_accessed"] = True
    spec.pop("content_hash")
    spec = sealed(**spec)
    p = plan(spec)
    flags = {"partition": ("--partition=gpu", "--partition=debug"),
             "memory": ("--mem=180000M", "--mem=90000M"),
             "gpu": ("--gres=gpu:l40s:1", "--gres=gpu:l40s:2")}
    if change in flags:
        a, b = flags[change]
        p["commands"][0]["command"] = [b if x == a else x for x in p["commands"][0]["command"]]
    if change == "dependency":
        p["commands"][0]["dependencies"] = ["other"]
    p.pop("content_hash")
    save(case.root / "science/campaign_spec.json", spec)
    save(case.root / "science/command_plan.json", sealed(**p))
    with pytest.raises((ValueError, PermissionError)):
        f.review(case.root, case.gate)


def test_original_cli_environment_and_source_boundary(case, monkeypatch):
    monkeypatch.setenv("SBATCH_PARTITION", "wrong")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    calls = []
    monkeypatch.setattr(f.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    f.command(case.gate, "plan", "--mode", "science")
    argv, options = calls[0]
    assert argv[3] == str(Path(case.gate["project_dir"]) / "scripts/jetclass2_context_full_ladder.py")
    assert options["check"] is True
    assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in options["env"])
    assert options["env"]["PYTHONNOUSERSITE"] == "1"
    assert "sys.path.insert(0" in f.INSPECT and "context_full as l" in f.INSPECT
    compile(f.INSPECT, "gate-inspect", "exec")
    compile(f.INSPECT_SCIENCE, "science-inspect", "exec")


def test_publication_immutable_and_lock(tmp_path, monkeypatch):
    p = tmp_path / "receipt.json"
    first = f.publish(p, kind="test", parents={"gate": "a" * 64})
    assert f.publish(p, kind="test", parents={"gate": "a" * 64}) == first
    with pytest.raises(FileExistsError):
        f.publish(p, kind="other")
    first["kind"] = "tampered"
    save(p, first)
    with pytest.raises(ValueError, match="checksum"):
        f.common.read(p)
    calls = []
    def flock(stream, flags):
        calls.append(flags)
        if calls.count(3) > 1 and flags == 3:
            raise BlockingIOError()
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=2, LOCK_UN=4))
    with f.common.lock(tmp_path / "controller"):
        with pytest.raises(PermissionError, match="already active"):
            with f.common.lock(tmp_path / "controller"):
                pytest.fail("Duplicate lock")


def test_gate_and_science_bridges_check_real_journals(case, monkeypatch, capsys):
    from hlt_classification.cms_proxy_ladder import context_full as l, submission as s
    from hlt_classification.scouting.hcwdl_recovery import build_submission_event, build_submission_ledger
    monkeypatch.setattr(l, "validate_gate", lambda *a, **kw: None)
    monkeypatch.setattr(l, "validate_campaign", lambda *a, **kw: None)
    for mode, code in (("gate", f.INSPECT), ("science", f.INSPECT_SCIENCE)):
        jobs = case.binding["jobs"] if mode == "gate" else {f"task_{i}": str(200+i) for i in range(16)}
        root = case.root / mode
        spec = case.gate if mode == "gate" else sealed(campaign_root=str(root))
        path = case.path if mode == "gate" else root / "campaign_spec.json"
        save(path, spec)
        rows = [dict(task_id=task, dependencies=[] if i == 0 else [list(jobs)[i-1]],
            command=["sbatch", "--wrap=true"] + ([] if i == 0 else
            ["--dependency=afterok:${JOB_" + list(jobs)[i-1] + "}"])) for i, task in enumerate(jobs)]
        p = sealed(commands=rows)
        monkeypatch.setattr(s, "gate_plan" if mode == "gate" else "science_plan", lambda *a: p)
        commands = {}
        for i, row in enumerate(rows):
            argv = row["command"]
            for parent in row["dependencies"]:
                argv = [a.replace("${JOB_" + parent + "}", jobs[parent]) for a in argv]
            commands[row["task_id"]] = argv
            event = build_submission_event(campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
                job_id=jobs[row["task_id"]], command=argv, sequence=i)
            save(root / f"submission_ledger_journal/{i:04d}_{row['task_id']}.json", event)
        save(root / "command_plan.json", p)
        save(root / "submission_ledger.json", build_submission_ledger(campaign_spec_sha256=spec["content_hash"],
            jobs=jobs, commands=commands, dry_run=False))
        monkeypatch.setattr(sys, "argv", ["inspect", case.gate["project_dir"], str(path)])
        monkeypatch.setattr(sys, "path", list(sys.path))
        exec(compile(code, mode, "exec"), {})
        assert json.loads(capsys.readouterr().out)["jobs"] == jobs
        journal = root / f"submission_ledger_journal/0000_{rows[0]['task_id']}.json"
        event = f.common.read(journal)
        event.pop("content_hash")
        event["job_id"] = "999"
        save(journal, sealed(**event))
        with pytest.raises(ValueError, match="journal differs"):
            exec(compile(code, mode, "exec"), {})


def test_launcher_and_controller_lock_files():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/start_jetclass2_context_full_followup.sh").read_text()
    assert "nohup" in launcher and '</dev/null &' in launcher
    assert 'JC2_SITE=oscar_l40s' in launcher and '"${PROJECT_DIR}/sbatch/' in launcher
    assert '--execute' in launcher and 'FOLLOWUP_EXIT=' in launcher
    assert 'scancel' not in launcher and 'git pull' not in launcher
    assert all((root / p).is_file() for p in f.CONTROLLER_FILES)
