"""No scheduler writes: fake accounting, pinned-CLI orchestration and failures."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hlt_classification import literature_ladder_followup as f


def sealed(**value):
    value["content_hash"] = f.digest(value)
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    gate = sealed(contract="JETCLASS2_CMS_PROXY_LADDER_GATE_SPEC/v7",
                  gate_root=str(tmp_path / "gate"), project_dir=str(tmp_path / "original-code"),
                  source_commit="a" * 40)
    path = tmp_path / "gate/gate_spec.json"
    save(path, gate)
    evidence = dict(jobs={"authenticate_release": "101", "build_foundation": "102", "preflight": "103"},
                    ledger_hash="b" * 64)
    monkeypatch.setenv("USER", "ryreu")
    return SimpleNamespace(root=tmp_path, path=path, gate=gate, evidence=evidence,
        kwargs=dict(gate_hash=gate["content_hash"], source_commit="a" * 40, preflight_job="103"))


def accounting(case, *, state="COMPLETED", exit_code="0:0"):
    return "\n".join(f"{job}|jc2lvg_{task}|ryreu|reu-aisocial|{state}|{exit_code}"
                     for task, job in case.evidence["jobs"].items())


def fake_lock(root):
    @contextmanager
    def held():
        root.mkdir(exist_ok=True)
        yield
    return held()


def test_bind_authenticates_original_source_and_exact_id(case, monkeypatch):
    calls = []
    def capture(argv, **kw):
        calls.append(argv)
        return "ClusterName = sporc\n" if argv[0] == "scontrol" else json.dumps(case.evidence)
    monkeypatch.setattr(f, "capture", capture)
    assert f.bind(case.path, **case.kwargs) == (case.gate, case.evidence)
    assert calls[1][-2:] == [case.gate["project_dir"], str(case.path.resolve())]
    assert calls[1][1:3] == ["-s", "-c"]
    with pytest.raises(ValueError, match="Preflight ID"):
        f.bind(case.path, **{**case.kwargs, "preflight_job": "999"})
    with pytest.raises(ValueError, match="Gate/source"):
        f.bind(case.path, **{**case.kwargs, "gate_hash": "c" * 64})


def test_wrong_cluster_does_not_import_or_submit(case, monkeypatch):
    monkeypatch.setattr(f, "capture", lambda *a, **kw: "ClusterName = tigris\n")
    with pytest.raises(PermissionError, match="SPORC"):
        f.bind(case.path, **case.kwargs)


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("CANCELLED by 42", "0:0"),
    ("TIMEOUT", "0:0"), ("OUT_OF_MEMORY", "0:125"), ("COMPLETED", "1:0"), ("NODE_FAIL", "0:0")])
def test_bad_accounting_stops(case, state, code):
    with pytest.raises(RuntimeError, match="no science submitted"):
        f.states(accounting(case, state=state, exit_code=code), case.evidence["jobs"], "ryreu")


@pytest.mark.parametrize("before,after", [("ryreu", "other"), ("reu-aisocial", "foreign"),
    ("jc2lvg_preflight", "another_preflight"), ("103|", "999|")])
def test_accounting_identity_rejected(case, before, after):
    with pytest.raises((ValueError, PermissionError)):
        f.states(accounting(case).replace(before, after), case.evidence["jobs"], "ryreu")


def test_missing_accounting_not_success_and_exact_jobs_wait(case, monkeypatch):
    outputs = iter(["", accounting(case, state="PENDING"), accounting(case, state="RUNNING"), accounting(case)])
    calls, sleeps = [], []
    def capture(argv):
        calls.append(argv)
        return next(outputs)
    monkeypatch.setattr(f, "capture", capture)
    f.wait_gate(case.evidence["jobs"], sleep=sleeps.append)
    assert sleeps == [60, 60, 60]
    assert all("101,102,103" in row and row[0] == "sacct" for row in calls)
    with pytest.raises(ValueError, match="duplicate"):
        f.states(accounting(case) + "\n" + accounting(case), case.evidence["jobs"], "ryreu")


def test_bounded_wait(case, monkeypatch):
    monkeypatch.setattr(f, "capture", lambda *a: "")
    times = iter([0, 0, 14 * 86400])
    with pytest.raises(TimeoutError):
        f.wait_gate(case.evidence["jobs"], sleep=lambda n: None, monotonic=lambda: next(times))


def harness(case, monkeypatch, *, fail=None):
    calls = []
    monkeypatch.setattr(f, "bind", lambda *a: (copy.deepcopy(case.gate), copy.deepcopy(case.evidence)))
    monkeypatch.setattr(f, "lock", fake_lock)
    monkeypatch.setattr(f, "wait_gate", lambda jobs: calls.append("wait"))
    def command(gate, *args):
        assert gate == case.gate
        action = args[0]
        calls.append(action)
        if action == fail:
            raise ValueError("simulated invalid gate or interrupted submit")
        science = case.root / "science"
        if action == "create-campaign":
            save(science / "campaign_spec.json", sealed(parents={"gate": gate["content_hash"]},
                source_commit=gate["source_commit"], project_dir=gate["project_dir"],
                fresh_fit_count=9, reducer_count=5, selected_branches=["DIRECT", "COARSE"],
                final_test_accessed=False))
        elif action == "plan":
            spec = f.read(science / "campaign_spec.json")
            save(science / "command_plan.json", sealed(parents={"subject": spec["content_hash"]},
                mode="science", commands=[dict(task_id=f"task_{i}", command=["sbatch", "--partition=debug",
                "--account=reu-aisocial", "--qos=qos_tier3"]) for i in range(16)]))
        elif action == "submit":
            plan = f.read(science / "command_plan.json")
            assert args[args.index("--plan-hash") + 1] == plan["content_hash"]
            assert args[-1] == f.SCIENCE_PHRASE
            assert (case.root / "followup/review.json").is_file()
            # Model original CLI's completed-ledger idempotence.
            if not (science / "submission_ledger.json").exists():
                save(science / "submission_ledger.json", sealed(dry_run=False,
                    campaign_spec_sha256=plan["parents"]["subject"],
                    jobs={f"task_{i}": str(1000 + i) for i in range(16)}))
    monkeypatch.setattr(f, "command", command)
    return calls


def test_full_followup_and_repeat_preserves_old_ledger(case, monkeypatch):
    calls = harness(case, monkeypatch)
    f.run(case.path, **case.kwargs, execute=True)
    assert calls == ["wait", "create-campaign", "plan", "submit"]
    completed = (case.root / "followup/complete.json").read_bytes()
    assert f.read(case.root / "followup/complete.json")["training_complete"] is False
    assert len(f.read(case.root / "science/submission_ledger.json")["jobs"]) == 16
    f.run(case.path, **case.kwargs, execute=True)
    assert calls[-3:] == ["wait", "plan", "submit"]
    assert (case.root / "followup/complete.json").read_bytes() == completed


def test_review_only_does_not_wait_or_write(case, monkeypatch):
    calls = harness(case, monkeypatch)
    f.run(case.path, **case.kwargs)
    assert calls == [] and not (case.root / "followup").exists()


def test_failed_gate_never_creates_science(case, monkeypatch):
    calls = harness(case, monkeypatch)
    def fail(jobs):
        raise RuntimeError("preflight failed")
    monkeypatch.setattr(f, "wait_gate", fail)
    with pytest.raises(RuntimeError, match="preflight failed"):
        f.run(case.path, **case.kwargs, execute=True)
    assert not calls and not (case.root / "science").exists()


@pytest.mark.parametrize("fail", ["create-campaign", "plan", "submit"])
def test_validation_or_submission_failure_never_retries(case, monkeypatch, fail):
    calls = harness(case, monkeypatch, fail=fail)
    with pytest.raises(ValueError, match="simulated"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls.count(fail) == 1
    assert "submit" not in calls if fail != "submit" else calls[-1] == "submit"
    assert not (case.root / "followup/complete.json").exists()


def test_gate_mutation_while_waiting(case, monkeypatch):
    calls = harness(case, monkeypatch)
    original = copy.deepcopy(case.gate)
    bindings = iter([(original, case.evidence), ({**original, "source_commit": "c" * 40}, case.evidence)])
    monkeypatch.setattr(f, "bind", lambda *a: next(bindings))
    with pytest.raises(ValueError, match="changed while waiting"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls == ["wait"]


def test_altered_plan_stops_before_submit(case, monkeypatch):
    calls = harness(case, monkeypatch)
    command = f.command
    def mutate(gate, *args):
        command(gate, *args)
        if args[0] == "plan":
            path = case.root / "science/command_plan.json"
            value = f.read(path)
            value.pop("content_hash")
            value["commands"][0]["command"][1] = "--partition=tier3"
            save(path, sealed(**value))
    monkeypatch.setattr(f, "command", mutate)
    with pytest.raises(PermissionError, match="site differs"):
        f.run(case.path, **case.kwargs, execute=True)
    assert "submit" not in calls


def test_publish_rejects_corruption_and_immutable_change(tmp_path):
    path = tmp_path / "receipt.json"
    first = f.publish(path, kind="test", parents={"gate": "a" * 64})
    assert f.publish(path, kind="test", parents={"gate": "a" * 64}) == first
    with pytest.raises(FileExistsError):
        f.publish(path, kind="different", parents={"gate": "a" * 64})
    first["kind"] = "corrupt"
    save(path, first)
    with pytest.raises(ValueError, match="checksum"):
        f.read(path)
    assert not list(tmp_path.glob(".followup-*"))


def test_environment_and_pinned_command(case, monkeypatch):
    monkeypatch.setenv("SBATCH_PARTITION", "tier3")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    calls = []
    monkeypatch.setattr(f.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    f.command(case.gate, "plan", "--mode", "science")
    argv, kwargs = calls[0]
    assert argv[3] == str(Path(case.gate["project_dir"]) / "scripts/jetclass2_literature_proxy_ladder.py")
    assert kwargs["check"] is True
    assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kwargs["env"])
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_exclusive_lock_with_linux_fcntl_api(tmp_path, monkeypatch):
    calls = []
    def flock(stream, flags):
        calls.append(flags)
        if calls.count(3) > 1 and flags == 3:
            raise BlockingIOError()
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=2, LOCK_UN=4))
    with f.lock(tmp_path / "followup"):
        with pytest.raises(PermissionError, match="already active"):
            with f.lock(tmp_path / "followup"):
                pytest.fail("Second controller obtained the lock")
    assert calls == [3, 3, 4]


def test_bridge_compiles_and_launcher_is_detached():
    compile(f.INSPECT, "pinned-inspect", "exec")
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/start_jetclass2_literature_ladder_followup.sh").read_text()
    assert "nohup" in script and '</dev/null &' in script
    assert "--execute" in script and "LD_LIBRARY_PATH" in script
    assert "scancel" not in script and "git pull" not in script


def test_bridge_authenticates_live_ledger_and_journal(case, monkeypatch, capsys):
    from hlt_classification.cms_proxy_ladder import literature as l, submission as s
    from hlt_classification.scouting.hcwdl_recovery import build_submission_event, build_submission_ledger
    jobs = case.evidence["jobs"]
    plan = sealed(commands=[dict(task_id=task, dependencies=[] if i == 0 else [list(jobs)[i-1]],
        command=["sbatch", "--wrap=true"] + ([] if i == 0 else
        ["--dependency=afterok:${JOB_" + list(jobs)[i-1] + "}"])) for i, task in enumerate(jobs)])
    monkeypatch.setattr(l, "validate_gate", lambda g, check_source: check_source is True)
    monkeypatch.setattr(s, "gate_plan", lambda g: plan)
    commands = {}
    for i, row in enumerate(plan["commands"]):
        command = row["command"]
        for task in row["dependencies"]:
            command = [a.replace("${JOB_" + task + "}", jobs[task]) for a in command]
        commands[row["task_id"]] = command
        event = build_submission_event(campaign_spec_sha256=case.gate["content_hash"],
            task_id=row["task_id"], job_id=jobs[row["task_id"]], command=command, sequence=i)
        save(case.root / f"gate/submission_ledger_journal/{i:04d}_{row['task_id']}.json", event)
    save(case.root / "gate/command_plan.json", plan)
    save(case.root / "gate/submission_ledger.json", build_submission_ledger(
        campaign_spec_sha256=case.gate["content_hash"], jobs=jobs, commands=commands, dry_run=False))
    monkeypatch.setattr(sys, "argv", ["inspect", case.gate["project_dir"], str(case.path)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(f.INSPECT, "inspect", "exec"), {})
    assert json.loads(capsys.readouterr().out)["jobs"] == jobs
    journal = case.root / "gate/submission_ledger_journal/0002_preflight.json"
    bad = f.read(journal)
    bad["job_id"] = "999"
    bad.pop("content_hash")
    save(journal, sealed(**bad))
    with pytest.raises(ValueError, match="ledger/journal differs"):
        exec(compile(f.INSPECT, "inspect", "exec"), {})
