"""Real S3 plan/submit journals, simulated Slurm and frozen preflight evidence."""
import copy
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hlt_classification import s3_ladder_followup as f
from hlt_classification.s3_ladder import campaign as c
from hlt_classification.s3_ladder.contracts import artifact, write_json
from hlt_classification.data.cache_contracts import with_content_hash


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@contextmanager
def fake_lock(root):
    root.mkdir(exist_ok=True)
    yield


def bridge(spec, path, mode):
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = ["bridge", spec["project_dir"], str(path), mode]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(compile(f.INSPECT, "pinned-s3-inspect", "exec"), {})
        return json.loads(buffer.getvalue())
    finally:
        sys.argv, sys.path = previous_argv, previous_path


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "jc2_s3_100k50k_aaaaaaaa_r1"
    root.mkdir()
    nodes = [dict(node_id=f.DIRECT, branch="DIRECT", coordinate="D000", teacher="OFFLINE")]
    for index, name in enumerate(f.COARSE):
        nodes.append(dict(node_id=name, branch="COARSE", coordinate=("D066", "D033", "D000")[index],
                          teacher="OFFLINE" if index == 0 else f.COARSE[index-1]))
    spec = artifact("SPEC", root=str(root), source_commit="a"*40, project_dir=str(tmp_path/"old-source"),
        counts=dict(train=100000, validation=50000), workers=6, cpus=6, memory_mb=90000,
        execution_site=dict(name="sporc_a100_debug", partition="debug", account="reu-aisocial",
                            qos="qos_tier3", gres="gpu:a100:1"),
        graph=dict(nodes=nodes, reducers=list(f.COARSE[:2])), dataset_export=False,
        development_only=True, automatic_followup=False)
    path = root / "study_spec.json"
    write_json(path, spec)
    accepted = dict(content_hash="b"*64, slurm_job_id="21826026", train_minutes=240, reduce_minutes=60)
    monkeypatch.setattr(c, "validate_spec", lambda s: {})
    monkeypatch.setattr(c, "preflight", lambda s, parent: accepted)
    monkeypatch.setattr(c, "check_submission_site", lambda value: None)
    sbatch = []
    def submit(argv, **kwargs):
        assert argv[0] == "sbatch"
        sbatch.append((argv, kwargs))
        return SimpleNamespace(stdout=str(21826025+len(sbatch))+"\n")
    monkeypatch.setattr(c.subprocess, "run", submit)
    c.submit(spec, mode="gate")
    gate = c.plan(spec, "gate")
    c.submit(spec, mode="gate", execute=True, plan_hash=gate["content_hash"], authorization=c.AUTHORIZATION)
    monkeypatch.setattr(f, "inspect", bridge)
    monkeypatch.setattr(f.common, "capture", lambda *a, **kw: "ClusterName = sporc\n")
    monkeypatch.setattr(f, "controller_source", lambda commit: dict(commit=commit, files={"controller": "c"*64}))
    monkeypatch.setattr(f.common, "lock", fake_lock)
    monkeypatch.setenv("USER", "ryreu")
    calls = []
    monkeypatch.setattr(f, "wait_gate", lambda job: calls.append(("wait", job)))
    def command(subject, *args):
        assert subject == spec
        calls.append(args)
        assert args[:5] == ("submit", "--spec", str(path.resolve()), "--mode", "science")
        live = "--execute" in args
        if live:
            assert (root / "science_followup/review.json").is_file()
        c.submit(subject, mode="science", execute=live,
            plan_hash=args[-1] if live else None, authorization=c.AUTHORIZATION if live else None)
    monkeypatch.setattr(f, "command", command)
    return SimpleNamespace(root=root, spec=spec, path=path, accepted=accepted, sbatch=sbatch, calls=calls,
        kwargs=dict(preflight_job="21826026", controller_commit="d"*40))


def test_read_only_then_full_dag_and_completed_repeat(case, capsys):
    assert f.discover(case.root.parent, "21826026") == case.path
    f.run(case.path, **case.kwargs)
    assert len(case.sbatch) == 1 and not (case.root/"science_followup").exists()
    f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 8
    assert "ARMED" in capsys.readouterr().out
    ledger = f.common.read(case.root/"submission_science/submission_ledger.json")
    assert list(ledger["jobs"]) != [] and set(ledger["jobs"]) == set(f.expected_tasks())
    for task, dependencies in f.expected_tasks().items():
        command = next(argv for argv, _ in case.sbatch if "--job-name=jc2s3_"+task in argv)
        actual = [a for a in command if a.startswith("--dependency=")]
        expected = ["--dependency=afterok:"+":".join(ledger["jobs"][p] for p in dependencies)] if dependencies else []
        assert actual == expected
    receipt = (case.root/"science_followup/complete.json").read_bytes()
    assert f.common.read(case.root/"science_followup/complete.json")["training_complete"] is False
    f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 8 and receipt == (case.root/"science_followup/complete.json").read_bytes()


def test_original_ledger_and_journal_authentication(case):
    spec, bound = f.bind(case.path, "21826026")
    assert spec == case.spec and bound["jobs"] == {"preflight": "21826026"}
    journal = case.root/"submission_gate/submission_ledger_journal/0000_preflight.json"
    value = f.common.read(journal)
    save(journal, with_content_hash(dict(value, job_id="999")))
    with pytest.raises(ValueError, match="ledger/journal differs"):
        f.bind(case.path, "21826026")


@pytest.mark.parametrize("mutation", ["wrong_job", "scope", "corrupt", "cluster", "canonical"])
def test_bad_binding_fails_before_arm(case, monkeypatch, mutation):
    path, kwargs = case.path, case.kwargs
    if mutation == "wrong_job":
        kwargs = dict(kwargs, preflight_job="999")
    elif mutation == "scope":
        save(path, with_content_hash(dict(case.spec, counts=dict(train=200000, validation=50000))))
    elif mutation == "corrupt":
        save(path, dict(case.spec, memory_mb=1))
    elif mutation == "cluster":
        monkeypatch.setattr(f.common, "capture", lambda *a, **kw: "ClusterName = tigris")
    else:
        path = case.root/"copy.json"
        save(path, case.spec)
    with pytest.raises((ValueError, PermissionError)):
        f.run(path, **kwargs, execute=True)
    assert len(case.sbatch) == 1 and not (case.root/"science_followup").exists()


def raw(state="COMPLETED", code="0:0"):
    return f"21826026|jc2s3_preflight|ryreu|reu-aisocial|debug|qos_tier3|{state}|{code}\n"


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("CANCELLED by 2914834", "0:0"),
    ("TIMEOUT", "0:0"), ("OUT_OF_MEMORY", "0:125"), ("COMPLETED", "1:0"), ("NODE_FAIL", "0:0"),
    ("UNKNOWN", "0:0")])
def test_failed_accounting_stops(state, code):
    with pytest.raises(RuntimeError, match="no science submitted"):
        f.state(raw(state, code), "21826026", "ryreu")


@pytest.mark.parametrize("before,after", [("21826026", "999"), ("jc2s3_preflight", "other"),
    ("ryreu", "other"), ("reu-aisocial", "wrong"), ("debug", "tier3"), ("qos_tier3", "wrong")])
def test_accounting_identity(before, after):
    with pytest.raises(PermissionError):
        f.state(raw().replace(before, after), "21826026", "ryreu")


def test_wait_uses_exact_job_missing_is_not_success_and_timeout(monkeypatch):
    output = iter(["", raw("PENDING"), raw("RUNNING"), raw()])
    calls, sleeps = [], []
    def capture(argv):
        calls.append(argv)
        return next(output)
    monkeypatch.setattr(f.common, "capture", capture)
    monkeypatch.setenv("USER", "ryreu")
    f.wait_gate("21826026", sleep=sleeps.append)
    assert sleeps == [60]*3
    assert all(argv[argv.index("-j")+1] == "21826026" for argv in calls)
    with pytest.raises(ValueError, match="duplicate"):
        f.state(raw()+raw(), "21826026", "ryreu")
    ticks = iter([0, 14*86400])
    with pytest.raises(TimeoutError):
        f.wait_gate("21826026", monotonic=lambda: next(ticks))


@pytest.mark.parametrize("failure", ["scheduler", "source", "acceptance", "acceptance_job"])
def test_gate_failures_never_submit(case, monkeypatch, failure):
    if failure == "scheduler":
        def wait(job):
            raise RuntimeError("Failed scheduler gate")
        monkeypatch.setattr(f, "wait_gate", wait)
    elif failure == "source":
        counter = iter([dict(commit="d"*40, files={"controller": "c"*64}), dict(commit="d"*40, files={})])
        monkeypatch.setattr(f, "controller_source", lambda commit: next(counter))
    elif failure == "acceptance":
        def invalid(*a):
            raise ValueError("Invalid GPU evidence")
        monkeypatch.setattr(c, "preflight", invalid)
    else:
        case.accepted["slurm_job_id"] = "999"
    with pytest.raises((ValueError, RuntimeError)):
        f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 1 and not (case.root/"submission_science").exists()


def test_spec_change_during_wait(case, monkeypatch):
    def wait(job):
        save(case.path, with_content_hash(dict(case.spec, source_commit="e"*40)))
    monkeypatch.setattr(f, "wait_gate", wait)
    with pytest.raises(ValueError, match="Saved command plan differs|changed while waiting"):
        f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 1


def test_controller_change_after_dry_review_stops_before_live(case, monkeypatch):
    original = dict(commit="d"*40, files={"controller": "c"*64})
    answers = iter([original, original, dict(commit="d"*40, files={})])
    monkeypatch.setattr(f, "controller_source", lambda commit: next(answers))
    with pytest.raises(ValueError, match="during dry review"):
        f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 1
    assert (case.root/"submission_science/dry_run_submission_ledger.json").exists()
    assert not (case.root/"submission_science/live_submission_claim.json").exists()


@pytest.mark.parametrize("change", ["extra", "dependency", "partition", "cpu", "memory", "gpu", "time", "duplicate", "parent"])
def test_plan_policy_rejects_expansion(case, change):
    plan = c.plan(case.spec, "science")
    row = plan["commands"][0]
    if change == "extra":
        plan["commands"].append(copy.deepcopy(row))
    elif change == "dependency":
        row["dependencies"] = ["fit_UNKNOWN"]
    elif change == "duplicate":
        row["command"].append("--partition=debug")
    elif change == "parent":
        plan["parents"]["preflight"] = "f"*64
    else:
        key, value = {"partition": ("--partition", "tier3"), "cpu": ("--cpus-per-task", "4"),
            "memory": ("--mem", "64000M"), "gpu": ("--gres", "gpu:h100:1"),
            "time": ("--time", "2-00:00:00")}[change]
        row["command"] = [key+"="+value if arg.startswith(key+"=") else arg for arg in row["command"]]
    with pytest.raises((ValueError, PermissionError)):
        f.review(plan, case.spec, bridge(case.spec, case.path, "preflight"))


def test_24_hour_measured_cap_accepted_over_cap_rejected(case):
    case.accepted["train_minutes"] = 1440
    f.review(c.plan(case.spec, "science"), case.spec, bridge(case.spec, case.path, "preflight"))
    case.accepted["train_minutes"] = 1441
    with pytest.raises(ValueError, match="ceiling"):
        f.review(c.plan(case.spec, "science"), case.spec, bridge(case.spec, case.path, "preflight"))


def test_interrupted_live_submit_refuses_repeat_without_duplicates(case, monkeypatch):
    original = c.subprocess.run
    def fail(argv, **kwargs):
        if len(case.sbatch) >= 2:
            raise RuntimeError("ambiguous Slurm response")
        return original(argv, **kwargs)
    monkeypatch.setattr(c.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="ambiguous"):
        f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 2
    assert (case.root/"submission_science/submission_ledger_journal").is_dir()
    assert not (case.root/"science_followup/complete.json").exists()
    with pytest.raises(PermissionError, match="no automatic retry"):
        f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 2


def test_completed_but_corrupt_journal_blocks_reuse(case):
    f.run(case.path, **case.kwargs, execute=True)
    journal = next((case.root/"submission_science/submission_ledger_journal").glob("*.json"))
    value = f.common.read(journal)
    save(journal, with_content_hash(dict(value, job_id="999")))
    with pytest.raises(ValueError):
        f.run(case.path, **case.kwargs, execute=True)
    assert len(case.sbatch) == 8


def test_atomic_publish_and_immutable_authorization(tmp_path):
    path = tmp_path/"authorization.json"
    receipt = f.publish(path, kind="authorization", job="123")
    assert receipt["contract"] == "JC2_S3_FOLLOWUP/v1"
    assert f.publish(path, kind="authorization", job="123") == receipt
    with pytest.raises(FileExistsError):
        f.publish(path, kind="authorization", job="124")
    save(path, dict(receipt, job="124"))
    with pytest.raises(ValueError, match="checksum"):
        f.common.read(path)
    assert not list(tmp_path.glob(".s3-followup-*"))


def test_duplicate_controller_fails_under_real_lock_api(tmp_path, monkeypatch):
    def blocked(*a):
        raise BlockingIOError()
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=blocked, LOCK_EX=1, LOCK_NB=2, LOCK_UN=4))
    with pytest.raises(PermissionError, match="already active"):
        with f.common.lock(tmp_path/"followup"):
            pytest.fail("Second controller ran")


def test_discovery_rejects_missing_and_duplicate(case):
    with pytest.raises(ValueError, match="found 0"):
        f.discover(case.root.parent, "999")
    duplicate = case.root.parent/"jc2_s3_100k50k_other_r1/submission_gate/submission_ledger.json"
    save(duplicate, f.common.read(case.root/"submission_gate/submission_ledger.json"))
    with pytest.raises(ValueError, match="found 2"):
        f.discover(case.root.parent, "21826026")


def test_original_pinned_cli_and_scrubbed_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("SBATCH_PARTITION", "tier3")
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.setattr(f.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    f.command(dict(project_dir=str(tmp_path/"OLD")), "submit", "--mode", "science")
    argv, kwargs = calls[0]
    assert argv[3] == str(tmp_path/"OLD/scripts/jetclass2_s3_ladder.py")
    assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kwargs["env"])
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1" and kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_controller_source_requires_clean_exact_pushed_commit(monkeypatch):
    captures = []
    def capture(argv):
        captures.append(argv)
        if "rev-parse" in argv:
            return "d"*40
        return ""
    monkeypatch.setattr(f.common, "capture", capture)
    value = f.controller_source("d"*40)
    assert value["commit"] == "d"*40 and set(value["files"]) == set(f.CONTROLLER_FILES)
    assert any("--is-ancestor" in argv for argv in captures)
    with pytest.raises(PermissionError):
        f.controller_source("e"*40)
    monkeypatch.setattr(f.common, "capture", lambda argv: "d"*40 if "rev-parse" in argv else " M dirty")
    with pytest.raises(PermissionError):
        f.controller_source("d"*40)


def test_bridge_syntax_and_launcher_safety():
    compile(f.INSPECT, "pinned-bridge", "exec")
    path = Path(__file__).resolve().parents[1]/"scripts/start_jetclass2_s3_ladder_followup.sh"
    script = path.read_text()
    for expected in ("nohup", '</dev/null &', "PYTHONNOUSERSITE=1", "LD_LIBRARY_PATH", "--execute"):
        assert expected in script
    assert "scancel" not in script and "git pull" not in script
