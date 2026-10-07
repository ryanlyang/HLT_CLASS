"""Fake scheduler coverage for the exact debug-gate -> tier3 controller."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hlt_classification import correlated_ladder_followup as f
from hlt_classification.cms_proxy_ladder.contracts import artifact, write_json
from hlt_classification.data.cache_contracts import with_content_hash
from test_correlated_tier3 import transferred


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    manifest = artifact("TEST_MANIFEST")
    monkeypatch.setattr(f, "DATASET_HASH", manifest["content_hash"])
    save(tmp_path / "dataset/dataset_manifest.json", manifest)
    gate = artifact("GATE_SPEC", version=9, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path / "original-code"), source_commit=f.tier3.BASE_COMMIT,
        request=dict(study_root=str(tmp_path / "dataset")))
    path = tmp_path / "gate/gate_spec.json"
    save(path, gate)
    binding = dict(jobs={"authenticate_release": "101", "build_foundation": "102", "preflight": "103"},
                   ledger_hash="b" * 64)
    monkeypatch.setenv("USER", "ryreu")
    return SimpleNamespace(root=tmp_path, path=path, gate=gate, binding=binding,
        kwargs=dict(gate_hash=gate["content_hash"], preflight_job="103",
                    project_dir=str(tmp_path / "new-code"), executor_commit="f"*40))


def accounting(c, state="COMPLETED", code="0:0"):
    return "\n".join(f"{job}|jc2crg_{task}|ryreu|reu-aisocial|{state}|{code}"
                     for task, job in c.binding["jobs"].items())


def test_bind_original_pinned_gate_not_new_worktree(case, monkeypatch):
    calls = []
    def capture(argv, **kwargs):
        calls.append(argv)
        return "ClusterName = sporc\n" if argv[0] == "scontrol" else json.dumps(case.binding)
    monkeypatch.setattr(f.common, "capture", capture)
    assert f.bind(case.path, case.kwargs["gate_hash"], "103") == (case.gate, case.binding)
    assert calls[1][-2:] == [case.gate["project_dir"], str(case.path)]
    with pytest.raises(ValueError, match="Preflight ID"):
        f.bind(case.path, case.kwargs["gate_hash"], "999")
    with pytest.raises(ValueError, match="identity"):
        f.bind(case.path, "0"*64, "103")


def test_wrong_cluster_or_manifest_stops(case, monkeypatch):
    monkeypatch.setattr(f.common, "capture", lambda *a, **kw: "ClusterName = tigris\n")
    with pytest.raises(PermissionError, match="SPORC"):
        f.bind(case.path, case.kwargs["gate_hash"], "103")
    monkeypatch.setattr(f.common, "capture", lambda argv, **kw:
        "ClusterName = sporc\n" if argv[0] == "scontrol" else json.dumps(case.binding))
    monkeypatch.setattr(f, "DATASET_HASH", "0"*64)
    with pytest.raises(ValueError, match="dataset manifest"):
        f.bind(case.path, case.kwargs["gate_hash"], "103")


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("CANCELLED by 42", "0:0"),
    ("TIMEOUT", "0:0"), ("OUT_OF_MEMORY", "0:125"), ("NODE_FAIL", "0:0"), ("COMPLETED", "1:0")])
def test_failed_gate_never_admits_science(case, state, code):
    with pytest.raises(RuntimeError, match="no science"):
        f.states(accounting(case, state, code), case.binding["jobs"], "ryreu")


@pytest.mark.parametrize("before,after", [("ryreu", "other"), ("reu-aisocial", "wrong"),
    ("jc2crg_preflight", "unrelated"), ("103|", "999|")])
def test_wrong_job_identity(case, before, after):
    with pytest.raises((PermissionError, ValueError)):
        f.states(accounting(case).replace(before, after), case.binding["jobs"], "ryreu")


def test_missing_accounting_waits_and_known_ids_only(case, monkeypatch):
    output = iter(["", accounting(case, "PENDING"), accounting(case, "RUNNING"), accounting(case)])
    calls, sleeps = [], []
    def capture(argv):
        calls.append(argv)
        return next(output)
    monkeypatch.setattr(f.common, "capture", capture)
    f.wait_gate(case.binding["jobs"], sleep=sleeps.append)
    assert sleeps == [60, 60, 60]
    assert all("101,102,103" in a for a in calls)
    with pytest.raises(ValueError, match="duplicate"):
        f.states(accounting(case) + "\n" + accounting(case), case.binding["jobs"], "ryreu")


def test_wait_is_bounded(case, monkeypatch):
    monkeypatch.setattr(f.common, "capture", lambda *a: "")
    ticks = iter([0, 0, 14*86400])
    with pytest.raises(TimeoutError):
        f.wait_gate(case.binding["jobs"], sleep=lambda n: None, monotonic=lambda: next(ticks))


def harness(c, monkeypatch):
    calls = []
    @contextmanager
    def lock(path):
        path.mkdir(exist_ok=True)
        yield
    monkeypatch.setattr(f, "controller_source", lambda *a: dict(commit="f"*40, files={}))
    monkeypatch.setattr(f, "bind", lambda *a: (copy.deepcopy(c.gate), copy.deepcopy(c.binding)))
    monkeypatch.setattr(f.tier3, "source_transfer", lambda *a: None)
    monkeypatch.setattr(f.common, "lock", lock)
    monkeypatch.setattr(f, "wait_gate", lambda jobs: calls.append("wait"))
    def original(gate, root):
        calls.append("original")
        path = root / "science/campaign_spec.json"
        save(path, artifact("ORIGINAL"))
        return path
    monkeypatch.setattr(f, "original_campaign", original)
    def create(**kwargs):
        calls.append("create")
        path = Path(kwargs["campaign_root"])/"campaign_spec.json"
        save(path, artifact("CAMPAIGN_SPEC", version=6, campaign_root=str(path.parent),
            gate_root=c.gate["gate_root"], source_commit="f"*40, project_dir=c.kwargs["project_dir"],
            measurement_campaign={"path": str(kwargs["measurement_spec"])}, fresh_fit_count=6, reducer_count=3,
            selected_branches=["DIRECT", "COARSE"], runtime_profile=dict(train_minutes=60, reduce_minutes=30)))
    monkeypatch.setattr(f.tier3, "create_campaign", create)
    def submit(spec, **kwargs):
        calls.append("submit" if kwargs else "dry")
        plan = artifact("COMMAND_PLAN", mode="science", parents={"subject": spec["content_hash"]},
            commands=[dict(task_id=f"train_{i}", command=["sbatch", "--partition=tier3", "--account=reu-aisocial",
                "--qos=qos_tier3", "--gres=gpu:a100:1", "--cpus-per-task=6", "--mem=90000M"])
                for i in range(11)])
        if not kwargs:
            return plan
        assert kwargs == dict(execute=True, plan_hash=plan["content_hash"], authorization_phrase=f.tier3.AUTHORIZATION)
        assert (c.root / "followup_tier3/review.json").is_file()
        return artifact("LEDGER", jobs={str(i): str(1000+i) for i in range(11)})
    monkeypatch.setattr(f.tier3, "submit", submit)
    return calls


def test_full_controller_dry_review_then_live_and_repeat(case, monkeypatch):
    calls = harness(case, monkeypatch)
    f.run(case.path, **case.kwargs, execute=True)
    assert calls == ["wait", "original", "create", "dry", "submit"]
    done = (case.root / "followup_tier3/complete.json").read_bytes()
    assert f.common.read(case.root / "followup_tier3/complete.json")["training_complete"] is False
    f.run(case.path, **case.kwargs, execute=True)
    assert calls[-4:] == ["wait", "original", "dry", "submit"]
    assert (case.root / "followup_tier3/complete.json").read_bytes() == done


def test_read_only_review_no_wait_or_write(case, monkeypatch):
    calls = harness(case, monkeypatch)
    f.run(case.path, **case.kwargs)
    assert calls == [] and not (case.root / "followup_tier3").exists()


def test_failed_gate_controller_stops_before_science(case, monkeypatch):
    calls = harness(case, monkeypatch)
    def fail(jobs):
        raise RuntimeError("preflight failed")
    monkeypatch.setattr(f, "wait_gate", fail)
    with pytest.raises(RuntimeError):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls == [] and not (case.root / "science_tier3").exists()


@pytest.mark.parametrize("name", ["submission_ledger.json", "live_submission_claim.json", "submission_ledger_journal"])
def test_existing_debug_science_blocks_arming(case, monkeypatch, name):
    calls = harness(case, monkeypatch)
    save(case.root / "science" / name, {})
    with pytest.raises(PermissionError, match="Debug science"):
        f.run(case.path, **case.kwargs, execute=True)
    assert not calls


def test_gate_drift_and_ambiguous_submit_stop_without_retry(case, monkeypatch):
    calls = harness(case, monkeypatch)
    values = iter([(case.gate, case.binding), ({**case.gate, "source_commit": "a"*40}, case.binding)])
    monkeypatch.setattr(f, "bind", lambda *a: next(values))
    with pytest.raises(ValueError, match="changed"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls == ["wait"]
    monkeypatch.setattr(f, "bind", lambda *a: (case.gate, case.binding))
    real_submit = f.tier3.submit
    def submit(spec, **kwargs):
        if kwargs:
            calls.append("failure")
            raise RuntimeError("ambiguous sbatch")
        return real_submit(spec)
    monkeypatch.setattr(f.tier3, "submit", submit)
    with pytest.raises(RuntimeError, match="ambiguous"):
        f.run(case.path, **case.kwargs, execute=True)
    assert calls.count("failure") == 1 and not (case.root / "followup_tier3/complete.json").exists()


def test_review_uses_actual_exact_tier3_plan(transferred):
    c = transferred
    plan = f.tier3.submit(c.spec)
    assert f.review(plan, c.spec) == plan["content_hash"]
    for mutation in ("partition", "cpu", "jobs"):
        bad = copy.deepcopy(plan)
        if mutation == "jobs":
            bad["commands"].pop()
        else:
            old, new = ("--partition=tier3", "--partition=debug") if mutation == "partition" else (
                "--cpus-per-task=6", "--cpus-per-task=36")
            bad["commands"][0]["command"] = [new if a == old else a for a in bad["commands"][0]["command"]]
        with pytest.raises((ValueError, PermissionError)):
            f.review(with_content_hash(bad), c.spec)


def test_inspection_bridge_and_launcher_no_implicit_arm():
    compile(f.INSPECT, "inspect", "exec")
    assert "import correlated as l, submission as s" in f.INSPECT
    assert "import literature as l" not in f.INSPECT
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/start_jetclass2_correlated_tier3_followup.sh").read_text()
    assert 'if [[ $# == 4 ]]' in script and '[[ "$5" == --execute ]]' in script
    assert "nohup" in script and '</dev/null &' in script
    assert 'JC2_SITE=sporc_a100' in script and '"${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"' in script
    assert "scancel" not in script and "git pull" not in script


def test_bridge_authenticates_correlated_ledger_journal(case, monkeypatch, capsys):
    from hlt_classification.cms_proxy_ladder import correlated as c, submission as s
    from hlt_classification.scouting.hcwdl_recovery import build_submission_event, build_submission_ledger
    jobs = case.binding["jobs"]
    plan = artifact("COMMAND_PLAN", commands=[dict(task_id=task, dependencies=[] if i == 0 else [list(jobs)[i-1]],
        command=["sbatch", "--wrap=true"] + ([] if i == 0 else
        ["--dependency=afterok:${JOB_" + list(jobs)[i-1] + "}"])) for i, task in enumerate(jobs)])
    monkeypatch.setattr(c, "validate_gate", lambda g, check_source: None)
    monkeypatch.setattr(s, "gate_plan", lambda g: plan)
    commands = {}
    for i, row in enumerate(plan["commands"]):
        command = row["command"]
        for parent in row["dependencies"]:
            command = [a.replace("${JOB_" + parent + "}", jobs[parent]) for a in command]
        commands[row["task_id"]] = command
        save(case.root / f"gate/submission_ledger_journal/{i:04d}_{row['task_id']}.json",
             build_submission_event(campaign_spec_sha256=case.gate["content_hash"],
                 task_id=row["task_id"], job_id=jobs[row["task_id"]], command=command, sequence=i))
    save(case.root / "gate/command_plan.json", plan)
    save(case.root / "gate/submission_ledger.json", build_submission_ledger(
        campaign_spec_sha256=case.gate["content_hash"], jobs=jobs, commands=commands, dry_run=False))
    monkeypatch.setattr(sys, "argv", ["inspect", case.gate["project_dir"], str(case.path)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(f.INSPECT, "inspect", "exec"), {})
    assert json.loads(capsys.readouterr().out)["jobs"] == jobs
    journal = case.root / "gate/submission_ledger_journal/0002_preflight.json"
    save(journal, with_content_hash(dict(f.common.read(journal), job_id="999")))
    with pytest.raises(ValueError, match="ledger/journal"):
        exec(compile(f.INSPECT, "inspect", "exec"), {})
