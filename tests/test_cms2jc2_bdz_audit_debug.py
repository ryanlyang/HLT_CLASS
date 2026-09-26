from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from hlt_classification.cms2jc2_response import (
    bdz_audit_debug as debug, bdz_audit_debug_worker as adapter,
    bdz_audit_campaign as original, bdz_audit_worker as worker,
    bdz_campaign as bdz, bdz_worker as old, dev_campaign as dev,
    dev_submission as submission, dev_data as data)
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.storage import GIB
from hlt_classification.cms2jc2_response.c_diagnostic_worker import historical_replay_equal
from test_cms2jc2_bdz_tuning import completed_bdz
from test_cms2jc2_bounded import bounded_root, LocalMeasurement
from test_cms2jc2_b_tracking import b_root
from test_cms2jc2_frozen_c import frozen_root, free_disk


def test_debug_dag_and_explicit_helper_contract():
    tasks = debug.tasks()
    assert [t["task_id"] for t in tasks] == list(debug.TARGETS)
    assert [t["cpus"] for t in tasks] == [36]*4+[1]
    assert [t["memory_gib"] for t in tasks] == [128]*4+[32]
    assert [t["hours"] for t in tasks] == [8]*4+[4]
    assert all(not t["depends_on"] and t["action"] == "ba_evaluate" for t in tasks[:4])
    assert tasks[-1]["depends_on"] == list(debug.TARGETS[:4])
    helper = (Path(__file__).parents[1]/"scripts/queue_cms2jc2_bdz_audit_debug.sh").read_text()
    assert "still_running" in helper and 'MODE="${4:-}"' in helper
    assert helper.index("run_phase retire") < helper.index("run_phase submit")
    assert "git push" not in helper and "scontrol update" not in helper
    cli = (Path(__file__).parents[1]/"scripts/cms2jc2_response_dev.py").read_text()
    for name in ("create-bdz-audit-debug", "retire-bdz-audit-debug", "bdz-audit-results"):
        assert name in cli


@pytest.mark.parametrize("status", [0, 7])
def test_helper_phase_preserves_stdin_and_exit_status(status):
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files"))/"Git/bin/bash.exe"
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is not installed")
    helper = (Path(__file__).parents[1]/"scripts/queue_cms2jc2_bdz_audit_debug.sh").read_text()
    function = "run_phase() {"+helper.split("run_phase() {", 1)[1].split("\nPROJECT_DIR=", 1)[0]
    script = "set -euo pipefail\n"+function+"\nsleep() { command sleep 0.01; }\n"
    script += f"""
result=0
run_phase fixture bash -c 'IFS= read -r value; printf "%s\\n" "$value"; exit {status}' <<'INPUT' || result=$?
preserved child input
INPUT
printf 'RESULT=%s\\n' "$result"
"""
    result = subprocess.run([bash, "--noprofile", "--norc", "-s"], input=script,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "preserved child input" in result.stdout and f"RESULT={status}" in result.stdout
    assert "phase=fixture starting" in result.stderr and f"phase=fixture exit={status}" in result.stderr


@pytest.mark.parametrize("kind", ["BDZ_STAGE", "BDZ_TIER3", "BDZ_AUDIT_DEBUG", "DEV_C_TOPOLOGY"])
def test_wrong_family_cannot_be_retired(kind):
    with pytest.raises(ValueError, match="original tier3 significance"):
        debug.subject_metadata(dict(contract=f"CMS2JC2_RESPONSE_{kind}/v1"))


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETING", "COMPLETED", "UNKNOWN", "SUSPENDED", "CONFIGURING"])
def test_active_successful_unknown_targets_are_preserved(state, monkeypatch):
    jobs = {t: str(i+100) for i, t in enumerate(("ba_acceptance", *debug.TARGETS))}
    states = {job: ("PENDING", "0:0") for job in jobs.values()}
    states[jobs["ba_acceptance"]] = ("COMPLETED", "0:0")
    states[jobs["ba_eval_0"]] = (state, "0:0")
    monkeypatch.setattr(submission, "states", lambda _: states)
    with pytest.raises(PermissionError, match="Preserve original audit"):
        debug.old_states(dict(migration=dict(subject_jobs=jobs)))


def test_acceptance_accounting_must_be_successful(monkeypatch):
    jobs = {t: str(i+100) for i, t in enumerate(("ba_acceptance", *debug.TARGETS))}
    monkeypatch.setattr(submission, "states", lambda _: {j: ("COMPLETED", "1:0") for j in jobs.values()})
    with pytest.raises(PermissionError, match="COMPLETED/0:0"):
        debug.old_states(dict(migration=dict(subject_jobs=jobs)))


@pytest.mark.parametrize("change", [dict(projected_seconds=1.), dict(projected_peak_bytes=1.),
    dict(projected_seconds=float("nan")), dict(measurement=dict(wall_seconds=-1, sampled_peak_tree_rss_bytes=1))])
def test_acceptance_projections_are_recomputed(change, monkeypatch):
    row = dict(jets=32, measurement=dict(wall_seconds=1., sampled_peak_tree_rss_bytes=1024),
               projected_seconds=2*(original.COUNTS["evaluation"]//4)/32, projected_peak_bytes=1.5*1024*18+2*GIB)
    monkeypatch.setattr(worker, "inputs", lambda s: (None, None, None, {}))
    monkeypatch.setattr(worker, "accepted", lambda s, r: dict(row, **change))
    with pytest.raises((PermissionError, ValueError), match="acceptance"):
        debug.acceptance({})


@pytest.mark.parametrize("marker", ["claims/ba_eval_0", "receipts/ba_report.json"])
def test_original_claims_and_outputs_block_migration(marker, tmp_path, monkeypatch):
    subject = dict(contract=original.CONTRACT, root=str(tmp_path), name="audit")
    monkeypatch.setattr(original, "validate_stage", lambda *a, **k: {})
    monkeypatch.setattr(dev, "verified_outputs", lambda *a: {})
    path = dev.stage_dir(subject)/marker
    if marker.startswith("claims"):
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    with pytest.raises(PermissionError, match="Preserve"):
        debug.subject_metadata(subject)


def test_adapter_does_not_expose_acceptance_rerun():
    with pytest.raises(ValueError, match="cannot rerun"):
        adapter.dispatch({}, {}, dict(action="ba_acceptance"))


@pytest.mark.parametrize("fault", ["dry_run", "wrong_id", "missing_acceptance"])
def test_original_ledger_requires_all_acknowledged_jobs(fault, tmp_path, monkeypatch):
    subject = dict(contract=original.CONTRACT, root=str(tmp_path), name="original", content_hash="f"*64)
    plan = dict(content_hash="e"*64)
    jobs = {t: str(100+i) for i, t in enumerate(("ba_acceptance", *debug.TARGETS))}
    ledger_jobs = dict(jobs)
    if fault == "wrong_id":
        ledger_jobs["ba_eval_0"] = "99999"
    if fault == "missing_acceptance":
        del jobs["ba_acceptance"]
        del ledger_jobs["ba_acceptance"]
    ledger = artifact("DEV_LEDGER", parents=dict(stage=subject["content_hash"], plan=plan["content_hash"]),
                      jobs=ledger_jobs, dry_run=fault == "dry_run")
    monkeypatch.setattr(original, "validate_stage", lambda *a, **k: {})
    monkeypatch.setattr(debug, "acceptance", lambda s: {})
    monkeypatch.setattr(dev, "command_plan", lambda *a: plan)
    monkeypatch.setattr(debug, "load_json", lambda p: plan if p.name == "command_plan.json" else ledger)
    monkeypatch.setattr(submission, "submitted_jobs", lambda *a: jobs)
    with pytest.raises(ValueError, match="ledger/journals"):
        debug.subject_metadata(subject)


def test_worker_requires_retirement_before_claiming_work(tmp_path, monkeypatch):
    from hlt_classification.cms2jc2_response import dev_worker as runtime
    spec = dict(contract=debug.CONTRACT, name=debug.NAME, root=str(tmp_path))
    monkeypatch.setattr(runtime, "validate_stage", lambda s: {})
    def refuse(s):
        assert s is spec
        raise PermissionError("Missing retirement evidence")
    monkeypatch.setattr(debug, "verify_retirement", refuse)
    with pytest.raises(PermissionError, match="Missing retirement"):
        runtime.run(spec, "ba_eval_0")
    assert not list(tmp_path.iterdir())


def test_full_debug_migration_real_root_and_replay(completed_bdz, tmp_path, monkeypatch):
    parent, ctx, put, _ = completed_bdz
    for module in (original, worker):
        monkeypatch.setattr(module, "COUNTS", data.COUNTS)
    monkeypatch.setattr(worker, "Measurement", LocalMeasurement)
    gate = bdz.create(parent_spec=dev.stage_dir(parent)/"stage_spec.json", project_dir=tmp_path/"bp",
        source_commit="c"*40, root=tmp_path/"bz")
    put(gate, "bz_acceptance", dict(result=old.acceptance(ctx, gate)))
    put(gate, "bz_calibrate", dict(result=old.calibrate(ctx, gate, dict(cpus=1))))
    donor = bdz.advance(dev.stage_dir(gate)/"stage_spec.json")
    for t in donor["tasks"][:4]:
        put(donor, t["task_id"], dict(result=old.evaluate(ctx, donor, dict(t, cpus=1))))
    monkeypatch.setattr(old, "plot_pages", lambda *a, **k: [])
    put(donor, "bz_select", dict(result=old.report(ctx, donor)))
    subject = original.create(parent_spec=dev.stage_dir(donor)/"stage_spec.json", project_dir=tmp_path/"ap",
        source_commit="d"*40, root=tmp_path/"audit")
    subject_path = dev.stage_dir(subject)/"stage_spec.json"
    put(subject, "ba_acceptance", dict(result=worker.acceptance(ctx, subject)))
    old_study = original.validate_stage(subject)
    calls, states, jobs = [], {"99999": ("RUNNING", "0:0")}, {}
    flags = dict(race=None, reject=False, identity=False)
    monkeypatch.setenv("USER", "fixture")

    def scheduler(argv):
        calls.append(argv)
        out = ""
        if argv[:3] == ["scontrol", "show", "partition"]:
            out = f"PartitionName={argv[3]} State=UP"
        elif argv[0] == "sbatch":
            if "--test-only" in argv:
                if flags["reject"]:
                    return SimpleNamespace(returncode=1, stdout="", stderr="Rejected debug shape")
            else:
                job = str(5000+len(jobs))
                task = next(a.split("=", 1)[1][5:] for a in argv if a.startswith("--job-name=c2jd_"))
                jobs[job] = task
                states[job] = ("PENDING", "0:0")
                out = job+"\n"
        elif argv[0] == "sacct":
            out = "\n".join(f"{j}|{s}|{c}" for j, (s, c) in states.items())
        elif argv[:3] == ["scontrol", "show", "job"]:
            job, task = argv[-1], jobs[argv[-1]]
            t = next(t for t in subject["tasks"] if t["task_id"] == task)
            state = "RUNNING" if flags["identity"] else states[job][0]
            out = (f"JobId={job} JobState={state} JobName=c2jd_{task} UserId=fixture(1) "
                f"Comment=c2jd:{subject['content_hash']}:{task} WorkDir={old_study['project_dir']} "
                f"Command={Path(old_study['project_dir'])/'sbatch/run_cms2jc2_response_dev_cpu.sh'} "
                f"Partition=tier3 Account=reu-aisocial QOS=qos_tier3 NumCPUs={t['cpus']} NumNodes=1-1 "
                "ReqTRES=cpu=36,mem=128G,node=1")
        elif argv[0] == "scancel":
            assert argv[1:5] == ["--ctld", "--state=PENDING", "--partition=tier3", "--account=reu-aisocial"]
            assert set(argv[5:]) <= {ledger["jobs"][t] for t in debug.TARGETS}
            assert ledger["jobs"]["ba_acceptance"] not in argv and "99999" not in argv
            for job in argv[5:]:
                states[job] = ("CANCELLED", "0:0")
            if flags["race"]:
                states[ledger["jobs"]["ba_eval_0"]] = (flags["race"], "0:0")
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(submission, "scheduler", scheduler)
    plan = submission.submit(subject)
    ledger = submission.submit(subject, execute=True, authorization_phrase=dev.PHRASES["bdz_audit"],
                               reviewed_plan_hash=plan["content_hash"])
    states[ledger["jobs"]["ba_acceptance"]] = ("COMPLETED", "0:0")
    before = {p: sha256_file(p) for root in (Path(donor["root"]), Path(subject["root"]))
              for p in root.rglob("*") if p.is_file()}
    spec = debug.create(parent_spec=subject_path, project_dir=tmp_path/"dp", source_commit="e"*40, root=tmp_path/"debug")
    study = debug.validate_stage(spec)
    assert spec["migration"]["acceptance_reused"] and not spec["migration"]["acceptance_rerun"]
    changed = deepcopy(study); changed["source"]["files"]["scientific.py"] = "e"*64
    with pytest.raises(ValueError, match="scientific source"):
        debug.reuse(subject, changed)
    bad = with_content_hash(dict(spec, tasks=subject["tasks"]))
    with pytest.raises(ValueError, match="registration"):
        debug.validate_stage(bad)
    # Scheduler race tests reuse already-authenticated metadata to avoid repeating
    # full donor decoding for each mocked scheduler transition. Restore below.
    real_metadata = debug.subject_metadata
    metadata = real_metadata(subject)
    with monkeypatch.context() as patch:
        patch.setattr(debug, "validate_stage", lambda s, **k: study)
        patch.setattr(debug, "subject_metadata", lambda s: metadata)
        newplan = submission.submit(spec)
        assert len(newplan["commands"]) == 5 and all("--partition=debug" in r["argv"] for r in newplan["commands"])
        kw = dict(execute=True, authorization_phrase=debug.RETIRE_PHRASE, reviewed_plan_hash=newplan["content_hash"])
        submit_kw = dict(execute=True, authorization_phrase=dev.PHRASES["bdz_audit"], reviewed_plan_hash=newplan["content_hash"])
        n = len(calls)
        with pytest.raises(FileNotFoundError):
            submission.submit(spec, **submit_kw)
        assert not any(a[0] == "sbatch" for a in calls[n:])
        assert debug.retire(spec)["read_only"]
        with pytest.raises(PermissionError, match="phrase"):
            debug.retire(spec, execute=True)
        flags["reject"] = True
        with pytest.raises(PermissionError, match="Site rejected"):
            debug.retire(spec, **kw)
        assert not any(a[0] == "scancel" for a in calls)
        flags.update(reject=False, identity=True)
        with pytest.raises(PermissionError, match="provenance/allocation differs|nothing cancelled"):
            debug.retire(spec, **kw)
        assert not any(a[0] == "scancel" for a in calls)
        flags.update(identity=False, race="PENDING")
        with pytest.raises(RuntimeError, match="not yet terminal"):
            debug.retire(spec, **kw)
        assert not debug.retirement_path(spec).exists()
        for t in debug.TARGETS:
            states[ledger["jobs"][t]] = ("PENDING", "0:0")
        flags["race"] = "RUNNING"
        with pytest.raises(PermissionError, match="Preserve original audit"):
            debug.retire(spec, **kw)
        assert not debug.retirement_path(spec).exists()
        states[ledger["jobs"]["ba_eval_0"]] = ("CANCELLED", "0:0")
        flags["race"] = None
        retired = debug.retire(spec, **kw)
        assert debug.retire(spec, **kw) == retired
        states[ledger["jobs"]["ba_eval_0"]] = ("PENDING", "0:0")
        with pytest.raises(PermissionError, match="can still run"):
            submission.submit(spec, **submit_kw)
        states[ledger["jobs"]["ba_eval_0"]] = ("CANCELLED", "0:0")
        n = len(calls)
        fresh = submission.submit(spec, **submit_kw)
        live = [a for a in calls[n:] if a[0] == "sbatch" and "--test-only" not in a]
        assert len(live) == 5
        assert all(not any(v.startswith("--dependency") for v in a) for a in live[:4])
        assert "--dependency=afterok:"+":".join(fresh["jobs"][t] for t in debug.TARGETS[:4]) in live[-1]
        n = len(calls)
        assert submission.submit(spec, **submit_kw) == fresh
        assert not any(a[0] == "sbatch" for a in calls[n:])
        assert states["99999"] == ("RUNNING", "0:0")
    assert debug.verify_retirement(spec, live=True) == retired
    assert before == {p: sha256_file(p) for p in before}
    real_rows, baseline_rows = [], []
    for t in spec["tasks"][:4]:
        row = adapter.evaluate(ctx, spec, dict(t, cpus=2 if t["params"]["shard"] == 0 else 1))
        expected = worker.evaluate(ctx, subject, dict(t, cpus=1))
        baseline_rows.append(expected)
        expected = with_content_hash(dict(expected, parents=dict(expected["parents"], stage=spec["content_hash"])))
        assert historical_replay_equal(row, expected)
        real_rows.append(row)
        put(spec, t["task_id"], dict(result=row))
    result = adapter.report(ctx, spec)
    assert result["parents"]["acceptance"] == spec["migration"]["parents"]["acceptance"]
    assert result["jets"] == data.COUNTS["evaluation"] and len(result["figures"]) == 2
    assert result["shard_hashes"] == [r["content_hash"] for r in real_rows]
    assert not (dev.stage_dir(spec)/"receipts/ba_acceptance.json").exists()
    # Run the original report logic without publishing into its preserved root.
    # Only shard lookup and figure publication are redirected; scientific
    # validation, merging and all diagnostics remain the real donor functions.
    real_product = dev.product
    def baseline_product(s, task, key):
        if s["content_hash"] == subject["content_hash"] and task.startswith("ba_eval_"):
            assert key == "result"
            return baseline_rows[int(task.rsplit("_", 1)[1])]
        return real_product(s, task, key)
    with monkeypatch.context() as patch:
        patch.setattr(dev, "product", baseline_product)
        patch.setattr(worker, "figures", lambda payload: [])
        baseline_report = worker.report(ctx, subject)
    execution_fields = {"parents", "content_hash", "shard_hashes", "figures"}
    assert {k: v for k, v in result.items() if k not in execution_fields} == {
        k: v for k, v in baseline_report.items() if k not in execution_fields}
    put(spec, "ba_report", dict(result=result))
    assert debug.read(spec) == result
    assert before == {p: sha256_file(p) for p in before}
    with pytest.raises(PermissionError):
        with data.sample_stream(ctx, "response_confirm") as stream:
            next(stream)
    receipt = dev.verified_outputs(subject, "ba_acceptance")
    path = Path(subject["root"])/receipt["outputs"]["result"]["relative"]
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="Corrupt"):
        debug.validate_stage(spec)
