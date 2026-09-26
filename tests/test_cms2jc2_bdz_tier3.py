"""B_DZ partition-only migration; exact scheduler races and unchanged science."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.cms2jc2_response import (
    bdz_tier3 as migration, bdz_campaign as original, bdz_worker as worker,
    dev_campaign as dev, dev_submission as submission, dev_data)
from hlt_classification.cms2jc2_response.contracts import with_content_hash, sha256_file
from test_cms2jc2_bdz_tuning import completed_bdz
from test_cms2jc2_bounded import bounded_root, LocalMeasurement
from test_cms2jc2_b_tracking import b_root
from test_cms2jc2_frozen_c import frozen_root, free_disk


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETING", "COMPLETED", "UNKNOWN", "SUSPENDED", "CONFIGURING"])
def test_preserve_nonpending_work(state, monkeypatch):
    jobs = {t: str(i+100) for i, t in enumerate(migration.TARGETS)}
    statuses = {j: ("PENDING", "0:0") for j in jobs.values()}
    statuses[jobs["bz_eval_0"]] = (state, "0:0")
    monkeypatch.setattr(submission, "states", lambda _: statuses)
    with pytest.raises(PermissionError, match="Preserve original B_DZ"):
        migration.old_states(dict(migration=dict(subject_jobs=jobs)))


@pytest.mark.parametrize("subject", [dict(), dict(contract="CMS2JC2_RESPONSE_BDZ_STAGE/v1", stage="bdz_gate"),
                                    dict(contract="CMS2JC2_RESPONSE_BOUNDED_STAGE/v1", stage="bounded_compare")])
def test_wrong_subject_family_or_gate(subject):
    with pytest.raises(ValueError, match="original debug B_DZ comparison"):
        migration.subject_metadata(subject)


def test_tier3_cli_and_worker_registration():
    root = Path(__file__).parents[1]
    cli = (root/"scripts/cms2jc2_response_dev.py").read_text()
    for command in ("create-bdz-tier3", "retire-bdz-tier3"):
        assert command in cli
    for module in ("dev_submission.py", "dev_worker.py"):
        text = (root/"src/hlt_classification/cms2jc2_response"/module).read_text()
        assert migration.CONTRACT in text and "from .bdz_tier3 import verify_retirement" in text
    script = (root/"scripts/queue_cms2jc2_bdz_tier3.sh").read_text()
    assert 'MODE="${4:-}"' in script
    assert "scontrol update" not in script and "scancel" not in script


def test_tier3_reuse_retirement_and_full_scientific_replay(completed_bdz, tmp_path, monkeypatch):
    parent, ctx, put, _ = completed_bdz
    monkeypatch.setattr(migration, "COUNTS", dev_data.COUNTS)
    gate = original.create(parent_spec=dev.stage_dir(parent)/"stage_spec.json",
        project_dir=tmp_path/"bdz_project", source_commit="c"*40, root=tmp_path/"bdz")
    put(gate, "bz_acceptance", dict(result=worker.acceptance(ctx, gate)))
    put(gate, "bz_calibrate", dict(result=worker.calibrate(ctx, gate, dict(cpus=1))))
    subject = original.advance(dev.stage_dir(gate)/"stage_spec.json")
    old_study = dev.validate_stage(subject)
    calls, statuses, race = [], {"99901": ("RUNNING", "0:0"), "99902": ("PENDING", "0:0")}, {"state": None}
    job_counter = [3100]
    old_ledger = None
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
            out = "\n".join(f"{j}|{state}|{code}" for j, (state, code) in statuses.items())
        elif argv[:3] == ["scontrol", "show", "job"]:
            job = argv[-1]
            task = next(t for t, j in old_ledger["jobs"].items() if j == job)
            t = next(t for t in subject["tasks"] if t["task_id"] == task)
            out = (f"JobId={job} JobState={statuses[job][0]} JobName=c2jd_{task} UserId=fixture(123) "
                f"Comment=c2jd:{subject['content_hash']}:{task} WorkDir={old_study['project_dir']} "
                f"Command={Path(old_study['project_dir'])/'sbatch/run_cms2jc2_response_dev_cpu.sh'} "
                f"Partition=debug Account=reu-aisocial QOS=qos_tier3 NumCPUs={t['cpus']} NumNodes=1-1 "
                "ReqTRES=cpu=36,mem=128G,node=1")
        elif argv[0] == "scancel":
            assert argv[1:5] == ["--ctld", "--state=PENDING", "--partition=debug", "--account=reu-aisocial"]
            targets = set(old_ledger["jobs"].values())
            assert set(argv[5:]) <= targets
            for job in argv[5:]:
                statuses[job] = ("CANCELLED", "0:0")
            if race["state"]:
                statuses[old_ledger["jobs"]["bz_eval_0"]] = (race["state"], "0:0")
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(submission, "scheduler", scheduler)
    old_plan = submission.submit(subject)
    old_ledger = submission.submit(subject, execute=True, authorization_phrase=dev.PHRASES["bdz_compare"],
                                   reviewed_plan_hash=old_plan["content_hash"])
    source_path = dev.stage_dir(subject)/"stage_spec.json"
    roots = [Path(subject["root"]), Path(parent["root"])]
    before = {p: sha256_file(p) for root in roots for p in root.rglob("*") if p.is_file()}
    spec = migration.create(parent_spec=source_path, project_dir=tmp_path/"tier3_project",
                             source_commit="d"*40, root=tmp_path/"tier3")
    plan = submission.submit(spec)
    assert original.registry(spec) == original.registry(subject)
    assert original.parent_compare(spec) == parent
    assert dev.preparation_stage(spec) == dev.preparation_stage(subject)
    assert spec["tasks"] == subject["tasks"] and spec["protocol"] == subject["protocol"]
    assert spec["migration"]["calibration_reused"] and spec["migration"]["acceptance_reused"]
    assert len(plan["commands"]) == 5 and plan["gpus"] == 0
    assert all("--partition=tier3" in c["argv"] for c in plan["commands"])
    assert all("--time=08:00:00" in c["argv"] for c in plan["commands"][:4])
    real_product, real_jobs = dev.product, submission.submitted_jobs
    with monkeypatch.context() as m:
        def bad_acceptance(s, task, key):
            value = real_product(s, task, key)
            return with_content_hash(dict(value, serial_process_parity=False)) if task == "bz_acceptance" else value
        m.setattr(dev, "product", bad_acceptance)
        with pytest.raises(ValueError, match="acceptance differs"):
            migration.validate_stage(spec)
    with monkeypatch.context() as m:
        m.setattr(submission, "submitted_jobs", lambda s, p: {} if s == subject else real_jobs(s, p))
        with pytest.raises(ValueError, match="ledger/journals"):
            migration.validate_stage(spec)
    with monkeypatch.context() as m:
        def bad_map(s, task, key):
            value = real_product(s, task, key)
            if task == "bz_calibrate":
                value["mapping"]["cells"]["all/0"]["x"] = [1., 0.]
            return value
        m.setattr(dev, "product", bad_map)
        with pytest.raises(ValueError):
            migration.validate_stage(spec)
    for key, value in (("name", "other"), ("stage", "bdz_gate"), ("tasks", []), ("protocol", {})):
        with pytest.raises(ValueError, match="registration"):
            dev.validate_stage(with_content_hash(dict(spec, **{key: value})))
    changed = deepcopy(dev.validate_stage(spec))
    changed["source"]["files"]["scientific.py"] = "e"*64
    with pytest.raises(ValueError, match="scientific source"):
        migration.reuse(subject, changed)
    with pytest.raises(PermissionError, match="disjoint"):
        migration.create(parent_spec=source_path, project_dir="unused", source_commit="d"*40, root=subject["root"])
    # The real immutable source/receipt/registry chains were checked above,
    # including negative mutations. Exercise scheduler races against those
    # already-validated inputs without re-parsing the large legacy reports for
    # every simulated state transition. Restore full validation before replay.
    real_validate, real_metadata = migration.validate_stage, migration.subject_metadata
    validated_study, validated_metadata = real_validate(spec), real_metadata(subject)
    def validated(s, **kwargs):
        assert s == spec
        return validated_study
    def metadata(s):
        assert s == subject
        return validated_metadata
    monkeypatch.setattr(migration, "validate_stage", validated)
    monkeypatch.setattr(migration, "subject_metadata", metadata)
    prior = len(calls)
    with pytest.raises(FileNotFoundError):
        submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["bdz_compare"],
                          reviewed_plan_hash=plan["content_hash"])
    assert not any(c[0] in ("scancel", "sbatch") for c in calls[prior:])
    assert migration.retire(spec)["read_only"]
    with pytest.raises(PermissionError, match="phrase"):
        migration.retire(spec, execute=True, reviewed_plan_hash=plan["content_hash"])
    kwargs = dict(execute=True, authorization_phrase=migration.RETIRE_PHRASE, reviewed_plan_hash=plan["content_hash"])

    def reject(*a, **kw):
        raise PermissionError("Site rejected tier3 request")
    with monkeypatch.context() as m:
        m.setattr(submission, "site_checks", reject)
        with pytest.raises(PermissionError, match="Site rejected"):
            migration.retire(spec, **kwargs)
    assert not any(c[0] == "scancel" for c in calls)
    with monkeypatch.context() as m:
        m.setattr(submission, "scheduler_identity", lambda *a, **kw: dict(JobState="RUNNING"))
        with pytest.raises(PermissionError, match="nothing cancelled"):
            migration.retire(spec, **kwargs)
    assert not any(c[0] == "scancel" for c in calls)
    for state, error in (("PENDING", RuntimeError), ("RUNNING", PermissionError), ("COMPLETED", PermissionError)):
        for job in old_ledger["jobs"].values(): statuses[job] = ("PENDING", "0:0")
        race["state"] = state
        with pytest.raises(error): migration.retire(spec, **kwargs)
        assert not migration.retirement_path(spec).exists()
    race["state"] = None
    for job in old_ledger["jobs"].values(): statuses[job] = ("CANCELLED", "0:0")
    evidence = migration.retire(spec, **kwargs)
    assert migration.verify_retirement(spec, live=True) == evidence
    assert migration.retire(spec, **kwargs) == evidence
    job = old_ledger["jobs"]["bz_eval_0"]
    statuses[job] = ("PENDING", "0:0")
    with pytest.raises(PermissionError, match="can still run"):
        submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["bdz_compare"],
                          reviewed_plan_hash=plan["content_hash"])
    statuses[job] = ("CANCELLED", "0:0")
    prior = len(calls)
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["bdz_compare"],
                               reviewed_plan_hash=plan["content_hash"])
    live = [c for c in calls[prior:] if c[0] == "sbatch" and "--test-only" not in c]
    assert len(live) == 5
    assert not any(arg.startswith("--dependency") for c in live[:4] for arg in c)
    assert "--dependency=afterok:"+":".join(ledger["jobs"][t] for t in migration.TARGETS[:4]) in live[-1]
    assert submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["bdz_compare"],
                             reviewed_plan_hash=plan["content_hash"]) == ledger
    assert statuses["99901"] == ("RUNNING", "0:0") and statuses["99902"] == ("PENDING", "0:0")
    assert all(j not in c for j in ("99901", "99902") for c in calls if c[0] == "scancel")
    monkeypatch.setattr(migration, "validate_stage", real_validate)
    monkeypatch.setattr(migration, "subject_metadata", real_metadata)
    migration.validate_stage(spec)

    # Execute the unchanged worker against the original frozen map/real ROOT fixture.
    for task in spec["tasks"][:4]:
        result = worker.evaluate(ctx, spec, dict(task, cpus=1))
        reference = worker.evaluate(ctx, subject, dict(task, cpus=1))
        for key in result.keys() - {"parents", "content_hash"}:
            assert result[key] == reference[key]
        put(spec, task["task_id"], dict(result=result))
    monkeypatch.setattr(worker, "plot_pages", lambda *a, **kw: [])
    put(spec, "bz_select", dict(result=worker.report(ctx, spec)))
    assert "Frozen development choice: B_DZ" in migration.render(spec)
    assert before == {p: sha256_file(p) for root in roots for p in root.rglob("*") if p.is_file()}
    # Even an old failed/partial worker claim prevents silently throwing work away.
    claim = dev.stage_dir(subject)/"claims/bz_eval_0"
    claim.mkdir(parents=True)
    with pytest.raises(PermissionError, match="Preserve original B_DZ work"):
        migration.validate_stage(spec)
