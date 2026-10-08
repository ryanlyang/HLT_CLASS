"""Operational continuation only; S3 science stays in its original pinned checkout."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from hlt_classification import literature_ladder_followup as common

POLICY = dict(train=100000, validation=50000, fits=4, reducers=2, jobs=7,
              branches=["DIRECT", "COARSE"], partition="debug", account="reu-aisocial",
              qos="qos_tier3", gpu="gpu:a100:1", cpus=6, memory_mb=90000,
              max_minutes=1440, final_test_accessed=False, dataset_export=False)
DIRECT = "S3_DIRECT_D000_from_OFFLINE"
COARSE = ("S3_COARSE_D066_from_OFFLINE", "S3_COARSE_D033_from_D066", "S3_COARSE_D000_from_D033")
CONTROLLER_FILES = (
    "src/hlt_classification/s3_ladder_followup.py",
    "src/hlt_classification/literature_ladder_followup.py",
    "scripts/jetclass2_s3_ladder_followup.py",
    "scripts/start_jetclass2_s3_ladder_followup.sh",
    "tests/test_s3_ladder_followup.py",
    "docs/plans/JETCLASS2_S3_FOLLOWUP_PLAN.md",
    "docs/contracts/JETCLASS2_S3_FOLLOWUP.md",
)

# Only this separate interpreter imports science, always from the OLD checkout.
INSPECT = r'''
import json, sys
from pathlib import Path
project, path, mode = sys.argv[1:]
sys.path.insert(0, str(Path(project) / 'src'))
from hlt_classification.s3_ladder import campaign as c
from hlt_classification.s3_ladder.contracts import load_json
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
spec = load_json(path)
if Path(spec['project_dir']).resolve() != Path(project).resolve():
    raise ValueError('Pinned project differs')
if mode == 'preflight':
    parent = c.validate_spec(spec)
    value = c.preflight(spec, parent)
    print(json.dumps(dict(hash=value['content_hash'], job=value['slurm_job_id'],
        train_minutes=value['train_minutes'], reduce_minutes=value['reduce_minutes'])))
else:
    if mode not in ('gate', 'science'):
        raise ValueError('Unknown inspection mode')
    plan = c.plan(spec, mode)
    root = Path(spec['root']) / ('submission_' + mode)
    if load_json(root / 'command_plan.json') != plan:
        raise ValueError('Saved command plan differs')
    ledger = load_json(root / 'submission_ledger.json')
    jobs = ledger['jobs']
    if set(jobs) != {r['task_id'] for r in plan['commands']} or len(set(jobs.values())) != len(jobs):
        raise ValueError('Exact live job set differs')
    commands = {}
    for row in plan['commands']:
        argv = row['command']
        for task in row['dependencies']:
            argv = [a.replace('${JOB_' + task + '}', jobs[task]) for a in argv]
        commands[row['task_id']] = argv
    expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
        jobs=jobs, commands=commands, dry_run=False)
    _, journal_jobs = load_exact_dag_journal(root / 'submission_ledger_journal',
        identity=spec['content_hash'], plan=plan)
    if ledger != expected or journal_jobs != jobs:
        raise ValueError('Exact live ledger/journal differs')
    print(json.dumps(dict(jobs=jobs, ledger_hash=ledger['content_hash'], plan_hash=plan['content_hash'])))
'''


def publish(path, **fields):
    value = dict(contract="JC2_S3_FOLLOWUP/v1", schema_version=1,
                 final_test_accessed=False, **fields)
    value["content_hash"] = common.digest(value)
    path = Path(path)
    if path.exists():
        if common.read(path) != value:
            raise FileExistsError(f"Follow-up evidence differs; preserve and inspect: {path}")
        return value
    fd, temporary = tempfile.mkstemp(prefix=".s3-followup-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return value


def controller_source(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Exact controller commit required")
    project = Path(__file__).resolve().parents[2]
    head = common.capture(["git", "-C", str(project), "rev-parse", "HEAD"]).strip()
    dirty = common.capture(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=all"])
    common.capture(["git", "-C", str(project), "merge-base", "--is-ancestor", commit, "origin/main"])
    if head != commit or dirty.strip():
        raise PermissionError("Controller must use its own exact clean pushed checkout")
    return dict(commit=commit, files={p: hashlib.sha256((project / p).read_bytes()).hexdigest()
                                    for p in CONTROLLER_FILES})


def discover(checkpoints, job):
    if not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("Exact non-array preflight job required")
    matches = []
    for path in Path(checkpoints).glob("jc2_s3_100k50k_*/submission_gate/submission_ledger.json"):
        ledger = common.read(path)
        if ledger.get("dry_run") is False and ledger.get("jobs", {}).get("preflight") == job:
            matches.append(path.parent.parent / "study_spec.json")
    if len(matches) != 1:
        raise ValueError(f"Expected one S3 study for preflight {job}; found {len(matches)}")
    return matches[0]


def inspect(spec, path, mode):
    return json.loads(common.capture([sys.executable, "-s", "-c", INSPECT,
        spec["project_dir"], str(path), mode], timeout=3600))


def bind(path, job):
    if not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("Exact non-array preflight job required")
    path = Path(path).resolve(strict=True)
    spec = common.read(path)
    if (spec.get("contract") != "JC2_S3_LADDER_SPEC/v1"
            or path != Path(spec["root"]).resolve() / "study_spec.json"
            or spec["counts"] != dict(train=100000, validation=50000)
            or (spec["workers"], spec["cpus"], spec["memory_mb"]) != (6, 6, 90000)
            or spec["dataset_export"] is not False or spec["final_test_accessed"] is not False
            or spec["automatic_followup"] is not False or spec["development_only"] is not True
            or any(spec["execution_site"][k] != POLICY[k] for k in ("partition", "account", "qos"))
            or spec["execution_site"]["gres"] != POLICY["gpu"]):
        raise ValueError("S3 study exceeds the authorized follow-up scope")
    config = common.capture(["scontrol", "show", "config"])
    if not re.search(r"(?m)^\s*ClusterName\s*=\s*sporc\s*$", config):
        raise PermissionError("Arm the S3 follow-up on SPORC")
    print("Authenticating original S3 source and exact preflight ledger/journal; this can take several minutes.", flush=True)
    evidence = inspect(spec, path, "gate")
    if evidence["jobs"] != {"preflight": job}:
        raise ValueError("Preflight ID differs from authenticated ledger")
    return spec, evidence


def state(raw, job, user):
    rows = [line.strip().split("|") for line in raw.splitlines() if line.strip()]
    if not rows:
        return None
    if len(rows) != 1 or len(rows[0]) != 8:
        raise ValueError("Unexpected/duplicate accounting rows")
    identity, name, owner, account, partition, qos, status, code = rows[0]
    if (identity != job or name != "jc2s3_preflight" or owner != user
            or account != POLICY["account"] or partition != POLICY["partition"] or qos != POLICY["qos"]):
        raise PermissionError("Preflight scheduler identity differs")
    status = status.split()[0].rstrip("+")
    if status not in common.ACTIVE and not (status == "COMPLETED" and code == "0:0"):
        raise RuntimeError(f"Preflight {job}: {status}, exit={code}; no science submitted")
    return status


def wait_gate(job, *, sleep=time.sleep, monotonic=time.monotonic):
    deadline = monotonic() + 14 * 86400
    while monotonic() < deadline:
        raw = common.capture(["sacct", "-X", "-n", "-P", "-j", job, "-o",
            "JobIDRaw,JobName%100,User%100,Account%100,Partition%40,QOS%40,State%40,ExitCode"])
        current = state(raw, job, os.environ["USER"])
        print(f"Preflight {job}: {current or 'accounting not yet available'}", flush=True)
        if current == "COMPLETED":
            return
        print("Waiting; existing job unchanged; no science submitted yet.", flush=True)
        sleep(60)
    raise TimeoutError("14-day waiting limit reached; jobs preserved")


def expected_tasks():
    direct, a, b, end = ("fit_" + n for n in (DIRECT, *COARSE))
    ra, rb = ("reduce_" + n for n in COARSE[:2])
    return {direct: [], a: [], ra: [a], b: [ra], rb: [b], end: [rb],
            "summary": [direct, a, b, end]}


def review(plan, spec, acceptance):
    if (plan["contract"] != "JC2_S3_LADDER_PLAN/v1" or plan["mode"] != "science"
            or plan["parents"] != dict(spec=spec["content_hash"], preflight=acceptance["hash"])
            or plan["final_test_accessed"] is not False or plan["automatic_followup"] is not False
            or [r["task_id"] for r in plan["commands"]] != list(expected_tasks())):
        raise ValueError("Science plan exceeds the authorized seven-job S3 graph")
    for row in plan["commands"]:
        task, argv = row["task_id"], row["command"]
        if argv[0] != "sbatch" or row["dependencies"] != expected_tasks()[task]:
            raise ValueError("S3 dependency graph differs")
        flags = {}
        for arg in argv[1:]:
            key, _, value = arg.partition("=")
            if key in flags:
                raise ValueError("Duplicate scheduler flag")
            flags[key] = value
        gpu = task != "summary"
        minutes = acceptance["reduce_minutes" if task.startswith("reduce_") else "train_minutes"] if gpu else 60
        if not 1 <= minutes <= POLICY["max_minutes"]:
            raise ValueError("Measured walltime exceeds authorized ceiling")
        days, rem = divmod(minutes, 1440)
        hours, mins = divmod(rem, 60)
        walltime = (f"{days}-" if days else "") + f"{hours:02d}:{mins:02d}:00"
        required = {"--partition": "debug", "--account": "reu-aisocial", "--qos": "qos_tier3",
            "--nodes": "1", "--ntasks": "1", "--export": "NONE", "--no-requeue": "", "--parsable": "",
            "--cpus-per-task": "6" if gpu else "1", "--mem": "90000M" if gpu else "16000M",
            "--time": walltime, "--job-name": "jc2s3_" + task,
            "--comment": "jc2s3:" + spec["content_hash"] + ":" + task,
            "--chdir": spec["project_dir"], "--output": str(Path(spec["root"]) / "slurm-%j.out")}
        if gpu:
            required["--gres"] = "gpu:a100:1"
        if row["dependencies"]:
            required["--dependency"] = "afterok:" + ":".join("${JOB_" + p + "}" for p in row["dependencies"])
        if set(flags) != set(required) | {"--wrap"} or any(flags[k] != v for k, v in required.items()):
            raise PermissionError("S3 resource/site/dependency scope differs")
        # The original CLI regenerates and checks the entire plan, including wrap, again on submit.
    print("FULL MEASURED SEVEN-JOB DRY PLAN:\n" + json.dumps(plan, indent=2), flush=True)
    return plan["content_hash"]


def command(spec, *args):
    print("Running original pinned S3 CLI: " + " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", "-s", str(Path(spec["project_dir"]) /
        "scripts/jetclass2_s3_ladder.py"), *args], check=True, env=common.environment())


def refuse_ambiguous(root):
    if not (root / "submission_ledger.json").exists() and any(
        (root / name).exists() for name in ("live_submission_claim.json", "submission_ledger_journal")
    ):
        raise PermissionError("Science submission is active/interrupted; preserve claims and journals; no automatic retry")


def run(path, *, preflight_job, controller_commit, execute=False):
    source = controller_source(controller_commit)
    path = Path(path).resolve(strict=True)
    spec, binding = bind(path, preflight_job)
    print(f"Bound S3 preflight {preflight_job}; policy={json.dumps(POLICY)}", flush=True)
    if not execute:
        print("Read-only review. --execute authorizes this bounded automatic follow-up.", flush=True)
        return
    root = Path(spec["root"])
    with common.lock(root / "science_followup"):
        refuse_ambiguous(root / "submission_science")
        authorization = publish(root / "science_followup/authorization.json", kind="authorization",
            parents=dict(spec=spec["content_hash"], gate_ledger=binding["ledger_hash"]),
            spec_path=str(path), science_commit=spec["source_commit"], project_dir=spec["project_dir"],
            job=preflight_job, controller=source, policy=POLICY)
        if (root / "science_followup/complete.json").exists():
            finish(root, authorization, inspect(spec, path, "science"))
            return
        print("ARMED: after successful preflight and artifact validation, all seven S3 science jobs will submit automatically.", flush=True)
        wait_gate(preflight_job)
        if bind(path, preflight_job) != (spec, binding) or controller_source(controller_commit) != source:
            raise ValueError("Source/spec/gate changed while waiting")
        acceptance = inspect(spec, path, "preflight")
        if str(acceptance["job"]) != preflight_job:
            raise ValueError("Preflight artifact belongs to another job")
        refuse_ambiguous(root / "submission_science")
        command(spec, "submit", "--spec", str(path), "--mode", "science")  # Dry only.
        plan = common.read(root / "submission_science/command_plan.json")
        plan_hash = review(plan, spec, acceptance)
        publish(root / "science_followup/review.json", kind="review",
            parents=dict(authorization=authorization["content_hash"], preflight=acceptance["hash"], plan=plan_hash),
            policy=POLICY)
        if controller_source(controller_commit) != source:
            raise ValueError("Controller source changed during dry review")
        command(spec, "submit", "--spec", str(path), "--mode", "science", "--execute", "--plan-hash", plan_hash)
        finish(root, authorization, inspect(spec, path, "science"))


def finish(root, authorization, evidence):
    if set(evidence["jobs"]) != set(expected_tasks()):
        raise ValueError("Completed science job set differs")
    publish(root / "science_followup/complete.json", kind="submitted",
        parents=dict(authorization=authorization["content_hash"], plan=evidence["plan_hash"], ledger=evidence["ledger_hash"]),
        jobs=evidence["jobs"], training_complete=False)
    print("ALL SEVEN SCIENCE JOBS QUEUED: " + json.dumps(evidence["jobs"], sort_keys=True), flush=True)
    print("Controller finished. Training follows the saved DAG; no further automatic campaign.", flush=True)
