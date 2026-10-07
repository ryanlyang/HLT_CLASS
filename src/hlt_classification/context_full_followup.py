"""Execution-only Oscar continuation; science always runs in the original pin."""
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
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout

SCIENCE_PHRASE = "AUTHORIZE JETCLASS2 CONTEXT 1M OSCAR DIRECT COARSE SCIENCE"
POLICY = dict(train=1000000, validation=250000, branches=["DIRECT", "COARSE"],
              fits=9, reducers=5, jobs=16, partition="gpu", account="default",
              qos="norm-gpu", gpu="gpu:l40s:1", cpus=6, memory_mb=180000,
              max_train_minutes=2880, max_reduce_minutes=1440, final_test_accessed=False)
CONTROLLER_FILES = (
    "src/hlt_classification/context_full_followup.py",
    "src/hlt_classification/literature_ladder_followup.py",
    "src/hlt_classification/scouting/hcwdl_authorization.py",
    "scripts/jetclass2_context_full_followup.py",
    "scripts/start_jetclass2_context_full_followup.sh",
    "docs/plans/JETCLASS2_CONTEXT_1M_FOLLOWUP_PLAN.md",
    "docs/contracts/JETCLASS2_CONTEXT_1M_FOLLOWUP.md",
)
INSPECT = common.INSPECT.replace("import literature as l, submission as s",
                               "import context_full as l, submission as s")
INSPECT_SCIENCE = r'''
import json, sys
from pathlib import Path
project, path = sys.argv[1:]
sys.path.insert(0, str(Path(project) / 'src'))
from hlt_classification.cms_proxy_ladder import context_full as l, submission as s
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger, validate_submission_ledger
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
spec = load_json(path)
l.validate_campaign(spec, check_source=True)
plan = s.science_plan(spec)
root = Path(spec['campaign_root'])
if load_json(root / 'command_plan.json') != plan:
    raise ValueError('Science plan differs')
ledger = load_json(root / 'submission_ledger.json')
validate_submission_ledger(ledger)
jobs = ledger['jobs']
commands = {}
for row in plan['commands']:
    argv = list(row['command'])
    for parent in row['dependencies']:
        argv = [a.replace('${JOB_' + parent + '}', jobs[parent]) for a in argv]
    commands[row['task_id']] = argv
expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
    jobs=jobs, commands=commands, dry_run=False)
_, journal_jobs = load_exact_dag_journal(root / 'submission_ledger_journal',
    identity=spec['content_hash'], plan=plan)
if (ledger != expected or journal_jobs != jobs or set(jobs) != set(commands)
        or len(jobs) != 16 or len(set(jobs.values())) != 16):
    raise ValueError('Science live ledger/journal differs')
print(json.dumps(dict(jobs=jobs, ledger_hash=ledger['content_hash'])))
'''


def controller_source(project, commit):
    project = Path(project).resolve(strict=True)
    validate_source_checkout(project, expected_commit=commit)
    if project != Path(__file__).resolve().parents[2]:
        raise ValueError("Controller must run from the exact executor checkout")
    return dict(commit=commit, project_dir=str(project), files={
        p: hashlib.sha256((project / p).read_bytes()).hexdigest() for p in CONTROLLER_FILES})


def publish(path, **fields):
    value = dict(contract="JC2_CONTEXT_1M_FOLLOWUP/v1", schema_version=1,
                 final_test_accessed=False, **fields)
    value["content_hash"] = common.digest(value)
    path = Path(path)
    if path.exists():
        if common.read(path) != value:
            raise FileExistsError("Follow-up evidence differs; preserve and inspect: " + str(path))
        return value
    fd, temporary = tempfile.mkstemp(prefix=".context1m-followup-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return value


def bind(spec_path, gate_hash, source_commit, preflight_job):
    if (not re.fullmatch(r"[0-9a-f]{64}", gate_hash)
            or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or not re.fullmatch(r"[1-9][0-9]*", preflight_job)):
        raise ValueError("Exact gate hash, source commit and preflight ID required")
    path = Path(spec_path).resolve(strict=True)
    gate = common.read(path)
    if (gate["content_hash"] != gate_hash or gate["source_commit"] != source_commit
            or gate["contract"] != "JETCLASS2_CMS_PROXY_LADDER_GATE_SPEC/v11"
            or path != Path(gate["gate_root"]).resolve() / "gate_spec.json"
            or path.parent.name != "gate"):
        raise ValueError("Original 1M gate/source identity differs")
    config = common.capture(["scontrol", "show", "config"])
    if not re.search(r"(?m)^\s*ClusterName\s*=\s*slurmctld\s*$", config):
        raise PermissionError("Arm this follow-up on Oscar")
    print("Authenticating original pinned gate and exact submission journal...", flush=True)
    evidence = json.loads(common.capture([sys.executable, "-s", "-c", INSPECT,
        gate["project_dir"], str(path)], timeout=3600))
    if evidence["jobs"].get("preflight") != preflight_job:
        raise ValueError("Preflight ID differs from authenticated gate ledger")
    return gate, evidence


def states(text, jobs, user):
    expected, result = {job: task for task, job in jobs.items()}, {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) != 6 or fields[0] not in expected or fields[0] in result:
            raise ValueError("Unexpected/duplicate accounting row")
        job, name, owner, account, state, code = fields
        if name != "jc2ctxmg_" + expected[job] or owner != user or account != "default":
            raise PermissionError("Gate scheduler identity differs: " + job)
        state = state.split()[0].rstrip("+")
        if state not in common.ACTIVE and not (state == "COMPLETED" and code == "0:0"):
            raise RuntimeError(f"Gate job {job}: {state}, exit={code}; no science submitted")
        result[job] = state
    return result


def wait_gate(jobs, *, sleep=time.sleep, monotonic=time.monotonic):
    deadline = monotonic() + 14 * 86400
    while monotonic() < deadline:
        raw = common.capture(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()),
            "-o", "JobIDRaw,JobName%100,User%100,Account%100,State%40,ExitCode"])
        current = states(raw, jobs, os.environ["USER"])
        print("Gate accounting: " + json.dumps(current, sort_keys=True), flush=True)
        if set(current) == set(jobs.values()) and all(s == "COMPLETED" for s in current.values()):
            return
        print("Waiting for authentication, foundation AND GPU preflight; no science submitted yet.", flush=True)
        sleep(60)
    raise TimeoutError("14-day wait bound reached; existing jobs unchanged")


def no_live_science(root):
    science = Path(root) / "science"
    for name in ("submission_ledger.json", "live_submission_claim.json"):
        if (science / name).exists():
            raise PermissionError("Science already submitted or ambiguous; preserve and inspect")
    journal = science / "submission_ledger_journal"
    if journal.exists() and (not journal.is_dir() or any(journal.iterdir())):
        raise PermissionError("Science submission journal already exists; preserve and inspect")


def command(gate, *arguments):
    argv = [sys.executable, "-u", "-s", str(Path(gate["project_dir"]) /
        "scripts/jetclass2_context_full_ladder.py"), *arguments]
    print("Running original pinned CLI: " + " ".join(arguments), flush=True)
    subprocess.run(argv, check=True, env=common.environment())


def review(root, gate):
    spec = common.read(root / "science/campaign_spec.json")
    plan = common.read(root / "science/command_plan.json")
    if (spec["parents"]["gate"] != gate["content_hash"] or spec.get("schema_version") != 8
            or spec["source_commit"] != gate["source_commit"] or spec["project_dir"] != gate["project_dir"]
            or Path(spec["campaign_root"]).resolve() != (root / "science").resolve()
            or Path(spec["gate_root"]).resolve() != Path(gate["gate_root"]).resolve()
            or spec["foundation"]["role_counts"] != {"train": POLICY["train"], "validation": POLICY["validation"]}
            or spec["fresh_fit_count"] != 9 or spec["reducer_count"] != 5
            or spec["selected_branches"] != POLICY["branches"] or spec["final_test_accessed"] is not False
            or plan["parents"] != {"subject": spec["content_hash"]} or plan["mode"] != "science"
            or plan["final_test_accessed"] is not False or len(plan["commands"]) != 16):
        raise ValueError("Science plan exceeds authorized 1M DIRECT/COARSE scope")
    profile = spec["runtime_profile"]
    if not (60 <= profile["train_minutes"] <= 2880 and 30 <= profile["reduce_minutes"] <= 1440):
        raise ValueError("Measured runtime exceeds authorized bounds")
    tasks = {r["task_id"]: r for r in spec["tasks"]}
    if (len(tasks) != 16 or len({r["task_id"] for r in plan["commands"]}) != 16
            or sum(r["kind"] == "train" for r in tasks.values()) != 9
            or sum(r["kind"] == "reduce" for r in tasks.values()) != 5):
        raise ValueError("Science task counts differ")
    for row in plan["commands"]:
        args, task = row["command"], tasks[row["task_id"]]
        gpu = task["kind"] in ("train", "reduce")
        minutes = profile["train_minutes"] if task["kind"] == "train" else profile["reduce_minutes"] if gpu else 60
        days, remainder = divmod(minutes, 1440)
        hours, remainder = divmod(remainder, 60)
        wall = (f"{days}-" if days else "") + f"{hours:02d}:{remainder:02d}:00"
        expected = dict(partition="gpu", account="default", qos="norm-gpu",
            **{"cpus-per-task": "6" if gpu else "4", "mem": "180000M" if gpu else "32000M", "time": wall})
        for key, value in expected.items():
            if [a for a in args if a.startswith("--" + key + "=")] != [f"--{key}={value}"]:
                raise PermissionError("Science submission resources/site differ: " + key)
        if (args[0] != "sbatch" or row["dependencies"] != task["dependencies"]
                or [a for a in args if a.startswith("--gres=")] != (["--gres=gpu:l40s:1"] if gpu else [])):
            raise ValueError("Science command/dependency/GPU scope differs")
    return plan


def run(spec_path, *, gate_hash, source_commit, preflight_job, project_dir, executor_commit, execute=False):
    controller = controller_source(project_dir, executor_commit)
    args = (spec_path, gate_hash, source_commit, preflight_job)
    gate, binding = bind(*args)
    root = Path(gate["gate_root"]).parent
    no_live_science(root)
    print(f"Bound preflight {preflight_job}; policy=" + json.dumps(POLICY, sort_keys=True), flush=True)
    if not execute:
        print("Read-only review; --execute authorizes automatic exact-plan science after the full gate.", flush=True)
        return
    directory = root / "context1m_followup"
    with common.lock(directory):
        no_live_science(root)
        authorization = publish(directory / "authorization.json", kind="authorization",
            parents={"gate": gate_hash, "gate_ledger": binding["ledger_hash"]}, controller=controller,
            gate_spec=str(Path(spec_path).resolve()), source_commit=source_commit,
            project_dir=gate["project_dir"], jobs=binding["jobs"], policy=POLICY)
        print("ARMED: after all three gate jobs pass, all 16 Oscar science jobs will submit automatically.", flush=True)
        wait_gate(binding["jobs"])
        if bind(*args) != (gate, binding) or controller_source(project_dir, executor_commit) != controller:
            raise ValueError("Source/gate changed while waiting")
        no_live_science(root)
        spec = root / "science/campaign_spec.json"
        if not spec.exists():
            command(gate, "create-campaign", "--gate-root", gate["gate_root"], "--campaign-root", str(spec.parent))
        command(gate, "plan", "--spec", str(spec), "--mode", "science")
        plan = review(root, gate)
        publish(directory / "review.json", kind="review",
            parents={"authorization": authorization["content_hash"], "plan": plan["content_hash"]}, policy=POLICY)
        no_live_science(root)
        command(gate, "submit", "--spec", str(spec), "--mode", "science",
            "--plan-hash", plan["content_hash"], "--authorization-phrase", SCIENCE_PHRASE)
        evidence = json.loads(common.capture([sys.executable, "-s", "-c", INSPECT_SCIENCE,
            gate["project_dir"], str(spec)], timeout=3600))
        publish(directory / "complete.json", kind="submitted", parents={
            "authorization": authorization["content_hash"], "plan": plan["content_hash"],
            "ledger": evidence["ledger_hash"]}, jobs=evidence["jobs"], training_complete=False)
        print("ALL 16 SCIENCE JOBS QUEUED: " + json.dumps(evidence["jobs"], sort_keys=True), flush=True)
        print("Existing gate and old 100k jobs unchanged. Submission is not training completion.", flush=True)
