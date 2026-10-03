"""Execution-only continuation; scientific commands run in the original worktree.

This module intentionally lives outside cms_proxy_ladder's scientific source
snapshot. Arming a follow-up must not invalidate an already submitted preflight.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

SCIENCE_PHRASE = "AUTHORIZE JETCLASS2 LITERATURE V3 200K DIRECT COARSE SCIENCE"
POLICY = dict(train=200000, validation=50000, branches=["DIRECT", "COARSE"],
              fits=9, reducers=5, jobs=16, partition="debug", max_minutes=1440,
              final_test_accessed=False)
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED"}

# Only this subprocess imports scientific code, from the gate's original pin.
INSPECT = r'''
import json, sys
from pathlib import Path
project, spec_path = sys.argv[1:]
sys.path.insert(0, str(Path(project) / 'src'))
from hlt_classification.cms_proxy_ladder import literature as l, submission as s
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger, validate_submission_ledger
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
gate = load_json(spec_path)
l.validate_gate(gate, check_source=True)
plan = s.gate_plan(gate)
root = Path(gate['gate_root'])
if load_json(root / 'command_plan.json') != plan:
    raise ValueError('Gate command plan differs')
ledger = load_json(root / 'submission_ledger.json')
validate_submission_ledger(ledger)
jobs = ledger['jobs']
commands = {}
for row in plan['commands']:
    argv = list(row['command'])
    for parent in row['dependencies']:
        argv = [a.replace('${JOB_' + parent + '}', jobs[parent]) for a in argv]
    commands[row['task_id']] = argv
expected = build_submission_ledger(campaign_spec_sha256=gate['content_hash'],
    jobs=jobs, commands=commands, dry_run=False)
_, journal_jobs = load_exact_dag_journal(root / 'submission_ledger_journal',
    identity=gate['content_hash'], plan=plan)
if (ledger != expected or journal_jobs != jobs or len(set(jobs.values())) != 3
        or set(jobs) != {'authenticate_release', 'build_foundation', 'preflight'}):
    raise ValueError('Exact live gate ledger/journal differs')
print(json.dumps(dict(jobs=jobs, ledger_hash=ledger['content_hash'])))
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if value.get("content_hash") != digest({k: v for k, v in value.items() if k != "content_hash"}):
        raise ValueError(f"Metadata checksum differs: {path}")
    return value


def publish(path, **fields):
    """Atomic immutable operational evidence, separate from scientific artifacts."""
    value = dict(contract="JC2_LITERATURE_LADDER_FOLLOWUP/v1", schema_version=1,
                 final_test_accessed=False, **fields)
    value["content_hash"] = digest(value)
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise FileExistsError(f"Follow-up evidence differs; preserve and inspect: {path}")
        return value
    fd, temporary = tempfile.mkstemp(prefix=".followup-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # No overwrite, even if another publisher races.
    finally:
        os.unlink(temporary)
    return value


def environment():
    result = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    result.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return result


def capture(argv, *, timeout=60):
    result = subprocess.run(argv, capture_output=True, text=True, env=environment(), timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {argv[0]}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def bind(spec_path, gate_hash, source_commit, preflight_job):
    if not re.fullmatch(r"[0-9a-f]{64}", gate_hash) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Exact gate hash and scientific source commit required")
    if not re.fullmatch(r"[1-9][0-9]*", preflight_job):
        raise ValueError("Exact non-array preflight ID required")
    path = Path(spec_path).resolve(strict=True)
    gate = read(path)
    if (gate["content_hash"] != gate_hash or gate["source_commit"] != source_commit
            or gate["contract"] != "JETCLASS2_CMS_PROXY_LADDER_GATE_SPEC/v7"
            or path != Path(gate["gate_root"]).resolve() / "gate_spec.json"
            or path.parent.name != "gate"):
        raise ValueError("Gate/source identity differs")
    config = capture(["scontrol", "show", "config"])
    if not re.search(r"(?m)^\s*ClusterName\s*=\s*sporc\s*$", config):
        raise PermissionError("Follow-up must run on SPORC")
    print("Authenticating the original pinned gate source and submission journal; this can take several minutes.", flush=True)
    evidence = json.loads(capture([sys.executable, "-s", "-c", INSPECT,
                                   gate["project_dir"], str(path)], timeout=3600))
    if evidence["jobs"].get("preflight") != preflight_job:
        raise ValueError("Preflight ID differs from authenticated gate ledger")
    return gate, evidence


def states(text, jobs, user):
    expected = {job: task for task, job in jobs.items()}
    result = {}
    for line in text.splitlines():
        fields = line.strip().split("|")
        if not line.strip():
            continue
        if len(fields) != 6 or fields[0] not in expected or fields[0] in result:
            raise ValueError("Unexpected/duplicate accounting row; stop follow-up")
        job, name, owner, account, state, exit_code = fields
        if (name != "jc2lvg_" + expected[job] or owner != user or account != "reu-aisocial"):
            raise PermissionError(f"Scheduler job identity differs: {job}")
        state = state.split()[0].rstrip("+")
        if state not in ACTIVE and not (state == "COMPLETED" and exit_code == "0:0"):
            raise RuntimeError(f"Gate task {expected[job]} ({job}) is {state}, exit={exit_code}; no science submitted")
        result[job] = state
    return result


def wait_gate(jobs, *, sleep=time.sleep, monotonic=time.monotonic):
    deadline = monotonic() + 14 * 86400
    while monotonic() < deadline:
        text = capture(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()),
                        "-o", "JobIDRaw,JobName%100,User%100,Account%100,State%40,ExitCode"])
        current = states(text, jobs, os.environ["USER"])
        print("Gate accounting: " + json.dumps(current, sort_keys=True), flush=True)
        if set(current) == set(jobs.values()) and all(s == "COMPLETED" for s in current.values()):
            return
        print("Waiting; existing jobs unchanged; no science submitted yet.", flush=True)
        sleep(60)
    raise TimeoutError("14-day waiting limit reached; jobs preserved")


@contextmanager
def lock(root):
    import fcntl  # SPORC/Linux only; released by kernel after a crash/disconnect.
    if root.is_symlink():
        raise PermissionError("Follow-up output must not be a symlink")
    root.mkdir(exist_ok=True)
    with (root / "controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PermissionError("A follow-up controller is already active") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def command(gate, *arguments):
    argv = [sys.executable, "-u", "-s",
            str(Path(gate["project_dir"]) / "scripts/jetclass2_literature_proxy_ladder.py"), *arguments]
    print("Running pinned scientific CLI: " + " ".join(arguments), flush=True)
    # Inherit the durable log. Do not retry after an ambiguous live sbatch failure.
    subprocess.run(argv, check=True, env=environment())


def review(root, gate):
    science = root / "science"
    spec = read(science / "campaign_spec.json")
    plan = read(science / "command_plan.json")
    if (spec["parents"]["gate"] != gate["content_hash"]
            or spec["source_commit"] != gate["source_commit"] or spec["project_dir"] != gate["project_dir"]
            or spec["fresh_fit_count"] != 9 or spec["reducer_count"] != 5
            or spec["selected_branches"] != ["DIRECT", "COARSE"]
            or spec["final_test_accessed"] is not False
            or plan["parents"] != {"subject": spec["content_hash"]}
            or plan["mode"] != "science" or len(plan["commands"]) != 16):
        raise ValueError("Science plan exceeds authorized direct/coarse scope")
    # The pinned plan CLI has already run validate_campaign, including exact
    # tasks, recipe, source hashes, measured resources, gate_complete and parity.
    for row in plan["commands"]:
        args = row["command"]
        if (args[0] != "sbatch" or "--partition=debug" not in args
                or sum(a.startswith("--partition=") for a in args) != 1
                or "--account=reu-aisocial" not in args or "--qos=qos_tier3" not in args):
            raise PermissionError("Science submission site differs")
    return plan


def run(spec_path, *, gate_hash, source_commit, preflight_job, execute=False):
    args = (spec_path, gate_hash, source_commit, preflight_job)
    gate, evidence = bind(*args)
    root = Path(gate["gate_root"]).parent
    print(f"Bound preflight {preflight_job}; policy={json.dumps(POLICY)}", flush=True)
    if not execute:
        print("Read-only review; use --execute to authorize automatic science submission.", flush=True)
        return
    with lock(root / "followup"):
        authorization = publish(root / "followup/authorization.json", kind="authorization",
            parents={"gate": gate_hash, "gate_ledger": evidence["ledger_hash"]},
            gate_spec=str(Path(spec_path).resolve()), source_commit=source_commit,
            project_dir=gate["project_dir"], jobs=evidence["jobs"], policy=POLICY,
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        print("ARMED: after successful gate, all 16 science jobs will submit automatically.", flush=True)
        wait_gate(evidence["jobs"])
        if bind(*args) != (gate, evidence):
            raise ValueError("Gate changed while waiting")
        science = root / "science/campaign_spec.json"
        if not science.exists():
            command(gate, "create-campaign", "--gate-root", gate["gate_root"],
                    "--campaign-root", str(root / "science"))
        command(gate, "plan", "--spec", str(science), "--mode", "science")
        plan = review(root, gate)
        publish(root / "followup/review.json", kind="review",
                parents={"authorization": authorization["content_hash"], "plan": plan["content_hash"]},
                policy=POLICY)
        command(gate, "submit", "--spec", str(science), "--mode", "science",
                "--plan-hash", plan["content_hash"], "--authorization-phrase", SCIENCE_PHRASE)
        ledger = read(root / "science/submission_ledger.json")
        if (ledger["dry_run"] is not False or ledger["campaign_spec_sha256"] != plan["parents"]["subject"]
                or set(ledger["jobs"]) != {row["task_id"] for row in plan["commands"]}):
            raise ValueError("Completed live ledger differs")
        publish(root / "followup/complete.json", kind="submitted",
                parents={"authorization": authorization["content_hash"], "plan": plan["content_hash"],
                         "ledger": ledger["content_hash"]}, jobs=ledger["jobs"], training_complete=False)
        print("ALL 16 SCIENCE JOBS QUEUED: " + json.dumps(ledger["jobs"], sort_keys=True), flush=True)
        print("Controller finished; training proceeds through its saved dependencies.", flush=True)
