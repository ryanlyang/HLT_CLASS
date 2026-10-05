"""Reviewed, three-task restart-zero adapter for the pinned v6 fusion campaign.

The CLI loads this adapter by filename with the ORIGINAL checkout on sys.path.
All scientific execution and validators therefore remain at the original commit.
Only exact-job authentication is redirected to a separate replacement ledger.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import subprocess

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, _resolved
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
from hlt_classification.jetclass2_delphes import dzfix_fusion_chain as chain
from hlt_classification.jetclass2_delphes import dzfix_fusion_runtime as runtime
from hlt_classification.jetclass2_delphes import dzfix_fusion_submit as scheduler
from hlt_classification.jetclass2_delphes.contracts import artifact as _artifact, validate as _validate
from hlt_classification.jetclass2_delphes.submission import _guarded_exact_submission


DONOR = "fd1c05786287c57b080b2d229dd64664f830a571"
TARGET = "train_FINAL_DIRECT_D000"
RETRY = (TARGET, "aggregate", "complete")
KEEP = ("train_FUSION_D000_D000", "reduce_FUSION_D000_D000", "train_FINAL_BRIDGE_D000")
AUTHORIZE = "AUTHORIZE DZFIX FUSION FINAL DIRECT NODE FAILURE RECOVERY"
FAILED = {"NODE_FAIL", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "TIMEOUT", "PREEMPTED", "BOOT_FAIL"}
FILES = ("scripts/recover_jetclass2_dzfix_fusion.py", "sbatch/run_jetclass2_dzfix_fusion_recovery.sh",
         "src/hlt_classification/jetclass2_delphes/dzfix_fusion_recovery.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(kind, **fields):
    return _artifact("DZFIX_FUSION_RECOVERY_" + kind, final_test_accessed=False, **fields)


def validate(value, kind):
    result = _validate(value, "DZFIX_FUSION_RECOVERY_" + kind)
    require(value["final_test_accessed"] is False, "Recovery cannot access final test")
    return result


def run(args):
    result = subprocess.run(list(map(str, args)), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("Command failed: " + " ".join(map(str, args)) + "\n" + result.stdout + result.stderr)
    return result.stdout


def safe(root, relative):
    root = Path(root)
    path = root / relative
    require(root.resolve() == root and path.resolve() == path and root in path.parents,
            "Redirected or escaping recovery path")
    return path


def inventory(path):
    """Bind partial output bytes and empty directories before moving anything."""
    path = Path(path)
    if not path.exists():
        require(not path.is_symlink(), "Dangling output symlink")
        return None
    require(path.is_dir() and path.resolve() == path, "Invalid partial output directory")
    result = {}
    for item in sorted(path.rglob("*")):
        require(not item.is_symlink() and item.resolve() == item, "Redirected partial output")
        require(item.is_file() or item.is_dir(), "Special file in partial output")
        result[item.relative_to(path).as_posix()] = sha256_file(item) if item.is_file() else None
    return result


def source_context(path):
    source = load_json(path)
    require(source["source_commit"] == DONOR, "Recovery supports only the reviewed fd1c0578 source")
    require(Path(path).resolve() == Path(source["campaign_root"]) / "campaign_spec.json",
            "Original spec location differs")
    original = Path(source["project_dir"]).resolve()
    for module in (chain, runtime, scheduler):
        require(original in Path(module.__file__).resolve().parents,
                "Scientific module did not load from the original pinned checkout; use recovery CLI")
    chain.validate_campaign(source)
    path = Path(source["campaign_root"]) / "submissions_science/submission_ledger.json"
    ledger = load_json(path)
    validate_submission_ledger(ledger)
    require(not ledger["dry_run"] and ledger["campaign_spec_sha256"] == source["content_hash"],
            "Original live ledger identity differs")
    plan = scheduler.plan(source, "science")
    expected = {r["task_id"]: _resolved(r, ledger["jobs"]) for r in plan["commands"]}
    require(ledger["commands"] == expected and set(ledger["jobs"]) == set(expected),
            "Original ledger commands differ from its exact DAG")
    require(len(set(ledger["jobs"].values())) == len(expected), "Duplicate original job IDs")
    return source, ledger


def accounting(jobs):
    raw = run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs),
               "--format=JobIDRaw,State%40,ExitCode"])
    result = {}
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3 or fields[0] not in jobs:
            continue
        value = dict(state=fields[1].strip().split()[0].rstrip("+"), exit_code=fields[2].strip())
        require(fields[0] not in result or result[fields[0]] == value, "Ambiguous accounting attempts")
        result[fields[0]] = value
    require(set(result) == set(jobs), "Missing exact-job accounting; no recovery authorized")
    return result


def live_job(source, name, job):
    raw = run(["scontrol", "show", "job", "-o", job])
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    require(fields.get("JobId") == job and fields.get("JobName") == "jc2fc_" + name
            and fields.get("Account") == source["execution_site"]["account"]
            and fields.get("WorkDir") == source["project_dir"]
            and fields.get("UserId", "").endswith(f"({os.getuid()})"), "Foreign scheduler job")
    return fields


def create(*, source_spec, recovery_root, project, source_commit):
    source, ledger = source_context(Path(source_spec).resolve())
    project, root = Path(project).resolve(), Path(recovery_root).resolve()
    chain._source(project, source_commit)
    require(root.parent == Path(source["campaign_root"]) / "recoveries", "Use campaign/recoveries/<new-name>")
    require(not root.exists(), "Recovery root already exists; use submit to continue, not create")
    print("Authenticating existing gate and completed science outputs...", flush=True)
    runtime.science_gate(source)
    bound = {name: ledger["jobs"][name] for name in RETRY + KEEP}
    states = accounting(list(bound.values()))
    require(states[bound[TARGET]]["state"] in FAILED, "Final-direct is not terminal unsuccessful")
    for name in RETRY:
        require(runtime.completed(source, name) is None, "Replacement task already complete: " + name)
    for name in ("aggregate", "complete"):
        require(states[bound[name]]["state"] in FAILED | {"PENDING"}, "Summary is not pending/failed")
    reused = {}
    for row in source["tasks"]:
        name = row["task_id"]
        if name in RETRY:
            continue
        receipt = runtime.completed(source, name)
        if receipt is None:
            require(name in KEEP and states[bound[name]]["state"] in {"PENDING", "RUNNING", "COMPLETING"},
                    "Unexpected incomplete/failed prerequisite: " + name)
        else:
            reused[name] = receipt["content_hash"]
    log = safe(source["campaign_root"], "slurm-" + bound[TARGET] + ".out")
    require("DUE TO NODE FAILURE" in log.read_text(errors="replace"), "Expected node-failure evidence missing")
    partial = inventory(safe(source["campaign_root"], "outputs/" + TARGET))
    require(partial is not None and not any(k in partial for k in ("selected.pt", "training_report.json", "result.json")),
            "Partial final-direct output needs separate review")
    for name in ("aggregate", "complete"):
        require(inventory(safe(source["campaign_root"], "outputs/" + name)) is None, "Existing summary output")
    external = []
    if KEEP[-1] not in reused:
        fields = live_job(source, KEEP[-1], bound[KEEP[-1]])
        require(fields["JobState"] in {"PENDING", "RUNNING", "COMPLETING"}, "Bridge not live")
        external = [bound[KEEP[-1]]]
    value = artifact("SPEC", source_spec=str(Path(source_spec).resolve()), source_sha256=source["content_hash"],
        original_ledger_sha256=ledger["content_hash"], original_jobs=bound, accounting=states,
        original_commit=DONOR, project_dir=str(project), source_commit=source_commit,
        helper_sha256={p: sha256_file(project / p) for p in FILES}, recovery_root=str(root),
        retry_tasks=list(RETRY), retained_tasks=list(KEEP), reused_tasks=reused,
        external_dependencies=external, partial_outputs=partial, failure_log=str(log), failure_log_sha256=sha256_file(log))
    write_immutable_json(root / "recovery_spec.json", value)
    submit(value, execute=False)
    return value


def validate_recovery(spec):
    validate(spec, "SPEC")
    source, ledger = source_context(Path(spec["source_spec"]))
    require(source["content_hash"] == spec["source_sha256"] and ledger["content_hash"] == spec["original_ledger_sha256"],
            "Recovery source/ledger changed")
    require(spec["retry_tasks"] == list(RETRY) and spec["retained_tasks"] == list(KEEP)
            and spec["original_commit"] == DONOR, "Recovery scope changed")
    require(spec["original_jobs"] == {n: ledger["jobs"][n] for n in RETRY + KEEP}, "Recovery job binding differs")
    root, project = Path(spec["recovery_root"]), Path(spec["project_dir"])
    require(root.resolve() == root and root.parent == Path(source["campaign_root"]) / "recoveries", "Recovery location differs")
    chain._source(project, spec["source_commit"])
    require(spec["helper_sha256"] == {p: sha256_file(project / p) for p in FILES}, "Recovery helper bytes changed")
    require(Path(__file__).resolve() == project / FILES[-1], "Wrong recovery adapter checkout")
    require(sha256_file(spec["failure_log"]) == spec["failure_log_sha256"], "Original failure log changed")
    require(spec["external_dependencies"] in ([], [ledger["jobs"][KEEP[-1]]]), "Foreign external dependency")
    expected_reuse = {r["task_id"] for r in source["tasks"]} - set(RETRY) - set(KEEP)
    require(expected_reuse <= set(spec["reused_tasks"]) <= expected_reuse | set(KEEP), "Missing reuse evidence")
    require(bool(spec["external_dependencies"]) == (KEEP[-1] not in spec["reused_tasks"]),
            "Unfinished bridge requires its registered external dependency")
    for name, digest in spec["reused_tasks"].items():
        receipt = runtime.completed(source, name)
        require(receipt is not None and receipt["content_hash"] == digest, "Reused output changed: " + name)
    runtime.science_gate(source)
    return source


def plan(spec, source):
    rows = []
    for name in RETRY:
        parent = [] if name == TARGET else [TARGET if name == "aggregate" else "aggregate"]
        dependencies = ["${JOB_" + p + "}" for p in parent]
        if name == "aggregate":
            dependencies += spec["external_dependencies"]
        row = runtime.task(source, name)
        cmd = scheduler.command(source, name, source["resources"][row["resource"]])
        cmd = cmd[:-5]
        cmd = ["--job-name=jc2fcr_" + name if c.startswith("--job-name=") else
               "--output=" + str(Path(spec["recovery_root"]) / "slurm-%j.out") if c.startswith("--output=") else c for c in cmd]
        if name == TARGET:
            cmd.append("--hold")
        if dependencies:
            cmd.append("--dependency=afterok:" + ":".join(dependencies))
        cmd += [str(Path(spec["project_dir"]) / FILES[1]), source["project_dir"], spec["project_dir"],
                str(Path(spec["recovery_root"]) / "recovery_spec.json"), name]
        rows.append(dict(task_id=name, dependencies=parent, command=cmd))
    return artifact("PLAN", recovery_sha256=spec["content_hash"], commands=rows)


@contextmanager
def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink()


def reservation(spec, source):
    return artifact("RESERVATION", recovery_sha256=spec["content_hash"], source_sha256=source["content_hash"],
                    retry_tasks=list(RETRY))


def submit(spec, *, execute=False, authorization=None):
    require(not execute or authorization == AUTHORIZE, "Explicit recovery authorization required")
    source = validate_recovery(spec)
    root = Path(spec["recovery_root"])
    directory = root / "submissions_science"
    commands = plan(spec, source)
    write_immutable_json(root / "command_plan.json", commands)
    dry = directory / "dry_run_submission_ledger.json"
    if not execute:
        return submit_exact_dag(identity=spec["content_hash"], plan=commands, output=dry,
                                canonical_dry_run=dry, execute=False)
    with claim(root / "submission.claim"):
        # Cross-recovery exclusion: never let two recovery roots own this task.
        write_immutable_json(safe(source["campaign_root"], "recovery_reservations/" + TARGET + ".json"),
                             reservation(spec, source))
        if (root / "released.json").exists():
            released = load_json(root / "released.json")
            validate(released, "RELEASED")
            require(released["recovery_sha256"] == spec["content_hash"], "Release identity differs")
            return _guarded_exact_submission(spec, commands, directory)
        states = accounting(list(spec["original_jobs"].values()))
        require(states[spec["original_jobs"][TARGET]]["state"] in FAILED, "Original fit is not terminal")
        for name in KEEP:
            if runtime.completed(source, name) is None:
                require(states[spec["original_jobs"][name]]["state"] in {"PENDING", "RUNNING", "COMPLETING"},
                        "Preserved fusion branch failed; stop for review")
        for name in ("aggregate", "complete"):
            state = states[spec["original_jobs"][name]]["state"]
            require(state in FAILED | {"PENDING"}, "Original summary cannot be retired")
            if state == "PENDING":
                info = live_job(source, name, spec["original_jobs"][name])
                require(info["JobState"] == "PENDING", "Original summary started unexpectedly")
        if spec["external_dependencies"]:
            live_job(source, KEEP[-1], spec["external_dependencies"][0])
        # Only the dependency-free replacement can be tested before new IDs
        # exist. This validates the unchanged fit request without submitting.
        if not (directory / "submission_ledger.json").exists():
            test_command = list(commands["commands"][0]["command"])
            test_command.insert(1, "--test-only")
            run(test_command)
        ledger = _guarded_exact_submission(spec, commands, directory)
        # Replacement training stays held until ALL receipts exist and both
        # original summaries have become terminal. No broad cancellation.
        for name in ("complete", "aggregate"):
            job = spec["original_jobs"][name]
            if states[job]["state"] == "PENDING":
                info = live_job(source, name, job)
                require(info["JobState"] == "PENDING", "Original summary is no longer pending")
                run(["scancel", job])
        retired = accounting([spec["original_jobs"][n] for n in ("aggregate", "complete")])
        require(all(r["state"] in FAILED for r in retired.values()), "Cancellation not terminal yet; retry submit")
        job = ledger["jobs"][TARGET]
        run(["scontrol", "release", job])
        write_immutable_json(root / "released.json", artifact("RELEASED", recovery_sha256=spec["content_hash"],
            replacement_ledger_sha256=ledger["content_hash"], job_id=job, retired_jobs=retired))
        return ledger


def authenticate(spec, source, name):
    require(name in RETRY, "Recovery task is outside its registered scope")
    # Reuse the original strict allocation/environment AND exact-ID validator,
    # but bind it to the separate, reviewed recovery ledger, not a forged ID.
    subject = dict(source, campaign_root=spec["recovery_root"], content_hash=spec["content_hash"])
    job = scheduler.authenticate_job(subject, name)
    raw = run(["scontrol", "show", "job", "-o", job])
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    require(fields.get("Command") == str(Path(spec["project_dir"]) / FILES[1])
            and fields.get("WorkDir") == source["project_dir"]
            and fields.get("JobName") == "jc2fcr_" + name, "Recovery scheduler command differs")
    return job


def run_task(spec, name):
    source = validate_recovery(spec)
    job = authenticate(spec, source, name)
    root = Path(spec["recovery_root"])
    with claim(safe(root, "worker_claims/" + name + ".claim")):
        return _run_task(spec, source, name, job)


def _run_task(spec, source, name, job):
    root = Path(spec["recovery_root"])
    require(load_json(safe(source["campaign_root"], "recovery_reservations/" + TARGET + ".json")) == reservation(spec, source),
            "Recovery reservation differs")
    directory = safe(source["campaign_root"], "outputs/" + name)
    receipt_path = root / "receipts" / (name + ".json")
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        validate(receipt, "EXECUTION")
        done = runtime.completed(source, name)
        require(receipt["job_id"] == job and receipt["recovery_sha256"] == spec["content_hash"]
                and done is not None and receipt["task_sha256"] == done["content_hash"], "Execution receipt differs")
        return receipt
    require(runtime.completed(source, name) is None, "Unrecorded completed replacement; review, do not overwrite")
    # Persistent intent fails closed after a killed worker; no automatic retry.
    intent = safe(root, "execution_intents/" + name + ".json")
    require(not intent.exists(), "Previous recovery execution interrupted; operator review required")
    write_immutable_json(intent, artifact("INTENT", recovery_sha256=spec["content_hash"], task_id=name, job_id=job))
    if name == TARGET:
        require(inventory(directory) == spec["partial_outputs"], "Original partial outputs changed")
        destination = safe(root, "archived/" + name)
        require(not destination.exists(), "Archived attempt already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(destination)
    else:
        require(not directory.exists(), "Summary output already exists")
    # Only this process's dispatcher is adapted. The original on-disk module,
    # SLURM_JOB_ID, spec, scientific functions, and original ledger stay intact.
    original_auth = scheduler.authenticate_job
    def authorized(original_spec, task_name, *, launch=False):
        require(not launch and original_spec == source and task_name == name, "Unexpected delegated scientific call")
        return job
    try:
        scheduler.authenticate_job = authorized
        report = runtime.run_task(source, name)
    finally:
        scheduler.authenticate_job = original_auth
    value = artifact("EXECUTION", recovery_sha256=spec["content_hash"], task_id=name, job_id=job,
                     source_sha256=source["content_hash"], task_sha256=report["content_hash"], restart_zero=name == TARGET)
    write_immutable_json(receipt_path, value)
    return value
