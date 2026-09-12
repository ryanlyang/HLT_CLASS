"""Exact campaign-only Slurm plans, submission journals and restart-zero recovery."""
from pathlib import Path
import os
import shutil
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, _journal, _resolved
from hlt_classification.scouting.hcwdl_recovery import (
    validate_submission_ledger, build_submission_event, assemble_submission_ledger,
)
from .contracts import artifact, validate
from .production import AUTHORIZE, completed_task, validate_campaign
from .execution import slurm_options

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def monitor(spec: dict, ledger: dict) -> dict:
    validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["dry_run"]:
        raise ValueError("Monitor requires this exact campaign's live ledger")
    jobs = ledger["jobs"]
    expected = {r["task_id"] for r in spec["tasks"]}
    if not set(jobs) <= expected or not jobs:
        raise ValueError("Ledger has unknown campaign tasks")
    raw = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()),
                          "--format=JobID,State%40"], capture_output=True, text=True, check=True).stdout
    states = {}
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0].strip() in jobs.values():
            states[parts[0].strip()] = parts[1].strip().split()[0].rstrip("+")
    return artifact(
        "MONITOR", campaign_sha256=spec["content_hash"], ledger_sha256=ledger["content_hash"],
        rows=[dict(task_id=task, job_id=job, state=states.get(job, "UNKNOWN"),
                   outputs_complete=completed_task(spec, task) is not None) for task, job in sorted(jobs.items())],
        remote_mutations=False,
    )


def command_plan(spec: dict, *, tasks: list[str] | None = None) -> dict:
    known = {r["task_id"]: r for r in spec["tasks"]}
    chosen = list(known) if tasks is None else tasks
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(known):
        raise ValueError("Command plan task coverage differs")
    profile = spec["runtime_profile"]
    site = profile["execution_site"]
    rows = []
    for task in spec["tasks"]:
        name = task["task_id"]
        if name not in chosen:
            continue
        dependencies = [p for p in task["dependencies"] if p in chosen]
        for parent in task["dependencies"]:
            if parent not in chosen and completed_task(spec, parent) is None:
                raise ValueError("Omitted dependency is not durably completed")
        gpu = task["kind"] in {"train", "reduce"}
        minutes = profile["train_minutes"] if task["kind"] == "train" else profile["reduce_minutes"]
        command = slurm_options(site) + [f"--cpus-per-task={profile['cpus'] if gpu else 1}",
                   f"--mem={profile['memory_mb'] if gpu else 8192}M", f"--time={minutes if gpu else 60}",
                   "--job-name=jc2_" + name, "--chdir=" + spec["project_dir"],
                   "--output=" + str(Path(spec["campaign_root"]) / "slurm-%j.out")]
        if gpu:
            command += ["--gres=" + site["gres"]]
        if dependencies:
            command += ["--dependency=afterok:" + ":".join("${JOB_" + p + "}" for p in dependencies)]
        command += [str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes.sh"), spec["project_dir"],
                    str(Path(spec["campaign_root"]) / "campaign_spec.json"), name, site["name"]]
        rows.append(dict(task_id=name, dependencies=dependencies, command=command))
    return artifact("COMMAND_PLAN", campaign_sha256=spec["content_hash"], commands=rows,
                    single_gpu=True, restart_from_zero=True, mutations_outside_campaign=False)


def prepare_recovery(spec: dict, ledger: dict, *, output_root: Path) -> dict:
    """Refuse to compete with any live/unknown old job; never cancel automatically."""
    validate_campaign(spec)
    observed = monitor(spec, ledger)
    unsafe = [r for r in observed["rows"] if r["state"] not in TERMINAL]
    if unsafe:
        raise PermissionError("Recovery requires old attempts to be terminal; leave running jobs alone or explicitly cancel exact pending IDs first: "
                              + ", ".join(f"{r['job_id']}={r['state']}" for r in unsafe))
    _other_attempts_terminal(spec)
    tasks = [r["task_id"] for r in spec["tasks"] if completed_task(spec, r["task_id"]) is None]
    if not tasks:
        raise ValueError("Campaign has no unfinished tasks")
    root = Path(output_root).resolve()
    if root.exists() or not root.is_relative_to(Path(spec["campaign_root"]).resolve()):
        raise ValueError("Recovery bookkeeping requires a fresh subdirectory inside this campaign")
    plan = command_plan(spec, tasks=tasks)
    report = artifact("RECOVERY", campaign_sha256=spec["content_hash"], ledger_sha256=ledger["content_hash"],
                      monitor_sha256=observed["content_hash"], source_commit=spec["source_commit"],
                      command_plan_sha256=plan["content_hash"], remaining_tasks=tasks,
                      completed_tasks_reused=True, resume_optimizer=False, jobs_cancelled=[])
    write_immutable_json(root / "monitor.json", observed)
    write_immutable_json(root / "recovery.json", report)
    write_immutable_json(root / "command_plan.json", plan)
    return report


def _other_attempts_terminal(spec, *, exclude_root=None):
    """Do not let an old ledger hide a newer live or partially submitted recovery."""
    root = Path(spec["campaign_root"]).resolve()
    directories = {p.parent for p in root.rglob("submission_ledger.json")}
    directories |= {p.parent for p in root.rglob("submission_ledger_journal") if p.is_dir()}
    directories |= {p.parent for p in root.rglob("submission_intents") if p.is_dir()}
    for directory in sorted(directories):
        if directory.resolve() == exclude_root:
            continue
        if not directory.resolve().is_relative_to(root):
            raise ValueError("Submission ancestry escaped campaign")
        path = directory / "submission_ledger.json"
        if not path.is_file():
            raise PermissionError("Unfinished submission must be reconciled before another recovery: " + str(directory))
        observed = monitor(spec, load_json(path))
        unsafe = [r for r in observed["rows"] if r["state"] not in TERMINAL]
        if unsafe:
            raise PermissionError("Another campaign attempt is active or unknown: "
                                  + ", ".join(f"{r['job_id']}={r['state']}" for r in unsafe))


def _guarded_exact_submission(spec, plan, root):
    """Reuse exact-DAG receipts, with a fail-closed sbatch acknowledgement window.

    A durable intent written before sbatch prevents duplicate submission if the
    process dies after Slurm accepts a job but before its receipt is recorded.
    Such an ambiguous submission needs operator reconciliation, not blind retry.
    """
    dry = root / "dry_run_submission_ledger.json"
    destination = root / "submission_ledger.json"
    # Validates the mandatory canonical dry plan without ever calling sbatch.
    if not dry.is_file():
        raise ValueError("Canonical dry run is required before live submission")
    submit_exact_dag(identity=spec["content_hash"], plan=plan, output=dry,
                     canonical_dry_run=dry, execute=False)
    if destination.exists():
        return submit_exact_dag(identity=spec["content_hash"], plan=plan, output=destination,
                                canonical_dry_run=dry, execute=True)
    directory = root / "submission_ledger_journal"
    events, jobs = _journal(directory, identity=spec["content_hash"], plan=plan)
    intents = root / "submission_intents"
    for path in intents.glob("*.json") if intents.exists() else ():
        receipt = directory / path.name
        if not receipt.is_file():
            raise PermissionError("Ambiguous sbatch acknowledgement; reconcile exact job before retry: " + str(path))
    for sequence, row in enumerate(plan["commands"][len(events):], start=len(events)):
        command = _resolved(row, jobs)
        name = f"{sequence:04d}_{row['task_id']}.json"
        intent = artifact("SUBMISSION_INTENT", campaign_sha256=spec["content_hash"],
                          task_id=row["task_id"], sequence=sequence, command=command)
        write_immutable_json(intents / name, intent)
        job_id = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
        event = build_submission_event(campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
                                       job_id=job_id, command=command, sequence=sequence)
        write_immutable_json(directory / name, event)
        events.append(event)
        jobs[row["task_id"]] = job_id
    ledger = assemble_submission_ledger(events, campaign_spec_sha256=spec["content_hash"])
    write_immutable_json(destination, ledger)
    return ledger


def submit(spec: dict, *, bookkeeping_root: Path, execute: bool, authorization_phrase: str | None = None) -> dict:
    validate_campaign(spec)
    root = Path(bookkeeping_root).resolve()
    campaign_root = Path(spec["campaign_root"]).resolve()
    if not root.is_relative_to(campaign_root):
        raise ValueError("Submission bookkeeping cannot escape campaign")
    if root == campaign_root:
        plan = command_plan(spec)
        write_immutable_json(root / "command_plan.json", plan)
    else:
        recovery = load_json(root / "recovery.json")
        validate(recovery, "RECOVERY")
        plan = load_json(root / "command_plan.json")
        if recovery["campaign_sha256"] != spec["content_hash"] or recovery["source_commit"] != spec["source_commit"]:
            raise ValueError("Recovery subject differs")
        if plan != command_plan(spec, tasks=recovery["remaining_tasks"]) or plan["content_hash"] != recovery["command_plan_sha256"]:
            raise ValueError("Recovery dependency plan differs")
    validate(plan, "COMMAND_PLAN")
    if execute:
        if authorization_phrase != AUTHORIZE:
            raise PermissionError("Exact new-dataset campaign authorization phrase required")
        n = sum(spec["foundation"]["splits"]["role_counts"][r] for r in ("train", "validation"))
        needed = 2 * (n * 76 * 26 + spec["runtime_profile"]["selected_state_bytes"] * 31) + 2**30
        if shutil.disk_usage(campaign_root).free < needed:
            raise OSError(f"Insufficient campaign-volume headroom; require {needed} free bytes")
    claim = campaign_root / "submission_in_progress.claim"
    # Prevent simultaneous submitters. A killed submitter leaves a small claim
    # for explicit operator investigation; it is never auto-bypassed.
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if execute:
            _other_attempts_terminal(spec, exclude_root=root)
            return _guarded_exact_submission(spec, plan, root)
        return submit_exact_dag(identity=spec["content_hash"], plan=plan,
                                output=root / ("submission_ledger.json" if execute else "dry_run_submission_ledger.json"),
                                canonical_dry_run=root / "dry_run_submission_ledger.json", execute=execute)
    finally:
        claim.unlink()
