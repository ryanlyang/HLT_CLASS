"""Canonical dry plans, durable submit intents, exact-ID monitoring/recovery.

Never cancels, holds, reprioritizes or edits an existing job. An ambiguous
acknowledgement requires explicit reconciliation; no automatic retry exists.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from .campaign import PHRASES, command_plan, stage_dir, validate_spec, file_ref, _publish_spec
from .contracts import artifact, load_json, validate
from .storage import publish_json

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "DEADLINE", "REVOKED"}


def _journal(spec, task):
    return stage_dir(spec)/"submission"/task


def _write(spec, path, value, kind):
    return publish_json(Path(spec["campaign_root"]), path.relative_to(Path(spec["campaign_root"])).as_posix(),
                        value, kind, remaining_bytes=spec["estimated_remaining_writes"])


def submitted_jobs(spec):
    plan = command_plan(spec)
    jobs = {}
    for row in plan["commands"]:
        path = _journal(spec, row["task_id"])/"receipt.json"
        if not path.exists():
            continue
        receipt = load_json(path)
        intent = load_json(path.with_name("intent.json"))
        validate(intent, "SUBMISSION_INTENT", parents={"spec": spec["content_hash"], "plan": plan["content_hash"]})
        validate(receipt, "SUBMISSION_RECEIPT", parents={"intent": intent["content_hash"]})
        if receipt["task_id"] != row["task_id"] or not re.fullmatch(r"[1-9][0-9]*", receipt["job_id"]):
            raise ValueError("Submission receipt identity differs")
        argv = _argv(spec, row, jobs)
        if intent["argv"] != argv or intent["task_id"] != row["task_id"]:
            raise ValueError("Submission journal differs from the canonical dependency command")
        if receipt["job_id"] in jobs.values():
            raise ValueError("One job ID claimed by multiple tasks")
        jobs[row["task_id"]] = receipt["job_id"]
    return jobs


def _argv(spec, row, jobs):
    ids = []
    for parent in row["depends_on"]:
        if parent in spec["reusable_tasks"]:
            from .orchestration import verified_receipt
            verified_receipt(spec, parent)
        elif parent not in jobs:
            raise ValueError("Parent has no exact acknowledged job ID")
        else:
            ids.append(jobs[parent])
    return row["argv"][:1]+(["--dependency=afterok:"+":".join(ids)] if ids else [])+row["argv"][1:]


def submit(spec, *, execute=False, authorization_phrase=None, reviewed_plan_hash=None):
    validate_spec(spec)
    plan = command_plan(spec)
    if load_json(stage_dir(spec)/"command_plan.json") != plan:
        raise ValueError("Dry command plan changed")
    if not execute:
        return plan
    if authorization_phrase != PHRASES[spec["stage"]] or reviewed_plan_hash != plan["content_hash"]:
        raise PermissionError("Exact stage phrase and reviewed dry-plan hash required")
    partition = spec["site"]["partition"]
    site = subprocess.run(["scontrol", "show", "partition", partition, "-o"], text=True, capture_output=True, check=True)
    if f"PartitionName={partition}" not in site.stdout or "State=UP" not in site.stdout:
        raise PermissionError(f"CPU {partition} availability is not established")
    jobs = submitted_jobs(spec)
    for row in plan["commands"]:
        task = row["task_id"]
        if task in jobs:
            continue
        argv = _argv(spec, row, jobs)
        directory = _journal(spec, task)
        directory.parent.mkdir(exist_ok=True)
        if directory.exists():
            raise RuntimeError(f"Unacknowledged intent for {task}; reconcile exact job, do not retry")
        directory.mkdir(exist_ok=False)
        intent = artifact("SUBMISSION_INTENT", parents={"spec": spec["content_hash"], "plan": plan["content_hash"]},
                          task_id=task, argv=argv, reviewed_plan_hash=reviewed_plan_hash)
        _write(spec, directory/"intent.json", intent, "SUBMISSION_INTENT")
        result = subprocess.run(argv, text=True, capture_output=True, check=False)
        if result.returncode or not re.fullmatch(r"[1-9][0-9]*(;[A-Za-z0-9_-]+)?\n?", result.stdout):
            raise RuntimeError(f"Ambiguous/failed submission for {task}; retained intent, no retry. "
                               +result.stdout+result.stderr)
        job = result.stdout.strip().split(";")[0]
        if job in jobs.values():
            raise ValueError("Scheduler returned a duplicate job ID")
        receipt = artifact("SUBMISSION_RECEIPT", parents={"intent": intent["content_hash"]},
                           task_id=task, job_id=job, scheduler_receipt=result.stdout.strip(), reconciled=False)
        _write(spec, directory/"receipt.json", receipt, "SUBMISSION_RECEIPT")
        jobs[task] = job
    ledger = artifact("SUBMISSION_LEDGER", parents={"spec": spec["content_hash"], "plan": plan["content_hash"]},
                      jobs=jobs, dry_run=False)
    _write(spec, stage_dir(spec)/"submission_ledger.json", ledger, "SUBMISSION_LEDGER")
    return ledger


def reconcile(spec, *, task_id, job_id):
    validate_spec(spec)
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("Reconciliation requires one explicit numeric job ID")
    directory = _journal(spec, task_id)
    if (directory/"receipt.json").exists():
        raise ValueError("Task already has an acknowledged receipt")
    intent = load_json(directory/"intent.json")
    plan = command_plan(spec)
    row = next(r for r in plan["commands"] if r["task_id"] == task_id)
    validate(intent, "SUBMISSION_INTENT", parents={"spec": spec["content_hash"], "plan": plan["content_hash"]})
    if intent["argv"] != _argv(spec, row, submitted_jobs(spec)):
        raise ValueError("Unacknowledged command differs")
    result = subprocess.run(["scontrol", "show", "job", "-o", job_id], text=True, capture_output=True, check=True)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)", result.stdout))
    if (fields.get("JobId") != job_id or fields.get("Comment") != f"c2jr:{spec['content_hash']}:{task_id}"
            or fields.get("WorkDir") != spec["project_dir"]
            or fields.get("Command") != str(Path(spec["project_dir"])/"sbatch/run_cms2jc2_response_cpu.sh")
            or fields.get("UserId", "").split("(")[0] != os.environ.get("USER")
            or fields.get("Partition") != spec["site"]["partition"]
            or fields.get("Account") != spec["site"]["account"]):
        raise PermissionError("Exact scheduler provenance could not authenticate this receipt")
    value = artifact("SUBMISSION_RECEIPT", parents={"intent": intent["content_hash"]}, task_id=task_id,
                     job_id=job_id, scheduler_receipt=job_id, reconciled=True,
                     scheduler_evidence=fields)
    _write(spec, directory/"receipt.json", value, "SUBMISSION_RECEIPT")
    return value


def monitor(spec):
    validate_spec(spec)
    jobs = submitted_jobs(spec)
    states = {}
    if jobs:
        result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()),
                                 "--format=JobIDRaw,State,ExitCode"], text=True, capture_output=True, check=True)
        for line in result.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 3 and fields[0] in jobs.values():
                states[fields[0]] = (fields[1].split()[0].rstrip("+"), fields[2])
    from .orchestration import verified_receipt
    rows = []
    for task in spec["tasks"]:
        name = task["task_id"]
        job = jobs.get(name)
        state, code = states.get(job, ("UNKNOWN" if job else "NOT_SUBMITTED", None))
        try:
            receipt = verified_receipt(spec, name)
            output = receipt["content_hash"]
        except FileNotFoundError:
            output = None
        # Corrupt outputs fail loudly; they are not reusable completion.
        rows.append(dict(task_id=name, job_id=job, state=state, exit_code=code, output_receipt=output,
                         reused=name in spec["reusable_tasks"]))
    return artifact("MONITOR", parents={"spec": spec["content_hash"]}, tasks=rows, read_only=True)


def create_recovery(spec_path: Path, *, attempt: str, approved_job_ids: list[str]):
    spec = load_json(spec_path)
    validate_spec(spec)
    if not re.fullmatch(r"r(?:[2-9]|[1-9][0-9]+)", attempt):
        raise ValueError("Recovery needs a fresh explicit r2-or-later attempt name")
    # Deliberately same pinned source: changed scientific/source semantics require
    # a new campaign and new acceptance, not an undocumented recovery override.
    state = monitor(spec)
    reusable, rerun = {}, []
    for row in state["tasks"]:
        task = row["task_id"]
        if row["output_receipt"]:
            if row["job_id"] and row["state"] not in TERMINAL:
                raise PermissionError("Wait for output-writing allocation to become terminal")
            reusable[task] = dict(spec=file_ref(spec_path), receipt_hash=row["output_receipt"])
        elif row["job_id"]:
            if row["state"] not in TERMINAL:
                raise PermissionError("Recovery leaves active/pending jobs untouched; resolve exact IDs first")
            rerun.append(row["job_id"])
        elif (_journal(spec, task)/"intent.json").exists():
            raise PermissionError("Reconcile ambiguous submission before recovery")
    if len(set(approved_job_ids)) != len(approved_job_ids) or set(approved_job_ids) != set(rerun):
        raise PermissionError("Recovery requires the exact terminal IDs being replaced")
    from .contracts import with_content_hash
    new = {k: v for k, v in spec.items() if k != "content_hash"}
    new.update(attempt=attempt, reusable_tasks=reusable,
               parents={**spec["parents"], "recovery_subject": spec["content_hash"], "monitor": state["content_hash"]},
               recovery_monitor=state, replaced_terminal_job_ids=sorted(rerun), restart_incomplete_from_zero=True)
    value = with_content_hash(new)
    validate_spec(value)
    return _publish_spec(value)
