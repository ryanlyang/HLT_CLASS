"""Source-aware exact submission and separately authorized dense retirement."""
from __future__ import annotations

import getpass
from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import _resolved, submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger, build_submission_event, validate_submission_ledger
from .contracts import SHARED_TASKS
from .shared_import import dense_ledger, source_task_outputs, validate_shared_source
from .storage import receipt_path

RETIRE_AUTHORIZATION = "CANCEL CMS DENSE LADDER KEEP SHARED"


def queue_rows(source, jobs):
    """Exact IDs only. Reject owner/name/account mismatches before mutation."""
    if not jobs:
        return {}
    if any(re.fullmatch(r"[1-9][0-9]*", str(j)) is None for j in jobs.values()):
        raise ValueError("Invalid source job IDs")
    # Read this user's queue, then intersect exact ledger IDs. Unlike --jobs,
    # this remains valid when every requested ID has aged out of Slurm.
    result = subprocess.run(["squeue", "--noheader", "--user=" + getpass.getuser(),
        "--format=%i|%T|%u|%j|%a|%R"], check=True, text=True, capture_output=True)
    by_id = {job: task for task, job in jobs.items()}
    rows = {}
    for line in result.stdout.splitlines():
        job, state, owner, name, account, reason = (s.strip() for s in line.split("|", 5))
        if job not in by_id:
            continue
        if (job in rows or owner != getpass.getuser()
            or name != "cmslfh_" + by_id[job] or account != source["site"]["account"]):
            raise ValueError("Live source job identity differs from authenticated ledger")
        rows[job] = dict(task=by_id[job], state=state, reason=reason)
    return rows


def _completed(source, task):
    if not receipt_path(source, task).exists():
        return False
    source_task_outputs(source, task)
    return True


def _external_job(source, task, job):
    if _completed(source, task):
        return None
    try:
        rows = queue_rows(source, {task: job})
    except subprocess.CalledProcessError:
        # squeue may reject an ID that just completed/aged out. An error alone
        # is not completion evidence and never licenses bypassing a dependency.
        if _completed(source, task):
            return None
        raise
    row = rows.get(job)
    if row is None:
        if _completed(source, task):  # completion between the first check and squeue
            return None
        raise ValueError(f"Source {task}/{job} is absent without authenticated completion")
    if row["state"] not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"} or "DependencyNeverSatisfied" in row["reason"]:
        raise ValueError(f"Source {task}/{job} cannot satisfy the import: {row}")
    return job


def _command(row, jobs, external):
    command = _resolved(row, jobs)
    token = "${SOURCE_JOB_" + row["task_id"] + "}"
    result = []
    for arg in command:
        if token in arg:
            if not arg.startswith("--dependency=afterok:"):
                raise ValueError("Invalid source dependency position")
            ids = arg.removeprefix("--dependency=afterok:").split(":")
            ids = [external if j == token else j for j in ids]
            ids = [j for j in ids if j is not None]
            if not ids:
                continue
            arg = "--dependency=afterok:" + ":".join(ids)
        result.append(arg)
    if any("${" in arg for arg in result):
        raise ValueError("Unresolved science dependency")
    return result


def submit_shared_dag(spec, plan, *, execute):
    base = Path(spec["campaign_root"])
    dry_path = base / "science_dry_run_submission_ledger.json"
    if execute and not dry_path.is_file():
        raise ValueError("Canonical science dry-run evidence is missing")
    # This canonical dry ledger deliberately shows symbolic SOURCE_JOB edges.
    dry = submit_exact_dag(identity=spec["content_hash"], plan=plan, output=dry_path,
        canonical_dry_run=dry_path, execute=False)
    if not execute:
        return dry
    source = validate_shared_source(spec)
    external_jobs = spec["shared_source"]["jobs"]
    directory = base / "science_submission_ledger_journal"
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    if len(paths) > len(plan["commands"]):
        raise ValueError("Excess science journal events")
    events, jobs = [], {}
    for index, path in enumerate(paths):
        row = plan["commands"][index]
        event = load_json(path)
        task = row["task_id"]
        external = external_jobs.get(task)
        commands = [_command(row, jobs, external)]
        if external is not None and _completed(source, task):
            commands.append(_command(row, jobs, None))
        if (path.name != f"{index:04d}_{task}.json" or event["command"] not in commands
            or re.fullmatch(r"[1-9][0-9]*", str(event["job_id"])) is None
            or event["job_id"] in jobs.values()
            or event != build_submission_event(campaign_spec_sha256=spec["content_hash"],
                task_id=task, job_id=event["job_id"], command=event["command"], sequence=index)):
            raise ValueError("Science submission journal differs")
        events.append(event); jobs[task] = event["job_id"]
    ledger_path = base / "science_submission_ledger.json"
    if ledger_path.exists() and len(events) != len(plan["commands"]):
        raise ValueError("Live ledger lacks a complete exact journal")
    for index, row in enumerate(plan["commands"][len(events):], start=len(events)):
        task = row["task_id"]
        external = _external_job(source, task, external_jobs[task]) if task in external_jobs else None
        command = _command(row, jobs, external)
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            # A completed source can age out of Slurm between resolution and
            # submission. Retry only a rejected dependency, with verified data.
            if external is None or "dependency" not in (error.stderr or "").lower() or not _completed(source, task):
                print(error.stderr or str(error), flush=True)
                raise
            command = _command(row, jobs, None)
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            external = None
        job = result.stdout.strip().split(";")[0]
        if re.fullmatch(r"[1-9][0-9]*", job) is None or job in jobs.values():
            raise ValueError("Slurm returned an invalid/duplicate job ID; inspect before retrying")
        event = build_submission_event(campaign_spec_sha256=spec["content_hash"], task_id=task,
            job_id=job, command=command, sequence=index)
        write_immutable_json(directory / f"{index:04d}_{task}.json", event)
        events.append(event); jobs[task] = job
        print(f"CMS-COARSE submitted task={task} job={job} source_dependency={external or 'none'}", flush=True)
    ledger = assemble_submission_ledger(events, campaign_spec_sha256=spec["content_hash"])
    if ledger_path.exists():
        stored = load_json(ledger_path); validate_submission_ledger(stored)
        if stored != ledger:
            raise ValueError("Live science ledger differs from its journal")
    else:
        write_immutable_json(ledger_path, ledger)
    return ledger


def retire_dense(spec, *, execute=False, authorization_phrase=None):
    from .campaign import gate_check, validate_campaign
    validate_campaign(spec, check_source=True)
    if spec.get("shared_source") is None:
        raise ValueError("Retirement requires an explicit shared dense source")
    source = validate_shared_source(spec)
    ledger = dense_ledger(source)
    targets = {t: j for t, j in ledger["jobs"].items() if t not in SHARED_TASKS}
    rows = queue_rows(source, targets)
    if execute:
        if authorization_phrase != RETIRE_AUTHORIZATION:
            raise PermissionError("Exact dense-retirement authorization required")
        gate_check(spec)
        # Do not retire a usable dense chain if a shared parent has meanwhile
        # failed: that must be diagnosed before changing any queued work.
        for task in SHARED_TASKS:
            _external_job(source, task, ledger["jobs"][task])
        if rows:
            # Only exact dense-specific IDs; do not cancel shared jobs even if
            # they are pending, and never delete any campaign output.
            subprocess.run(["scancel", *rows], check=True)
    return dict(source_campaign=source["campaign_root"], execute=execute,
        dense_jobs=rows, preserved_jobs={t: ledger["jobs"][t] for t in SHARED_TASKS})
