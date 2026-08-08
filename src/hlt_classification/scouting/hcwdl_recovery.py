"""Exact-ID HCWDL submission journals, monitoring, resume, and cancellation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash,
)


LEDGER_CONTRACT: Final = "HCWDL_SUBMISSION_LEDGER/v2"
MONITOR_CONTRACT: Final = "HCWDL_MONITOR_REPORT/v1"
TASK_ATTESTATION_CONTRACT: Final = "HCWDL_TASK_ATTESTATION/v1"
SUBMISSION_EVENT_CONTRACT: Final = "HCWDL_SUBMISSION_EVENT/v1"
_JOB_ID = re.compile(r"^[1-9][0-9]*(?:_[0-9]+)?$")
TERMINAL_SUCCESS: Final = frozenset({"COMPLETED"})
TERMINAL_FAILURE: Final = frozenset({"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"})


def _job_id(value: object) -> str:
    result = str(value)
    if not _JOB_ID.fullmatch(result):
        raise ValueError(f"invalid exact Slurm job ID {result!r}")
    return result


def build_submission_ledger(
    *, campaign_spec_sha256: str, jobs: Mapping[str, str], commands: Mapping[str, Sequence[str]],
    dry_run: bool,
    parent_ledger_sha256: str | None = None,
    monitor_report_sha256: str | None = None,
    superseded_jobs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if set(jobs) != set(commands) or not jobs:
        raise ValueError("HCWDL ledger jobs/commands differ")
    normalized = {task: _job_id(job) for task, job in jobs.items()} if not dry_run else {
        task: f"DRY_RUN_{index:04d}" for index, task in enumerate(sorted(jobs))
    }
    return with_content_hash({
        "contract": LEDGER_CONTRACT, "schema_version": 2,
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign SHA-256"),
        "dry_run": bool(dry_run), "jobs": dict(sorted(normalized.items())),
        "commands": {task: list(commands[task]) for task in sorted(commands)},
        "exact_ids_only": not dry_run,
        "parent_ledger_sha256": (
            None if parent_ledger_sha256 is None
            else require_sha256(parent_ledger_sha256, name="parent ledger SHA-256")
        ),
        "monitor_report_sha256": (
            None if monitor_report_sha256 is None
            else require_sha256(monitor_report_sha256, name="monitor report SHA-256")
        ),
        "superseded_jobs": {
            task: _job_id(job) for task, job in sorted((superseded_jobs or {}).items())
        },
    })


def validate_submission_ledger(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=LEDGER_CONTRACT, expected_schema_version=2)
    require_sha256(value.get("campaign_spec_sha256"), name="campaign SHA-256")
    if set(value.get("jobs", {})) != set(value.get("commands", {})):
        raise ValueError("HCWDL ledger jobs/commands differ")
    if not value.get("dry_run"):
        for job in value["jobs"].values():
            _job_id(job)
    parent = value.get("parent_ledger_sha256")
    monitor = value.get("monitor_report_sha256")
    superseded = value.get("superseded_jobs")
    if (parent is None) != (monitor is None):
        raise ValueError("HCWDL recovery ledger parent/monitor lineage differs")
    if not isinstance(superseded, Mapping):
        raise ValueError("HCWDL superseded-job registry differs")
    for task, job in superseded.items():
        if task not in value["jobs"]:
            raise ValueError("HCWDL superseded job lacks a replacement")
        _job_id(job)
    if parent is not None:
        require_sha256(parent, name="parent ledger SHA-256")
        require_sha256(monitor, name="monitor report SHA-256")
    return digest


def build_submission_event(
    *, campaign_spec_sha256: str, task_id: str, job_id: str,
    command: Sequence[str], sequence: int,
) -> dict[str, Any]:
    if not task_id or not command or sequence < 0:
        raise ValueError("HCWDL submission event differs")
    return with_content_hash({
        "contract": SUBMISSION_EVENT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign SHA-256",
        ),
        "task_id": task_id, "job_id": _job_id(job_id),
        "command": list(command), "sequence": int(sequence),
    })


def assemble_submission_ledger(
    events: Sequence[Mapping[str, Any]], *, campaign_spec_sha256: str,
) -> dict[str, Any]:
    if not events:
        raise ValueError("HCWDL submission journal is empty")
    ordered = sorted(events, key=lambda row: int(row.get("sequence", -1)))
    if [int(row.get("sequence", -1)) for row in ordered] != list(range(len(ordered))):
        raise ValueError("HCWDL submission journal sequence differs")
    jobs = {}; commands = {}
    for row in ordered:
        validate_content_hash(
            row, expected_contract=SUBMISSION_EVENT_CONTRACT, expected_schema_version=1,
        )
        if row.get("campaign_spec_sha256") != campaign_spec_sha256:
            raise ValueError("HCWDL submission event belongs to another campaign")
        task = str(row["task_id"])
        if task in jobs:
            raise ValueError("HCWDL submission journal repeats a task")
        jobs[task] = _job_id(row["job_id"]); commands[task] = list(row["command"])
    return build_submission_ledger(
        campaign_spec_sha256=campaign_spec_sha256, jobs=jobs,
        commands=commands, dry_run=False,
    )


def build_monitor_report(
    ledger: Mapping[str, Any], *, states_by_job_id: Mapping[str, str],
    artifact_validity: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    ledger_hash = validate_submission_ledger(ledger)
    if ledger.get("dry_run"):
        raise ValueError("a dry-run HCWDL ledger cannot be monitored")
    rows = []
    for task, job in ledger["jobs"].items():
        state = str(states_by_job_id.get(job, "UNKNOWN")).split("+")[0].upper()
        artifacts_valid = True if artifact_validity is None else artifact_validity.get(task) is True
        if state in TERMINAL_SUCCESS and artifacts_valid:
            disposition = "complete"
        elif state in TERMINAL_SUCCESS:
            state = "ARTIFACT_INVALID"; disposition = "retryable_failure"
        elif state in TERMINAL_FAILURE:
            disposition = "retryable_failure"
        else:
            disposition = "active_or_unknown"
        rows.append({
            "task_id": task, "job_id": _job_id(job), "state": state,
            "disposition": disposition, "artifacts_valid": artifacts_valid,
        })
    return with_content_hash({
        "contract": MONITOR_CONTRACT, "schema_version": 1,
        "submission_ledger_sha256": ledger_hash, "rows": rows,
    })


def exact_cancel_ids(ledger: Mapping[str, Any]) -> tuple[str, ...]:
    validate_submission_ledger(ledger)
    if ledger.get("dry_run"):
        raise PermissionError("dry-run ledger has no cancellable job IDs")
    return tuple(_job_id(value) for _, value in sorted(ledger["jobs"].items()))


def resume_tasks(
    monitor: Mapping[str, Any], *, dependency_graph: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    validate_content_hash(monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1)
    rows = {str(row["task_id"]): row for row in monitor["rows"]}
    if set(rows) - set(dependency_graph):
        raise ValueError("HCWDL monitor contains a task outside the dependency graph")
    retry = set(dependency_graph) - set(rows)
    retry |= {task for task, row in rows.items() if row["disposition"] == "retryable_failure"}
    # Downstream tasks whose dependency was retried must receive a fresh exact dependency.
    changed = True
    while changed:
        changed = False
        for task, parents in dependency_graph.items():
            if task not in retry and any(parent in retry for parent in parents):
                retry.add(task); changed = True
    return tuple(task for task in dependency_graph if task in retry)


def task_attestation_path(
    campaign_root: str | Path, task_id: str, array_index: int | None,
) -> Path:
    suffix = "single" if array_index is None else f"array_{array_index:06d}"
    return Path(campaign_root) / "tasks" / task_id / f"{suffix}.json"


def build_task_attestation(
    *, campaign_spec_sha256: str, task_id: str, array_index: int | None,
    outputs: Sequence[str | Path],
) -> dict[str, Any]:
    if not task_id or not outputs:
        raise ValueError("HCWDL task attestation requires a task and outputs")
    rows = []
    for raw in outputs:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"HCWDL task output is absent: {path}")
        record: dict[str, Any] = {"path": str(path), "byte_sha256": sha256_file(path)}
        if path.suffix.lower() == ".json":
            value = load_json(path); supplied = require_sha256(
                value.get("content_hash"), name=f"task output content hash {path}",
            )
            unhashed = dict(value); unhashed.pop("content_hash", None)
            if canonical_sha256(unhashed) != supplied:
                raise ValueError(f"HCWDL task JSON content hash differs: {path}")
            record["content_hash"] = supplied
        rows.append(record)
    return with_content_hash({
        "contract": TASK_ATTESTATION_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign spec SHA-256",
        ),
        "task_id": task_id, "array_index": array_index, "outputs": rows,
    })


def validate_task_attestation(
    value: Mapping[str, Any], *, campaign_spec_sha256: str, task_id: str,
    array_index: int | None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TASK_ATTESTATION_CONTRACT, expected_schema_version=1,
    )
    if (
        value.get("campaign_spec_sha256") != campaign_spec_sha256
        or value.get("task_id") != task_id
        or value.get("array_index") != array_index
    ):
        raise ValueError("HCWDL task attestation identity or lineage differs")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("HCWDL task attestation output inventory is empty")
    for record in outputs:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record.get("byte_sha256"):
            raise ValueError("HCWDL attested task output bytes differ")
        if "content_hash" in record:
            payload = load_json(path); supplied = payload.get("content_hash")
            unhashed = dict(payload); unhashed.pop("content_hash", None)
            if supplied != record["content_hash"] or canonical_sha256(unhashed) != supplied:
                raise ValueError("HCWDL attested JSON output content differs")
    return digest


__all__ = [
    "LEDGER_CONTRACT", "MONITOR_CONTRACT", "SUBMISSION_EVENT_CONTRACT",
    "TASK_ATTESTATION_CONTRACT",
    "TERMINAL_FAILURE", "TERMINAL_SUCCESS", "build_monitor_report",
    "assemble_submission_ledger", "build_submission_event", "build_submission_ledger",
    "build_task_attestation", "exact_cancel_ids",
    "resume_tasks", "task_attestation_path", "validate_submission_ledger",
    "validate_task_attestation",
]
