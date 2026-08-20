"""Exact-ledger monitoring and cancellation for HCWDL-MHPE TRI60."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json

from .hcwdl_mhpe_tri60_contracts import (
    CANCELLATION_CONTRACT,
    MONITOR_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_recovery import (
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
    task_attestation_path,
    validate_submission_ledger,
    validate_task_attestation,
)


def build_monitor(
    *, subject: Mapping[str, Any], ledger: Mapping[str, Any],
    states_by_job_id: Mapping[str, str], attestation_root: str | Path,
) -> dict[str, Any]:
    """Build an immutable monitor from exact Slurm IDs and output attestations."""

    ledger_hash = validate_submission_ledger(ledger)
    if ledger.get("dry_run") is not False:
        raise ValueError("TRI60 monitor requires a live ledger")
    if ledger.get("campaign_spec_sha256") != subject.get("content_hash"):
        raise ValueError("TRI60 monitor subject/ledger differs")
    rows = []
    root = Path(attestation_root)
    for task_id, job_id in ledger["jobs"].items():
        raw_state = str(states_by_job_id.get(str(job_id), "UNKNOWN"))
        state = raw_state.split()[0].split("+")[0].upper()
        attestation_path = task_attestation_path(root, task_id, None)
        artifact_valid = False
        attestation_hash = None
        artifact_error = None
        if attestation_path.is_file():
            try:
                attestation = load_json(attestation_path)
                attestation_hash = validate_task_attestation(
                    attestation,
                    campaign_spec_sha256=subject["content_hash"],
                    task_id=task_id,
                    array_index=None,
                )
                artifact_valid = True
            except Exception as error:  # monitor records, recovery fails closed later
                artifact_error = f"{type(error).__name__}: {error}"
        if state in TERMINAL_SUCCESS and artifact_valid:
            disposition = "complete"
        elif state in TERMINAL_SUCCESS or state in TERMINAL_FAILURE:
            disposition = "retryable_failure"
        else:
            disposition = "active_or_unknown"
        rows.append({
            "task_id": task_id,
            "job_id": str(job_id),
            "state": state,
            "disposition": disposition,
            "artifact_valid": artifact_valid,
            "attestation_path": str(attestation_path.resolve()),
            "attestation_sha256": attestation_hash,
            "artifact_error": artifact_error,
        })
    return artifact({
        "parents": {"submission_ledger": ledger_hash},
        "subject_spec_sha256": subject["content_hash"],
        "rows": rows,
        "exact_job_ids_only": True,
        "poor_metrics_do_not_affect_disposition": True,
        "final_test_accessed": False,
    }, contract=MONITOR_CONTRACT)


def validate_monitor(
    value: Mapping[str, Any], *, subject_sha256: str,
    ledger_sha256: str,
) -> str:
    digest = validate_artifact(value, contract=MONITOR_CONTRACT)
    rows = value.get("rows")
    if (
        value.get("parents", {}).get("submission_ledger") != ledger_sha256
        or value.get("subject_spec_sha256") != subject_sha256
        or not isinstance(rows, list)
        or not rows
        or len({row.get("task_id") for row in rows}) != len(rows)
        or value.get("exact_job_ids_only") is not True
        or value.get("poor_metrics_do_not_affect_disposition") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 monitor semantics differ")
    allowed = {"complete", "retryable_failure", "active_or_unknown"}
    if any(row.get("disposition") not in allowed for row in rows):
        raise ValueError("TRI60 monitor disposition differs")
    return digest


def build_cancellation(
    *, ledger: Mapping[str, Any], monitor: Mapping[str, Any],
    task_ids: tuple[str, ...] | None = None, executed: bool,
) -> dict[str, Any]:
    ledger_hash = validate_submission_ledger(ledger)
    if ledger.get("dry_run") is not False:
        raise PermissionError("TRI60 dry-run ledger cannot be cancelled")
    monitor_hash = validate_artifact(monitor, contract=MONITOR_CONTRACT)
    if monitor.get("parents", {}).get("submission_ledger") != ledger_hash:
        raise ValueError("TRI60 cancellation monitor/ledger differs")
    monitor_rows = {str(row.get("task_id")): row for row in monitor.get("rows", ())}
    if (
        set(monitor_rows) != set(ledger["jobs"])
        or any(
            str(monitor_rows[task].get("job_id")) != str(job_id)
            for task, job_id in ledger["jobs"].items()
        )
    ):
        raise ValueError("TRI60 cancellation monitor coverage differs")
    selected = tuple(sorted(ledger["jobs"])) if task_ids is None else tuple(task_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(ledger["jobs"]):
        raise ValueError("TRI60 cancellation task registry differs")
    rows = []
    for task in selected:
        state = str(monitor_rows[task]["state"])
        if state in TERMINAL_SUCCESS or state in TERMINAL_FAILURE:
            category = "terminal"
        elif state in {"PENDING", "CONFIGURING"}:
            category = "pending"
        elif state in {"RUNNING", "COMPLETING", "SUSPENDED"}:
            category = "running"
        else:
            category = "unknown"
        rows.append({
            "task_id": task, "job_id": ledger["jobs"][task],
            "state": state, "state_category": category,
        })
    return artifact({
        "parents": {
            "submission_ledger": ledger_hash, "monitor": monitor_hash,
        },
        "task_ids": list(selected),
        "job_ids": [ledger["jobs"][task] for task in selected],
        "rows": rows,
        "executed": bool(executed),
        "exact_ids_only": True,
        "recoverable_from_ledger": True,
        "final_test_accessed": False,
    }, contract=CANCELLATION_CONTRACT)


__all__ = [
    "build_cancellation", "build_monitor", "validate_monitor",
]
