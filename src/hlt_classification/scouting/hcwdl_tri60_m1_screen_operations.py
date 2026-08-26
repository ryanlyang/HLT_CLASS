"""Exact-ledger monitor for the standalone TRI60 M1 screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json

from .hcwdl_recovery import (
    TERMINAL_FAILURE, TERMINAL_SUCCESS, task_attestation_path,
    validate_submission_ledger, validate_task_attestation,
)
from .hcwdl_tri60_m1_screen_contracts import (
    MONITOR_CONTRACT, artifact, validate_artifact,
)


def build_monitor(
    *, subject: Mapping[str, Any], ledger: Mapping[str, Any],
    states_by_job_id: Mapping[str, str], attestation_root: str | Path,
) -> dict[str, Any]:
    ledger_hash = validate_submission_ledger(ledger)
    if (
        ledger.get("dry_run") is not False
        or ledger.get("campaign_spec_sha256") != subject.get("content_hash")
    ):
        raise ValueError("TRI60 M1 screen monitor subject differs")
    root = Path(attestation_root)
    rows = []
    for task_id, job_id in ledger["jobs"].items():
        state = str(states_by_job_id.get(str(job_id), "UNKNOWN"))
        state = state.split()[0].split("+")[0].upper()
        path = task_attestation_path(root, task_id, None)
        valid, digest, error_text = False, None, None
        if path.is_file():
            try:
                value = load_json(path)
                digest = validate_task_attestation(
                    value, campaign_spec_sha256=subject["content_hash"],
                    task_id=task_id, array_index=None,
                )
                valid = True
            except Exception as error:  # diagnostic only
                error_text = f"{type(error).__name__}: {error}"
        if state in TERMINAL_SUCCESS and valid:
            disposition = "complete"
        elif state in TERMINAL_SUCCESS or state in TERMINAL_FAILURE:
            disposition = "retryable_failure"
        else:
            disposition = "active_or_unknown"
        rows.append({
            "task_id": task_id, "job_id": str(job_id), "state": state,
            "disposition": disposition, "artifact_valid": valid,
            "attestation_path": str(path.resolve()),
            "attestation_sha256": digest, "artifact_error": error_text,
        })
    return artifact({
        "parents": {"submission_ledger": ledger_hash},
        "subject_spec_sha256": subject["content_hash"], "rows": rows,
        "exact_job_ids_only": True,
        "poor_metrics_do_not_affect_disposition": True,
        "final_test_accessed": False,
    }, contract=MONITOR_CONTRACT)


def validate_monitor(
    value: Mapping[str, Any], *, subject_sha256: str, ledger_sha256: str,
) -> str:
    digest = validate_artifact(value, contract=MONITOR_CONTRACT)
    rows = value.get("rows")
    if (
        value.get("parents", {}).get("submission_ledger") != ledger_sha256
        or value.get("subject_spec_sha256") != subject_sha256
        or not isinstance(rows, list) or not rows
        or len({row.get("task_id") for row in rows}) != len(rows)
        or value.get("exact_job_ids_only") is not True
        or value.get("poor_metrics_do_not_affect_disposition") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 screen monitor differs")
    return digest


__all__ = ["build_monitor", "validate_monitor"]
