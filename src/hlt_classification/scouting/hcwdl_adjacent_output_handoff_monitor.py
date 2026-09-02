"""Ledger-bound monitoring for adjacent output handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json

from .hcwdl_adjacent_output_handoff_campaign import validate_campaign
from .hcwdl_adjacent_output_handoff_workflow import task_outputs
from .hcwdl_recovery import (
    build_monitor_report, task_attestation_path, validate_submission_ledger,
    validate_task_attestation,
)


def build_monitor(
    *, campaign_spec: str | Path, submission_ledger: str | Path,
    states_by_job_id: Mapping[str, str],
) -> dict[str, Any]:
    spec = load_json(campaign_spec); validate_campaign(spec)
    ledger = load_json(submission_ledger); validate_submission_ledger(ledger)
    if ledger.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("output-handoff monitor ledger belongs to another campaign")
    validity = {}
    for task_id in ledger["jobs"]:
        path = task_attestation_path(spec["campaign_root"], task_id, None)
        try:
            attestation = load_json(path)
            validate_task_attestation(
                attestation, campaign_spec_sha256=spec["content_hash"],
                task_id=task_id, array_index=None,
            )
            validity[task_id] = all(item.is_file() for item in task_outputs(spec, task_id))
        except (OSError, KeyError, TypeError, ValueError):
            validity[task_id] = False
    return build_monitor_report(
        ledger, states_by_job_id=states_by_job_id, artifact_validity=validity,
    )


__all__ = ["build_monitor"]
