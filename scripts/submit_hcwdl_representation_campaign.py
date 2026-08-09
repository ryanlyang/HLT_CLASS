#!/usr/bin/env python3
"""Submit an explicitly authorized HCWDL-RKD command plan and publish its ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import (
    artifact, load_json_mapping, publish, scheduler,
)
from hlt_classification.scouting.hcwdl_representation_campaign import submit_command_plan


def _validate_reconciled_scheduler_job(intent, job_id: str) -> None:
    output = scheduler(["scontrol", "show", "job", "-o", job_id])
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("scheduler reconciliation did not return one exact job")
    fields = {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in lines[0].split()
        if "=" in token
    }
    command = list(intent["command"])
    comments = [token.split("=", 1)[1] for token in command if token.startswith("--comment=")]
    names = [token.split("=", 1)[1] for token in command if token.startswith("--job-name=")]
    if (
        fields.get("JobId") != job_id or len(comments) != 1 or len(names) != 1
        or fields.get("Comment") != comments[0]
        or fields.get("JobName") != names[0]
        or fields.get("Command") != command[-1]
    ):
        raise PermissionError(
            "scheduler job does not carry the exact reviewed submission intent"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reconciliation", type=Path,
        help="JSON mapping of one unresolved intent SHA-256 to its exact Slurm job ID",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("submission requires the explicit --execute flag")
    spec = artifact(args.campaign_spec)
    event_root = Path(spec["campaign_root"]) / "submission_events"
    prior_events = [artifact(path) for path in sorted(event_root.glob("*.json"))]
    reconciliation = (
        {} if args.reconciliation is None
        else load_json_mapping(args.reconciliation)
    )
    def write_event(event):
        publish(
            event_root / (
                f"{int(event['sequence']):06d}_{event['content_hash']}.json"
            ),
            event,
        )
    ledger = submit_command_plan(
        spec=spec, command_plan=artifact(args.command_plan),
        scheduler=scheduler, execute=True,
        campaign_spec_path=args.campaign_spec,
        event_writer=write_event,
        prior_events=prior_events,
        reconciled_job_ids=reconciliation,
        reconciliation_validator=_validate_reconciled_scheduler_job,
    )
    publish(args.output, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
