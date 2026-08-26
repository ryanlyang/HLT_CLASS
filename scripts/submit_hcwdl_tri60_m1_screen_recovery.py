#!/usr/bin/env python3
"""Dry-run or submit the exact TRI60 M1 screen recovery DAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    assemble_submission_ledger, build_submission_event, build_submission_ledger,
)
from hlt_classification.scouting.hcwdl_tri60_m1_screen_contracts import COMMAND_PLAN_CONTRACT, validate_artifact  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_m1_screen_recovery import RECOVERY_SUBMISSION_PHRASE, validate_recovery  # noqa: E402


def _resolved(row, jobs):
    command = list(row["command"])
    for index, item in enumerate(command):
        for parent in row["dependencies"]:
            item = item.replace(f"${{JOB_{parent}}}", jobs[parent])
        command[index] = item
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    recovery = load_json(args.recovery_spec)
    validate_recovery(recovery)
    plan = load_json(Path(recovery["recovery_root"]) / "command_plan.json")
    validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT)
    commands = {row["task_id"]: list(row["command"]) for row in plan["commands"]}
    dry = build_submission_ledger(
        campaign_spec_sha256=recovery["content_hash"],
        jobs={task: "1" for task in commands}, commands=commands, dry_run=True,
    )
    if not args.execute:
        write_immutable_json(args.output, dry)
        return 0
    if args.authorization_phrase != RECOVERY_SUBMISSION_PHRASE:
        raise PermissionError("TRI60 M1 screen recovery phrase differs")
    if load_json(Path(recovery["recovery_root"]) / "dry_run_submission_ledger.json") != dry:
        raise ValueError("TRI60 M1 screen recovery dry-run evidence differs")
    validate_source_checkout(ROOT, expected_commit=recovery["source_commit"])
    events, jobs = [], {}
    for sequence, row in enumerate(plan["commands"]):
        command = _resolved(row, jobs)
        job = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
        event = build_submission_event(
            campaign_spec_sha256=recovery["content_hash"],
            task_id=row["task_id"], job_id=job, command=command,
            sequence=sequence,
        )
        events.append(event)
        jobs[row["task_id"]] = job
    write_immutable_json(args.output, assemble_submission_ledger(
        events, campaign_spec_sha256=recovery["content_hash"],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
