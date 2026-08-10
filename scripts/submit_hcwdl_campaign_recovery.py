#!/usr/bin/env python3
"""Dry-run or submit an exact primary HCWDL failed-closure recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_campaign_recovery import (  # noqa: E402
    CAMPAIGN_RECOVERY_SUBMISSION_PHRASE, build_campaign_recovery_plan,
    validate_campaign_recovery_inputs, validate_campaign_recovery_spec,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_submission_event, build_submission_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.recovery_spec)
    validate_campaign_recovery_spec(spec, executable=args.execute)
    validate_campaign_recovery_inputs(spec)
    if args.recovery_spec.resolve() != (
        Path(spec["recovery_root"]) / "recovery_spec.json"
    ).resolve():
        raise PermissionError("campaign recovery submit path is not canonical")
    plan = build_campaign_recovery_plan(spec)
    commands = {row["task_id"]: list(row["command"]) for row in plan["commands"]}
    if not args.execute:
        ledger = build_submission_ledger(
            campaign_spec_sha256=spec["content_hash"],
            jobs={task: "1" for task in commands}, commands=commands,
            dry_run=True,
            parent_ledger_sha256=spec["parent_submission_ledger"]["content_hash"],
            monitor_report_sha256=spec["failure_monitor"]["content_hash"],
            superseded_jobs=spec["superseded_jobs"],
        )
        write_immutable_json(args.output, ledger)
        return 0
    if args.authorization_phrase != CAMPAIGN_RECOVERY_SUBMISSION_PHRASE:
        raise PermissionError("campaign recovery submission phrase differs")
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("campaign recovery submitter is in another checkout")
    validate_source_checkout(REPO_ROOT, expected_commit=spec["source_commit"])
    jobs: dict[str, str] = {}
    emitted: dict[str, list[str]] = {}
    events = []
    journal = args.output.parent / f"{args.output.stem}_journal"
    try:
        for sequence, row in enumerate(plan["commands"]):
            command = list(row["command"])
            for index, argument in enumerate(command):
                for parent in row["dependencies"]:
                    argument = argument.replace(
                        f"${{JOB_{parent}}}", jobs[parent],
                    )
                command[index] = argument
            output = subprocess.run(
                command, check=True, capture_output=True, text=True,
            ).stdout.strip()
            job_id = output.split(";")[0]
            emitted[row["task_id"]] = command
            event = build_submission_event(
                campaign_spec_sha256=spec["content_hash"],
                task_id=row["task_id"], job_id=job_id,
                command=command, sequence=sequence,
            )
            write_immutable_json(
                journal / f"{sequence:04d}_{row['task_id']}.json", event,
            )
            events.append(event)
            jobs[row["task_id"]] = job_id
    except BaseException:
        if events and not args.output.exists():
            write_immutable_json(
                args.output,
                build_submission_ledger(
                    campaign_spec_sha256=spec["content_hash"],
                    jobs=jobs, commands=emitted, dry_run=False,
                    parent_ledger_sha256=spec[
                        "parent_submission_ledger"
                    ]["content_hash"],
                    monitor_report_sha256=spec["failure_monitor"]["content_hash"],
                    superseded_jobs={
                        task: spec["superseded_jobs"][task] for task in jobs
                    },
                ),
            )
        raise
    write_immutable_json(
        args.output,
        build_submission_ledger(
            campaign_spec_sha256=spec["content_hash"],
            jobs=jobs, commands=emitted, dry_run=False,
            parent_ledger_sha256=spec["parent_submission_ledger"]["content_hash"],
            monitor_report_sha256=spec["failure_monitor"]["content_hash"],
            superseded_jobs=spec["superseded_jobs"],
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
