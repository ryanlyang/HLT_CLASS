#!/usr/bin/env python3
"""Dry-run or submit the exact supplemental dense cold 300k DAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_dense import (  # noqa: E402
    build_dense_command_plan, dense_profile_for_spec, validate_dense_spec,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    assemble_submission_ledger, build_submission_event, build_submission_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_dense_spec(spec, executable=args.execute)
    profile = dense_profile_for_spec(spec)
    if args.campaign_spec.resolve() != (
        Path(spec["campaign_root"]) / "campaign_spec.json"
    ).resolve():
        raise PermissionError("dense cold submitter requires canonical campaign spec path")
    plan = build_dense_command_plan(spec)
    commands = {row["task_id"]: list(row["command"]) for row in plan["commands"]}
    if not args.execute:
        ledger = build_submission_ledger(
            campaign_spec_sha256=spec["content_hash"],
            jobs={task: "1" for task in commands}, commands=commands, dry_run=True,
        )
        write_immutable_json(args.output, ledger)
        return 0
    if args.authorization_phrase != profile.submission_phrase:
        raise PermissionError("dense cold submission phrase differs")
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("dense cold submitter is not running from bound worktree")
    validate_source_checkout(REPO_ROOT, expected_commit=str(spec["source_commit"]))
    jobs: dict[str, str] = {}
    events = []
    journal = args.output.parent / f"{args.output.stem}_journal"
    try:
        for sequence, row in enumerate(plan["commands"]):
            command = list(row["command"])
            for index, argument in enumerate(command):
                for parent in row["dependencies"]:
                    argument = argument.replace(f"${{JOB_{parent}}}", jobs[parent])
                command[index] = argument
            output = subprocess.run(
                command, check=True, capture_output=True, text=True,
            ).stdout.strip()
            job_id = output.split(";")[0]
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
                assemble_submission_ledger(
                    events, campaign_spec_sha256=spec["content_hash"],
                ),
            )
        raise
    write_immutable_json(
        args.output,
        assemble_submission_ledger(
            events, campaign_spec_sha256=spec["content_hash"],
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
