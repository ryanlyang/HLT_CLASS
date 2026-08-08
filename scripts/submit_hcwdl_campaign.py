#!/usr/bin/env python3
"""Submit the pre-acknowledgement phase of an authorized HCWDL DAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_campaign import (  # noqa: E402
    split_submission_commands, validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    AUTOMATIC_ENDPOINT_CONTINUATION, require_canonical_campaign_spec_path,
    validate_source_checkout,
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
    commands, _ = split_submission_commands(spec)
    by_task = {row["task_id"]: list(row["command"]) for row in commands}
    if not args.execute:
        ledger = build_submission_ledger(
            campaign_spec_sha256=spec["content_hash"],
            jobs={task: "1" for task in by_task}, commands=by_task, dry_run=True,
        )
        write_immutable_json(args.output, ledger)
        return 0
    validate_campaign_spec(spec, executable=True)
    require_canonical_campaign_spec_path(
        args.campaign_spec, campaign_root=spec["campaign_root"],
    )
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("HCWDL submitter is not running from the bound project worktree")
    expected_phrase = (
        "SUBMIT HCWDL EXACT SPEC WITH PREAUTHORIZED ENDPOINT CONTINUATION"
        if spec.get("endpoint_continuation") == AUTOMATIC_ENDPOINT_CONTINUATION
        else "SUBMIT HCWDL EXACT SPEC"
    )
    if args.authorization_phrase != expected_phrase:
        raise PermissionError("live HCWDL submission requires the exact authorization phrase")
    validate_source_checkout(REPO_ROOT, expected_commit=str(spec["source_commit"]))
    jobs: dict[str, str] = {}
    submitted_commands: dict[str, list[str]] = {}
    events = []
    journal = args.output.parent / f"{args.output.stem}_journal"
    try:
        for sequence, row in enumerate(commands):
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
                campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
                job_id=job_id, command=command, sequence=sequence,
            )
            write_immutable_json(journal / f"{sequence:04d}_{row['task_id']}.json", event)
            events.append(event); jobs[row["task_id"]] = job_id
            submitted_commands[row["task_id"]] = command
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
        assemble_submission_ledger(events, campaign_spec_sha256=spec["content_hash"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
