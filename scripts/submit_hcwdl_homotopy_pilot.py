#!/usr/bin/env python3
"""Render or submit the exact HCWDL-UJ validation-only DAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_campaign import (  # noqa: E402
    SUBMISSION_PHRASE, build_command_plan, validate_campaign,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    assemble_submission_ledger, build_submission_event, build_submission_ledger,
    validate_submission_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.campaign_spec); validate_campaign(spec, executable=args.execute)
    if args.campaign_spec.resolve() != (Path(spec["campaign_root"]) / "campaign_spec.json").resolve():
        raise PermissionError("HCWDL-UJ submitter requires the canonical campaign spec")
    plan = build_command_plan(spec)
    commands = {row["task_id"]: list(row["command"]) for row in plan["commands"]}
    if not args.execute:
        write_immutable_json(args.output, build_submission_ledger(
            campaign_spec_sha256=spec["content_hash"], jobs={task: "1" for task in commands},
            commands=commands, dry_run=True,
        )); return 0
    if args.authorization_phrase != SUBMISSION_PHRASE:
        raise PermissionError("HCWDL-UJ submission phrase differs")
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("HCWDL-UJ submitter is not running from its bound worktree")
    validate_source_checkout(REPO_ROOT, expected_commit=spec["source_commit"])
    if args.output.exists():
        completed = load_json(args.output); validate_submission_ledger(completed)
        if (
            completed.get("campaign_spec_sha256") != spec["content_hash"]
            or set(completed.get("jobs", {})) != set(commands)
        ):
            raise FileExistsError("existing HCWDL-UJ ledger is not the complete exact campaign")
        return 0
    jobs = {}; journal = args.output.parent / f"{args.output.stem}_journal"
    event_paths = sorted(journal.glob("*.json")) if journal.is_dir() else []
    events = [load_json(path) for path in event_paths]
    if events:
        prior = assemble_submission_ledger(
            events, campaign_spec_sha256=spec["content_hash"],
        )
        jobs.update({str(task): str(job) for task, job in prior["jobs"].items()})
    try:
        for sequence, row in enumerate(plan["commands"]):
            command = [str(item) for item in row["command"]]
            for i, item in enumerate(command):
                for parent in row["dependencies"]:
                    item = item.replace(f"${{JOB_{parent}}}", jobs[parent])
                command[i] = item
            if sequence < len(events):
                event = events[sequence]
                if (
                    event.get("task_id") != row["task_id"]
                    or event.get("command") != command
                ):
                    raise ValueError("HCWDL-UJ submission journal differs from command plan")
                jobs[row["task_id"]] = str(event["job_id"])
                continue
            raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
            job = raw.split(";")[0]
            event = build_submission_event(
                campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
                job_id=job, command=command, sequence=sequence,
            )
            write_immutable_json(journal / f"{sequence:04d}_{row['task_id']}.json", event)
            events.append(event); jobs[row["task_id"]] = job
    except BaseException:
        if events:
            partial = args.output.parent / f"{args.output.stem}_partial_{len(events):04d}.json"
            if not partial.exists():
                write_immutable_json(
                    partial,
                    assemble_submission_ledger(
                        events, campaign_spec_sha256=spec["content_hash"],
                    ),
                )
        raise
    write_immutable_json(args.output, assemble_submission_ledger(events, campaign_spec_sha256=spec["content_hash"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
