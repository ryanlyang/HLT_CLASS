#!/usr/bin/env python3
"""Dry-run or resubmit exactly the failed HCWDL task closure."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_campaign import slurm_commands, validate_campaign_spec  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    require_canonical_campaign_spec_path, validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_submission_event, build_submission_ledger, resume_tasks,
    validate_submission_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.campaign_spec); ledger = load_json(args.submission_ledger)
    ledger_hash = validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"]:
        raise ValueError("HCWDL recovery ledger belongs to another campaign")
    monitor = load_json(args.monitor_report)
    if monitor.get("submission_ledger_sha256") != ledger_hash:
        raise ValueError("HCWDL recovery monitor belongs to another submission ledger")
    commands = slurm_commands(spec)
    graph = {row["task_id"]: tuple(row["dependencies"]) for row in commands}
    retry = set(resume_tasks(monitor, dependency_graph=graph))
    selected = [row for row in commands if row["task_id"] in retry]
    if not selected:
        raise ValueError("HCWDL recovery has no failed or missing task to submit")
    if args.execute:
        validate_campaign_spec(spec, executable=True)
        require_canonical_campaign_spec_path(
            args.campaign_spec, campaign_root=spec["campaign_root"],
        )
        validate_source_checkout(REPO_ROOT, expected_commit=str(spec["source_commit"]))
        if args.authorization_phrase != "RESUME HCWDL EXACT TASKS":
            raise PermissionError("HCWDL resume requires the exact authorization phrase")
    jobs = dict(ledger["jobs"])
    emitted: dict[str, list[str]] = {}
    new_jobs: dict[str, str] = {}
    monitor_hash = str(monitor["content_hash"])
    superseded = {task: jobs[task] for task in retry if task in jobs}
    journal = args.output.parent / f"{args.output.stem}_journal"
    events = []
    try:
        for index, row in enumerate(selected):
            command = list(row["command"])
            for position, argument in enumerate(command):
                for parent in row["dependencies"]:
                    parent_job = new_jobs[parent] if parent in new_jobs else jobs[parent]
                    argument = argument.replace(f"${{JOB_{parent}}}", parent_job)
                command[position] = argument
            emitted[row["task_id"]] = command
            if args.execute:
                output = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
                new_jobs[row["task_id"]] = output.split(";")[0]
                event = build_submission_event(
                    campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
                    job_id=new_jobs[row["task_id"]], command=command, sequence=index,
                )
                write_immutable_json(journal / f"{index:04d}_{row['task_id']}.json", event)
                events.append(event)
            else:
                new_jobs[row["task_id"]] = str(index + 1)
    except BaseException:
        if new_jobs and not args.output.exists():
            write_immutable_json(args.output, build_submission_ledger(
                campaign_spec_sha256=spec["content_hash"], jobs=new_jobs,
                commands={task: emitted[task] for task in new_jobs}, dry_run=False,
                parent_ledger_sha256=ledger_hash,
                monitor_report_sha256=monitor_hash,
                superseded_jobs={task: superseded[task] for task in new_jobs if task in superseded},
            ))
        raise
    result = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=new_jobs,
        commands=emitted, dry_run=not args.execute,
        parent_ledger_sha256=ledger_hash,
        monitor_report_sha256=monitor_hash,
        superseded_jobs=superseded,
    )
    write_immutable_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
