#!/usr/bin/env python3
"""Dry-run or resumably submit one exact HCWDL-MHPE recovery closure."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, validate_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_contracts import COMMAND_PLAN_CONTRACT  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_recovery import validate_recovery  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_submission_event, build_submission_ledger, validate_submission_ledger,
)

PHRASE = "SUBMIT HCWDL MHPE RECOVERY EXACT CLOSURE"


def _resolved_command(row, jobs):
    command = [str(item) for item in row["command"]]
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
    spec = load_json(args.recovery_spec)
    validate_recovery(spec)
    root = Path(spec["recovery_root"])
    plan = load_json(root / "command_plan.json")
    validate_content_hash(
        plan, expected_contract=COMMAND_PLAN_CONTRACT, expected_schema_version=1,
    )
    commands = {row["task_id"]: list(row["command"]) for row in plan["commands"]}
    if [row["task_id"] for row in plan["commands"]] != list(spec["recovery_tasks"]):
        raise ValueError("HCWDL-MHPE recovery command order differs")
    if not args.execute:
        value = build_submission_ledger(
            campaign_spec_sha256=spec["content_hash"],
            jobs={task: "1" for task in commands}, commands=commands, dry_run=True,
        )
        if args.output.exists() and load_json(args.output) != value:
            raise FileExistsError("existing MHPE recovery dry-run ledger differs")
        if not args.output.exists():
            write_immutable_json(args.output, value)
        return 0
    if args.authorization_phrase != PHRASE:
        raise PermissionError("MHPE recovery submission phrase differs")
    dry = load_json(root / "dry_run_submission_ledger.json")
    validate_submission_ledger(dry)
    expected_dry = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: "1" for task in commands}, commands=commands, dry_run=True,
    )
    if dry != expected_dry:
        raise ValueError("HCWDL-MHPE recovery dry-run evidence differs")
    if ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("recovery submitter is outside bound worktree")
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    if args.output.exists():
        existing = load_json(args.output)
        validate_submission_ledger(existing)
        if (existing.get("dry_run") is not False
                or existing.get("campaign_spec_sha256") != spec["content_hash"]
                or set(existing.get("jobs", {})) != set(commands)):
            raise FileExistsError("existing MHPE recovery ledger differs")
        return 0
    journal = args.output.parent / f"{args.output.stem}_journal"
    events = []
    jobs = {}
    existing_events = sorted(journal.glob("*.json")) if journal.is_dir() else []
    for sequence, path in enumerate(existing_events):
        row = plan["commands"][sequence]
        event = load_json(path)
        command = _resolved_command(row, jobs)
        expected = build_submission_event(
            campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
            job_id=event.get("job_id", ""), command=command, sequence=sequence,
        )
        if event != expected:
            raise ValueError("MHPE recovery submission journal differs")
        events.append(event)
        jobs[row["task_id"]] = event["job_id"]
    for sequence, row in enumerate(
        plan["commands"][len(events):], start=len(events),
    ):
        command = _resolved_command(row, jobs)
        job_id = subprocess.run(
            command, check=True, capture_output=True, text=True,
        ).stdout.strip().split(";")[0]
        event = build_submission_event(
            campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
            job_id=job_id, command=command, sequence=sequence,
        )
        write_immutable_json(journal / f"{sequence:04d}_{row['task_id']}.json", event)
        events.append(event)
        jobs[row["task_id"]] = job_id
    original_ledger = load_json(spec["original_ledger_path"])
    final = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands={row["task_id"]: list(row["command"]) for row in events},
        dry_run=False, parent_ledger_sha256=spec["original_ledger_sha256"],
        monitor_report_sha256=spec["monitor_report_sha256"],
        superseded_jobs={
            task: original_ledger["jobs"][task]
            for task in jobs if task in original_ledger["jobs"]
        },
    )
    write_immutable_json(args.output, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
