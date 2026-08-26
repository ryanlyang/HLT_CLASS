#!/usr/bin/env python3
"""Dry-run or submit an exact TRI60 CE5 recovery DAG."""

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
    assemble_submission_ledger, build_submission_event,
    build_submission_ledger, validate_submission_ledger,
)
from hlt_classification.scouting.hcwdl_tri60_ce5_contracts import (  # noqa: E402
    COMMAND_PLAN_CONTRACT, validate_artifact,
)
from hlt_classification.scouting.hcwdl_tri60_ce5_recovery import (  # noqa: E402
    RECOVERY_SUBMISSION_PHRASE, validate_recovery,
)


def _resolved(row, jobs):
    command = list(map(str, row["command"]))
    for index, item in enumerate(command):
        for parent in row["dependencies"]:
            if parent not in jobs:
                raise ValueError("TRI60 CE5 recovery lacks a dependency")
            item = item.replace(f"${{JOB_{parent}}}", jobs[parent])
        command[index] = item
    if any("${JOB_" in item for item in command):
        raise ValueError("TRI60 CE5 recovery retains a dependency placeholder")
    return command


def _ledger(spec, plan, *, jobs, commands, dry_run):
    return build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=dry_run,
        parent_ledger_sha256=spec["subject_ledger_sha256"],
        monitor_report_sha256=spec["monitor_report_sha256"],
        superseded_jobs={
            task: load_json(spec["subject_ledger_path"])["jobs"][task]
            for task in jobs
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.spec)
    validate_recovery(spec)
    if args.spec.resolve() != (Path(spec["recovery_root"]) / "recovery_spec.json").resolve():
        raise PermissionError("TRI60 CE5 recovery submitter requires canonical spec")
    plan = load_json(Path(spec["recovery_root"]) / "command_plan.json")
    validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT)
    commands = {row["task_id"]: list(map(str, row["command"])) for row in plan["commands"]}
    dry = _ledger(
        spec, plan, jobs={task: "1" for task in commands},
        commands=commands, dry_run=True,
    )
    if not args.execute:
        if args.output.exists() and load_json(args.output) != dry:
            raise FileExistsError("existing TRI60 CE5 recovery dry run differs")
        if not args.output.exists():
            write_immutable_json(args.output, dry)
        return 0
    if args.authorization_phrase != RECOVERY_SUBMISSION_PHRASE:
        raise PermissionError("TRI60 CE5 recovery submission phrase differs")
    canonical_dry = Path(spec["recovery_root"]) / "dry_run_submission_ledger.json"
    if not canonical_dry.is_file() or load_json(canonical_dry) != dry:
        raise ValueError("TRI60 CE5 recovery canonical dry run differs")
    if ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("TRI60 CE5 recovery submitter is outside bound worktree")
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    if args.output.exists():
        existing = load_json(args.output)
        validate_submission_ledger(existing)
        expected_commands = {
            row["task_id"]: _resolved(row, existing.get("jobs", {}))
            for row in plan["commands"]
        }
        expected = _ledger(
            spec, plan, jobs=existing.get("jobs", {}),
            commands=expected_commands, dry_run=False,
        )
        if existing != expected:
            raise FileExistsError("existing TRI60 CE5 recovery live ledger differs")
        return 0
    journal = args.output.parent / f"{args.output.stem}_journal"
    events = []
    jobs = {}
    paths = sorted(journal.glob("*.json")) if journal.is_dir() else []
    for sequence, path in enumerate(paths):
        row = plan["commands"][sequence]
        command = _resolved(row, jobs)
        event = load_json(path)
        expected = build_submission_event(
            campaign_spec_sha256=spec["content_hash"],
            task_id=row["task_id"], job_id=event.get("job_id", ""),
            command=command, sequence=sequence,
        )
        if event != expected:
            raise ValueError("TRI60 CE5 recovery journal differs")
        events.append(event)
        jobs[row["task_id"]] = event["job_id"]
    for sequence, row in enumerate(plan["commands"][len(events):], start=len(events)):
        command = _resolved(row, jobs)
        job = subprocess.run(
            command, check=True, capture_output=True, text=True,
        ).stdout.strip().split(";")[0]
        event = build_submission_event(
            campaign_spec_sha256=spec["content_hash"],
            task_id=row["task_id"], job_id=job, command=command,
            sequence=sequence,
        )
        write_immutable_json(
            journal / f"{sequence:04d}_{row['task_id']}.json", event,
        )
        events.append(event)
        jobs[row["task_id"]] = job
    base = assemble_submission_ledger(
        events, campaign_spec_sha256=spec["content_hash"],
    )
    write_immutable_json(
        args.output,
        _ledger(
            spec, plan, jobs=base["jobs"], commands=base["commands"],
            dry_run=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
