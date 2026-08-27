#!/usr/bin/env python3
"""Dry-run or submit the exact TRI60 D000 optimization-budget screen DAG."""

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
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_campaign import SUBMISSION_PHRASE, validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_contracts import COMMAND_PLAN_CONTRACT, validate_artifact  # noqa: E402


def _resolved(row, jobs: dict[str, str]) -> list[str]:
    command = list(map(str, row["command"]))
    for index, item in enumerate(command):
        for parent in row["dependencies"]:
            if parent not in jobs:
                raise ValueError("TRI60 D000 budget-screen submission lacks dependency")
            item = item.replace(f"${{JOB_{parent}}}", jobs[parent])
        command[index] = item
    if any("${JOB_" in item for item in command):
        raise ValueError("TRI60 D000 budget-screen command retains placeholder")
    return command


def _dry_ledger(spec, plan) -> dict:
    commands = {
        row["task_id"]: list(map(str, row["command"]))
        for row in plan["commands"]
    }
    return build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task_id: "1" for task_id in commands},
        commands=commands, dry_run=True,
    )


def _load_journal(journal: Path, spec, plan):
    paths = sorted(journal.glob("*.json")) if journal.is_dir() else []
    if len(paths) > len(plan["commands"]):
        raise ValueError("TRI60 D000 budget-screen journal has excess events")
    events, jobs = [], {}
    for sequence, path in enumerate(paths):
        row = plan["commands"][sequence]
        command = _resolved(row, jobs)
        event = load_json(path)
        expected = build_submission_event(
            campaign_spec_sha256=spec["content_hash"],
            task_id=row["task_id"], job_id=event.get("job_id", ""),
            command=command, sequence=sequence,
        )
        if event != expected or path.name != f"{sequence:04d}_{row['task_id']}.json":
            raise ValueError("TRI60 D000 budget-screen journal differs")
        events.append(event)
        jobs[row["task_id"]] = event["job_id"]
    return events, jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.spec)
    validate_campaign(spec, executable=args.execute)
    canonical = Path(spec["campaign_root"]) / "campaign_spec.json"
    if args.spec.resolve() != canonical.resolve():
        raise PermissionError("TRI60 D000 budget screen requires canonical spec")
    plan = load_json(Path(spec["campaign_root"]) / "command_plan.json")
    validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT)
    if plan.get("spec_sha256") != spec["content_hash"]:
        raise ValueError("TRI60 D000 budget-screen plan belongs elsewhere")
    expected_dry = _dry_ledger(spec, plan)
    if not args.execute:
        if args.output.exists() and load_json(args.output) != expected_dry:
            raise FileExistsError("existing TRI60 D000 budget-screen dry ledger differs")
        if not args.output.exists():
            write_immutable_json(args.output, expected_dry)
        return 0
    if args.authorization_phrase != SUBMISSION_PHRASE:
        raise PermissionError("TRI60 D000 budget-screen submission phrase differs")
    dry_path = Path(spec["campaign_root"]) / "dry_run_submission_ledger.json"
    if not dry_path.is_file() or load_json(dry_path) != expected_dry:
        raise ValueError("TRI60 D000 budget-screen dry-run evidence differs")
    validate_submission_ledger(load_json(dry_path))
    if ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("TRI60 D000 budget-screen submitter is outside worktree")
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    if args.output.exists():
        ledger = load_json(args.output)
        validate_submission_ledger(ledger)
        expected_commands = {
            row["task_id"]: _resolved(row, ledger.get("jobs", {}))
            for row in plan["commands"]
        }
        expected_live = build_submission_ledger(
            campaign_spec_sha256=spec["content_hash"],
            jobs=ledger.get("jobs", {}), commands=expected_commands,
            dry_run=False,
        )
        if (
            set(ledger.get("jobs", {}))
            != {row["task_id"] for row in plan["commands"]}
            or ledger != expected_live
        ):
            raise FileExistsError("existing TRI60 D000 budget-screen ledger differs")
        return 0
    journal = args.output.parent / f"{args.output.stem}_journal"
    events, jobs = _load_journal(journal, spec, plan)
    for sequence, row in enumerate(plan["commands"][len(events):], start=len(events)):
        command = _resolved(row, jobs)
        job = subprocess.run(
            command, check=True, capture_output=True, text=True,
        ).stdout.strip().split(";")[0]
        event = build_submission_event(
            campaign_spec_sha256=spec["content_hash"], task_id=row["task_id"],
            job_id=job, command=command, sequence=sequence,
        )
        write_immutable_json(
            journal / f"{sequence:04d}_{row['task_id']}.json", event,
        )
        events.append(event)
        jobs[row["task_id"]] = job
    write_immutable_json(args.output, assemble_submission_ledger(
        events, campaign_spec_sha256=spec["content_hash"],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
