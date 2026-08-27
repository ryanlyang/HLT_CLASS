#!/usr/bin/env python3
"""Dry-run or submit the standalone TRI60 D000 long180 ledger."""

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
    build_submission_event, build_submission_ledger,
    validate_submission_ledger,
)
from hlt_classification.scouting.hcwdl_tri60_d000_long180 import (  # noqa: E402
    SUBMISSION_PHRASE, validate_campaign,
)
from hlt_classification.scouting.hcwdl_tri60_d000_long180_contracts import (  # noqa: E402
    COMMAND_PLAN_CONTRACT, validate_artifact,
)


TASK_ID = "train_D000_from_D033E_180"


def _expected(spec, plan, *, job_id: str, dry_run: bool):
    command = list(map(str, plan["commands"][0]["command"]))
    return build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs={TASK_ID: job_id},
        commands={TASK_ID: command}, dry_run=dry_run,
    )


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
        raise PermissionError("D000 long180 submitter requires canonical spec")
    plan = load_json(Path(spec["campaign_root"]) / "command_plan.json")
    validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT)
    if plan["spec_sha256"] != spec["content_hash"] or len(plan["commands"]) != 1:
        raise ValueError("D000 long180 command plan differs")

    if not args.execute:
        expected = _expected(spec, plan, job_id="1", dry_run=True)
        if args.output.exists() and load_json(args.output) != expected:
            raise FileExistsError("existing D000 long180 dry ledger differs")
        write_immutable_json(args.output, expected)
        return 0

    if args.authorization_phrase != SUBMISSION_PHRASE:
        raise PermissionError("D000 long180 submission phrase differs")
    dry_path = Path(spec["campaign_root"]) / "dry_run_submission_ledger.json"
    dry = load_json(dry_path)
    if dry != _expected(spec, plan, job_id="1", dry_run=True):
        raise ValueError("D000 long180 canonical dry-run evidence differs")
    validate_submission_ledger(dry)
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    if args.output.exists():
        existing = load_json(args.output)
        validate_submission_ledger(existing)
        jobs = existing.get("jobs", {})
        if (
            set(jobs) != {TASK_ID}
            or existing != _expected(
                spec, plan, job_id=str(jobs.get(TASK_ID, "")), dry_run=False,
            )
        ):
            raise FileExistsError("existing D000 long180 live ledger differs")
        return 0
    command = list(map(str, plan["commands"][0]["command"]))
    journal_path = (
        args.output.parent / f"{args.output.stem}_journal"
        / f"0000_{TASK_ID}.json"
    )
    if journal_path.is_file():
        event = load_json(journal_path)
        job_id = str(event.get("job_id", ""))
        expected_event = build_submission_event(
            campaign_spec_sha256=spec["content_hash"], task_id=TASK_ID,
            job_id=job_id, command=command, sequence=0,
        )
        if event != expected_event:
            raise ValueError("D000 long180 submission journal differs")
    else:
        job_id = subprocess.run(
            command, check=True, capture_output=True, text=True,
        ).stdout.strip().split(";")[0]
        write_immutable_json(journal_path, build_submission_event(
            campaign_spec_sha256=spec["content_hash"], task_id=TASK_ID,
            job_id=job_id, command=command, sequence=0,
        ))
    write_immutable_json(
        args.output, _expected(spec, plan, job_id=job_id, dry_run=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
