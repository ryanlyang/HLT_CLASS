#!/usr/bin/env python3
"""Create or execute an explicit exact-ID recovery submission plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import (  # noqa: E402
    build_submission_job_record,
    build_resume_plan,
    submit_plan,
    validate_monitor_report,
    validate_submission_ledger,
)
from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    write_immutable_json,
)
from hlt_classification.provenance import validate_campaign_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--cancel-stale-exact", action="store_true")
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    # This remains mandatory even if every task is reusable.
    validate_campaign_source(spec, repository=args.repository)
    ledger = load_json(args.submission_ledger)
    validate_submission_ledger(ledger, campaign_spec=spec)
    monitor = load_json(args.monitor_report)
    validate_monitor_report(
        monitor,
        campaign_spec=spec,
        submission_ledger=ledger,
    )
    plan = build_resume_plan(
        campaign_spec=spec,
        monitor_report=monitor,
        submission_ledger=ledger,
    )
    write_immutable_json(args.output, plan)
    if not args.submit:
        print(json.dumps({"mutated": False, "resume_plan": plan}, indent=2))
        return 0
    stale = plan["cancel_exact_job_ids"]
    if stale and not args.cancel_stale_exact:
        raise PermissionError(
            "stale exact campaign job ids require --cancel-stale-exact"
        )
    if stale:
        subprocess.run(["scancel", *stale], check=True)
    if not plan["rerun_tasks"]:
        print(json.dumps({"submitted": False, "reason": "all_tasks_reusable"}))
        return 0
    root = Path(spec["site"]["campaign_root"])
    output = root / "ledgers" / f"resume_{monitor['content_hash']}.json"
    if output.is_file():
        resumed = load_json(output)
        validate_submission_ledger(resumed, campaign_spec=spec)
        print(
            json.dumps(
                {
                    "submitted": False,
                    "reused": True,
                    "ledger_path": str(output),
                    **resumed,
                }
            )
        )
        return 0
    journal_root = (
        root
        / "ledgers"
        / f"resume_{monitor['content_hash']}_jobs"
    )
    if any(journal_root.glob("*.json")):
        raise RuntimeError(
            "partial resume submission journal exists; assemble and recover "
            "it instead of duplicating submitted jobs"
        )
    sequence = 0

    def persist_job(job):
        nonlocal sequence
        record = build_submission_job_record(
            campaign_spec=spec,
            sequence=sequence,
            job=job,
            submission_kind="resume",
        )
        write_immutable_json(
            journal_root / f"{sequence:04d}_{job['task']}.json",
            record,
        )
        sequence += 1

    resumed = submit_plan(
        campaign_spec_path=(
            Path(spec["site"]["campaign_root"]) / "campaign_spec.json"
        ),
        campaign_spec=spec,
        task_names=plan["rerun_tasks"],
        on_submitted=persist_job,
    )
    write_immutable_json(output, resumed)
    print(json.dumps({"submitted": True, "ledger_path": str(output), **resumed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
