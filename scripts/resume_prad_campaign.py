#!/usr/bin/env python3
"""Resubmit failed PRAD nodes and descendants while reusing authenticated jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.prad.campaign import submit_prad_plan, validate_prad_campaign_spec, validate_prad_monitor_report, validate_prad_submission_ledger  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--prior-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--attempt", type=int, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.attempt <= 0:
        raise ValueError("PRAD resume attempt must be positive")
    spec = load_json(args.campaign_spec)
    validate_prad_campaign_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=args.repository, require_clean=True)
    prior = load_json(args.prior_ledger)
    validate_prad_submission_ledger(prior, spec=spec)
    monitor = load_json(args.monitor_report)
    validate_prad_monitor_report(monitor, spec=spec, ledger=prior)
    if (
        monitor.get("campaign_spec_sha256") != spec["content_hash"]
        or monitor.get("submission_ledger_sha256") != prior["content_hash"]
    ):
        raise ValueError("PRAD resume monitor lineage differs")
    reusable_names = {row["task"] for row in monitor["jobs"] if row.get("reusable") is True}
    existing_jobs = [row for row in prior["jobs"] if row["task"] in reusable_names]
    task_lookup = {row["name"]: row for row in spec["tasks"]}
    for row in existing_jobs:
        if any(dependency not in reusable_names for dependency in task_lookup[row["task"]]["dependencies"]):
            raise ValueError("PRAD monitor marks a task reusable without its dependencies")
    if args.dry_run:
        print(json.dumps({"campaign_spec_sha256": spec["content_hash"], "attempt": args.attempt, "reused_tasks": sorted(reusable_names), "resubmitted_tasks": [row["name"] for row in spec["tasks"] if row["name"] not in reusable_names], "mutated": False}, indent=2, sort_keys=True))
        return 0
    root = Path(spec["site"]["campaign_root"])
    journal = root / "ledgers" / f"resume_{args.attempt}_jobs"
    sequence = 0

    def persist(job):
        nonlocal sequence
        write_immutable_json(
            journal / f"{sequence:04d}_{job['task']}.json",
            with_content_hash({"contract": "hlt_classification_prad_resume_job_v1", "schema_version": 1, "campaign_spec_sha256": spec["content_hash"], "attempt": args.attempt, "sequence": sequence, **job}),
        )
        sequence += 1

    ledger = submit_prad_plan(
        campaign_spec_path=args.campaign_spec,
        spec=spec,
        existing_jobs=existing_jobs,
        on_submitted=persist,
    )
    output = root / "ledgers" / f"resume_{args.attempt}_submission.json"
    write_immutable_json(output, ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
