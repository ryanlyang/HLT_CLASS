#!/usr/bin/env python3
"""Dry-run or guarded-submit one exact PRAD Slurm DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.prad.campaign import (  # noqa: E402
    render_prad_submission_plan,
    submit_prad_plan,
    validate_prad_campaign_spec,
    validate_prad_resource_evidence,
    validate_prad_storage_evidence,
)
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke-submit", action="store_true")
    modes.add_argument("--full-production-submit", action="store_true")
    parser.add_argument("--dry-run-report", type=Path)
    parser.add_argument("--production-dry-run-report", type=Path)
    parser.add_argument("--resource-evidence", type=Path)
    parser.add_argument("--storage-evidence", type=Path)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_prad_campaign_spec(spec)
    validate_source_snapshot(
        spec["source_snapshot"], repository=args.repository, require_clean=True
    )
    plan = render_prad_submission_plan(
        campaign_spec_path=args.campaign_spec, spec=spec
    )
    if args.dry_run:
        report = with_content_hash(
            {
                "contract": "hlt_classification_prad_dry_run_v1",
                "schema_version": 1,
                "campaign_spec_sha256": spec["content_hash"],
                "mutated": False,
                "plan": plan,
            }
        )
        if args.dry_run_report is not None:
            write_immutable_json(args.dry_run_report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.smoke_submit and spec["mode"] != "smoke":
        raise PermissionError("--smoke-submit requires a smoke PRAD spec")
    if args.full_production_submit and (
        spec["mode"] != "production" or not spec["production_authorized"]
    ):
        raise PermissionError("full PRAD submission requires an authorized production spec")
    if args.full_production_submit:
        if any(
            value is None
            for value in (
                args.production_dry_run_report,
                args.resource_evidence,
                args.storage_evidence,
            )
        ):
            raise PermissionError(
                "full PRAD submission requires exact dry-run, resource, and storage evidence"
            )
        dry_run = load_json(args.production_dry_run_report)
        validate_content_hash(
            dry_run, expected_contract="hlt_classification_prad_dry_run_v1"
        )
        if (
            dry_run.get("contract") != "hlt_classification_prad_dry_run_v1"
            or dry_run.get("campaign_spec_sha256") != spec["content_hash"]
            or dry_run.get("mutated") is not False
            or dry_run.get("plan") != plan
        ):
            raise PermissionError("production PRAD dry-run evidence differs")
        resource = load_json(args.resource_evidence)
        storage = load_json(args.storage_evidence)
        resource_hash = validate_prad_resource_evidence(resource)
        storage_hash = validate_prad_storage_evidence(
            storage, resource_evidence=resource
        )
        if (
            resource_hash
            != spec["production_evidence"]["resource_evidence_sha256"]
            or storage_hash
            != spec["production_evidence"]["storage_evidence_sha256"]
        ):
            raise PermissionError("production PRAD evidence hashes differ")
    root = Path(spec["site"]["campaign_root"])
    for name in ("logs", "ledgers", "task_attestations"):
        (root / name).mkdir(parents=True, exist_ok=True)
    canonical_spec = root / "campaign_spec.json"
    write_immutable_json(canonical_spec, spec)
    journal = root / "ledgers" / "submission_jobs"
    existing_jobs = []
    for path in sorted(journal.glob("*.json")):
        payload = load_json(path)
        if payload.get("campaign_spec_sha256") != spec["content_hash"]:
            raise ValueError("partial PRAD submission journal campaign differs")
        existing_jobs.append(
            {
                "task": payload["task"],
                "job_id": payload["job_id"],
                "command": payload["command"],
            }
        )
    sequence = len(existing_jobs)

    def persist(job):
        nonlocal sequence
        write_immutable_json(
            journal / f"{sequence:04d}_{job['task']}.json",
            with_content_hash(
                {
                    "contract": "hlt_classification_prad_submission_job_v1",
                    "schema_version": 1,
                    "campaign_spec_sha256": spec["content_hash"],
                    "sequence": sequence,
                    **job,
                }
            ),
        )
        sequence += 1

    ledger = submit_prad_plan(
        campaign_spec_path=canonical_spec,
        spec=spec,
        on_submitted=persist,
        existing_jobs=existing_jobs,
    )
    write_immutable_json(root / "ledgers" / "submission.json", ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
