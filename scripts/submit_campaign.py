#!/usr/bin/env python3
"""Dry-run, simulate, or submit one exact baseline campaign DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import (  # noqa: E402
    build_submission_job_record,
    render_submission_plan,
    simulate_failure,
    submit_plan,
    validate_campaign_spec,
    validate_storage_measurement,
    validate_submission_ledger,
)
from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    write_immutable_json,
)
from hlt_classification.provenance import validate_campaign_source  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-spec", type=Path, required=True)
    result.add_argument("--repository", type=Path, default=REPO_ROOT)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke-simulate", action="store_true")
    modes.add_argument("--smoke-submit", action="store_true")
    modes.add_argument("--full-production-submit", action="store_true")
    result.add_argument("--storage-measurement", type=Path)
    result.add_argument("--resource-evidence", type=Path)
    result.add_argument("--simulate-failure-task")
    return result


def main() -> int:
    args = parser().parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign_spec(spec)
    # This is intentionally before every dry-run, reuse, or submission branch.
    validate_campaign_source(spec, repository=args.repository)
    canonical_spec_path = (
        Path(spec["site"]["campaign_root"]) / "campaign_spec.json"
    )
    measured_resources = None
    if args.storage_measurement is not None:
        if args.resource_evidence is None:
            raise PermissionError(
                "a storage measurement with resource lineage requires "
                "--resource-evidence"
            )
        storage = load_json(args.storage_measurement)
        validate_storage_measurement(
            storage,
            campaign_spec=spec,
            resource_evidence=load_json(args.resource_evidence),
        )
        measured_resources = storage["resources"]
    plan = render_submission_plan(
        campaign_spec_path=canonical_spec_path,
        campaign_spec=spec,
        measured_resources=measured_resources,
    )
    if args.dry_run:
        print(json.dumps({"mutated": False, "plan": plan}, indent=2, sort_keys=True))
        return 0
    if args.smoke_simulate:
        if spec["mode"] != "smoke":
            raise PermissionError("smoke simulation requires a smoke specification")
        counter = iter(range(90001, 90001 + len(plan)))
        ledger = submit_plan(
            campaign_spec_path=args.campaign_spec,
            campaign_spec=spec,
            executor=lambda command: str(next(counter)),
        )
        monitor, resume = simulate_failure(
            campaign_spec=spec,
            submission_ledger=ledger,
            failed_task=args.simulate_failure_task,
        )
        print(
            json.dumps(
                {
                    "mutated": False,
                    "ledger": ledger,
                    "monitor": monitor,
                    "resume": resume,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.smoke_submit:
        if spec["mode"] != "smoke":
            raise PermissionError("smoke submission requires a smoke specification")
    else:
        if spec["mode"] != "production" or not spec["production_authorized"]:
            raise PermissionError(
                "full production requires an explicitly authorized production spec"
            )
        if args.storage_measurement is None or args.resource_evidence is None:
            raise PermissionError(
                "full production requires authenticated storage and smoke "
                "resource evidence"
            )
    root = Path(spec["site"]["campaign_root"])
    for directory in ("logs", "ledgers", "task_attestations"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    write_immutable_json(canonical_spec_path, spec)
    ledger_path = root / "ledgers" / "submission.json"
    if ledger_path.is_file():
        ledger = load_json(ledger_path)
        validate_submission_ledger(ledger, campaign_spec=spec)
        print(
            json.dumps(
                {
                    "reused": True,
                    "ledger_path": str(ledger_path),
                    **ledger,
                },
                indent=2,
            )
        )
        return 0
    journal_root = root / "ledgers" / "submission_initial_jobs"
    existing_journal = sorted(journal_root.glob("*.json"))
    if existing_journal:
        raise RuntimeError(
            "partial initial submission journal exists; assemble and recover "
            "it instead of resubmitting the graph"
        )
    sequence = 0

    def persist_job(job):
        nonlocal sequence
        record = build_submission_job_record(
            campaign_spec=spec,
            sequence=sequence,
            job=job,
            submission_kind="initial",
        )
        write_immutable_json(
            journal_root / f"{sequence:04d}_{job['task']}.json",
            record,
        )
        sequence += 1

    ledger = submit_plan(
        campaign_spec_path=canonical_spec_path,
        campaign_spec=spec,
        measured_resources=measured_resources,
        on_submitted=persist_job,
    )
    write_immutable_json(ledger_path, ledger)
    print(json.dumps({"ledger_path": str(ledger_path), **ledger}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
