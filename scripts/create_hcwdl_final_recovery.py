#!/usr/bin/env python3
"""Create an exact recovery spec for an interrupted sealed HCWDL evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_final_recovery import (  # noqa: E402
    FINAL_RECOVERY_AUTHORIZATION_PHRASE, create_final_recovery_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign-spec", type=Path, required=True)
    parser.add_argument("--parent-submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if REPO_ROOT.resolve() != args.project_dir.resolve():
        raise PermissionError("HCWDL final recovery must bind its creating checkout")
    if args.output.resolve() != (args.recovery_root / "recovery_spec.json").resolve():
        raise PermissionError("HCWDL final recovery spec path is not canonical")
    validate_source_checkout(REPO_ROOT, expected_commit=args.source_commit)
    phrase = args.authorization_phrase if args.authorize_live_submission else None
    if not args.authorize_live_submission and args.authorization_phrase is not None:
        raise ValueError("recovery phrase supplied without live authorization")
    spec = create_final_recovery_spec(
        parent_campaign_spec=args.parent_campaign_spec,
        parent_submission_ledger=args.parent_submission_ledger,
        monitor_report=args.monitor_report, recovery_root=args.recovery_root,
        project_dir=args.project_dir, source_commit=args.source_commit,
        authorization_phrase=phrase,
    )
    write_immutable_json(args.output, spec)
    print(f"Failed job: {spec['failed_job_id']} ({spec['failed_state']})")
    print(f"Frozen finalists: {spec['frozen_finalist_count']}")
    print(f"Existing claim: {spec['execution_claim_sha256']}")
    print(f"Creation phrase: {FINAL_RECOVERY_AUTHORIZATION_PHRASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
