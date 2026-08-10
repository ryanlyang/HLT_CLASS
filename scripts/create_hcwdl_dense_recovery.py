#!/usr/bin/env python3
"""Create a source-pinned recovery for one failed HCWDL dense closure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_dense_recovery import (  # noqa: E402
    DENSE_RECOVERY_AUTHORIZATION_PHRASE, create_dense_recovery_spec,
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
        raise PermissionError("dense recovery must bind its creating checkout")
    if args.output.resolve() != (args.recovery_root / "recovery_spec.json").resolve():
        raise PermissionError("dense recovery spec path is not canonical")
    validate_source_checkout(REPO_ROOT, expected_commit=args.source_commit)
    phrase = args.authorization_phrase if args.authorize_live_submission else None
    if not args.authorize_live_submission and args.authorization_phrase is not None:
        raise ValueError("dense recovery phrase supplied without live authorization")
    spec = create_dense_recovery_spec(
        parent_campaign_spec=args.parent_campaign_spec,
        parent_submission_ledger=args.parent_submission_ledger,
        monitor_report=args.monitor_report, recovery_root=args.recovery_root,
        project_dir=args.project_dir, source_commit=args.source_commit,
        authorization_phrase=phrase,
    )
    write_immutable_json(args.output, spec)
    print(f"Parent: {spec['parent_campaign']}")
    print(f"Retry tasks: {', '.join(spec['retry_tasks'])}")
    print(f"Failed jobs: {', '.join(spec['failed_job_ids'])}")
    print(f"Creation phrase: {DENSE_RECOVERY_AUTHORIZATION_PHRASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

