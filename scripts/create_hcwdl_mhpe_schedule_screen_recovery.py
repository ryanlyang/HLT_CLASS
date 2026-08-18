#!/usr/bin/env python3
"""Create an exact failed/downstream D066 schedule-screen recovery."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.scouting.hcwdl_mhpe_schedule_screen_recovery import create_recovery  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorization-phrase", required=True)
    args = parser.parse_args()
    recovery = create_recovery(
        campaign_spec=args.campaign_spec, submission_ledger=args.submission_ledger,
        monitor_report=args.monitor_report, recovery_root=args.recovery_root,
        project_dir=args.project_dir, source_commit=args.source_commit,
        authorization_phrase=args.authorization_phrase,
    )
    print(recovery["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
