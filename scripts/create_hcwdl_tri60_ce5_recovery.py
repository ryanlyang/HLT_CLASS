#!/usr/bin/env python3
"""Create an exact failed/downstream TRI60 CE5 recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.scouting.hcwdl_tri60_ce5_recovery import create_recovery  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--source-repair-phrase")
    args = parser.parse_args()
    value = create_recovery(
        campaign_spec=args.campaign_spec,
        submission_ledger=args.submission_ledger,
        monitor_report=args.monitor_report, recovery_root=args.recovery_root,
        project_dir=args.project_dir, source_commit=args.source_commit,
        changed_files=args.changed_file,
        source_repair_phrase=args.source_repair_phrase,
    )
    print(value["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
