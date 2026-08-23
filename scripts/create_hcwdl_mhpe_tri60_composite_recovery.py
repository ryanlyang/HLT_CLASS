#!/usr/bin/env python3
"""Create an authenticated TRI60 recovery across split execution ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.scouting.hcwdl_mhpe_tri60_composite_recovery import (  # noqa: E402
    create_composite_recovery,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logit-subject-spec", type=Path, required=True)
    parser.add_argument("--logit-subject-ledger", type=Path, required=True)
    parser.add_argument("--logit-monitor-report", type=Path, required=True)
    parser.add_argument("--representation-subject-spec", type=Path, required=True)
    parser.add_argument("--representation-subject-ledger", type=Path, required=True)
    parser.add_argument("--representation-monitor-report", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--source-repair-phrase", required=True)
    parser.add_argument("--cpus", type=int, default=72)
    args = parser.parse_args()
    value = create_composite_recovery(
        logit_subject_spec=args.logit_subject_spec,
        logit_subject_ledger=args.logit_subject_ledger,
        logit_monitor_report=args.logit_monitor_report,
        representation_subject_spec=args.representation_subject_spec,
        representation_subject_ledger=args.representation_subject_ledger,
        representation_monitor_report=args.representation_monitor_report,
        recovery_root=args.recovery_root, project_dir=args.project_dir,
        source_commit=args.source_commit, changed_files=args.changed_file,
        source_repair_phrase=args.source_repair_phrase, cpus=args.cpus,
    )
    print(value["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
