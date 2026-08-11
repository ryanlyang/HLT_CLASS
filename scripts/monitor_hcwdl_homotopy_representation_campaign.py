#!/usr/bin/env python3
"""Query exact campaign job IDs and publish one HCWDL-U-RKD monitor report."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_recovery import build_monitor_report, query_scheduler_states  # noqa: E402


def _scheduler(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec); ledger = load_json(args.submission_ledger)
    report = build_monitor_report(
        spec=spec, ledger=ledger,
        scheduler_states=query_scheduler_states(ledger, runner=_scheduler),
    )
    write_immutable_json(args.output, report)
    for row in report["rows"]:
        print(f"{row['task_id']:<28} {row['state']:<12} {row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
