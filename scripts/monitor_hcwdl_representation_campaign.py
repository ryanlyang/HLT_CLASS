#!/usr/bin/env python3
"""Publish one append-only HCWDL-RKD monitor report from exact scheduler states."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, load_json_mapping, scheduler
from hlt_classification.scouting.hcwdl_representation_recovery import (
    build_monitor_report, load_monitor_chain, publish_monitor_report,
    query_scheduler_states,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--recovery-ledger", type=Path, action="append", default=[])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scheduler-states", type=Path,
        help="local/mock exact-ID state mapping",
    )
    source.add_argument(
        "--query-scheduler", action="store_true",
        help="query sacct for the authenticated exact-ID union",
    )
    args = parser.parse_args()
    chain = load_monitor_chain(args.campaign_root)
    original = artifact(args.submission_ledger)
    recoveries = [artifact(path) for path in args.recovery_ledger]
    states = (
        query_scheduler_states(original, recoveries, runner=scheduler)
        if args.query_scheduler
        else load_json_mapping(args.scheduler_states)
    )
    report = build_monitor_report(
        original_ledger=original,
        recovery_ledgers=recoveries,
        scheduler_states=states,
        previous_report_sha256=None if not chain else chain[-1]["content_hash"],
        sequence=len(chain),
    )
    print(publish_monitor_report(args.campaign_root, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
