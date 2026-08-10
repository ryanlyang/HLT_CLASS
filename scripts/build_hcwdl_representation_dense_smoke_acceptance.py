#!/usr/bin/env python3
"""Publish pilot-gating evidence from one complete non-final dense smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_dense_acceptance import (
    build_dense_smoke_acceptance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--terminal-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output, build_dense_smoke_acceptance(
        campaign_spec=artifact(args.campaign_spec),
        command_plan=artifact(args.command_plan),
        submission_ledger=artifact(args.submission_ledger),
        monitor_report=artifact(args.monitor_report),
        output_audit=artifact(args.output_audit),
        terminal_aggregate=artifact(args.terminal_aggregate),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
