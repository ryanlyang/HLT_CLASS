#!/usr/bin/env python3
"""Authenticate an HCWDL-U-RKD partial/full ledger from submission events."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import (  # noqa: E402
    assemble_submission_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = [load_json(path) for path in sorted(args.event_root.glob("*.json"))]
    if not events:
        raise ValueError("HCWDL-U-RKD submission event journal is empty")
    ledger = assemble_submission_ledger(
        spec=load_json(args.campaign_spec),
        command_plan=load_json(args.command_plan), events=events,
    )
    write_immutable_json(args.output, ledger)
    print(f"Authenticated submitted tasks: {len(ledger['jobs'])}")
    print(f"Complete submission: {ledger['complete_submission']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
