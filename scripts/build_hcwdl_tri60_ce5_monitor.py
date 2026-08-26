#!/usr/bin/env python3
"""Build an immutable TRI60 CE5 monitor from exact job-id states."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_ce5_operations import build_monitor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state", action="append", default=[], metavar="JOB_ID=STATE",
    )
    args = parser.parse_args()
    states = {}
    for item in args.state:
        job_id, separator, state = item.partition("=")
        if not separator or not job_id or not state or job_id in states:
            raise ValueError("TRI60 CE5 monitor state assignment differs")
        states[job_id] = state
    spec = load_json(args.spec)
    ledger = load_json(args.ledger)
    value = build_monitor(
        subject=spec, ledger=ledger, states_by_job_id=states,
        attestation_root=spec["campaign_root"],
    )
    write_immutable_json(args.output, value)
    print(value["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
