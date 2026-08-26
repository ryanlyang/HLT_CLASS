#!/usr/bin/env python3
"""Build an exact-job monitor for the TRI60 M1 compression screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_m1_screen_operations import build_monitor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_monitor(
        subject=load_json(args.spec), ledger=load_json(args.ledger),
        states_by_job_id=load_json(args.states),
        attestation_root=args.spec.parent,
    )
    write_immutable_json(args.output, value)
    print(value["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
