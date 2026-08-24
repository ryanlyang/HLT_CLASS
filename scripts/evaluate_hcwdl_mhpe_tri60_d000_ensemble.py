#!/usr/bin/env python3
"""Evaluate the fixed four-model TRI60 D000 ensemble on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_d000_ensemble import (  # noqa: E402
    evaluate_d000_cross_track_ensemble,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_source_checkout(ROOT, expected_commit=args.producer_commit)
    report = evaluate_d000_cross_track_ensemble(
        campaign_spec_path=args.campaign_spec,
        output=args.output,
        producer_commit=args.producer_commit,
        device=args.device,
    )
    print(json.dumps({
        "content_hash": report["content_hash"],
        "output": str(args.output.resolve()),
        "summary": report["primary_ensemble"]["summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
