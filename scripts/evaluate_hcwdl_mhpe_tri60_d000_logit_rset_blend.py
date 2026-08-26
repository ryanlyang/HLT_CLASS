#!/usr/bin/env python3
"""Evaluate the fixed 50/50 LOGIT_D000E/RSET_D000E validation blend."""

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
from hlt_classification.scouting.hcwdl_mhpe_tri60_d000_logit_rset_blend import (  # noqa: E402
    evaluate_d000_logit_rset_blend,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--ce-control-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    args = parser.parse_args()
    validate_source_checkout(ROOT, expected_commit=args.producer_commit)
    report = evaluate_d000_logit_rset_blend(
        campaign_spec_path=args.campaign_spec,
        ce_control_spec_path=args.ce_control_spec,
        output=args.output,
        producer_commit=args.producer_commit,
    )
    print(json.dumps({
        "content_hash": report["content_hash"],
        "output": str(args.output.resolve()),
        "summary": report["primary_ensemble"]["summary"],
        "recovery": report["primary_ensemble"][
            "recovery_m0ce60_to_u000"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
