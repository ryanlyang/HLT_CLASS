#!/usr/bin/env python3
"""Plot original TRI60 LOGIT_D000E Hbb/Hcc validation rejection curves."""

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
from hlt_classification.scouting.hcwdl_mhpe_tri60_original_logit_roc import (  # noqa: E402
    evaluate_original_logit_d000e_roc,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--ce-control-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_source_checkout(ROOT, expected_commit=args.producer_commit)
    report = evaluate_original_logit_d000e_roc(
        campaign_spec_path=args.campaign_spec,
        ce_control_spec_path=args.ce_control_spec,
        output_dir=args.output_dir,
        producer_commit=args.producer_commit,
        device=args.device,
    )
    print(json.dumps({
        "content_hash": report["content_hash"],
        "curves": report["curves_path"],
        "pdf": report["figures"]["pdf"]["path"],
        "png": report["figures"]["png"]["path"],
        "working_points": report["working_points"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
