#!/usr/bin/env python3
"""Plot zoomed source/DX LOGIT ladder Hbb/Hcc rejection curves."""

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
from hlt_classification.scouting.hcwdl_mhpe_tri60_logit_ladder_zoom_roc import (  # noqa: E402
    evaluate_logit_ladder_zoom_roc,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign-spec", type=Path, required=True)
    parser.add_argument("--dense-campaign-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_source_checkout(ROOT, expected_commit=args.producer_commit)
    report = evaluate_logit_ladder_zoom_roc(
        source_campaign_spec_path=args.source_campaign_spec,
        dense_campaign_spec_path=args.dense_campaign_spec,
        output_dir=args.output_dir,
        producer_commit=args.producer_commit,
        device=args.device,
    )
    print(json.dumps({
        "content_hash": report["content_hash"],
        "curves": report["curves_path"],
        "figures": report["figures"],
        "display_order": report["display_order"],
        "source_registry": report["source_registry"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
