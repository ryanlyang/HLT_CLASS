#!/usr/bin/env python3
"""Aggregate a completed locked PRAD campaign into final artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.prad.reporting import aggregate_prad_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_prad_campaign(
                args.campaign_root, bootstrap_samples=args.bootstrap_samples
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
