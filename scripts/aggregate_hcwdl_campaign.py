#!/usr/bin/env python3
"""Run the registered HCWDL aggregation task for a campaign spec."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_workflow import HcwdlWorkflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen_aggregate", "aggregate_report"), required=True)
    args = parser.parse_args()
    HcwdlWorkflow(load_json(args.campaign_spec), repository=REPO_ROOT).run(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
