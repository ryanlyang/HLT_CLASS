#!/usr/bin/env python3
"""Build the train-only HCWDL-UJ Cartesian-edge scale calibration."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_workflow import HomotopyWorkflow  # noqa: E402
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--campaign-spec", type=Path, required=True)
    args = parser.parse_args(); HomotopyWorkflow(load_json(args.campaign_spec), repository=ROOT).run("upper_calibration"); return 0
if __name__ == "__main__": raise SystemExit(main())
