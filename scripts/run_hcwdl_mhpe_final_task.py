#!/usr/bin/env python3
"""Run the separately authorized sealed HCWDL-MHPE final evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_final import run_sealed_final_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--finalist-lock", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    spec = load_json(args.spec)
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    run_sealed_final_evaluation(
        campaign_spec_path=args.spec,
        finalist_lock_path=args.finalist_lock,
        execution_lock_path=args.execution_lock,
        output=args.output,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
