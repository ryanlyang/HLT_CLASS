#!/usr/bin/env python3
"""Publish the final immutable completion record for the six-arm HCWDL-UB study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import (  # noqa: E402
    validate_finalist_lock, validate_sweep_aggregate,
)
from hlt_classification.scouting.hcwdl_unified_balanced_reporting import (  # noqa: E402
    completion_payload, validate_campaign_completion, validate_final_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-aggregate", type=Path, required=True)
    parser.add_argument("--finalist-lock", type=Path, required=True)
    parser.add_argument("--final-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sweep = load_json(args.sweep_aggregate)
    finalist = load_json(args.finalist_lock)
    final = load_json(args.final_evaluation)
    sweep_hash = validate_sweep_aggregate(sweep)
    finalist_hash = validate_finalist_lock(finalist)
    final_hash = validate_final_evaluation(final, finalist_lock=finalist)
    if (
        final["finalist_lock_sha256"] != finalist_hash
        or finalist["sweep_aggregate_sha256"] != sweep_hash
    ):
        raise ValueError("HCWDL-UB final evaluation/finalist lineage differs")
    payload = completion_payload(
        sweep_aggregate_sha256=sweep_hash,
        finalist_lock_sha256=finalist_hash,
        final_evaluation_sha256=final_hash,
    )
    validate_campaign_completion(payload)
    write_immutable_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
