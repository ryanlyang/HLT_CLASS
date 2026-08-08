#!/usr/bin/env python3
"""Exactly recompute a deterministic sample of one HCWDL assignment role."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_assignment import assignment_recomputer  # noqa: E402
from hlt_classification.scouting.highcov_cache import sampled_recomputation_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "final_test"), required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--completed-lock", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sampled_recomputation_audit(
        args.manifest,
        recompute=assignment_recomputer(
            split_manifest=load_json(args.split_manifest), data_root=args.data_root,
            role=args.role, completed_locks=args.completed_lock,
        ),
        sample_size=args.sample_size, seed=args.seed,
    )
    write_immutable_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
