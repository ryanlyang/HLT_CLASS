#!/usr/bin/env python3
"""Fit PRAD input moments and semantic weights from the train split only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.prad.cache import PradCacheDataset  # noqa: E402
from hlt_classification.prad.statistics import save_semantic_positive_weights  # noqa: E402


def _replica(value: str):
    raw, path = value.split("=", 1)
    return int(raw), Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-paired-cache", type=Path, required=True)
    parser.add_argument("--train-target-cache", action="append", type=_replica, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    target_paths = dict(args.train_target_cache)
    if len(target_paths) != len(args.train_target_cache) or set(target_paths) != {0, 1, 2, 3}:
        raise ValueError("PRAD statistics require target replicas 0,1,2,3")
    caches = tuple(
        PradCacheDataset(
            target_paths[replica],
            expected_kind="structural_targets",
            expected_role="train",
        )
        for replica in range(4)
    )
    paired = PradCacheDataset(
        args.train_paired_cache,
        expected_kind="paired_views",
        expected_role="train",
    )
    report = save_semantic_positive_weights(
        caches, args.output, paired_cache=paired
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
