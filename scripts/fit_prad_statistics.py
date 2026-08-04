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
from hlt_classification.prad.splits import load_prad_split_manifest  # noqa: E402
from hlt_classification.prad.streaming import (  # noqa: E402
    build_in_memory_paired_views,
    build_in_memory_structural_targets,
)


def _replica(value: str):
    raw, path = value.split("=", 1)
    return int(raw), Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-paired-cache", type=Path)
    parser.add_argument("--train-target-cache", action="append", type=_replica)
    parser.add_argument("--streaming-split-manifest", type=Path)
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.streaming_split_manifest is not None:
        if (
            args.train_paired_cache is not None
            or args.train_target_cache
            or args.source_snapshot_sha256 is None
        ):
            raise ValueError("streaming PRAD statistics arguments differ")
        split = load_prad_split_manifest(args.streaming_split_manifest)
        paired_by_replica = build_in_memory_paired_views(
            split,
            logical_role="train",
            replica_ids=(0, 1, 2, 3),
            source_snapshot_sha256=args.source_snapshot_sha256,
        )
        target_by_replica = build_in_memory_structural_targets(
            split,
            paired_views=paired_by_replica,
            source_snapshot_sha256=args.source_snapshot_sha256,
        )
        caches = tuple(target_by_replica[replica] for replica in range(4))
        paired = paired_by_replica[0]
    else:
        if args.train_paired_cache is None or not args.train_target_cache:
            raise ValueError("durable PRAD statistics require cache paths")
        target_paths = dict(args.train_target_cache)
        if (
            len(target_paths) != len(args.train_target_cache)
            or set(target_paths) != {0, 1, 2, 3}
        ):
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
