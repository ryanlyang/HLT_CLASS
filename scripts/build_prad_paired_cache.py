#!/usr/bin/env python3
"""Build/resume one authenticated paired offline/HLT PRAD cache."""

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

from hlt_classification.prad.artifacts import build_prad_paired_view_cache  # noqa: E402
from hlt_classification.prad.splits import load_prad_split_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "val", "test"), required=True)
    parser.add_argument("--replica-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--realization-policy", choices=("R_FIXED", "R_MULTI"), default="R_MULTI")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-new-shards", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_prad_split_manifest(args.split_manifest)
    result = build_prad_paired_view_cache(
        manifest,
        logical_role=args.role,
        replica_id=args.replica_id,
        output_dir=args.output_dir,
        source_snapshot_sha256=args.source_snapshot_sha256,
        realization_policy=args.realization_policy,
        shard_size=args.shard_size,
        max_new_shards=args.max_new_shards,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
