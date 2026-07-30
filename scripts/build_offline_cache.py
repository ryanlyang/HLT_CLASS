#!/usr/bin/env python3
"""Build or resume one immutable, identity-bound offline cache."""

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

from hlt_classification.data.offline_cache import build_offline_cache  # noqa: E402
from hlt_classification.data.splits import load_split_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--read-chunk-size", type=int, default=4096)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--max-new-shards", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_offline_cache(
        load_split_manifest(args.split_manifest),
        logical_role=args.role,
        output_dir=args.output_dir,
        data_root=args.data_root,
        shard_size=args.shard_size,
        read_chunk_size=args.read_chunk_size,
        source_snapshot_sha256=args.source_snapshot_sha256,
        max_new_shards=args.max_new_shards,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
