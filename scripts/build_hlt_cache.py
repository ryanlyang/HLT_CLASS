#!/usr/bin/env python3
"""Build or resume one immutable deterministic HLT-v3 cache."""

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

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.data.hlt_cache import build_hlt_cache  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-contract", type=Path, required=True)
    parser.add_argument("--replica-manifest", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--replica-id", type=int, required=True)
    parser.add_argument("--realization-policy", default="R_MULTI")
    parser.add_argument("--profile-id", default="D_NOMINAL")
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--processing-batch-size", type=int, default=256)
    parser.add_argument("--max-new-shards", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_hlt_cache(
        offline_cache_dir=args.offline_cache_dir,
        output_dir=args.output_dir,
        profile_contract=load_json(args.profile_contract),
        replica_manifest=load_json(args.replica_manifest),
        logical_role=args.role,
        replica_id=args.replica_id,
        realization_policy=args.realization_policy,
        degradation_profile_id=args.profile_id,
        source_snapshot_sha256=args.source_snapshot_sha256,
        shard_size=args.shard_size,
        processing_batch_size=args.processing_batch_size,
        max_new_shards=args.max_new_shards,
        progress=lambda payload: print(json.dumps(dict(payload), sort_keys=True)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
