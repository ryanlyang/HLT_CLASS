#!/usr/bin/env python3
"""Revalidate every HLT shard and its exact scientific lineage."""

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

from hlt_classification.data.hlt_cache import audit_hlt_cache  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-profile-contract-sha256")
    parser.add_argument("--expected-replica-manifest-sha256")
    parser.add_argument("--expected-degradation-profile-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_hlt_cache(
        args.cache_dir,
        expected_profile_contract_sha256=args.expected_profile_contract_sha256,
        expected_replica_manifest_sha256=args.expected_replica_manifest_sha256,
        expected_degradation_profile_id=args.expected_degradation_profile_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
