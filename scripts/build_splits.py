#!/usr/bin/env python3
"""Build and atomically publish the authenticated five-role split manifest."""

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

from hlt_classification.data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_SPLIT_SIZES,
    DEFAULT_TREE_NAME,
    MAX_CONSTITUENTS,
    audit_split_manifest,
    build_balanced_split_manifest,
    discover_file_records,
    save_split_manifest,
)
from hlt_classification.data.splits import DEFAULT_BASE_SEED  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/split_manifest.json.gz"),
    )
    parser.add_argument("--tree-name", default=DEFAULT_TREE_NAME)
    parser.add_argument("--pattern", default="*.root")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    for role, default in DEFAULT_SPLIT_SIZES.items():
        parser.add_argument(f"--{role.replace('_', '-')}", type=int, default=default)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_sizes = {
        role: int(getattr(args, role)) for role in DEFAULT_SPLIT_SIZES
    }
    records = discover_file_records(
        args.data_root,
        pattern=args.pattern,
        tree_name=args.tree_name,
        require_all_classes=True,
        validate_branches=True,
        skip_unreadable=False,
    )
    manifest = build_balanced_split_manifest(
        records,
        data_root=str(args.data_root.resolve()),
        tree_name=args.tree_name,
        max_constituents=MAX_CONSTITUENTS,
        split_sizes=split_sizes,
        split_seeds=DEFAULT_SPLIT_SEEDS,
        base_seed=args.base_seed,
    )
    audit = audit_split_manifest(manifest)
    if not audit["ok"]:
        print(json.dumps(audit, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    save_split_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "contract": "hlt_classification_split_build_report_v1",
                "manifest_path": str(args.output.resolve()),
                "manifest_content_hash": manifest.content_hash,
                "file_count": len(manifest.files),
                "audit": audit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
