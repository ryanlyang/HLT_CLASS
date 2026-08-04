#!/usr/bin/env python3
"""Build the immutable 500k/150k/500k PRAD split with seed 1337."""

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
    DEFAULT_TREE_NAME,
    discover_file_records,
)
from hlt_classification.prad.splits import build_prad_split_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--tree-name", default=DEFAULT_TREE_NAME)
    parser.add_argument("--pattern", default="*.root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = discover_file_records(
        args.data_root,
        pattern=args.pattern,
        tree_name=args.tree_name,
        require_all_classes=True,
        validate_branches=True,
        skip_unreadable=False,
    )
    manifest = build_prad_split_manifest(
        records,
        data_root=str(args.data_root.resolve()),
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
