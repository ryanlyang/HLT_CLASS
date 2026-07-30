#!/usr/bin/env python3
"""Audit the canonical recursive JetClass ROOT inventory and class capacity."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data import (  # noqa: E402
    CLASS_LABELS,
    DEFAULT_DATA_ROOT,
    DEFAULT_SPLIT_SIZES,
    DEFAULT_TREE_NAME,
    discover_file_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tree-name", default=DEFAULT_TREE_NAME)
    parser.add_argument("--pattern", default="*.root")
    parser.add_argument(
        "--required-per-class",
        type=int,
        default=sum(DEFAULT_SPLIT_SIZES.values()) // len(CLASS_LABELS),
    )
    parser.add_argument(
        "--diagnostic-skip-unreadable",
        action="store_true",
        help=(
            "Report capacity after skipping unreadable files. This mode is marked "
            "non-production and may not authorize a split manifest."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.required_per_class < 0:
        raise ValueError("--required-per-class must be non-negative")
    records = discover_file_records(
        args.data_root,
        pattern=args.pattern,
        tree_name=args.tree_name,
        require_all_classes=True,
        validate_branches=True,
        skip_unreadable=args.diagnostic_skip_unreadable,
        diagnostic_only=args.diagnostic_skip_unreadable,
    )
    entries = Counter()
    files = Counter()
    for record in records:
        entries[record.label] += record.num_entries
        files[record.label] += 1
    required = int(args.required_per_class)
    classes = {
        CLASS_LABELS[label]: {
            "label": label,
            "file_count": files[label],
            "available_events": entries[label],
            "required_events": required,
            "enough": entries[label] >= required,
        }
        for label in range(len(CLASS_LABELS))
    }
    capacity_ok = all(item["enough"] for item in classes.values())
    production_eligible = capacity_ok and not args.diagnostic_skip_unreadable
    report = {
        "contract": "hlt_classification_data_preflight_v1",
        "data_root": str(args.data_root.resolve()),
        "tree_name": args.tree_name,
        "pattern": args.pattern,
        "diagnostic_skip_unreadable": bool(args.diagnostic_skip_unreadable),
        "production_eligible": production_eligible,
        "ok": production_eligible,
        "classes": classes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if production_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
