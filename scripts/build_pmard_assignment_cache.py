#!/usr/bin/env python
"""Build or finalize compact, persistent fitted-strict assignment shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.selective_assignment import (  # noqa: E402
    build_assignment_shard, finalize_assignment_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--completed-lock", action="append", default=[])
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--source", nargs=2, metavar=("ROLE", "INDEX"))
    actions.add_argument("--finalize", nargs="+", metavar="ROLE")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split = load_json(args.split_manifest); selection = load_json(args.selection_manifest)
    if args.source:
        role, index = args.source
        build_assignment_shard(
            split, selection, data_root=args.data_root, output_root=args.assignment_root,
            role=role, source_index=int(index), completed_locks=args.completed_lock,
        )
    else:
        if args.output is None:
            parser.error("--output is required with --finalize")
        finalize_assignment_manifest(
            split, selection, assignment_root=args.assignment_root,
            roles=args.finalize, output=args.output,
        )


if __name__ == "__main__":
    main()
