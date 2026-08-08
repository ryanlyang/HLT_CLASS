#!/usr/bin/env python3
"""Build one immutable HCWDL dense-assignment source shard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_assignment import (  # noqa: E402
    build_assignment_source, load_assignment_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--resources-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "final_test"), required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--completed-lock", action="append", default=[])
    args = parser.parse_args()
    split, selection, resources = load_assignment_inputs(
        split_manifest_path=args.split_manifest,
        selection_manifest_path=args.selection_manifest,
        resources_report_path=args.resources_report,
    )
    build_assignment_source(
        split_manifest=split, selection_manifest=selection, resources_report=resources,
        data_root=args.data_root, assignment_root=args.assignment_root,
        role=args.role, source_index=args.source_index,
        completed_locks=args.completed_lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
