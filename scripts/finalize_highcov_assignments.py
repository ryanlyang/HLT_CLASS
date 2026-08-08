#!/usr/bin/env python3
"""Validate and merge all dense HCWDL assignment shards for one role."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_assignment import (  # noqa: E402
    finalize_role_assignments, load_assignment_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--resources-report", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "final_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split, selection, resources = load_assignment_inputs(
        split_manifest_path=args.split_manifest,
        selection_manifest_path=args.selection_manifest,
        resources_report_path=args.resources_report,
    )
    finalize_role_assignments(
        split_manifest=split, selection_manifest=selection, resources_report=resources,
        assignment_root=args.assignment_root, role=args.role, output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
