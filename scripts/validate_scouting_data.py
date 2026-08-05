#!/usr/bin/env python3
"""Authenticate the immutable ScoutingAK8 ROOT inventory."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.scouting.audit import audit_source_inventory  # noqa: E402
from hlt_classification.scouting.streaming import discover_root_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miniature", action="store_true", help="allow non-reference file/count totals")
    args = parser.parse_args()
    report = audit_source_inventory(
        args.data_root, discover_root_files(args.data_root), strict_reference=not args.miniature,
    )
    write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
