#!/usr/bin/env python3
"""Validate and attest packaged high-coverage matcher resources."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.scouting.highcov_resources import resource_validation_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_immutable_json(args.output, resource_validation_report(args.resource_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
