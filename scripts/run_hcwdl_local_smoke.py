#!/usr/bin/env python3
"""Run the complete bounded local HCWDL fixture DAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_smoke import run_local_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_local_smoke(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
