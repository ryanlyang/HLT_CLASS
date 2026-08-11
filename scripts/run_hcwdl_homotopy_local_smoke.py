#!/usr/bin/env python3
"""Run the bounded synthetic 80-fit HCWDL-UJ behavioral smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_homotopy_smoke import (  # noqa: E402
    run_local_homotopy_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_local_homotopy_smoke(args.output_root)
    print(f"Completed {report['fit_count']} synthetic HCWDL-UJ fits")
    print(f"Report: {args.output_root / 'smoke_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
