#!/usr/bin/env python3
"""Print or execute cancellation for exact direct-KD campaign job IDs."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import exact_cancel_ids  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args(); ids = exact_cancel_ids(load_json(args.submission_ledger))
    if args.execute and args.authorization_phrase != "CANCEL HCWDL DIRECT KD EXACT IDS":
        raise PermissionError("direct KD cancellation phrase differs")
    command = ["scancel", *ids]; print(" ".join(command))
    if args.execute: subprocess.run(command, check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
