#!/usr/bin/env python3
"""Print or cancel only the exact job IDs in one HCWDL ledger."""

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
    args = parser.parse_args()
    ids = exact_cancel_ids(load_json(args.submission_ledger))
    if not args.execute:
        print("scancel " + " ".join(ids))
        return 0
    if args.authorization_phrase != "CANCEL HCWDL EXACT IDS":
        raise PermissionError("HCWDL cancellation requires the exact authorization phrase")
    subprocess.run(["scancel", *ids], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
