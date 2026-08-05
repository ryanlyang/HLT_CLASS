#!/usr/bin/env python3
"""Cancel only exact numeric job IDs recorded by one PMARD campaign ledger."""

from __future__ import annotations

import argparse, re, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, validate_content_hash  # noqa: E402
from hlt_classification.scouting.campaign import PMARD_LEDGER_CONTRACT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); ledger = load_json(args.submission_ledger)
    validate_content_hash(ledger, expected_contract=PMARD_LEDGER_CONTRACT)
    if ledger.get("dry_run") is not False:
        raise ValueError("dry-run PMARD IDs cannot be cancelled")
    ids = list(ledger["jobs"].values())
    if any(not re.fullmatch(r"[1-9][0-9]*", value) for value in ids): raise ValueError("ledger contains a nonnumeric job ID")
    print(" ".join(ids))
    if args.execute: subprocess.run(["scancel", *ids], check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
