#!/usr/bin/env python3
"""Cancel explicitly selected jobs by exact IDs from one PRAD ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.prad.campaign import validate_prad_campaign_spec, validate_prad_submission_ledger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_prad_campaign_spec(spec)
    ledger = load_json(args.submission_ledger)
    validate_prad_submission_ledger(ledger, spec=spec)
    by_task = {row["task"]: row["job_id"] for row in ledger["jobs"]}
    unknown = sorted(set(args.task) - set(by_task))
    if unknown:
        raise ValueError(f"tasks are absent from the PRAD ledger: {unknown}")
    ids = [by_task[name] for name in dict.fromkeys(args.task)]
    command = ["scancel", *ids]
    if args.execute:
        subprocess.run(command, check=True)
    print(json.dumps({"campaign_spec_sha256": spec["content_hash"], "tasks": args.task, "exact_job_ids": ids, "command": command, "executed": args.execute}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
