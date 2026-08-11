#!/usr/bin/env python3
"""Submit an explicitly authorized HCWDL-U-RKD failed-closure recovery."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_recovery import submit_recovery_command_plan  # noqa: E402
def _scheduler(command): return subprocess.run(command, check=True, capture_output=True, text=True).stdout
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--prior-submission-ledger", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute: raise PermissionError("recovery submission requires --execute")
    ledger = submit_recovery_command_plan(
        spec=load_json(args.campaign_spec),
        prior_ledger=load_json(args.prior_submission_ledger),
        recovery=load_json(args.recovery), command_plan=load_json(args.command_plan),
        scheduler=_scheduler,
    )
    write_immutable_json(args.output, ledger); return 0
if __name__ == "__main__": raise SystemExit(main())
