#!/usr/bin/env python3
"""Write exact future HCWDL Slurm commands without invoking Slurm."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_campaign import slurm_commands  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    commands = slurm_commands(spec)
    by_task = {row["task_id"]: row["command"] for row in commands}
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs={task: "1" for task in by_task},
        commands=by_task, dry_run=True,
    )
    write_immutable_json(args.output, ledger)
    for row in commands:
        print(row["task_id"] + ": " + " ".join(row["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
