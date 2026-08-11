#!/usr/bin/env python3
"""Submit one explicitly authorized HCWDL-U-RKD command plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import submit_command_plan  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import SUBMISSION_PHRASE  # noqa: E402


def _scheduler(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission-phrase", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("HCWDL-U-RKD submission requires --execute")
    if args.submission_phrase != SUBMISSION_PHRASE:
        raise PermissionError("HCWDL-U-RKD submission phrase differs")
    spec = load_json(args.campaign_spec)
    events = Path(spec["campaign_root"]) / "submission_events"
    prior_events = [load_json(path) for path in sorted(events.glob("*.json"))]
    def record(event):
        write_immutable_json(
            events / f"{int(event['sequence']):06d}_{event['content_hash']}.json",
            event,
        )
    ledger = submit_command_plan(
        spec=spec, command_plan=load_json(args.command_plan),
        scheduler=_scheduler, authorization_phrase=args.submission_phrase,
        event_writer=record, prior_events=prior_events,
    )
    write_immutable_json(args.output, ledger)
    print(f"Submitted {ledger['submitted_task_count']} exact jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
