#!/usr/bin/env python3
"""Run one source-pinned D066 schedule-screen task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_schedule_screen_runner import ScheduleScreenWorkflow  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation, task_attestation_path  # noqa: E402


def _outputs(root: Path, task: str) -> list[Path]:
    if task.startswith("train_"):
        node = task.removeprefix("train_")
        directory = root / "training" / node
        engine = load_json(directory / "training_report.json")
        return [
            directory / "training_report.json",
            directory / "screen_training_report.json",
            root / "reports/runtime" / f"{node}.json",
            directory / engine["selected_checkpoint"],
            directory / engine["final_checkpoint"],
        ]
    if task == "aggregate":
        return [root / "reports/validation_aggregate.json"]
    return [root / "reports/campaign_complete.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    spec = load_json(args.spec)
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    result = ScheduleScreenWorkflow(spec).run(args.task, device=args.device)
    root = Path(spec["campaign_root"])
    attestation = build_task_attestation(
        campaign_spec_sha256=spec["content_hash"], task_id=args.task,
        array_index=None, outputs=_outputs(root, args.task),
    )
    write_immutable_json(task_attestation_path(root, args.task, None), attestation)
    print(json.dumps({"task": args.task, "content_hash": result.get("content_hash"), "complete": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
