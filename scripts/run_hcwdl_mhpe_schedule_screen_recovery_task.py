#!/usr/bin/env python3
"""Run one source-pinned D066 schedule-screen recovery task."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_schedule_screen_recovery import validate_recovery  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_schedule_screen_runner import ScheduleScreenWorkflow  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation, task_attestation_path  # noqa: E402


def _outputs(root: Path, task: str) -> list[Path]:
    if task.startswith("train_"):
        node = task.removeprefix("train_"); directory = root / "training" / node
        engine = load_json(directory / "training_report.json")
        return [
            directory / "training_report.json", directory / "screen_training_report.json",
            root / "reports/runtime" / f"{node}.json",
            directory / engine["selected_checkpoint"], directory / engine["final_checkpoint"],
        ]
    if task == "aggregate":
        return [root / "reports/validation_aggregate.json"]
    return [root / "reports/campaign_complete.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    recovery = load_json(args.recovery_spec); validate_recovery(recovery)
    validate_source_checkout(ROOT, expected_commit=recovery["source_commit"])
    original = load_json(recovery["campaign_spec_path"])
    ScheduleScreenWorkflow(original, verify_source_tree=False).run(args.task, device=args.device)
    root = Path(original["campaign_root"])
    attestation = build_task_attestation(
        campaign_spec_sha256=original["content_hash"], task_id=args.task,
        array_index=None, outputs=_outputs(root, args.task),
    )
    path = task_attestation_path(root, args.task, None)
    if path.exists():
        if load_json(path) != attestation:
            raise FileExistsError("schedule-screen recovery attestation differs")
    else:
        write_immutable_json(path, attestation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
