#!/usr/bin/env python3
"""Run one exact task from an immutable PMARD campaign specification."""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.workflow import Workflow, write_task_attestation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args(); spec = load_json(args.campaign_spec)
    registered = {row["name"] for row in spec["tasks"]}
    if args.task not in registered: raise ValueError("task is not registered by the campaign")
    workflow = Workflow(spec, repository=REPO_ROOT); outputs = workflow.run(args.task)
    write_task_attestation(
        spec=spec, task=args.task, array_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        outputs=outputs, campaign_root=Path(spec["campaign_root"]),
    )
    return 0


if __name__ == "__main__": raise SystemExit(main())
