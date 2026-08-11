#!/usr/bin/env python3
"""Run one filesystem-audited same-owner shared-final recovery row."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _hcwdl_representation_common import artifact
from hlt_classification.scouting.hcwdl_representation_campaign import (
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_task_runtime import (
    execute_registered_task,
)
from hlt_classification.scouting.hcwdl_shared_final import (
    authorize_shared_final_recovery_dispatch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--execution-claim", type=Path, required=True)
    parser.add_argument("--task-registry", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--deterministic-worker", action="store_true")
    args = parser.parse_args()

    spec = artifact(args.campaign_spec)
    validate_campaign_spec(spec, executable=True)
    campaign_rows = [row for row in spec["tasks"] if row["task_key"] == args.task]
    if len(campaign_rows) != 1:
        raise KeyError("task is not an exact campaign registry row")
    environment_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_index = args.array_index
    if environment_index is not None:
        if array_index is not None and array_index != int(environment_index):
            raise ValueError("CLI and Slurm array indices differ")
        array_index = int(environment_index)
    plan = artifact(args.recovery_plan)
    claim = artifact(args.execution_claim)
    registry = artifact(args.task_registry)
    if str(args.output_root.resolve()) != str(Path(plan["output_root"]).resolve()):
        raise ValueError("recovery output root differs from immutable plan")
    authorize_shared_final_recovery_dispatch(
        plan, claim=claim, task_registry=registry,
        campaign_task_key=args.task, array_index=array_index,
        output_root=args.output_root,
    )
    execute_registered_task(
        spec=spec, task_key=args.task, array_index=array_index,
        deterministic_worker=args.deterministic_worker,
        local_planning_fixture=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
