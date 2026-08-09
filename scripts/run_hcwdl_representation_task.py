#!/usr/bin/env python3
"""Dispatch one exact task row from an immutable HCWDL-RKD campaign spec."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from _hcwdl_representation_common import artifact
from hlt_classification.scouting.hcwdl_representation_campaign import validate_campaign_spec
from hlt_classification.scouting.hcwdl_representation_task_runtime import execute_registered_task


def main(*, allowed_kinds: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--deterministic-worker", action="store_true")
    parser.add_argument("--local-planning-fixture", action="store_true")
    args = parser.parse_args()
    spec = artifact(args.campaign_spec)
    validate_campaign_spec(spec, executable=not args.local_planning_fixture)
    rows = [row for row in spec["tasks"] if row["task_key"] == args.task]
    if len(rows) != 1:
        raise KeyError("task is not an exact campaign registry row")
    if allowed_kinds is not None and rows[0]["kind"] not in set(allowed_kinds):
        raise PermissionError("this thin entry point does not own the registered task kind")
    environment_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_index = args.array_index
    if environment_index is not None:
        if array_index is not None and array_index != int(environment_index):
            raise ValueError("CLI and Slurm array indices differ")
        array_index = int(environment_index)
    execute_registered_task(
        spec=spec, task_key=args.task, array_index=array_index,
        deterministic_worker=args.deterministic_worker,
        local_planning_fixture=args.local_planning_fixture,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
