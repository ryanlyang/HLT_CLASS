#!/usr/bin/env python3
"""Run one exact TRI100 four-spine task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_task_attestation, task_attestation_path,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_workflow import (  # noqa: E402
    Spine4Workflow, task_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-world-size", type=int, default=1)
    args = parser.parse_args()
    spec = load_json(args.spec)
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if args.task not in tasks:
        raise KeyError("unknown TRI100 four-spine task")
    if args.execution_world_size != 1:
        raise ValueError("TRI100 four-spine execution world size differs")
    result = Spine4Workflow(spec).run(args.task, device=args.device)
    outputs = task_outputs(spec, args.task)
    attestation = build_task_attestation(
        campaign_spec_sha256=spec["content_hash"],
        task_id=args.task, array_index=None, outputs=outputs,
    )
    write_immutable_json(
        task_attestation_path(spec["campaign_root"], args.task, None),
        attestation,
    )
    print(json.dumps({
        "task": args.task, "content_hash": result.get("content_hash"),
        "complete": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
