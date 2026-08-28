#!/usr/bin/env python3
"""Run one restart-from-zero TRI100 four-spine recovery task."""

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
from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (  # noqa: E402
    destroy_tri60_distributed, initialize_tri60_distributed,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_task_attestation, task_attestation_path,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_recovery import (  # noqa: E402
    clean_incomplete_task_outputs, validate_recovery,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_workflow import (  # noqa: E402
    Spine4Workflow, task_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed-world-size", type=int, default=1)
    args = parser.parse_args()
    recovery = load_json(args.recovery)
    validate_recovery(recovery)
    validate_source_checkout(ROOT, expected_commit=recovery["source_commit"])
    if args.task not in recovery["retry_tasks"]:
        raise PermissionError("task is not registered for TRI100 spine4 recovery")
    subject = load_json(recovery["subject_spec_path"])
    tasks = {row["task_id"]: row for row in subject["tasks"]}
    needs_ddp = tasks[args.task]["kind"] in {"preflight", "train"}
    expected_world = 4 if needs_ddp else 1
    if args.distributed_world_size != expected_world:
        raise ValueError("TRI100 four-spine recovery world size differs")
    context = None
    try:
        if needs_ddp:
            context = initialize_tri60_distributed(
                expected_world_size=4, global_batch_size=256, backend="nccl",
            )
        if context is None or context.is_primary:
            clean_incomplete_task_outputs(subject, args.task)
        if context is not None:
            context.barrier()
        result = Spine4Workflow(
            subject, recovery_spec_sha256=recovery["content_hash"],
            execution_source_commit=recovery["source_commit"],
        ).run(args.task, device=args.device, distributed_context=context)
        if context is not None:
            context.barrier()
        if context is None or context.is_primary:
            outputs = task_outputs(subject, args.task)
            attestation = build_task_attestation(
                campaign_spec_sha256=recovery["content_hash"],
                task_id=args.task, array_index=None, outputs=outputs,
            )
            write_immutable_json(
                task_attestation_path(
                    recovery["recovery_root"], args.task, None,
                ),
                attestation,
            )
            print(json.dumps({
                "task": args.task, "content_hash": result.get("content_hash"),
                "restart_from_zero": True,
            }, sort_keys=True))
        if context is not None:
            context.barrier()
        return 0
    finally:
        destroy_tri60_distributed(context)


if __name__ == "__main__":
    raise SystemExit(main())
