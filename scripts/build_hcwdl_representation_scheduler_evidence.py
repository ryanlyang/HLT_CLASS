#!/usr/bin/env python3
"""Build HCWDL-RKD scheduler evidence from an immutable raw sacct capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference, build_scheduler_evidence_from_sacct, resource_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "production"), required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--resource-class", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--representation-recipe-sha256")
    parser.add_argument("--worker-role", choices=("ordinary", "deterministic"), required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument(
        "--sacct-output", type=Path, required=True,
        help="raw output of the frozen sacct --parsable2 capture command",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = resource_table(mode=args.mode)
    if args.resource_class not in requests:
        raise ValueError("unknown HCWDL-RKD resource class")
    evidence = build_scheduler_evidence_from_sacct(
        raw_accounting_record=artifact_reference(args.sacct_output),
        task_key=args.task_key,
        resource_class=args.resource_class, source_commit=args.source_commit,
        representation_recipe_sha256=args.representation_recipe_sha256,
        worker_role=args.worker_role, worker=artifact_reference(args.worker),
        request=requests[args.resource_class],
    )
    if evidence["job_id"] != args.job_id:
        raise PermissionError("requested job ID differs from raw sacct allocation")
    publish(args.output, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
