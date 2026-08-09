#!/usr/bin/env python3
"""Run exactly one task from an immutable dense cold HCWDL specification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_dense import validate_dense_spec  # noqa: E402
from hlt_classification.scouting.hcwdl_dense_workflow import DenseColdWorkflow  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_task_attestation, task_attestation_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_dense_spec(spec, executable=True)
    if args.campaign_spec.resolve() != (
        Path(spec["campaign_root"]) / "campaign_spec.json"
    ).resolve():
        raise PermissionError("dense cold worker requires canonical campaign spec path")
    validate_source_checkout(REPO_ROOT, expected_commit=str(spec["source_commit"]))
    outputs = DenseColdWorkflow(spec).run(args.task)
    raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_index = None if raw_index is None else int(raw_index)
    attestation = build_task_attestation(
        campaign_spec_sha256=spec["content_hash"], task_id=args.task,
        array_index=array_index, outputs=outputs,
    )
    write_immutable_json(
        task_attestation_path(spec["campaign_root"], args.task, array_index),
        attestation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
