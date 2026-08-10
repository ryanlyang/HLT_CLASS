#!/usr/bin/env python3
"""Run one task from an authenticated primary HCWDL recovery closure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_campaign_recovery import (  # noqa: E402
    validate_campaign_recovery_inputs, validate_campaign_recovery_spec,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_task_attestation, task_attestation_path,
)
from hlt_classification.scouting.hcwdl_workflow import HcwdlWorkflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    spec = load_json(args.recovery_spec)
    validate_campaign_recovery_spec(spec, executable=True)
    if args.task not in spec["retry_tasks"]:
        raise PermissionError(
            "task is outside the authenticated HCWDL campaign recovery closure"
        )
    evidence = validate_campaign_recovery_inputs(spec)
    validate_source_checkout(REPO_ROOT, expected_commit=spec["source_commit"])
    outputs = HcwdlWorkflow(evidence["parent"], repository=REPO_ROOT).run(args.task)
    raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_index = None if raw_index is None else int(raw_index)
    attestation = build_task_attestation(
        campaign_spec_sha256=spec["content_hash"], task_id=args.task,
        array_index=array_index, outputs=outputs,
    )
    write_immutable_json(
        task_attestation_path(spec["recovery_root"], args.task, array_index),
        attestation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
