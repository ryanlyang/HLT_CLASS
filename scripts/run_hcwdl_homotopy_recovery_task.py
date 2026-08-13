#!/usr/bin/env python3
"""Run one task from a source or resource HCWDL-UJ recovery spec."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_contracts import (  # noqa: E402
    RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_SPEC_CONTRACT,
)
from hlt_classification.scouting.hcwdl_homotopy_campaign import (  # noqa: E402
    validate_worker_semantics,
)
from hlt_classification.scouting.hcwdl_homotopy_recovery import (  # noqa: E402
    validate_recovery_spec, validate_recovery_worker_semantics,
    validate_resource_recovery_spec,
)
from hlt_classification.scouting.hcwdl_homotopy_workflow import HomotopyWorkflow  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_task_attestation, task_attestation_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--array-index", type=int)
    args = parser.parse_args()
    recovery = load_json(args.recovery_spec)
    source_recovery = recovery.get("contract") == RECOVERY_SPEC_CONTRACT
    if source_recovery:
        validate_recovery_spec(recovery, executable=True)
    elif recovery.get("contract") == RESOURCE_RECOVERY_SPEC_CONTRACT:
        validate_resource_recovery_spec(recovery, executable=True)
    else:
        raise ValueError("unknown HCWDL-UJ recovery specification")
    if args.task not in recovery["retry_tasks"]:
        raise PermissionError("task is outside the authenticated recovery closure")
    if REPO_ROOT.resolve() != Path(recovery["project_dir"]).resolve():
        raise PermissionError("recovery worker is not running from its bound worktree")
    validate_source_checkout(REPO_ROOT, expected_commit=recovery["source_commit"])
    campaign = load_json(recovery["campaign_spec"]["path"])
    if source_recovery:
        validate_recovery_worker_semantics(
            campaign, recovery, repository=REPO_ROOT,
        )
    else:
        validate_worker_semantics(campaign, repository=REPO_ROOT)
    index = args.array_index
    if index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    outputs = HomotopyWorkflow(
        campaign, repository=REPO_ROOT,
        producer_commit=recovery["source_commit"],
        recovery_spec_sha256=recovery["content_hash"],
        execution_semantic_source_sha256=(
            recovery["execution_semantic_source_sha256"]
            if source_recovery else None
        ),
    ).run(
        args.task, array_index=index,
    )
    attestation = build_task_attestation(
        campaign_spec_sha256=campaign["content_hash"], task_id=args.task,
        array_index=index, outputs=outputs,
    )
    path = task_attestation_path(campaign["campaign_root"], args.task, index)
    if path.exists():
        if load_json(path) != attestation:
            raise FileExistsError("existing recovery task attestation differs")
    else:
        write_immutable_json(path, attestation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
