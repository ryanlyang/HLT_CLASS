#!/usr/bin/env python3
"""Run one exact HCWDL-UJ task and publish its task attestation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_campaign import (  # noqa: E402
    validate_campaign, validate_worker_semantics,
)
from hlt_classification.scouting.hcwdl_homotopy_workflow import HomotopyWorkflow  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_task_attestation, task_attestation_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--array-index", type=int)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign(spec, executable=True)
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("HCWDL-UJ worker is not running from its bound worktree")
    validate_source_checkout(REPO_ROOT, expected_commit=spec["source_commit"])
    validate_worker_semantics(spec, repository=REPO_ROOT)
    index = args.array_index
    if index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    outputs = HomotopyWorkflow(spec, repository=REPO_ROOT).run(
        args.task, array_index=index,
    )
    attestation = build_task_attestation(
        campaign_spec_sha256=spec["content_hash"], task_id=args.task,
        array_index=index, outputs=outputs,
    )
    output = task_attestation_path(spec["campaign_root"], args.task, index)
    if output.exists():
        existing = load_json(output)
        if existing != attestation:
            raise FileExistsError("existing HCWDL-UJ task attestation differs")
    else:
        write_immutable_json(output, attestation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
