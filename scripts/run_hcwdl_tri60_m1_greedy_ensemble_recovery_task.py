#!/usr/bin/env python3
"""Run one source-pinned TRI60 M1 greedy reducer recovery task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation, task_attestation_path  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_m1_greedy_ensemble_recovery import validate_recovery  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_m1_greedy_ensemble_workflow import M1GreedyEnsembleWorkflow, task_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    recovery = load_json(args.recovery_spec)
    validate_recovery(recovery)
    validate_source_checkout(ROOT, expected_commit=recovery["source_commit"])
    campaign = load_json(recovery["campaign_spec_path"])
    outputs = task_outputs(campaign, args.task)
    if any(path.exists() for path in outputs):
        raise FileExistsError("TRI60 M1 greedy recovery output already exists")
    result = M1GreedyEnsembleWorkflow(
        campaign, recovery_spec_sha256=recovery["content_hash"],
        execution_source_commit=recovery["source_commit"],
    ).run(args.task, device="cpu")
    attestation = build_task_attestation(
        campaign_spec_sha256=recovery["content_hash"], task_id=args.task,
        array_index=None, outputs=outputs,
    )
    write_immutable_json(
        task_attestation_path(recovery["recovery_root"], args.task, None),
        attestation,
    )
    print(json.dumps({
        "task": args.task, "content_hash": result.get("content_hash"),
        "reused_prediction_shards": 5,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
