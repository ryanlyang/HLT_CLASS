#!/usr/bin/env python3
"""Run the standalone full-data 180-pass D000-from-D033E study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_task_attestation, task_attestation_path,
)
from hlt_classification.scouting.hcwdl_tri60_d000_long180 import (  # noqa: E402
    NODE_ID, run_training, task_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    spec = load_json(args.spec)
    validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
    report = run_training(spec, device=args.device)
    outputs = task_outputs(spec)
    task_id = "train_D000_from_D033E_180"
    attestation = build_task_attestation(
        campaign_spec_sha256=spec["content_hash"], task_id=task_id,
        array_index=None, outputs=outputs,
    )
    write_immutable_json(
        task_attestation_path(spec["campaign_root"], task_id, None),
        attestation,
    )
    print(json.dumps({
        "node_id": NODE_ID, "content_hash": report["content_hash"],
        "complete": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
