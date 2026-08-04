#!/usr/bin/env python3
"""Write reviewed PRAD production requests after inspecting smoke usage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.prad.campaign import prad_tasks, validate_prad_campaign_spec  # noqa: E402


def _override(value: str) -> tuple[str, int, str, str]:
    fields = value.split(",")
    if len(fields) != 4 or not fields[1].isdigit():
        raise argparse.ArgumentTypeError("override must be TASK,CPUS,MEMORY,WALLTIME")
    return fields[0], int(fields[1]), fields[2], fields[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-campaign-spec", type=Path, required=True)
    parser.add_argument("--override", action="append", type=_override, default=[])
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.smoke_campaign_spec)
    validate_prad_campaign_spec(spec)
    if spec["mode"] != "smoke":
        raise ValueError("PRAD resource requests require a smoke parent")
    requests = {
        task.name: {
            "cpus": task.cpus,
            "memory": task.memory,
            "walltime": task.walltime,
            "gpu": task.gpu,
            "array": task.array,
        }
        for task in prad_tasks(smoke=False)
    }
    for task, cpus, memory, walltime in args.override:
        if task not in requests:
            raise ValueError(f"unknown PRAD task {task!r}")
        requests[task].update(cpus=cpus, memory=memory, walltime=walltime)
    payload = with_content_hash(
        {
            "contract": "hlt_classification_prad_resource_request_review_v1",
            "schema_version": 1,
            "smoke_campaign_spec_sha256": spec["content_hash"],
            "source_snapshot_sha256": spec["source_snapshot"]["source_snapshot_sha256"],
            "review_note": args.review_note,
            "requests": requests,
        }
    )
    write_immutable_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
