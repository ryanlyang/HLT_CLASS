#!/usr/bin/env python3
"""Publish exact post-cancellation proof for reserved legacy final jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_shared_final import build_legacy_cancellation


def _load_jobs(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("legacy cancellation jobs must be a JSON array of objects")
    return [dict(row) for row in value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=Path,
        required=True,
        help="JSON array of job_id/scheduler_state_after rows from exact-ID audit",
    )
    parser.add_argument("--output-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_legacy_cancellation(
        reservation=artifact(args.reservation),
        jobs=_load_jobs(args.jobs),
        output_audit_sha256=args.output_audit_sha256,
    )
    publish(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
