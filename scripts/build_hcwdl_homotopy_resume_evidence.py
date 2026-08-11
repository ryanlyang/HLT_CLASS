#!/usr/bin/env python3
"""Authenticate a real Slurm USR1 interruption and exact HCWDL-UJ resume."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_campaign import (  # noqa: E402
    validate_campaign,
)
from hlt_classification.scouting.hcwdl_homotopy_resume import build_resume_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--preemption-event", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = load_json(args.campaign_spec); validate_campaign(spec)
    payload = build_resume_evidence(
        spec, node_id=args.node_id,
        preemption_event_path=args.preemption_event,
    )
    write_immutable_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
