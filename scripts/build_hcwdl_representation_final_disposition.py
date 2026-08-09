#!/usr/bin/env python3
"""Freeze the fail-closed HCWDL-RKD final disposition before campaign creation."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_shared_final import build_final_disposition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-final-state", type=Path, required=True)
    parser.add_argument(
        "--requested",
        choices=("combined_confirmatory", "validation_only_parent_claim_consumed"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_final_disposition(
        parent_final_state=artifact(args.parent_final_state),
        requested=args.requested,
    )
    publish(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
