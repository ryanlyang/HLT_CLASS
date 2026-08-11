#!/usr/bin/env python3
"""Reopen the exact four bounded HCWDL-RKD two-update action results."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_nonfinal_acceptance import (
    build_two_update_acceptance_proof,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--rset-m1c-action-result", type=Path, required=True)
    parser.add_argument("--rset-m1w-action-result", type=Path, required=True)
    parser.add_argument("--rrel-m1c-action-result", type=Path, required=True)
    parser.add_argument("--rrel-m1w-action-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build_two_update_acceptance_proof(
        authority=artifact_reference(args.authority),
        action_results={
            "RSET_M1c": artifact_reference(args.rset_m1c_action_result),
            "RSET_M1w": artifact_reference(args.rset_m1w_action_result),
            "RREL_M1c": artifact_reference(args.rrel_m1c_action_result),
            "RREL_M1w": artifact_reference(args.rrel_m1w_action_result),
        },
        require_genuine=True,
    )
    publish(args.output, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
