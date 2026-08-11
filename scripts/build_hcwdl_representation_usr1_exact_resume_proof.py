#!/usr/bin/env python3
"""Reopen the three bounded HCWDL-RKD USR1 action-result envelopes."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_usr1_exact_resume_proof,
)
from hlt_classification.scouting.hcwdl_representation_resources import artifact_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--reference-action-result", type=Path, required=True)
    parser.add_argument("--interrupt-action-result", type=Path, required=True)
    parser.add_argument("--resume-action-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build_usr1_exact_resume_proof(
        authority=artifact_reference(args.authority),
        action_results={
            "usr1_reference": artifact_reference(args.reference_action_result),
            "usr1_interrupt": artifact_reference(args.interrupt_action_result),
            "usr1_resume": artifact_reference(args.resume_action_result),
        },
        require_genuine=True,
    )
    publish(args.output, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
