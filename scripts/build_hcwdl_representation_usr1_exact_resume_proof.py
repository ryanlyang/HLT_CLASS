#!/usr/bin/env python3
"""Prove an HCWDL-RKD two-update USR1 resume equals uninterrupted execution."""

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
    parser.add_argument("--uninterrupted-report", type=Path, required=True)
    parser.add_argument("--resumed-report", type=Path, required=True)
    parser.add_argument("--resumed-state-directory", type=Path, required=True)
    parser.add_argument("--resumed-sequence", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--representation-recipe-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build_usr1_exact_resume_proof(
        uninterrupted_report=artifact_reference(args.uninterrupted_report),
        resumed_report=artifact_reference(args.resumed_report),
        resumed_state_directory=args.resumed_state_directory,
        resumed_sequence=args.resumed_sequence, source_commit=args.source_commit,
        representation_recipe_sha256=args.representation_recipe_sha256,
    )
    publish(args.output, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
