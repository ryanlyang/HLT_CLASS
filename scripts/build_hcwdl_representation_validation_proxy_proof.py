#!/usr/bin/env python3
"""Build the bounded validation-role proxy proof for HCWDL-RKD final streams."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_validation_proxy_proof,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--representation-recipe-sha256", required=True)
    parser.add_argument("--validation-population-sha256", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--branch-access", type=Path, action="append", required=True)
    parser.add_argument("--prediction-manifest-sha256", action="append", required=True)
    parser.add_argument("--metric-report-sha256", required=True)
    parser.add_argument("--runtime-signature-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build_validation_proxy_proof(
        source_commit=args.source_commit,
        representation_recipe_sha256=args.representation_recipe_sha256,
        validation_population_sha256=args.validation_population_sha256,
        rows=args.rows,
        branch_access_records=[artifact(path) for path in args.branch_access],
        prediction_manifest_sha256s=args.prediction_manifest_sha256,
        metric_report_sha256=args.metric_report_sha256,
        runtime_signature_sha256=args.runtime_signature_sha256,
    )
    publish(args.output, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
