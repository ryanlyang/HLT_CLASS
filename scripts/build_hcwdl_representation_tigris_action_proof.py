#!/usr/bin/env python3
"""Bind an action result to exact HCWDL-RKD scheduler and miniature evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    ACTION_RESULT_CONTRACTS, build_tigris_action_proof,
)
from hlt_classification.scouting.hcwdl_representation_resources import artifact_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-kind", choices=tuple(ACTION_RESULT_CONTRACTS), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--representation-recipe-sha256", required=True)
    parser.add_argument("--scheduler-evidence", type=Path, required=True)
    parser.add_argument("--miniature-evidence", type=Path, required=True)
    parser.add_argument("--result-artifact", type=Path, required=True)
    parser.add_argument("--resource-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scheduler = artifact(args.scheduler_evidence)
    profile = artifact(args.resource_profile)
    resource_class = scheduler["resource_class"]
    proof = build_tigris_action_proof(
        evidence_kind=args.evidence_kind, source_commit=args.source_commit,
        representation_recipe_sha256=args.representation_recipe_sha256,
        scheduler_evidence=artifact_reference(args.scheduler_evidence),
        miniature_evidence=artifact_reference(args.miniature_evidence),
        result_artifact=artifact_reference(args.result_artifact),
        resource_request=profile["requests"][resource_class],
        expected_workers=profile["measurement_environment"]["production_workers"],
    )
    publish(args.output, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
