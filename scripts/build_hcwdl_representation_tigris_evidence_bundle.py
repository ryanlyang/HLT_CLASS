#!/usr/bin/env python3
"""Assemble the seven exact HCWDL-RKD Tigris action proofs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_tigris_evidence_bundle,
)
from hlt_classification.scouting.hcwdl_representation_resources import artifact_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--representation-recipe-sha256", required=True)
    parser.add_argument("--resource-profile", type=Path, required=True)
    parser.add_argument("--storage-estimate", type=Path, required=True)
    parser.add_argument("--fixed-size-inventory", type=Path, required=True)
    parser.add_argument("--action-proofs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = load_json_mapping(args.action_proofs)
    bundle = build_tigris_evidence_bundle(
        source_commit=args.source_commit,
        representation_recipe_sha256=args.representation_recipe_sha256,
        resource_profile=artifact_reference(args.resource_profile),
        storage_estimate=artifact_reference(args.storage_estimate),
        fixed_size_inventory=artifact_reference(args.fixed_size_inventory),
        action_proofs={
            name: artifact_reference(Path(path)) for name, path in raw.items()
        },
    )
    publish(args.output, bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
