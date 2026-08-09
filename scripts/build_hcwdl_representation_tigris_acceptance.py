#!/usr/bin/env python3
"""Build the HCWDL-RKD Tigris acceptance gate from a validated evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_tigris_acceptance,
)
from hlt_classification.scouting.hcwdl_representation_resources import artifact_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = artifact(args.evidence_bundle)
    acceptance = build_tigris_acceptance(
        evidence_bundle=artifact_reference(args.evidence_bundle),
        source_commit=bundle["source_commit"],
        representation_recipe_sha256=bundle["representation_recipe_sha256"],
        resource_profile_sha256=bundle["resource_profile_sha256"],
        storage_estimate_sha256=bundle["storage_estimate_sha256"],
        fixed_size_inventory_sha256=bundle["fixed_size_inventory_sha256"],
    )
    publish(args.output, acceptance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
