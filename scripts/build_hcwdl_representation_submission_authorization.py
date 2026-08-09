#!/usr/bin/env python3
"""Bind explicit human authorization to one strict HCWDL-RKD candidate audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_campaign import (
    AUTHORIZATION_PHRASE,
    build_submission_authorization,
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_candidate import (
    validate_executable_candidate_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-spec", type=Path, required=True)
    parser.add_argument("--executable-candidate-audit", type=Path, required=True)
    parser.add_argument(
        "--authorization-phrase",
        required=True,
        help=f"must equal exactly: {AUTHORIZATION_PHRASE}",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    planning_spec = artifact(args.planning_spec)
    validate_campaign_spec(planning_spec, executable=False)
    candidate = artifact(args.executable_candidate_audit)
    candidate_hash = validate_executable_candidate_audit(
        candidate, campaign_spec=planning_spec,
    )
    authorization = build_submission_authorization(
        mode=str(planning_spec["mode"]),
        source_commit=str(planning_spec["source_commit"]),
        command_plan_sha256=str(planning_spec["command_plan_sha256"]),
        executable_candidate_audit_sha256=candidate_hash,
        resource_profile_sha256=str(planning_spec["resource_profile_sha256"]),
        storage_estimate_sha256=str(planning_spec["storage_estimate_sha256"]),
        tigris_acceptance_sha256=str(planning_spec["tigris_acceptance_sha256"]),
        parent_import_sha256=str(planning_spec["parent_import_sha256"]),
        representation_recipe_sha256=str(
            planning_spec["representation_recipe_sha256"]
        ),
        disposition_sha256=str(planning_spec["disposition_sha256"]),
        authorization_phrase=args.authorization_phrase,
    )
    publish(args.output, authorization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
