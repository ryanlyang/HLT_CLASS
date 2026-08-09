#!/usr/bin/env python3
"""Capture one completed non-final action's exact sacct evidence on Tigris."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import REPO_ROOT, artifact, publish
from hlt_classification.scouting.hcwdl_representation_nonfinal_acceptance import (
    ACTION_IDS,
    nonfinal_acceptance_raw_sacct_path,
    nonfinal_acceptance_scheduler_evidence_path,
    validate_nonfinal_acceptance_authority_static,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    capture_nonfinal_acceptance_scheduler_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--action", choices=ACTION_IDS, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    args = parser.parse_args()

    authority = artifact(args.authority)
    validate_nonfinal_acceptance_authority_static(
        authority, project_dir=REPO_ROOT,
    )
    action = authority["actions"][args.action]
    evidence = capture_nonfinal_acceptance_scheduler_evidence(
        authority_sha256=authority["content_hash"],
        action_id=args.action,
        job_id=args.job_id,
        raw_accounting_output=nonfinal_acceptance_raw_sacct_path(
            authority, action_id=args.action,
        ),
        resource_class=action["resource_class"],
        source_commit=authority["source_commit"],
        representation_recipe_sha256=authority[
            "representation_recipe_sha256"
        ],
        worker_role=action["worker_role"],
        worker=authority["workers"][action["worker_role"]],
        request=authority["resources"][action["resource_class"]],
    )
    publish(
        nonfinal_acceptance_scheduler_evidence_path(
            authority, action_id=args.action,
        ),
        evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
