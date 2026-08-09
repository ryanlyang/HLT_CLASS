#!/usr/bin/env python3
"""Execute one bounded non-final action and publish its worker receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import REPO_ROOT, artifact, publish
from hlt_classification.scouting.hcwdl_representation_nonfinal_acceptance import (
    ACTION_IDS,
    build_nonfinal_acceptance_execution_receipt,
    nonfinal_acceptance_execution_receipt_path,
)
from hlt_classification.scouting.hcwdl_representation_nonfinal_runtime import (
    execute_nonfinal_production_action,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--action", choices=ACTION_IDS, required=True)
    parser.add_argument("--deterministic-worker", action="store_true")
    args = parser.parse_args()

    authority = artifact(args.authority)
    result = execute_nonfinal_production_action(
        authority=authority,
        authority_path=args.authority,
        action_id=args.action,
        project_dir=REPO_ROOT,
        deterministic_worker=args.deterministic_worker,
    )
    receipt = build_nonfinal_acceptance_execution_receipt(
        authority=artifact_reference(args.authority),
        action_id=args.action,
        semantic_outputs=result.semantic_outputs,
        dependency_action_results=result.dependency_action_results,
        scheduler_job_id=result.scheduler_job_id,
        project_dir=REPO_ROOT,
        local_fixture=False,
    )
    publish(
        nonfinal_acceptance_execution_receipt_path(
            authority, action_id=args.action,
        ),
        receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
