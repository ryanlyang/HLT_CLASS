#!/usr/bin/env python3
"""Bind one completed non-final HCWDL-RKD action to post-job evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_nonfinal_acceptance import (
    ACTION_IDS,
    build_nonfinal_acceptance_action_result,
    nonfinal_acceptance_action_result_path,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--action", choices=ACTION_IDS, required=True)
    parser.add_argument("--scheduler-evidence", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    args = parser.parse_args()
    authority_value = artifact(args.authority)
    authority = artifact_reference(args.authority)
    result = build_nonfinal_acceptance_action_result(
        authority=authority,
        action_id=args.action,
        scheduler_evidence=artifact_reference(args.scheduler_evidence),
        execution_receipt=artifact_reference(args.execution_receipt),
        require_genuine=True,
    )
    publish(
        nonfinal_acceptance_action_result_path(
            authority_value, action_id=args.action,
        ),
        result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
