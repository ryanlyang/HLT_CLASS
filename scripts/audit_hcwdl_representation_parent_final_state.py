#!/usr/bin/env python3
"""Audit immutable parent final artifacts before freezing an RKD disposition."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_shared_final import audit_parent_final_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        action="append",
        default=[],
        help="immutable legacy final artifact to audit; repeat for every candidate",
    )
    parser.add_argument("--parent-campaign-sha256", required=True)
    parser.add_argument("--exploratory-campaign-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit_parent_final_state(
        candidate_artifacts=[artifact(path) for path in args.candidate_artifact],
        parent_campaign_sha256=args.parent_campaign_sha256,
        exploratory_campaign_sha256=args.exploratory_campaign_sha256,
    )
    publish(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
