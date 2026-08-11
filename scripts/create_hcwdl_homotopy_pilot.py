#!/usr/bin/env python3
"""Create an immutable HCWDL-UJ smoke or 300k validation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_campaign import (  # noqa: E402
    AUTHORIZATION_PHRASE, build_command_plan, create_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign-spec", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--weaver-parity", type=Path, required=True)
    parser.add_argument("--dense-d0-report", type=Path)
    parser.add_argument("--contextual-report", type=Path, action="append", default=[])
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    profile = None if args.resource_profile is None else load_json(args.resource_profile)
    spec = create_campaign(
        parent_campaign_spec=args.parent_campaign_spec,
        campaign_root=args.campaign_root, project_dir=args.project_dir,
        source_commit=args.source_commit, weaver_parity=args.weaver_parity,
        dense_d0_report=args.dense_d0_report,
        contextual_reports=args.contextual_report,
        resource_profile=profile,
        authorize_live_submission=args.authorize_live_submission,
        authorization_phrase=args.authorization_phrase,
    )
    plan = build_command_plan(spec)
    write_immutable_json(args.campaign_root / "command_plan.json", plan)
    print(f"Campaign: {spec['content_hash']}")
    print(f"Mode: {spec['mode']}")
    print(f"Tasks: {len(spec['tasks'])}")
    print(f"Authorized: {spec['live_submission_authorized']}")
    if not spec["live_submission_authorized"]:
        print(f"Creation authorization phrase: {AUTHORIZATION_PHRASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
