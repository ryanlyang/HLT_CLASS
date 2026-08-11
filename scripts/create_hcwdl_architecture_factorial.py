#!/usr/bin/env python3
"""Create an immutable HCWDL input-by-architecture factorial campaign."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_architecture_ablation import AUTHORIZATION_PHRASE  # noqa: E402
from hlt_classification.scouting.hcwdl_architecture_campaign import create_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign-spec", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = create_campaign(
        parent_campaign_spec=args.parent_campaign_spec,
        campaign_root=args.campaign_root, project_dir=args.project_dir,
        source_commit=args.source_commit,
        authorize_live_submission=args.authorize_live_submission,
        authorization_phrase=args.authorization_phrase,
    )
    print(f"Campaign: {spec['content_hash']}")
    print(f"Mode: {spec['mode']}")
    print("Fits: 4 (+ imported native TOFF reference)")
    print(f"Authorized: {spec['live_submission_authorized']}")
    if not spec["live_submission_authorized"]:
        print(f"Creation authorization phrase: {AUTHORIZATION_PHRASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
