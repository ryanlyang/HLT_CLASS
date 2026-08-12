#!/usr/bin/env python3
"""Create the immutable 300k direct offline-to-HLT KD campaign."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_direct_offline_kd_campaign import create_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign-spec", type=Path, required=True)
    parser.add_argument("--prerequisite-bundle", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = create_campaign(
        parent_campaign_spec=args.parent_campaign_spec,
        prerequisite_bundle=args.prerequisite_bundle,
        campaign_root=args.campaign_root, project_dir=args.project_dir,
        source_commit=args.source_commit,
        authorize_live_submission=args.authorize_live_submission,
        authorization_phrase=args.authorization_phrase,
    )
    print(spec["content_hash"])
    print("fits=5")
    print(f"campaign_root={spec['campaign_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
