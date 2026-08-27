#!/usr/bin/env python3
"""Create the source-pinned TRI60 M1 greedy ensemble campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.scouting.hcwdl_tri60_m1_greedy_ensemble_campaign import (  # noqa: E402
    create_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-campaign-spec", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = create_campaign(
        screen_campaign_spec=args.screen_campaign_spec,
        campaign_root=args.campaign_root, project_dir=args.project_dir,
        source_commit=args.source_commit,
        authorize_live_submission=args.authorize_live_submission,
        authorization_phrase=args.authorization_phrase,
    )
    print(spec["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
