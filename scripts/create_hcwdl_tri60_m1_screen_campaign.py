#!/usr/bin/env python3
"""Create the standalone full-data TRI60 M1 compression screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.scouting.hcwdl_tri60_m1_screen_campaign import create_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign-spec", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    value = create_campaign(
        source_campaign_spec=args.source_campaign_spec,
        campaign_root=args.campaign_root, project_dir=args.project_dir,
        source_commit=args.source_commit,
        authorize_live_submission=args.authorize_live_submission,
        authorization_phrase=args.authorization_phrase,
    )
    print(value["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
