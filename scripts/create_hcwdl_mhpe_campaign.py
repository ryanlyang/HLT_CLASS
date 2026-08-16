#!/usr/bin/env python3
"""Create an exact source-pinned HCWDL-MHPE campaign profile."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_mhpe_campaign import create_campaign  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_graph import SUPPORTED_PROFILES  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--foundation-lock",type=Path,required=True);p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--recipe-profile",choices=SUPPORTED_PROFILES,default="C25P75");p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase");p.add_argument("--dry-run",action="store_true");a=p.parse_args();spec=create_campaign(foundation_lock=a.foundation_lock,campaign_root=a.campaign_root,project_dir=a.project_dir,source_commit=a.source_commit,recipe_profile=a.recipe_profile,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase,publish=not a.dry_run);print(json.dumps({"root":spec["campaign_root"],"spec_sha256":spec["content_hash"],"recipe_profile":a.recipe_profile,"fresh_fits":16,"tasks":len(spec["tasks"])},indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
