#!/usr/bin/env python3
"""Create the controlled four-spine campaign over a bottleneck foundation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_campaign import create_campaign  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--foundation-spec",type=Path,required=True);p.add_argument("--established-campaign-spec",type=Path,required=True);p.add_argument("--m0ce60-report",type=Path,required=True);p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();value=create_campaign(foundation_spec=a.foundation_spec,established_campaign_spec=a.established_campaign_spec,m0ce60_report=a.m0ce60_report,campaign_root=a.campaign_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase);print(value["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
