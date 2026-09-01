#!/usr/bin/env python3
"""Create the isolated persistent-HLT MT20 four-spine campaign."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_tri100_spine4_mt20_campaign import create_campaign  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--foundation-spec",type=Path,required=True);p.add_argument("--immediate-campaign-spec",type=Path,required=True);p.add_argument("--m0ce60-report",type=Path,required=True);p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();spec=create_campaign(foundation_spec=a.foundation_spec,immediate_campaign_spec=a.immediate_campaign_spec,m0ce60_report=a.m0ce60_report,campaign_root=a.campaign_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase);print(spec["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
