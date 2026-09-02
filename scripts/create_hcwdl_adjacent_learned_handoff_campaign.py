#!/usr/bin/env python3
"""Create the exact 25-fit Strategy-B learned fusion campaign."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_campaign import create_campaign  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--source-campaign-spec",type=Path,required=True);p.add_argument("--u100-training-report",type=Path,required=True);p.add_argument("--u100-selected-checkpoint",type=Path,required=True);p.add_argument("--m0ce60-training-report",type=Path,required=True);p.add_argument("--pure-offline-u000-training-report",type=Path,required=True);p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase")
 a=p.parse_args();value=create_campaign(source_campaign_spec=a.source_campaign_spec,u100_training_report=a.u100_training_report,u100_selected_checkpoint=a.u100_selected_checkpoint,m0ce60_training_report=a.m0ce60_training_report,pure_offline_u000_training_report=a.pure_offline_u000_training_report,campaign_root=a.campaign_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase);print(value["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
