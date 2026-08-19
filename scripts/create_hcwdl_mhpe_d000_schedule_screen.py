#!/usr/bin/env python3
"""Create the authenticated full-data D000 teacher-distance schedule screen."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen import create_campaign  # noqa:E402
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-campaign-spec",type=Path,required=True);p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase");p.add_argument("--operational-evidence-waiver",action="store_true");p.add_argument("--waiver-phrase");a=p.parse_args();spec=create_campaign(source_campaign_spec=a.source_campaign_spec,campaign_root=a.campaign_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase,authorize_waiver=a.operational_evidence_waiver,waiver_phrase=a.waiver_phrase);print(spec["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())

