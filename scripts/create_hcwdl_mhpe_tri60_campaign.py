#!/usr/bin/env python3
"""Create the exact source-pinned HCWDL-MHPE TRI60 campaign."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_campaign import create_campaign  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--foundation-lock",type=Path,required=True);p.add_argument("--representation-recipe",type=Path,required=True);p.add_argument("--test-evidence",type=Path,required=True);p.add_argument("--installed-weaver-parity",type=Path,required=True);p.add_argument("--runtime-profile",type=Path,required=True);p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase")
 a=p.parse_args();validate_source_checkout(ROOT,expected_commit=a.source_commit)
 spec=create_campaign(foundation_lock=a.foundation_lock,representation_recipe=a.representation_recipe,test_evidence=a.test_evidence,installed_weaver_parity=a.installed_weaver_parity,runtime_profile=a.runtime_profile,campaign_root=a.campaign_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase)
 print(spec["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
