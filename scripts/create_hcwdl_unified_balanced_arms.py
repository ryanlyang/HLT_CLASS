#!/usr/bin/env python3
"""Create the six isolated HCWDL-UB arm specifications after foundation lock."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_unified_balanced_campaign import create_arm_specs  # noqa: E402
def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--foundation-lock",type=Path,required=True); p.add_argument("--arms-root",type=Path,required=True); p.add_argument("--project-dir",type=Path,required=True); p.add_argument("--source-commit",required=True); p.add_argument("--authorize-live-submission",action="store_true"); p.add_argument("--authorization-phrase"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
    specs=create_arm_specs(foundation_lock=a.foundation_lock,arms_root=a.arms_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase,publish=not a.dry_run)
    print(json.dumps({arm:{"root":s["campaign_root"],"spec_sha256":s["content_hash"],"fits":s["node_count"]} for arm,s in specs.items()},indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
