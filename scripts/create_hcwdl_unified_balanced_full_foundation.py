#!/usr/bin/env python3
"""Create or dry-run the all-mapped HCWDL-UB-FULL3 foundation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_unified_balanced_full_campaign import create_foundation  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parent-homotopy-spec",type=Path,required=True);p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase");p.add_argument("--dry-run",action="store_true");a=p.parse_args()
 spec=create_foundation(parent_homotopy_spec=a.parent_homotopy_spec,campaign_root=a.campaign_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase,publish=not a.dry_run)
 print(json.dumps({"foundation_spec_sha256":spec["content_hash"],"root":spec["campaign_root"],"role_counts":spec["role_counts"],"training_passes":20,"fit_count":38,"task_count":len(spec["tasks"]),"published":not a.dry_run},indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
