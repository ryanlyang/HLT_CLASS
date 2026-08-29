#!/usr/bin/env python3
"""Create the isolated full-cardinality bottleneck foundation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_foundation_campaign import create_foundation  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-campaign-spec",type=Path,required=True);p.add_argument("--foundation-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorize-live-submission",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();value=create_foundation(source_campaign_spec=a.source_campaign_spec,foundation_root=a.foundation_root,project_dir=a.project_dir,source_commit=a.source_commit,authorize_live_submission=a.authorize_live_submission,authorization_phrase=a.authorization_phrase);print(value["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
