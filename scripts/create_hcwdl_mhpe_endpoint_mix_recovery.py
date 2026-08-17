#!/usr/bin/env python3
"""Create an exact endpoint-mixture failed/downstream recovery."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix_recovery import create_recovery  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-spec",type=Path,required=True);p.add_argument("--submission-ledger",type=Path,required=True);p.add_argument("--monitor-report",type=Path,required=True);p.add_argument("--recovery-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--authorization-phrase",required=True);a=p.parse_args();result=create_recovery(campaign_spec=a.campaign_spec,submission_ledger=a.submission_ledger,monitor_report=a.monitor_report,recovery_root=a.recovery_root,project_dir=a.project_dir,source_commit=a.source_commit,authorization_phrase=a.authorization_phrase);print(result["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
