#!/usr/bin/env python3
"""Run explicitly authorized HCWDL-MHPE sealed final evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_mhpe_final import run_sealed_final_evaluation  # noqa:E402
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-spec",type=Path,required=True);p.add_argument("--finalist-lock",type=Path,required=True);p.add_argument("--execution-lock",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda");a=p.parse_args();spec=load_json(a.campaign_spec);validate_source_checkout(ROOT,expected_commit=spec["source_commit"]);run_sealed_final_evaluation(campaign_spec_path=a.campaign_spec,finalist_lock_path=a.finalist_lock,execution_lock_path=a.execution_lock,output=a.output,device=a.device);return 0
if __name__=="__main__":raise SystemExit(main())
