#!/usr/bin/env python3
"""Evaluate fixed D000E/M1 probability blends on validation only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_endpoint_refinement import evaluate_endpoint_refinement_blends  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--producer-commit",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();validate_source_checkout(ROOT,expected_commit=a.producer_commit);evaluate_endpoint_refinement_blends(campaign_spec_path=a.campaign_spec,output=a.output,producer_commit=a.producer_commit,device=a.device);return 0
if __name__=="__main__":raise SystemExit(main())
