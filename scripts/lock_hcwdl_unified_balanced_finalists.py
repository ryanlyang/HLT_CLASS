#!/usr/bin/env python3
"""Lock the top two validation-only HCWDL-UB recipe arms and HLT finalists."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_reporting import build_finalist_lock  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--sweep-aggregate",type=Path,required=True);p.add_argument("--foundation-lock",type=Path,required=True);p.add_argument("--arms-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();lock=build_finalist_lock(sweep_aggregate_path=a.sweep_aggregate,foundation_lock_path=a.foundation_lock,arms_root=a.arms_root);write_immutable_json(a.output,lock);print("Selected arms:",*lock["selected_arms"]);return 0
if __name__=="__main__":raise SystemExit(main())
