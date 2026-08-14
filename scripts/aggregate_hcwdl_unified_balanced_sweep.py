#!/usr/bin/env python3
"""Build the read-only six-arm HCWDL-UB validation aggregate."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_reporting import build_sweep_aggregate  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recipe-sweep",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();value=build_sweep_aggregate(a.recipe_sweep);write_immutable_json(a.output,value)
 for i,row in enumerate(value["rankings"],1):print(f"{i}. {row['arm_id']}: D0F AUC={row['d0f_macro_ovr_auc']:.6f} J100 AUC={row['j100_macro_ovr_auc']:.6f} M1F AUC={row['m1f_macro_ovr_auc']:.6f}")
 return 0
if __name__=="__main__":raise SystemExit(main())
