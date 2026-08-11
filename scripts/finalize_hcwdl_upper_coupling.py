#!/usr/bin/env python3
"""Finalize one authenticated HCWDL-UJ coupling stage."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_workflow import HomotopyWorkflow  # noqa: E402
TASKS=("train_base_manifest","switch_calibration","validation_base_manifest","train_manifest","validation_manifest","coupling_audit","coupling_lock","cache_miniature","endpoint_equality_lock","graph_recipe_lock")
def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--campaign-spec",type=Path,required=True); p.add_argument("--task",choices=TASKS,required=True); a=p.parse_args(); HomotopyWorkflow(load_json(a.campaign_spec),repository=ROOT).run(a.task); return 0
if __name__=="__main__": raise SystemExit(main())
