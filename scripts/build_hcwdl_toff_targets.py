#!/usr/bin/env python3
"""Build and lock the shared identity-ordered FP32 TOFF train logits."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_workflow import HomotopyWorkflow  # noqa: E402
def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--campaign-spec",type=Path,required=True); a=p.parse_args(); w=HomotopyWorkflow(load_json(a.campaign_spec),repository=ROOT); w.run("toff_target_shards"); w.run("toff_target_manifest"); w.run("toff_target_lock"); return 0
if __name__=="__main__": raise SystemExit(main())
