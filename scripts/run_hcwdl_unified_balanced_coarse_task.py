#!/usr/bin/env python3
"""Run one source-pinned HCWDL-UB-FULLCOARSE3 task."""
from __future__ import annotations
import argparse,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_coarse_workflow import UnifiedBalancedCoarseArmWorkflow  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arm-spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--array-index",type=int);a=p.parse_args();spec=load_json(a.arm_spec);validate_source_checkout(ROOT,expected_commit=spec["source_commit"]);index=a.array_index
 if index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:index=int(os.environ["SLURM_ARRAY_TASK_ID"])
 outputs=UnifiedBalancedCoarseArmWorkflow(spec).run(a.task,array_index=index);att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=index,outputs=outputs);write_immutable_json(task_attestation_path(spec["campaign_root"],a.task,index),att);return 0
if __name__=="__main__":raise SystemExit(main())
