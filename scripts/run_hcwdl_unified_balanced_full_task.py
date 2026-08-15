#!/usr/bin/env python3
"""Run one source-pinned HCWDL-UB-FULL3 foundation or arm task."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_full_workflow import UnifiedBalancedFullArmWorkflow,UnifiedBalancedFullFoundationWorkflow  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument("--foundation-spec",type=Path);g.add_argument("--arm-spec",type=Path);p.add_argument("--task",required=True);p.add_argument("--array-index",type=int);a=p.parse_args();index=a.array_index
 if index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:index=int(os.environ["SLURM_ARRAY_TASK_ID"])
 path=a.foundation_spec or a.arm_spec;spec=load_json(path);validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
 workflow=UnifiedBalancedFullFoundationWorkflow(spec) if a.foundation_spec else UnifiedBalancedFullArmWorkflow(spec);outputs=workflow.run(a.task,array_index=index)
 att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=index,outputs=outputs);write_immutable_json(task_attestation_path(spec["campaign_root"],a.task,index),att);return 0
if __name__=="__main__":raise SystemExit(main())
