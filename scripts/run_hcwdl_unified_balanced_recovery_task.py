#!/usr/bin/env python3
"""Run one task from an authenticated HCWDL-UB recovery closure."""
from __future__ import annotations
import argparse,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_recovery import validate_recovery_spec  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_workflow import UnifiedBalancedArmWorkflow,UnifiedBalancedFoundationWorkflow  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--array-index",type=int);a=p.parse_args();recovery=load_json(a.recovery_spec);validate_recovery_spec(recovery);validate_source_checkout(ROOT,expected_commit=recovery["source_commit"])
 if a.task not in recovery["task_ids"]:raise PermissionError("task is outside HCWDL-UB recovery closure")
 scope=load_json(recovery["scope_spec_path"]);index=a.array_index
 if index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:index=int(os.environ["SLURM_ARRAY_TASK_ID"])
 workflow=UnifiedBalancedFoundationWorkflow(scope,producer_commit=recovery["source_commit"]) if scope["contract"].endswith("FOUNDATION_SPEC/v1") else UnifiedBalancedArmWorkflow(scope,producer_commit=recovery["source_commit"],recovery_context={"root":recovery["recovery_root"],"spec_sha256":recovery["content_hash"],"task_ids":recovery["task_ids"]});outputs=workflow.run(a.task,array_index=index)
 att=build_task_attestation(campaign_spec_sha256=recovery["content_hash"],task_id=a.task,array_index=index,outputs=outputs);write_immutable_json(task_attestation_path(recovery["recovery_root"],a.task,index),att);return 0
if __name__=="__main__":raise SystemExit(main())
