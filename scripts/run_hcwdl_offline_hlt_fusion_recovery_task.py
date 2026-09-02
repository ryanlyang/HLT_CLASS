#!/usr/bin/env python3
"""Run one restart-zero fusion recovery task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_bottleneck_operations import publish_output_inventory  # noqa:E402
from hlt_classification.scouting.hcwdl_offline_hlt_fusion_recovery import clean_incomplete_task_outputs,validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_offline_hlt_fusion_workflow import FusionWithdrawalWorkflow,task_outputs  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();recovery=load_json(a.recovery);validate_recovery(recovery);validate_source_checkout(ROOT,expected_commit=recovery["source_commit"]);spec=load_json(recovery["subject_spec_path"]);clean_incomplete_task_outputs(spec,a.task);result=FusionWithdrawalWorkflow(spec,recovery_spec_sha256=recovery["content_hash"],execution_source_commit=recovery["source_commit"]).run(a.task,device=a.device);outputs=task_outputs(spec,a.task);inventory=publish_output_inventory(root=spec["campaign_root"],task_id=a.task,array_index=None,outputs=outputs);outputs=[*outputs,inventory];att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=None,outputs=outputs);write_immutable_json(task_attestation_path(spec["campaign_root"],a.task,None),att);print(json.dumps({"task":a.task,"content_hash":result.get("content_hash"),"restart_from_zero":True},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
