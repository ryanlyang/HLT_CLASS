#!/usr/bin/env python3
"""Run one exact TRI60 dense-extension task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_workflow import DenseWorkflow,task_outputs  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();spec=load_json(a.spec);validate_source_checkout(ROOT,expected_commit=spec["source_commit"]);result=DenseWorkflow(spec).run(a.task,device=a.device);outputs=task_outputs(spec,a.task);att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=None,outputs=outputs);write_immutable_json(task_attestation_path(spec["campaign_root"],a.task,None),att);print(json.dumps({"task":a.task,"content_hash":result.get("content_hash"),"complete":True},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
