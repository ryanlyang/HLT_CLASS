#!/usr/bin/env python3
"""Run one immutable full-cardinality foundation task or array element."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_bottleneck_operations import publish_output_inventory  # noqa:E402
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_foundation_workflow import FullCardinalityFoundationWorkflow  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--task",required=True);a=p.parse_args();spec=load_json(a.spec);validate_source_checkout(ROOT,expected_commit=spec["source_commit"]);raw=os.environ.get("SLURM_ARRAY_TASK_ID");index=None if raw is None else int(raw);outputs=FullCardinalityFoundationWorkflow(spec).run(a.task,array_index=index);inventory=publish_output_inventory(root=spec["campaign_root"],task_id=a.task,array_index=index,outputs=outputs);outputs=[*outputs,inventory];att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=index,outputs=outputs);write_immutable_json(task_attestation_path(spec["campaign_root"],a.task,index),att);print(json.dumps({"task":a.task,"array_index":index,"outputs":[str(x) for x in outputs]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
