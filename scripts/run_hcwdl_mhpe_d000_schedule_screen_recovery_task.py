#!/usr/bin/env python3
"""Run one source-pinned D000 teacher-distance screen recovery task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen_recovery import validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen_runner import D000ScheduleScreenWorkflow  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
from run_hcwdl_mhpe_d000_schedule_screen_task import _outputs  # noqa:E402
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();recovery=load_json(a.recovery_spec);validate_recovery(recovery);validate_source_checkout(ROOT,expected_commit=recovery["source_commit"]);spec=load_json(recovery["campaign_spec_path"]);result=D000ScheduleScreenWorkflow(spec,verify_source_tree=False).run(a.task,device=a.device);root=Path(recovery["recovery_root"]);att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=None,outputs=_outputs(Path(spec["campaign_root"]),a.task),recovery_spec_sha256=recovery["content_hash"]);write_immutable_json(task_attestation_path(root,a.task,None),att);print(json.dumps({"task":a.task,"content_hash":result.get("content_hash"),"complete":True},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

