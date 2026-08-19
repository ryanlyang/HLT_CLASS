#!/usr/bin/env python3
"""Run one source-pinned D000 teacher-distance screen task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen_runner import D000ScheduleScreenWorkflow  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
def _outputs(root,task):
 if task.startswith("train_"):
  node=task.removeprefix("train_");directory=root/"training"/node;engine=load_json(directory/"training_report.json")
  return [directory/"training_report.json",directory/"horizon_checkpoints.json",directory/"screen_training_report.json",root/"reports/runtime"/f"{node}.json",directory/engine["selected_checkpoint"],directory/engine["final_checkpoint"],*[directory/row["checkpoint"] for row in engine["selection_horizon_checkpoints"]]]
 return [root/"reports"/("validation_aggregate.json" if task=="aggregate" else "campaign_complete.json")]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();spec=load_json(a.spec);validate_source_checkout(ROOT,expected_commit=spec["source_commit"]);result=D000ScheduleScreenWorkflow(spec).run(a.task,device=a.device);root=Path(spec["campaign_root"]);att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=None,outputs=_outputs(root,a.task));write_immutable_json(task_attestation_path(root,a.task,None),att);print(json.dumps({"task":a.task,"content_hash":result.get("content_hash"),"complete":True},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

