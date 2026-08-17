#!/usr/bin/env python3
"""Run one source-pinned refined-continuation recovery task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_refined_recovery import validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_refined_runner import RefinedContinuationWorkflow  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402

def _outputs(root,task):
 if task.startswith("train_"):
  node=task.removeprefix("train_");directory=root/"training"/node;report=load_json(directory/"training_report.json");return [directory/"training_report.json",directory/"hcwdl_training_report.json",directory/report["selected_checkpoint"],directory/report["final_checkpoint"],root/"reports/runtime"/f"{node}.json"]
 if task.startswith("ensemble_"):
  ensemble=task.removeprefix("ensemble_");paths=[root/"reports"/f"{ensemble}_stage.json",root/"targets"/ensemble/"T1"/"lock.json"]
  for role in ("train","validation"):paths += [root/"targets"/ensemble/"T1"/f"{role}_all.npz",root/"targets"/ensemble/"T1"/f"{role}_all.json",root/"targets"/ensemble/"T1"/f"{role}_manifest.json"]
  return paths
 return [root/"reports"/("validation_aggregate.json" if task=="aggregate" else "campaign_complete.json")]
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();recovery=load_json(a.recovery_spec);validate_recovery(recovery)
 if a.task not in recovery["recovery_tasks"]:raise PermissionError("task is outside refined-continuation recovery closure")
 validate_source_checkout(ROOT,expected_commit=recovery["source_commit"]);spec=load_json(recovery["campaign_spec_path"]);result=RefinedContinuationWorkflow(spec,recovery_spec_sha256=recovery["content_hash"]).run(a.task,device=a.device);root=Path(spec["campaign_root"]);att=build_task_attestation(campaign_spec_sha256=recovery["content_hash"],task_id=a.task,array_index=None,outputs=_outputs(root,a.task));write_immutable_json(task_attestation_path(recovery["recovery_root"],a.task,None),att);print(json.dumps({"task":a.task,"content_hash":result.get("content_hash"),"complete":True},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
