#!/usr/bin/env python3
"""Run one source-pinned HCWDL-MHPE-FULL task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.data.cache_contracts import write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_workflow import MhpeWorkflow  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();spec=load_json(a.spec);validate_source_checkout(ROOT,expected_commit=spec["source_commit"]);result=MhpeWorkflow(spec).run(a.task,device=a.device);root=Path(spec["campaign_root"])
 if a.task.startswith("train_"):
  node=a.task.removeprefix("train_");report_path=root/"training"/node/"training_report.json";report=load_json(report_path);outputs=[report_path,root/"training"/node/"hcwdl_training_report.json",root/"reports/runtime"/f"{node}.json",root/"training"/node/report["selected_checkpoint"],root/"training"/node/report["final_checkpoint"]]
  if node=="U050_from_U000":outputs.extend([root/"targets/U050/train_all.npz",root/"targets/U050/train_all.json",root/"targets/U050/train_manifest.json"])
 elif a.task.startswith("ensemble_"):
  ensemble=a.task.removeprefix("ensemble_");outputs=[root/"reports"/f"{ensemble}_stage.json"]
  for temperature in ("T1","T2"):
   directory=root/"targets"/ensemble/temperature;outputs.append(directory/"lock.json")
   for role in ("train","validation"):outputs.extend([directory/f"{role}_all.npz",directory/f"{role}_all.json",directory/f"{role}_manifest.json"])
 elif a.task=="aggregate":outputs=[root/"reports/validation_aggregate.json"]
 elif a.task=="finalist_lock":outputs=[root/"locks/finalist_lock.json"]
 else:outputs=[root/"reports/campaign_complete.json"]
 outputs=list(dict.fromkeys(outputs));att=build_task_attestation(campaign_spec_sha256=spec["content_hash"],task_id=a.task,array_index=None,outputs=outputs);write_immutable_json(task_attestation_path(spec["campaign_root"],a.task,None),att);print(json.dumps({"task":a.task,"content_hash":result.get("content_hash"),"complete":True},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
