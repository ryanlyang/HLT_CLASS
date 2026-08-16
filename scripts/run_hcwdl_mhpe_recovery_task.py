#!/usr/bin/env python3
"""Run one exact HCWDL-MHPE recovery task against original artifacts."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_recovery import validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_workflow import MhpeWorkflow  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_contracts import campaign_profile  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_graph import direct_model_teacher  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--task",required=True);a=p.parse_args();recovery=load_json(a.recovery_spec);validate_recovery(recovery)
 if a.task not in recovery["recovery_tasks"]:raise PermissionError("task is outside MHPE recovery closure")
 validate_source_checkout(ROOT,expected_commit=recovery["source_commit"]);original=load_json(recovery["campaign_spec_path"]);MhpeWorkflow(original,recovery_spec_sha256=recovery["content_hash"]).run(a.task);campaign_root=Path(original["campaign_root"])
 if a.task.startswith("train_"):
  node=a.task.removeprefix("train_");report_path=campaign_root/"training"/node/"training_report.json";report=load_json(report_path);outputs=[report_path,campaign_root/"training"/node/"hcwdl_training_report.json",campaign_root/"reports/runtime"/f"{node}.json",campaign_root/"training"/node/report["selected_checkpoint"],campaign_root/"training"/node/report["final_checkpoint"]]
  direct=direct_model_teacher(campaign_profile(original))
  if node==f"{direct}_from_U000":outputs.extend([campaign_root/"targets"/direct/"train_all.npz",campaign_root/"targets"/direct/"train_all.json",campaign_root/"targets"/direct/"train_manifest.json"])
 elif a.task.startswith("ensemble_"):
  ensemble=a.task.removeprefix("ensemble_");outputs=[campaign_root/"reports"/f"{ensemble}_stage.json"]
  for temperature in ("T1","T2"):
   directory=campaign_root/"targets"/ensemble/temperature;outputs.append(directory/"lock.json")
   for role in ("train","validation"):outputs.extend([directory/f"{role}_all.npz",directory/f"{role}_all.json",directory/f"{role}_manifest.json"])
 elif a.task=="aggregate":outputs=[campaign_root/"reports/validation_aggregate.json"]
 elif a.task=="finalist_lock":outputs=[campaign_root/"locks/finalist_lock.json"]
 else:outputs=[campaign_root/"reports/campaign_complete.json"]
 outputs=list(dict.fromkeys(outputs));att=build_task_attestation(campaign_spec_sha256=recovery["content_hash"],task_id=a.task,array_index=None,outputs=outputs);write_immutable_json(task_attestation_path(recovery["recovery_root"],a.task,None),att);return 0
if __name__=="__main__":raise SystemExit(main())
