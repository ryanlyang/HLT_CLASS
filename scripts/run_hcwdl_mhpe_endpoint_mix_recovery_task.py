#!/usr/bin/env python3
"""Run one exact endpoint-mixture recovery task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix import NODES  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix_recovery import validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix_workflow import EndpointMixWorkflow  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_task_attestation,task_attestation_path  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();recovery=load_json(a.recovery_spec);validate_recovery(recovery);validate_source_checkout(ROOT,expected_commit=recovery["source_commit"]);spec=load_json(recovery["campaign_spec_path"]);result=EndpointMixWorkflow(spec,recovery_spec_sha256=recovery["content_hash"]).run(a.task,device=a.device)
 # The workflow writes into the immutable original campaign namespace; attest the durable primary JSON for this recovery.
 root=Path(spec["campaign_root"])
 if a.task=="build_targets":
  outputs=[root/"reports/target_build.json",root/"targets/lock.json"]
  for node in NODES:
   for role in ("train","validation"):outputs += [root/"targets"/node/f"{role}_all.npz",root/"targets"/node/f"{role}_all.json",root/"targets"/node/f"{role}_manifest.json"]
 elif a.task.startswith("train_"):
  node=a.task.removeprefix("train_");report=root/"training"/node/"training_report.json";value=load_json(report);outputs=[report,root/"training"/node/"hcwdl_training_report.json",root/"reports/runtime"/f"{node}.json",root/"training"/node/value["selected_checkpoint"],root/"training"/node/value["final_checkpoint"]]
 elif a.task=="aggregate":outputs=[root/"reports/validation_aggregate.json"]
 else:outputs=[root/"reports/campaign_complete.json"]
 att=build_task_attestation(campaign_spec_sha256=recovery["content_hash"],task_id=a.task,array_index=None,outputs=outputs);write_immutable_json(task_attestation_path(recovery["recovery_root"],a.task,None),att);print(json.dumps({"task":a.task,"complete":True,"content_hash":result.get("content_hash")},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
