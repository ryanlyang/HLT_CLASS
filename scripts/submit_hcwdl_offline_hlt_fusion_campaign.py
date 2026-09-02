#!/usr/bin/env python3
"""Dry-run or submit a staged fusion-withdrawal DAG."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag  # noqa:E402
from hlt_classification.scouting.hcwdl_offline_hlt_fusion_campaign import SUBMISSION_PHRASE,validate_campaign  # noqa:E402
from hlt_classification.scouting.hcwdl_offline_hlt_fusion_contracts import PLAN_CONTRACT,validate_artifact  # noqa:E402
from hlt_classification.scouting.hcwdl_offline_hlt_fusion_runner import validate_capacity_audit,validate_execution_acceptance  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import task_attestation_path,validate_task_attestation  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--stage",choices=("gate","science"),required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();spec=load_json(a.spec);validate_campaign(spec,executable=a.execute);root=Path(spec["campaign_root"]);plan=load_json(root/f"{a.stage}_command_plan.json");validate_artifact(plan,contract=PLAN_CONTRACT)
 if a.execute:
  if a.authorization_phrase!=SUBMISSION_PHRASE:raise PermissionError("fusion submission phrase differs")
  if ROOT.resolve()!=Path(spec["project_dir"]).resolve():raise PermissionError("fusion submitter is outside bound worktree")
  validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
  if a.stage=="science":
   capacity=load_json(spec["artifact_paths"]["capacity_audit"]);acceptance=load_json(spec["artifact_paths"]["execution_acceptance"]);capacity_hash=validate_capacity_audit(spec,capacity);validate_execution_acceptance(spec,acceptance)
   if acceptance.get("parents",{}).get("campaign_spec")!=spec["content_hash"] or acceptance.get("parents",{}).get("capacity_audit")!=capacity_hash:raise PermissionError("fusion science gates differ")
   attestation=load_json(task_attestation_path(root,"preflight",None));validate_task_attestation(attestation,campaign_spec_sha256=spec["content_hash"],task_id="preflight",array_index=None)
 submit_exact_dag(identity=spec["content_hash"],plan=plan,output=a.output,canonical_dry_run=root/f"{a.stage}_dry_run_submission_ledger.json",execute=a.execute);return 0
if __name__=="__main__":raise SystemExit(main())
