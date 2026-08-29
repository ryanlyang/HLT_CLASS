#!/usr/bin/env python3
"""Dry-run or submit the exact bottleneck-foundation DAG."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,validate_content_hash  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag  # noqa:E402
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_foundation_campaign import PLAN_CONTRACT,SCHEMA_VERSION,SUBMISSION_PHRASE,validate_foundation  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();spec=load_json(a.spec);validate_foundation(spec,executable=a.execute);plan=load_json(Path(spec["campaign_root"])/"command_plan.json");validate_content_hash(plan,expected_contract=PLAN_CONTRACT,expected_schema_version=SCHEMA_VERSION)
 if a.execute:
  if a.authorization_phrase!=SUBMISSION_PHRASE:raise PermissionError("foundation submission phrase differs")
  if ROOT.resolve()!=Path(spec["project_dir"]).resolve():raise PermissionError("foundation submitter is outside bound worktree")
  validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
 submit_exact_dag(identity=spec["content_hash"],plan=plan,output=a.output,canonical_dry_run=Path(spec["campaign_root"])/"dry_run_submission_ledger.json",execute=a.execute);return 0
if __name__=="__main__":raise SystemExit(main())
