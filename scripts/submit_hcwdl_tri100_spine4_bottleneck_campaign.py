#!/usr/bin/env python3
"""Dry-run or submit the exact bottleneck four-spine DAG."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag  # noqa:E402
from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_campaign import SUBMISSION_PHRASE,validate_campaign  # noqa:E402
from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_contracts import PLAN_CONTRACT,validate_artifact  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();spec=load_json(a.spec);validate_campaign(spec,executable=a.execute);plan=load_json(Path(spec["campaign_root"])/"command_plan.json");validate_artifact(plan,contract=PLAN_CONTRACT)
 if a.execute:
  if a.authorization_phrase!=SUBMISSION_PHRASE:raise PermissionError("bottleneck submission phrase differs")
  if ROOT.resolve()!=Path(spec["project_dir"]).resolve():raise PermissionError("bottleneck submitter is outside bound worktree")
  validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
 submit_exact_dag(identity=spec["content_hash"],plan=plan,output=a.output,canonical_dry_run=Path(spec["campaign_root"])/"dry_run_submission_ledger.json",execute=a.execute);return 0
if __name__=="__main__":raise SystemExit(main())
