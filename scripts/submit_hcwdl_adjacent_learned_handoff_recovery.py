#!/usr/bin/env python3
"""Dry-run or submit exact learned-handoff recovery."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_contracts import RECOVERY_PLAN_CONTRACT,validate_artifact  # noqa:E402
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_recovery import RECOVERY_SUBMISSION_PHRASE,validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();recovery=load_json(a.recovery_spec);validate_recovery(recovery);plan=load_json(Path(recovery["recovery_root"])/"command_plan.json");validate_artifact(plan,contract=RECOVERY_PLAN_CONTRACT)
 if a.execute:
  if a.authorization_phrase!=RECOVERY_SUBMISSION_PHRASE:raise PermissionError("learned-handoff recovery submission phrase differs")
  if ROOT.resolve()!=Path(recovery["project_dir"]).resolve():raise PermissionError("learned-handoff recovery submitter is outside bound worktree")
  validate_source_checkout(ROOT,expected_commit=recovery["source_commit"])
 submit_exact_dag(identity=recovery["content_hash"],plan=plan,output=a.output,canonical_dry_run=Path(recovery["recovery_root"])/"dry_run_submission_ledger.json",execute=a.execute);return 0
if __name__=="__main__":raise SystemExit(main())
