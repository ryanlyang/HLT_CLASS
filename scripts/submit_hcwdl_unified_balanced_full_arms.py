#!/usr/bin/env python3
"""Dry-run or submit the three independent HCWDL-UB-FULL3 arms."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,validate_content_hash  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_full_campaign import ARMS_SUBMISSION_PHRASE,validate_arm_campaign  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_full_contracts import SWEEP_CONTRACT  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_full_graph import ARM_IDS  # noqa:E402
PHRASE="SUBMIT HCWDL UB FULL3 THREE ARMS PARALLEL EXACT LEDGERS"
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arms-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();rows=[];source=None;lock_hash=None
 sweep=load_json(a.arms_root/"recipe_sweep.json");validate_content_hash(sweep,expected_contract=SWEEP_CONTRACT,expected_schema_version=1)
 if sweep.get("arm_order")!=list(ARM_IDS):raise ValueError("HCWDL-UB-FULL3 sweep arm order differs")
 for arm in ARM_IDS:
  path=(a.arms_root/arm/"arm_spec.json").resolve();spec=load_json(path);validate_arm_campaign(spec,executable=a.execute)
  if spec["content_hash"]!=sweep["arm_spec_sha256"][arm]:raise ValueError("HCWDL-UB-FULL3 arm/sweep differs")
  source=source or spec["source_commit"];lock_hash=lock_hash or spec["foundation_lock_sha256"]
  if spec["source_commit"]!=source or spec["foundation_lock_sha256"]!=lock_hash:raise ValueError("HCWDL-UB-FULL3 arm lineage differs")
  rows.append((arm,path,spec))
 if sweep.get("foundation_lock_sha256")!=lock_hash:raise ValueError("HCWDL-UB-FULL3 sweep/foundation lock differs")
 if a.execute:
  if a.authorization_phrase!=PHRASE:raise PermissionError("HCWDL-UB-FULL3 three-arm phrase differs")
  if ROOT.resolve()!=Path(rows[0][2]["project_dir"]).resolve():raise PermissionError("submitter is outside bound worktree")
  validate_source_checkout(ROOT,expected_commit=source)
 submitter=ROOT/"scripts/submit_hcwdl_unified_balanced_full.py"
 for arm,path,_ in rows:
  command=[sys.executable,"-s",str(submitter),"--spec",str(path),"--output",str(a.output_root/arm/"submission_ledger.json")]
  if a.execute:command.extend(("--execute","--authorization-phrase",ARMS_SUBMISSION_PHRASE))
  subprocess.run(command,cwd=ROOT,check=True)
 print("HCWDL-UB-FULL3 arms submitted independently." if a.execute else "HCWDL-UB-FULL3 arm dry runs complete.");return 0
if __name__=="__main__":raise SystemExit(main())
