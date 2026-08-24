#!/usr/bin/env python3
"""Dry-run or submit a TRI60 dense-extension recovery ledger."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_contracts import PLAN_CONTRACT,validate_artifact  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_recovery import validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_event,build_submission_ledger,validate_submission_ledger  # noqa:E402
PHRASE="SUBMIT HCWDL TRI60 DENSE EXTENSION RECOVERY EXACT LEDGER"
def resolved(row,jobs):
 value=list(map(str,row["command"]))
 for index,item in enumerate(value):
  for parent in row["dependencies"]:
   if parent not in jobs:raise ValueError("dense recovery dependency is absent")
   item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
  value[index]=item
 if any("${JOB_" in item for item in value):raise ValueError("dense recovery dependency unresolved")
 return value
def load_journal(path,recovery,plan):
 paths=sorted(path.glob("*.json")) if path.is_dir() else [];events=[];jobs={}
 if len(paths)>len(plan["commands"]):raise ValueError("dense recovery journal has excess events")
 for sequence,event_path in enumerate(paths):
  row=plan["commands"][sequence];command=resolved(row,jobs);event=load_json(event_path);expected=build_submission_event(campaign_spec_sha256=recovery["content_hash"],task_id=row["task_id"],job_id=event.get("job_id",""),command=command,sequence=sequence)
  if event!=expected or event_path.name!=f"{sequence:04d}_{row['task_id']}.json":raise ValueError("dense recovery journal differs")
  events.append(event);jobs[row["task_id"]]=event["job_id"]
 return events,jobs
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();recovery=load_json(a.recovery);validate_recovery(recovery);plan=load_json(Path(recovery["recovery_root"])/"command_plan.json");validate_artifact(plan,contract=PLAN_CONTRACT);commands={row["task_id"]:list(map(str,row["command"])) for row in plan["commands"]};dry=build_submission_ledger(campaign_spec_sha256=recovery["content_hash"],jobs={name:"1" for name in commands},commands=commands,dry_run=True)
 if not a.execute:
  if not a.output.exists():write_immutable_json(a.output,dry)
  elif load_json(a.output)!=dry:raise FileExistsError("dense recovery dry ledger differs")
  return 0
 if a.authorization_phrase!=PHRASE:raise PermissionError("dense recovery submission phrase differs")
 dry_path=Path(recovery["recovery_root"])/"dry_run_submission_ledger.json"
 if not dry_path.is_file() or load_json(dry_path)!=dry:raise ValueError("dense recovery dry evidence differs")
 validate_submission_ledger(load_json(dry_path))
 validate_source_checkout(ROOT,expected_commit=recovery["source_commit"])
 if a.output.exists():
  ledger=load_json(a.output);validate_submission_ledger(ledger);commands={row["task_id"]:resolved(row,ledger.get("jobs",{})) for row in plan["commands"]}
  if set(ledger.get("jobs",{}))!=set(commands) or ledger!=build_submission_ledger(campaign_spec_sha256=recovery["content_hash"],jobs=ledger["jobs"],commands=commands,dry_run=False):raise FileExistsError("dense recovery live ledger differs")
  return 0
 journal=a.output.parent/f"{a.output.stem}_journal";events,jobs=load_journal(journal,recovery,plan)
 for sequence,row in enumerate(plan["commands"][len(events):],start=len(events)):
  command=resolved(row,jobs);job=subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(";")[0];event=build_submission_event(campaign_spec_sha256=recovery["content_hash"],task_id=row["task_id"],job_id=job,command=command,sequence=sequence);write_immutable_json(journal/f"{sequence:04d}_{row['task_id']}.json",event);events.append(event);jobs[row["task_id"]]=job
 write_immutable_json(a.output,assemble_submission_ledger(events,campaign_spec_sha256=recovery["content_hash"]));return 0
if __name__=="__main__":raise SystemExit(main())
