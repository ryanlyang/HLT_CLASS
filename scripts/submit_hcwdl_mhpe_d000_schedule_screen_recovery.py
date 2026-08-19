#!/usr/bin/env python3
"""Dry-run or submit an exact D000 teacher-distance screen recovery."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,validate_content_hash,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen import RECOVERY_COMMAND_PLAN_CONTRACT  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen_recovery import validate_recovery  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_event,build_submission_ledger,validate_submission_ledger  # noqa:E402
def _resolved(row,jobs):
 command=[str(x) for x in row["command"]]
 for i,item in enumerate(command):
  for parent in row["dependencies"]:item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
  command[i]=item
 return command
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");a=p.parse_args();recovery=load_json(a.recovery_spec);validate_recovery(recovery);plan=load_json(Path(recovery["recovery_root"])/"command_plan.json");validate_content_hash(plan,expected_contract=RECOVERY_COMMAND_PLAN_CONTRACT,expected_schema_version=1);commands={row["task_id"]:list(row["command"]) for row in plan["commands"]}
 if not a.execute:
  ledger=build_submission_ledger(campaign_spec_sha256=recovery["content_hash"],jobs={task:"1" for task in commands},commands=commands,dry_run=True)
  if a.output.exists() and load_json(a.output)!=ledger:raise FileExistsError("D000-screen recovery dry ledger differs")
  if not a.output.exists():write_immutable_json(a.output,ledger)
  return 0
 validate_source_checkout(ROOT,expected_commit=recovery["source_commit"])
 dry=load_json(Path(recovery["recovery_root"])/"dry_run_submission_ledger.json");validate_submission_ledger(dry);expected=build_submission_ledger(campaign_spec_sha256=recovery["content_hash"],jobs={task:"1" for task in commands},commands=commands,dry_run=True)
 if dry!=expected:raise ValueError("D000-screen recovery dry ledger differs")
 if a.output.exists():
  live=load_json(a.output);validate_submission_ledger(live)
  if live.get("dry_run") is not False or live.get("campaign_spec_sha256")!=recovery["content_hash"]:raise FileExistsError("D000-screen recovery live ledger differs")
  return 0
 jobs={};events=[];journal=a.output.parent/f"{a.output.stem}_journal";existing=sorted(journal.glob("*.json")) if journal.is_dir() else []
 for sequence,path in enumerate(existing):
  row=plan["commands"][sequence];event=load_json(path);command=_resolved(row,jobs);expected_event=build_submission_event(campaign_spec_sha256=recovery["content_hash"],task_id=row["task_id"],job_id=event.get("job_id",""),command=command,sequence=sequence)
  if event!=expected_event:raise ValueError("D000-screen recovery submission journal differs")
  events.append(event);jobs[row["task_id"]]=event["job_id"]
 for sequence,row in enumerate(plan["commands"][len(events):],start=len(events)):
  command=_resolved(row,jobs);job=subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(";")[0];event=build_submission_event(campaign_spec_sha256=recovery["content_hash"],task_id=row["task_id"],job_id=job,command=command,sequence=sequence);write_immutable_json(journal/f"{sequence:04d}_{row['task_id']}.json",event);events.append(event);jobs[row["task_id"]]=job
 write_immutable_json(a.output,assemble_submission_ledger(events,campaign_spec_sha256=recovery["content_hash"]));return 0
if __name__=="__main__":raise SystemExit(main())
