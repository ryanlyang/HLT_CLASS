#!/usr/bin/env python3
"""Dry-run or submit an exact FULLCOARSE3 recovery closure."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_event,build_submission_ledger,validate_submission_ledger  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_coarse_recovery import RECOVERY_PHRASE,validate_recovery_command_plan,validate_recovery_spec  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();spec=load_json(a.recovery_spec);validate_recovery_spec(spec);plan=load_json(Path(spec["recovery_root"])/"recovery_command_plan.json");validate_recovery_command_plan(plan,recovery_spec=spec);commands={r["task_id"]:r["command"] for r in plan["commands"]}
 if not a.execute:
  expected=build_submission_ledger(campaign_spec_sha256=spec["content_hash"],jobs={k:"1" for k in commands},commands=commands,dry_run=True)
  if a.output.exists():
   if load_json(a.output)!=expected:raise FileExistsError("existing coarse recovery dry-run ledger differs")
  else:write_immutable_json(a.output,expected)
  return 0
 if a.authorization_phrase!=RECOVERY_PHRASE:raise PermissionError("coarse recovery phrase differs")
 if ROOT.resolve()!=Path(spec["project_dir"]).resolve():raise PermissionError("recovery submitter is outside bound worktree")
 validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
 if a.output.exists():
  ledger=load_json(a.output);validate_submission_ledger(ledger)
  if ledger.get("dry_run") is not False or ledger.get("campaign_spec_sha256")!=spec["content_hash"]:raise FileExistsError("existing coarse recovery ledger differs")
  return 0
 events=[];jobs={};journal=Path(spec["recovery_root"])/"submission_journal";existing=sorted(journal.glob("*.json")) if journal.is_dir() else []
 for sequence,path in enumerate(existing):
  row=plan["commands"][sequence];event=load_json(path);command=[str(item) for item in row["command"]]
  for i,item in enumerate(command):
   for parent in row["dependencies"]:item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
   command[i]=item
  expected=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=event.get("job_id",""),command=command,sequence=sequence)
  if path.name!=f"{sequence:04d}_{row['task_id']}.json" or event!=expected:raise ValueError("coarse recovery journal differs")
  events.append(event);jobs[row["task_id"]]=event["job_id"]
 for sequence,row in enumerate(plan["commands"][len(events):],start=len(events)):
  command=[str(item) for item in row["command"]]
  for i,item in enumerate(command):
   for parent in row["dependencies"]:item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
   command[i]=item
  job=subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(";")[0];event=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=job,command=command,sequence=sequence);write_immutable_json(journal/f"{sequence:04d}_{row['task_id']}.json",event);events.append(event);jobs[row["task_id"]]=job
 write_immutable_json(a.output,assemble_submission_ledger(events,campaign_spec_sha256=spec["content_hash"]));return 0
if __name__=="__main__":raise SystemExit(main())
