#!/usr/bin/env python3
"""Dry-run or submit an exact HCWDL-UB recovery closure."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_recovery import validate_recovery_command_plan,validate_recovery_spec  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_event,build_submission_ledger,validate_submission_ledger  # noqa:E402
PHRASE="SUBMIT HCWDL UB RECOVERY EXACT CLOSURE"
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--recovery-spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();spec=load_json(a.recovery_spec);validate_recovery_spec(spec)
 if a.recovery_spec.resolve()!=(Path(spec["recovery_root"])/"recovery_spec.json").resolve():raise ValueError("HCWDL-UB recovery specification is not canonical")
 plan=load_json(Path(spec["recovery_root"])/"recovery_command_plan.json");validate_recovery_command_plan(plan,recovery_spec=spec)
 commands={r["task_id"]:r["command"] for r in plan["commands"]}
 if not a.execute:
  expected=build_submission_ledger(campaign_spec_sha256=spec["content_hash"],jobs={k:"1" for k in commands},commands=commands,dry_run=True)
  if a.output.exists():
   if load_json(a.output)!=expected:raise FileExistsError("existing HCWDL-UB recovery dry-run ledger differs")
  else:write_immutable_json(a.output,expected)
  return 0
 if a.authorization_phrase!=PHRASE:raise PermissionError("HCWDL-UB recovery submission phrase differs")
 if ROOT.resolve()!=Path(spec["project_dir"]).resolve():raise PermissionError("HCWDL-UB recovery submitter is outside the bound worktree")
 validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
 if a.output.exists():
  ledger=load_json(a.output);validate_submission_ledger(ledger)
  if ledger.get("dry_run") is not False or ledger.get("campaign_spec_sha256")!=spec["content_hash"] or set(ledger["jobs"])!=set(commands):raise FileExistsError("existing HCWDL-UB recovery live ledger differs")
  return 0
 events=[];jobs={};journal=Path(spec["recovery_root"])/"submission_journal";existing=sorted(journal.glob("*.json")) if journal.is_dir() else []
 if len(existing)>len(plan["commands"]):raise ValueError("HCWDL-UB recovery journal is longer than its command plan")
 for seq,path in enumerate(existing):
  row=plan["commands"][seq]
  if path.name!=f"{seq:04d}_{row['task_id']}.json":raise ValueError("HCWDL-UB recovery journal order differs")
  event=load_json(path);command=list(row["command"])
  for i,item in enumerate(command):
   for parent in row["dependencies"]:item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
   command[i]=item
  expected=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=event.get("job_id",""),command=command,sequence=seq)
  if event!=expected:raise ValueError("HCWDL-UB recovery journal event differs")
  events.append(event);jobs[row["task_id"]]=event["job_id"]
 for seq,row in enumerate(plan["commands"][len(events):],start=len(events)):
  command=list(row["command"])
  for i,item in enumerate(command):
   for parent in row["dependencies"]:item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
   command[i]=item
  job=subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(";")[0];event=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=job,command=command,sequence=seq);write_immutable_json(journal/f"{seq:04d}_{row['task_id']}.json",event);events.append(event);jobs[row["task_id"]]=job
 write_immutable_json(a.output,assemble_submission_ledger(events,campaign_spec_sha256=spec["content_hash"]));return 0
if __name__=="__main__":raise SystemExit(main())
