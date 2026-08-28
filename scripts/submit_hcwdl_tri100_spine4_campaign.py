#!/usr/bin/env python3
"""Dry-run or submit the exact TRI100 four-spine DAG."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_event,build_submission_ledger,validate_submission_ledger  # noqa:E402
from hlt_classification.scouting.hcwdl_tri100_spine4_campaign import SUBMISSION_PHRASE,validate_campaign  # noqa:E402
from hlt_classification.scouting.hcwdl_tri100_spine4_contracts import PLAN_CONTRACT,validate_artifact  # noqa:E402
def commands(plan):return {str(row["task_id"]):list(map(str,row["command"])) for row in plan["commands"]}
def resolved(row,jobs):
 value=list(map(str,row["command"]))
 for index,item in enumerate(value):
  for parent in row["dependencies"]:
   if parent not in jobs:raise ValueError("TRI100 spine4 dependency is absent")
   item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
  value[index]=item
 if any("${JOB_" in item for item in value):raise ValueError("TRI100 spine4 dependency unresolved")
 return value
def journal(path,spec,plan):
 paths=sorted(path.glob("*.json")) if path.is_dir() else [];events=[];jobs={}
 if len(paths)>len(plan["commands"]):raise ValueError("TRI100 spine4 journal has excess events")
 for sequence,event_path in enumerate(paths):
  row=plan["commands"][sequence];command=resolved(row,jobs);event=load_json(event_path);expected=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=event.get("job_id",""),command=command,sequence=sequence)
  if event!=expected or event_path.name!=f"{sequence:04d}_{row['task_id']}.json":raise ValueError("TRI100 spine4 journal differs")
  events.append(event);jobs[row["task_id"]]=event["job_id"]
 return events,jobs
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();spec=load_json(a.spec);validate_campaign(spec,executable=a.execute);canonical=Path(spec["campaign_root"])/"campaign_spec.json"
 if a.spec.resolve()!=canonical.resolve():raise PermissionError("TRI100 spine4 submitter requires canonical spec")
 plan=load_json(Path(spec["campaign_root"])/"command_plan.json");validate_artifact(plan,contract=PLAN_CONTRACT);raw=commands(plan);dry=build_submission_ledger(campaign_spec_sha256=spec["content_hash"],jobs={name:"1" for name in raw},commands=raw,dry_run=True)
 if not a.execute:
  if a.output.exists() and load_json(a.output)!=dry:raise FileExistsError("TRI100 spine4 dry ledger differs")
  if not a.output.exists():write_immutable_json(a.output,dry)
  return 0
 if a.authorization_phrase!=SUBMISSION_PHRASE:raise PermissionError("TRI100 spine4 submission phrase differs")
 dry_path=Path(spec["campaign_root"])/"dry_run_submission_ledger.json"
 if not dry_path.is_file() or load_json(dry_path)!=dry:raise ValueError("TRI100 spine4 canonical dry-run evidence differs")
 validate_submission_ledger(load_json(dry_path))
 if ROOT.resolve()!=Path(spec["project_dir"]).resolve():raise PermissionError("TRI100 spine4 submitter is outside bound worktree")
 validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
 if a.output.exists():
  ledger=load_json(a.output);validate_submission_ledger(ledger);exact={row["task_id"]:resolved(row,ledger.get("jobs",{})) for row in plan["commands"]}
  if set(ledger.get("jobs",{}))!=set(raw) or ledger!=build_submission_ledger(campaign_spec_sha256=spec["content_hash"],jobs=ledger["jobs"],commands=exact,dry_run=False):raise FileExistsError("TRI100 spine4 live ledger differs")
  return 0
 event_dir=a.output.parent/f"{a.output.stem}_journal";events,jobs=journal(event_dir,spec,plan)
 for sequence,row in enumerate(plan["commands"][len(events):],start=len(events)):
  command=resolved(row,jobs);job=subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(";")[0];event=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=job,command=command,sequence=sequence);write_immutable_json(event_dir/f"{sequence:04d}_{row['task_id']}.json",event);events.append(event);jobs[row["task_id"]]=job
 write_immutable_json(a.output,assemble_submission_ledger(events,campaign_spec_sha256=spec["content_hash"]));return 0
if __name__=="__main__":raise SystemExit(main())
