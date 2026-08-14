#!/usr/bin/env python3
"""Dry-run or submit one exact HCWDL-UB foundation/arm DAG."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger, build_submission_event, build_submission_ledger, validate_submission_ledger  # noqa: E402
from hlt_classification.scouting.hcwdl_unified_balanced_campaign import ARM_SUBMISSION_PHRASE, FOUNDATION_SUBMISSION_PHRASE  # noqa: E402
from hlt_classification.scouting.hcwdl_unified_balanced_campaign import validate_arm_campaign, validate_foundation_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import ARM_COMMAND_PLAN_CONTRACT, FOUNDATION_COMMAND_PLAN_CONTRACT, validate_arm_spec, validate_foundation_spec  # noqa: E402
def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--spec",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--execute",action="store_true"); p.add_argument("--authorization-phrase"); a=p.parse_args(); spec=load_json(a.spec)
    foundation=spec["contract"].endswith("FOUNDATION_SPEC/v1"); (validate_foundation_campaign if foundation else validate_arm_campaign)(spec,executable=a.execute)
    canonical=Path(spec["campaign_root"])/("foundation_spec.json" if foundation else "arm_spec.json")
    if a.spec.resolve()!=canonical.resolve(): raise PermissionError("HCWDL-UB submitter requires the canonical spec")
    if spec.get("live_submission_authorized") is not True: raise PermissionError("HCWDL-UB spec is not live-authorized")
    plan_path=Path(spec["campaign_root"])/("foundation_command_plan.json" if foundation else "arm_command_plan.json"); plan=load_json(plan_path)
    expected_contract=FOUNDATION_COMMAND_PLAN_CONTRACT if foundation else ARM_COMMAND_PLAN_CONTRACT
    if plan.get("contract")!=expected_contract or plan.get("spec_sha256")!=spec["content_hash"]: raise ValueError("HCWDL-UB command plan differs")
    commands={row["task_id"]:list(row["command"]) for row in plan["commands"]}
    if not a.execute:
        expected=build_submission_ledger(campaign_spec_sha256=spec["content_hash"],jobs={task:"1" for task in commands},commands=commands,dry_run=True)
        if a.output.exists():
            if load_json(a.output)!=expected: raise FileExistsError("existing HCWDL-UB dry-run ledger differs")
        else: write_immutable_json(a.output,expected)
        return 0
    phrase=FOUNDATION_SUBMISSION_PHRASE if foundation else ARM_SUBMISSION_PHRASE
    if a.authorization_phrase!=phrase: raise PermissionError("HCWDL-UB submission phrase differs")
    if ROOT.resolve()!=Path(spec["project_dir"]).resolve(): raise PermissionError("HCWDL-UB submitter is outside the bound worktree")
    validate_source_checkout(ROOT,expected_commit=spec["source_commit"])
    if a.output.exists():
        ledger=load_json(a.output); validate_submission_ledger(ledger)
        if ledger.get("dry_run") is not False or ledger.get("campaign_spec_sha256")!=spec["content_hash"] or set(ledger["jobs"])!=set(commands): raise FileExistsError("existing HCWDL-UB live ledger differs")
        return 0
    events=[]; jobs={}; journal=a.output.parent/f"{a.output.stem}_journal"
    existing=sorted(journal.glob("*.json")) if journal.is_dir() else []
    if len(existing)>len(plan["commands"]): raise ValueError("HCWDL-UB submission journal is longer than its command plan")
    for sequence,path in enumerate(existing):
        row=plan["commands"][sequence]
        if path.name!=f"{sequence:04d}_{row['task_id']}.json": raise ValueError("HCWDL-UB submission journal order differs")
        event=load_json(path)
        command=[str(item) for item in row["command"]]
        for i,item in enumerate(command):
            for parent in row["dependencies"]: item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
            command[i]=item
        expected=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=event.get("job_id",""),command=command,sequence=sequence)
        if event!=expected: raise ValueError("HCWDL-UB submission journal event differs")
        events.append(event); jobs[row["task_id"]]=event["job_id"]
    for sequence,row in enumerate(plan["commands"][len(events):],start=len(events)):
        command=[str(item) for item in row["command"]]
        for i,item in enumerate(command):
            for parent in row["dependencies"]: item=item.replace(f"${{JOB_{parent}}}",jobs[parent])
            command[i]=item
        raw=subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip(); job=raw.split(";")[0]
        event=build_submission_event(campaign_spec_sha256=spec["content_hash"],task_id=row["task_id"],job_id=job,command=command,sequence=sequence)
        write_immutable_json(journal/f"{sequence:04d}_{row['task_id']}.json",event); events.append(event); jobs[row["task_id"]]=job
    write_immutable_json(a.output,assemble_submission_ledger(events,campaign_spec_sha256=spec["content_hash"])); return 0
if __name__=="__main__": raise SystemExit(main())
