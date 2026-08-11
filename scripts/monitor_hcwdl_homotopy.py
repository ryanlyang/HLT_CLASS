#!/usr/bin/env python3
"""Publish an immutable exact-ID monitor report for an HCWDL-UJ ledger."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_campaign import validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_recovery import aggregate_slurm_states  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_monitor_report, task_attestation_path, validate_submission_ledger,
    validate_task_attestation,
)
def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--campaign-spec",type=Path,required=True); p.add_argument("--submission-ledger",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--states-json",type=Path); p.add_argument("--query-slurm",action="store_true"); a=p.parse_args()
    if (a.states_json is None) == (not a.query_slurm): p.error("choose exactly one of --states-json or --query-slurm")
    spec=load_json(a.campaign_spec); validate_campaign(spec)
    ledger=load_json(a.submission_ledger); ledger_hash=validate_submission_ledger(ledger)
    if ledger.get("campaign_spec_sha256") != spec["content_hash"]: raise ValueError("HCWDL-UJ monitor ledger belongs to another campaign")
    tasks={row["task_id"]:row for row in spec["tasks"]}
    if a.states_json: states=json.loads(a.states_json.read_text())
    else:
        ids=",".join(ledger["jobs"].values()); raw=subprocess.run(["sacct","-n","-P","-j",ids,"--format=JobIDRaw,State"],check=True,capture_output=True,text=True).stdout; records=[]
        for line in raw.splitlines():
            parts=line.split("|");
            if len(parts)>=2: records.append((parts[0],parts[1]))
        states=aggregate_slurm_states(
            jobs=ledger["jobs"],
            array_counts={task:int(tasks[task]["array_count"]) for task in ledger["jobs"]},
            records=records,
        )
    validity={}
    for task_id in ledger["jobs"]:
        task=tasks[task_id]; count=int(task["array_count"]); indexes=(None,) if count==1 else range(count); valid=True
        for index in indexes:
            try:
                validate_task_attestation(load_json(task_attestation_path(spec["campaign_root"],task_id,index)),campaign_spec_sha256=spec["content_hash"],task_id=task_id,array_index=index)
            except (FileNotFoundError, KeyError, OSError, PermissionError, TypeError, ValueError):
                valid=False; break
        validity[task_id]=valid
    report=build_monitor_report(ledger,states_by_job_id=states,artifact_validity=validity)
    report["validated_submission_ledger_sha256"]=ledger_hash; report.pop("content_hash")
    from hlt_classification.data.cache_contracts import with_content_hash
    write_immutable_json(a.output,with_content_hash(report)); return 0
if __name__=="__main__": raise SystemExit(main())
