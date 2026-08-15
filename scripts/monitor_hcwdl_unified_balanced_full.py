#!/usr/bin/env python3
"""Authenticate and summarize one exact HCWDL-UB-FULL3 ledger."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import build_monitor_report,task_attestation_path,validate_submission_ledger,validate_task_attestation  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--submission-ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();spec=load_json(a.spec);ledger=load_json(a.submission_ledger);validate_submission_ledger(ledger)
 if ledger["campaign_spec_sha256"]!=spec["content_hash"]:raise ValueError("HCWDL-UB-FULL3 monitor ledger/spec differs")
 ids=list(ledger["jobs"].values());raw=subprocess.run(["sacct","-n","-P","-j",",".join(ids),"--format=JobIDRaw,State"],check=True,capture_output=True,text=True).stdout;states={}
 for line in raw.splitlines():
  fields=line.split("|")
  if len(fields)>=2 and fields[0] in ids and "." not in fields[0]:states[fields[0]]=fields[1]
 tasks={row["task_id"]:row for row in spec["tasks"]};validity={}
 for task,job in ledger["jobs"].items():
  count=int(tasks[task]["array_count"]);indexes=[None] if count==1 else list(range(count));ok=True
  for index in indexes:
   path=task_attestation_path(spec["campaign_root"],task,index)
   if not path.is_file():ok=False;continue
   try:validate_task_attestation(load_json(path),campaign_spec_sha256=spec["content_hash"],task_id=task,array_index=index)
   except Exception:ok=False
  validity[task]=ok
 report=build_monitor_report(ledger,states_by_job_id=states,artifact_validity=validity);write_immutable_json(a.output,report)
 for row in report["rows"]:print(f"{row['task_id']:<34} {row['job_id']:<12} {row['state']:<20} {row['disposition']}")
 return 0
if __name__=="__main__":raise SystemExit(main())
