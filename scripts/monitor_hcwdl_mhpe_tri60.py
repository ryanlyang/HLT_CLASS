#!/usr/bin/env python3
"""Query exact TRI60 ledger IDs and publish an immutable monitor report."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_operations import build_monitor  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--subject-spec",type=Path,required=True);p.add_argument("--ledger",type=Path,required=True);p.add_argument("--attestation-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();subject=load_json(a.subject_spec);ledger=load_json(a.ledger);validate_submission_ledger(ledger);ids=",".join(ledger["jobs"].values());raw=subprocess.run(["sacct","-n","-P","-j",ids,"--format=JobIDRaw,State"],check=True,capture_output=True,text=True).stdout;states={}
 for line in raw.splitlines():
  fields=line.split("|")
  if len(fields)>=2 and fields[0] in ledger["jobs"].values():states[fields[0]]=fields[1]
 report=build_monitor(subject=subject,ledger=ledger,states_by_job_id=states,attestation_root=a.attestation_root);write_immutable_json(a.output,report)
 for row in report["rows"]:print(f"{row['job_id']:>10} {row['task_id']:<44} {row['state']:<18} {row['disposition']}")
 return 0
if __name__=="__main__":raise SystemExit(main())
