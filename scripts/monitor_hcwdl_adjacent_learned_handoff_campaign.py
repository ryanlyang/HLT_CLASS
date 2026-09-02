#!/usr/bin/env python3
"""Monitor an exact learned-handoff submission ledger."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_monitor import build_monitor  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();ledger=load_json(a.ledger);jobs=list(ledger["jobs"].values());result=subprocess.run(["sacct","-X","-n","-P","-j",",".join(jobs),"-o","JobIDRaw,State"],capture_output=True,text=True,check=True);states={}
 for line in result.stdout.splitlines():
  fields=line.split("|");
  if len(fields)>=2 and fields[0] in jobs:states[fields[0]]=fields[1].split()[0]
 report=build_monitor(campaign_spec=a.spec,submission_ledger=a.ledger,states_by_job_id=states);write_immutable_json(a.output,report);print(json.dumps(report,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
