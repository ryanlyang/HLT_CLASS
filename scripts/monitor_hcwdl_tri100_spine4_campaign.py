#!/usr/bin/env python3
"""Write an exact-ID monitor for one TRI100 four-spine ledger."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_tri100_spine4_campaign import validate_campaign  # noqa:E402
from hlt_classification.scouting.hcwdl_tri100_spine4_recovery import build_monitor  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();spec=load_json(a.spec);validate_campaign(spec);ledger=load_json(a.ledger);ids=list(map(str,ledger["jobs"].values()));raw=subprocess.run(["sacct","-X","-n","-P","-j",",".join(ids),"-o","JobIDRaw,State"],check=True,capture_output=True,text=True).stdout;states={}
 for line in raw.splitlines():
  fields=line.split("|")
  if len(fields)>=2 and fields[0] in ids:states[fields[0]]=fields[1]
 report=build_monitor(spec=spec,ledger=ledger,states_by_job_id=states);write_immutable_json(a.output,report);return 0
if __name__=="__main__":raise SystemExit(main())
