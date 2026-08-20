#!/usr/bin/env python3
"""Print or execute exact-ID cancellation for one authenticated TRI60 ledger."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_operations import build_cancellation  # noqa:E402
PHRASE="CANCEL HCWDL MHPE TRI60 EXACT LEDGER IDS"
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--ledger",type=Path,required=True);p.add_argument("--monitor",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--task",action="append");p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();ledger=load_json(a.ledger);monitor=load_json(a.monitor);tasks=None if a.task is None else tuple(a.task);report=build_cancellation(ledger=ledger,monitor=monitor,task_ids=tasks,executed=a.execute);print("scancel "+" ".join(report["job_ids"]))
 if a.execute:
  if a.authorization_phrase!=PHRASE:raise PermissionError("TRI60 cancellation phrase differs")
  subprocess.run(["scancel",*report["job_ids"]],check=True)
 write_immutable_json(a.output,report);return 0
if __name__=="__main__":raise SystemExit(main())
