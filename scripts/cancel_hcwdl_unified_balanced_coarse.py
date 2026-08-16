#!/usr/bin/env python3
"""Print or execute exact-ID cancellation for one FULLCOARSE3 ledger."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import exact_cancel_ids  # noqa:E402
PHRASE="CANCEL HCWDL UB FULLCOARSE3 EXACT IDS"
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--submission-ledger",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--authorization-phrase");a=p.parse_args();ids=exact_cancel_ids(load_json(a.submission_ledger));print("scancel "+" ".join(ids))
 if a.execute:
  if a.authorization_phrase!=PHRASE:raise PermissionError("coarse cancellation phrase differs")
  subprocess.run(["scancel",*ids],check=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
