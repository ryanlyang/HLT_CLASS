#!/usr/bin/env python3
"""Print or execute exact-ID cancellation for one D000-screen ledger."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger  # noqa:E402
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--submission-ledger",type=Path,required=True);p.add_argument("--execute",action="store_true");a=p.parse_args();ledger=load_json(a.submission_ledger);validate_submission_ledger(ledger)
 if ledger.get("dry_run") is not False:raise PermissionError("cannot cancel a dry-run ledger")
 ids=list(dict.fromkeys(str(value) for value in ledger["jobs"].values()));print("scancel "+" ".join(ids))
 if a.execute:subprocess.run(["scancel",*ids],check=True)
 return 0
if __name__=="__main__":raise SystemExit(main())

