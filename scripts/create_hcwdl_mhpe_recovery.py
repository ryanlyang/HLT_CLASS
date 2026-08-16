#!/usr/bin/env python3
"""Create an exact HCWDL-MHPE failed/downstream recovery."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_mhpe_recovery import create_recovery  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--original-spec",type=Path,required=True);p.add_argument("--original-ledger",type=Path,required=True);p.add_argument("--monitor-report",type=Path,required=True);p.add_argument("--recovery-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--resource-overrides-json");p.add_argument("--changed-file",action="append",default=[]);p.add_argument("--repair-authorization-phrase");p.add_argument("--dry-run",action="store_true");a=p.parse_args();overrides=None if a.resource_overrides_json is None else json.loads(a.resource_overrides_json);value=create_recovery(original_spec=a.original_spec,original_ledger=a.original_ledger,monitor_report=a.monitor_report,recovery_root=a.recovery_root,project_dir=a.project_dir,source_commit=a.source_commit,resource_overrides=overrides,changed_files=a.changed_file,repair_authorization_phrase=a.repair_authorization_phrase,publish=not a.dry_run);print(json.dumps(value,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
