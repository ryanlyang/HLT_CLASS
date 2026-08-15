#!/usr/bin/env python3
"""Create an exact failed-closure or resource-only HCWDL-UB-FULL3 recovery."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_full_recovery import build_recovery_spec,recovery_command_plan  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scope-spec",type=Path,required=True);p.add_argument("--submission-ledger",type=Path,required=True);p.add_argument("--monitor-report",type=Path,required=True);p.add_argument("--recovery-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--resource-overrides-json");p.add_argument("--execution-repair");p.add_argument("--authorization-phrase");a=p.parse_args();overrides=None if a.resource_overrides_json is None else json.loads(a.resource_overrides_json)
 spec=build_recovery_spec(scope_spec_path=a.scope_spec,submission_ledger_path=a.submission_ledger,monitor_report_path=a.monitor_report,recovery_root=a.recovery_root,project_dir=a.project_dir,source_commit=a.source_commit,resource_overrides=overrides,execution_repair=a.execution_repair,authorization_phrase=a.authorization_phrase);plan=recovery_command_plan(spec);write_immutable_json(a.recovery_root/"recovery_spec.json",spec);write_immutable_json(a.recovery_root/"recovery_command_plan.json",plan);print(json.dumps({"contract":spec["contract"],"tasks":spec["task_ids"],"root":str(a.recovery_root)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
