#!/usr/bin/env python3
"""Create an exact failed/downstream TRI60 recovery specification."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_mhpe_tri60_recovery import create_recovery  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--subject-spec",type=Path,required=True);p.add_argument("--subject-ledger",type=Path,required=True);p.add_argument("--monitor-report",type=Path,required=True);p.add_argument("--recovery-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--changed-file",action="append",default=[]);p.add_argument("--source-repair-phrase");p.add_argument("--logit-memory");p.add_argument("--representation-memory");p.add_argument("--representation-walltime");a=p.parse_args();overrides={}
 if a.logit_memory:overrides["gpu_logit"]={"memory":a.logit_memory}
 if a.representation_memory:
  overrides["gpu_rset"]={"memory":a.representation_memory};overrides["gpu_rrel"]={"memory":a.representation_memory}
 if a.representation_walltime:
  overrides.setdefault("gpu_rset",{})["walltime"]=a.representation_walltime;overrides.setdefault("gpu_rrel",{})["walltime"]=a.representation_walltime
 value=create_recovery(subject_spec=a.subject_spec,subject_ledger=a.subject_ledger,monitor_report=a.monitor_report,recovery_root=a.recovery_root,project_dir=a.project_dir,source_commit=a.source_commit,changed_files=a.changed_file,source_repair_phrase=a.source_repair_phrase,resource_overrides=overrides or None);print(value["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
