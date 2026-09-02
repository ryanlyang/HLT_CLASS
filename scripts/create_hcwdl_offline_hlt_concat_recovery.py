#!/usr/bin/env python3
"""Create a source-pinned restart-zero tagged concatenation recovery."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.scouting.hcwdl_offline_hlt_concat_recovery import create_recovery  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--subject-spec",type=Path,required=True);p.add_argument("--subject-ledger",type=Path,required=True);p.add_argument("--monitor-report",type=Path,required=True);p.add_argument("--recovery-root",type=Path,required=True);p.add_argument("--project-dir",type=Path,required=True);p.add_argument("--source-commit",required=True);a=p.parse_args();value=create_recovery(subject_spec=a.subject_spec,subject_ledger=a.subject_ledger,monitor_report=a.monitor_report,recovery_root=a.recovery_root,project_dir=a.project_dir,source_commit=a.source_commit);print(value["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
