#!/usr/bin/env python3
"""Run the bounded real-worker TRI60 RAM/no-resume acceptance profile."""
from __future__ import annotations
import argparse,os,shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_acceptance import run_bounded_acceptance  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--foundation-lock",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda");a=p.parse_args();validate_source_checkout(ROOT,expected_commit=a.source_commit);base=Path(os.environ.get("SLURM_TMPDIR","/tmp")).resolve();temporary=base/f"hcwdl_tri60_acceptance_{os.environ.get('SLURM_JOB_ID','missing')}"
 if temporary.exists():raise FileExistsError("TRI60 acceptance temporary root already exists")
 try:report=run_bounded_acceptance(foundation_lock=a.foundation_lock,temporary_root=temporary,source_commit=a.source_commit,device=a.device)
 finally:
  if temporary.exists() and temporary.is_relative_to(base):shutil.rmtree(temporary)
 if not report["passed"]:raise MemoryError("TRI60 bounded acceptance resource projection failed")
 write_immutable_json(a.output,report);print(report["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
