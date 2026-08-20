#!/usr/bin/env python3
"""Run source-pinned focused/full TRI60 validation and publish compact evidence."""
from __future__ import annotations
import argparse,hashlib,os,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_contracts import TEST_EVIDENCE_CONTRACT,artifact  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_tri60_integration import SEMANTIC_SOURCE_FILES  # noqa:E402
CLIS=("create_hcwdl_mhpe_tri60_campaign.py","run_hcwdl_mhpe_tri60_task.py","submit_hcwdl_mhpe_tri60_campaign.py","run_hcwdl_mhpe_tri60_acceptance.py","create_hcwdl_mhpe_tri60_recovery.py","run_hcwdl_mhpe_tri60_recovery_task.py","submit_hcwdl_mhpe_tri60_recovery.py","monitor_hcwdl_mhpe_tri60.py","cancel_hcwdl_mhpe_tri60.py","validate_hcwdl_mhpe_tri60_source.py")
def _run(command,env):
 result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True);text=result.stdout+result.stderr
 if result.returncode:print(text);raise subprocess.CalledProcessError(result.returncode,command)
 return {"command":list(command),"output_sha256":hashlib.sha256(text.encode()).hexdigest(),"output_tail":text[-2000:]}
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-commit",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();validate_source_checkout(ROOT,expected_commit=a.source_commit);env=dict(os.environ);env["PYTHONPATH"]=str(ROOT/"src");env["PYTHONDONTWRITEBYTECODE"]="1";runs=[]
 runs.append(_run([
  sys.executable,"-m","pytest","-q",
  "tests/test_hcwdl_representation_math.py",
  "tests/test_hcwdl_representation_model.py",
  "tests/test_hcwdl_representation_training.py",
  "tests/test_hcwdl_mhpe_tri60.py",
  "tests/test_hcwdl_unified_balanced.py",
  "tests/test_hcwdl_unified_balanced_full.py",
 ],env))
 runs.append(_run([sys.executable,"-m","pytest","-q"],env))
 for cli in CLIS:runs.append(_run([sys.executable,str(ROOT/"scripts"/cli),"--help"],env))
 for name in SEMANTIC_SOURCE_FILES:
  if name.endswith(".py"):compile((ROOT/name).read_text(encoding="utf-8"),name,"exec")
 _run(["git","-C",str(ROOT),"diff","--check"],env)
 report=artifact({"source_commit":a.source_commit,"focused_and_full_runs":runs,"semantic_python_files_compiled":sum(name.endswith('.py') for name in SEMANTIC_SOURCE_FILES),"cli_help_count":len(CLIS),"passed":True,"final_test_accessed":False},contract=TEST_EVIDENCE_CONTRACT);write_immutable_json(a.output,report);print(report["content_hash"]);return 0
if __name__=="__main__":raise SystemExit(main())
