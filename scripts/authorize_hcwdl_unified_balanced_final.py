#!/usr/bin/env python3
"""Create the explicit HCWDL-UB sealed-final-test execution lock."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import (  # noqa:E402
    execution_lock_payload, validate_finalist_lock, validate_foundation_spec,
)
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--foundation-spec",type=Path,required=True);p.add_argument("--finalist-lock",type=Path,required=True);p.add_argument("--authorization-phrase",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();foundation=load_json(a.foundation_spec);validate_foundation_spec(foundation);finalist=load_json(a.finalist_lock);finalist_hash=validate_finalist_lock(finalist);payload=execution_lock_payload(finalist_lock_sha256=finalist_hash,source_commit=foundation["source_commit"],split_manifest_sha256=foundation["parents"]["split_manifest_sha256"],selection_manifest_sha256=foundation["parents"]["selection_manifest_sha256"],authorization_phrase=a.authorization_phrase);write_immutable_json(a.output,payload);return 0
if __name__=="__main__":raise SystemExit(main())
