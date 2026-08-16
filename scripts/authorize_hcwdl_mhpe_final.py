#!/usr/bin/env python3
"""Create the explicit HCWDL-MHPE sealed-final-test execution lock."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json,validate_content_hash,write_immutable_json  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_campaign import validate_campaign  # noqa:E402
from hlt_classification.scouting.hcwdl_mhpe_contracts import FINALIST_LOCK_CONTRACT,execution_lock_payload  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-spec",type=Path,required=True);p.add_argument("--finalist-lock",type=Path,required=True);p.add_argument("--authorization-phrase",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();spec=load_json(a.campaign_spec);spec_hash=validate_campaign(spec,verify_source_tree=False);finalist=load_json(a.finalist_lock);finalist_hash=validate_content_hash(finalist,expected_contract=FINALIST_LOCK_CONTRACT,expected_schema_version=1);payload=execution_lock_payload(campaign_spec_sha256=spec_hash,finalist_lock_sha256=finalist_hash,source_commit=spec["source_commit"],authorization_phrase=a.authorization_phrase);write_immutable_json(a.output,payload);return 0
if __name__=="__main__":raise SystemExit(main())
