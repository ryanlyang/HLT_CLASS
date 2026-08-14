#!/usr/bin/env python3
"""Build the exact no-new-smoke HCWDL-UB 300k operational waiver."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json, sha256_file, validate_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_unified_balanced_campaign import ARM_RESOURCES, FOUNDATION_RESOURCES, semantic_source_hashes  # noqa: E402
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import operational_waiver_payload  # noqa: E402
def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit",required=True); p.add_argument("--parent-completion",type=Path,required=True); p.add_argument("--parent-homotopy-spec",type=Path,required=True)
    p.add_argument("--prior-smoke-completion",type=Path,required=True); p.add_argument("--performance-guide",type=Path,required=True); p.add_argument("--readiness-evidence",type=Path,required=True); p.add_argument("--project-dir",type=Path,required=True)
    p.add_argument("--authorization-phrase",required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    parent=load_json(a.parent_completion); smoke=load_json(a.prior_smoke_completion); parent_spec=load_json(a.parent_homotopy_spec)
    for value in (parent,smoke,parent_spec): validate_content_hash(value,expected_contract=str(value["contract"]),expected_schema_version=int(value["schema_version"]))
    payload=operational_waiver_payload(source_commit=a.source_commit,parent_completion_sha256=parent["content_hash"],prior_smoke_completion_sha256=smoke["content_hash"],performance_guide_sha256=sha256_file(a.performance_guide),parent_weaver_parity_sha256=parent_spec["weaver_parity_sha256"],readiness_evidence_sha256=sha256_file(a.readiness_evidence),semantic_source_sha256=semantic_source_hashes(a.project_dir),resources={"foundation":{k:asdict(v) for k,v in FOUNDATION_RESOURCES.items()},"arm":{k:asdict(v) for k,v in ARM_RESOURCES.items()}},authorization_phrase=a.authorization_phrase)
    write_immutable_json(a.output,payload); print(a.output); return 0
if __name__=="__main__": raise SystemExit(main())
