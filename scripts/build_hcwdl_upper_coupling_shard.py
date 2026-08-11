#!/usr/bin/env python3
"""Build one immutable HCWDL-UJ base-coupling shard or switch sidecar."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_workflow import HomotopyWorkflow  # noqa: E402
def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--campaign-spec",type=Path,required=True); p.add_argument("--role",choices=("train","validation"),required=True); p.add_argument("--source-index",type=int,required=True); p.add_argument("--switch",action="store_true"); a=p.parse_args()
    task=(f"{a.role}_switch" if a.switch else f"{a.role}_base"); HomotopyWorkflow(load_json(a.campaign_spec),repository=ROOT).run(task,array_index=a.source_index); return 0
if __name__ == "__main__": raise SystemExit(main())
