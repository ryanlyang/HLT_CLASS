#!/usr/bin/env python
"""Thin task worker for the staged CMS-proxy ladder."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.gate import run_gate_task
from hlt_classification.cms_proxy_ladder.production import run_task


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gate-spec", type=Path)
    group.add_argument("--campaign-spec", type=Path)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    if args.gate_spec is not None:
        run_gate_task(load_json(args.gate_spec), args.task)
    else:
        attempt = "slurm_" + os.environ.get("SLURM_JOB_ID", "missing")
        run_task(load_json(args.campaign_spec), args.task, attempt=attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
