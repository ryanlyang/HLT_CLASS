#!/usr/bin/env python
"""Arm a one-time seven-job S3 continuation without replacing its pending gate."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.s3_ladder_followup import discover, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec")
    p.add_argument("--checkpoints", default="/home/ryreu/atlas/HLT_Classification/checkpoints")
    p.add_argument("--preflight-job", required=True)
    p.add_argument("--controller-commit", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    path = a.spec or discover(a.checkpoints, a.preflight_job)
    run(path, preflight_job=a.preflight_job, controller_commit=a.controller_commit, execute=a.execute)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"S3 FOLLOWUP STOPPED: {exc}\nPreserve artifacts and journals; do not blindly resubmit.",
              file=sys.stderr, flush=True)
        raise
