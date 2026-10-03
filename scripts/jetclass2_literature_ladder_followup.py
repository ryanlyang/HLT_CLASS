#!/usr/bin/env python
"""Wait for the exact existing literature gate, then submit its debug science DAG."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.literature_ladder_followup import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-spec", required=True)
    parser.add_argument("--gate-hash", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--preflight-job", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.gate_spec, gate_hash=args.gate_hash, source_commit=args.source_commit,
        preflight_job=args.preflight_job, execute=args.execute)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FOLLOWUP STOPPED: {exc}\nPreserve all artifacts and submission journals; do not blindly resubmit.",
              file=sys.stderr, flush=True)
        raise
