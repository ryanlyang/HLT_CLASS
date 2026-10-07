#!/usr/bin/env python
"""Arm conditional Oscar 1M science using the already submitted full gate."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.context_full_followup import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gate-spec", "gate-hash", "source-commit", "preflight-job", "project-dir", "executor-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.gate_spec, gate_hash=args.gate_hash, source_commit=args.source_commit,
        preflight_job=args.preflight_job, project_dir=args.project_dir,
        executor_commit=args.executor_commit, execute=args.execute)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FOLLOWUP STOPPED: {exc}\nPreserve all artifacts/journals; do not blindly resubmit.",
              file=sys.stderr, flush=True)
        raise
