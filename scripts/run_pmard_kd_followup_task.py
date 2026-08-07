#!/usr/bin/env python3
"""Run one task from an immutable paired KD schedule follow-up."""

from __future__ import annotations

import argparse, os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.kd_followup import (  # noqa: E402
    KD_FOLLOWUP_TASKS, validate_kd_followup_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-spec", type=Path, required=True)
    parser.add_argument("--task", choices=KD_FOLLOWUP_TASKS, required=True)
    args = parser.parse_args(); spec = load_json(args.followup_spec)
    validate_kd_followup_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    if args.task == "grid":
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None or not raw.isdigit() or int(raw) >= len(spec["registry"]):
            raise ValueError("grid task requires a valid SLURM_ARRAY_TASK_ID")
        command = [
            sys.executable, "-s", str(REPO_ROOT / "scripts/train_pmard_kd_followup.py"),
            "--followup-spec", str(args.followup_spec), "--index", raw, "--device", "cuda",
        ]
    else:
        command = [
            sys.executable, "-s", str(REPO_ROOT / "scripts/aggregate_pmard_kd_followup.py"),
            "--followup-spec", str(args.followup_spec),
        ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
