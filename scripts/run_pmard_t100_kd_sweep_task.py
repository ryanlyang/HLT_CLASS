#!/usr/bin/env python3
"""Run one registered task from an immutable supplemental T100 KD sweep."""

from __future__ import annotations

import argparse, os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.kd_sweep import (  # noqa: E402
    t100_sweep_grid, validate_t100_sweep_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-spec", type=Path, required=True)
    parser.add_argument("--task", choices=("teacher_targets", "grid", "aggregate"), required=True)
    args = parser.parse_args(); spec = load_json(args.sweep_spec)
    validate_t100_sweep_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    if args.task == "teacher_targets":
        command = [sys.executable, "-s", str(REPO_ROOT / "scripts/build_pmard_t100_kd_targets.py"),
                   "--sweep-spec", str(args.sweep_spec), "--device", "cuda"]
    elif args.task == "grid":
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None or not raw.isdigit() or int(raw) >= len(t100_sweep_grid()):
            raise ValueError("grid task requires a valid SLURM_ARRAY_TASK_ID")
        command = [sys.executable, "-s", str(REPO_ROOT / "scripts/train_pmard_t100_kd_sweep.py"),
                   "--sweep-spec", str(args.sweep_spec), "--index", raw, "--device", "cuda"]
    else:
        command = [sys.executable, "-s", str(REPO_ROOT / "scripts/aggregate_pmard_t100_kd_sweep.py"),
                   "--sweep-spec", str(args.sweep_spec)]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
