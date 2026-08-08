#!/usr/bin/env python3
"""Run one stage of the immutable 64-model exploratory comparison."""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.exploratory_test import (  # noqa: E402
    EXPECTED_MODEL_COUNT, EXPLORATORY_TEST_TASKS,
    aggregate_exploratory_test, authorize_exploratory_test,
    build_exploratory_row_selection, evaluate_exploratory_model,
    validate_exploratory_test_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploratory-test-spec", type=Path, required=True)
    parser.add_argument("--task", choices=EXPLORATORY_TEST_TASKS, required=True)
    args = parser.parse_args()
    spec = load_json(args.exploratory_test_spec)
    validate_exploratory_test_spec(spec)
    validate_source_snapshot(
        spec["source_snapshot"], repository=REPO_ROOT, require_clean=True,
    )
    if args.task == "authorize":
        authorize_exploratory_test(spec)
    elif args.task == "row_selection":
        build_exploratory_row_selection(spec)
    elif args.task == "evaluation":
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None or not raw.isdigit() or int(raw) >= EXPECTED_MODEL_COUNT:
            raise ValueError("evaluation requires a valid SLURM_ARRAY_TASK_ID")
        evaluate_exploratory_model(spec, index=int(raw), device="cuda")
    else:
        aggregate_exploratory_test(spec)
    return 0


if __name__ == "__main__": raise SystemExit(main())
