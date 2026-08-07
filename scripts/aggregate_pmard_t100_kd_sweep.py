#!/usr/bin/env python3
"""Aggregate and select the complete validation-only T100 KD sweep."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.kd_sweep import (  # noqa: E402
    aggregate_t100_sweep, validate_t100_sweep_inputs, validate_t100_sweep_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-spec", type=Path, required=True)
    args = parser.parse_args(); spec = load_json(args.sweep_spec)
    validate_t100_sweep_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    validate_t100_sweep_inputs(spec)
    report = aggregate_t100_sweep(spec)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
