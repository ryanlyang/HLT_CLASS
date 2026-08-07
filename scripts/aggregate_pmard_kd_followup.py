#!/usr/bin/env python3
"""Aggregate the complete paired CE/self-KD/T100 schedule follow-up."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.kd_followup import (  # noqa: E402
    aggregate_kd_followup, validate_kd_followup_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-spec", type=Path, required=True)
    args = parser.parse_args(); spec = load_json(args.followup_spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    validate_kd_followup_inputs(spec)
    report = aggregate_kd_followup(spec)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
