#!/usr/bin/env python3
"""Build a genuine HCWDL-RKD resource profile from authenticated evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference, build_measured_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "production"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--ordinary-worker", type=Path, required=True)
    parser.add_argument("--deterministic-worker", type=Path, required=True)
    parser.add_argument("--measurement-registry", type=Path, required=True)
    parser.add_argument("--array-concurrency-limits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = load_json_mapping(args.measurement_registry)
    measurements = {}
    for resource_class, row in raw.items():
        if not isinstance(row, dict) or set(row) != {
            "scheduler_evidence", "miniature_evidence",
        }:
            raise ValueError("measurement registry row differs")
        measurements[resource_class] = {
            name: artifact_reference(Path(value)) for name, value in row.items()
        }
    concurrency = (
        {} if args.array_concurrency_limits is None
        else load_json_mapping(args.array_concurrency_limits)
    )
    profile = build_measured_profile(
        mode=args.mode, source_commit=args.source_commit,
        production_workers={
            "ordinary": artifact_reference(args.ordinary_worker),
            "deterministic": artifact_reference(args.deterministic_worker),
        },
        measurements=measurements,
        array_concurrency_limits={key: int(value) for key, value in concurrency.items()},
    )
    publish(args.output, profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
