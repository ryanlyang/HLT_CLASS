#!/usr/bin/env python3
"""Build the genuine four-class resource profile for the dense-only DAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference, build_dense_measured_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--ordinary-worker", type=Path, required=True)
    parser.add_argument("--deterministic-worker", type=Path, required=True)
    parser.add_argument("--measurement-registry", type=Path, required=True)
    parser.add_argument("--array-concurrency-limits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = load_json_mapping(args.measurement_registry)
    measurements = {
        resource_class: {
            name: artifact_reference(Path(value)) for name, value in row.items()
        }
        for resource_class, row in raw.items()
        if isinstance(row, dict)
    }
    concurrency = (
        {} if args.array_concurrency_limits is None
        else {
            key: int(value) for key, value in load_json_mapping(
                args.array_concurrency_limits
            ).items()
        }
    )
    result = build_dense_measured_profile(
        source_commit=args.source_commit,
        production_workers={
            "ordinary": artifact_reference(args.ordinary_worker),
            "deterministic": artifact_reference(args.deterministic_worker),
        },
        measurements=measurements,
        array_concurrency_limits=concurrency,
    )
    publish(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
