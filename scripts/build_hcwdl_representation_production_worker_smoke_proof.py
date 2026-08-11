#!/usr/bin/env python3
"""Build the exact safe-prefix production-worker smoke proof."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_production_worker_smoke_proof,
)
from hlt_classification.scouting.hcwdl_representation_resources import artifact_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--planning-spec-sha256", required=True)
    parser.add_argument("--runtime-binding-sha256", required=True)
    parser.add_argument("--ordinary-runtime-measurement", type=Path, required=True)
    parser.add_argument("--deterministic-runtime-measurement", type=Path, required=True)
    parser.add_argument("--completed-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = load_json_mapping(args.completed_rows)
    ordered = raw.get("rows")
    if not isinstance(ordered, list):
        raise ValueError("completed-row registry must contain an ordered rows list")
    rows = []
    for row in ordered:
        normalized = dict(row)
        normalized["output"] = artifact_reference(Path(normalized["output"]))
        rows.append(normalized)
    proof = build_production_worker_smoke_proof(
        source_commit=args.source_commit,
        planning_spec_sha256=args.planning_spec_sha256,
        runtime_binding_sha256=args.runtime_binding_sha256,
        ordinary_runtime_measurement=artifact_reference(
            args.ordinary_runtime_measurement
        ),
        deterministic_runtime_measurement=artifact_reference(
            args.deterministic_runtime_measurement
        ),
        completed_rows=rows,
    )
    publish(args.output, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
