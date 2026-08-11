#!/usr/bin/env python3
"""Run the non-authorizing scientific and synthetic-final HCWDL-RKD smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.hcwdl_representation_campaign import LOCAL_SMOKE_CONTRACT
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    LOCAL_SEMANTIC_COVERAGE, build_local_planning_handlers,
    local_scientific_probe,
)
from hlt_classification.scouting.hcwdl_representation_workflow import exercise_registered_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--scientific-probe-output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device != "cpu":
        raise PermissionError(
            "local planning smoke is fixed to CPU; CUDA belongs to genuine-worker acceptance"
        )
    spec = artifact(args.campaign_spec)
    kinds = {row["kind"] for row in spec["tasks"]}
    handlers = build_local_planning_handlers(kinds)
    rows = exercise_registered_rows(spec, handlers=handlers)
    invalid = [
        row for row in rows
        if row["result"].get("semantic_fixture_executed") is not True
        or row["result"].get("generic_fallback") is not False
        or row["result"].get("semantic_surface")
        != LOCAL_SEMANTIC_COVERAGE[
            next(
                task["kind"] for task in spec["tasks"]
                if task["task_key"] == row["task_key"]
            )
        ]
    ]
    if invalid:
        raise RuntimeError(
            "local smoke contains generic/structural/mock semantic success"
        )
    scientific_probe = local_scientific_probe()
    probe_output = (
        args.scientific_probe_output
        if args.scientific_probe_output is not None
        else args.output.parent / "smoke_probe.json"
    )
    publish(probe_output, scientific_probe)
    report = with_content_hash(
        {
            "contract": LOCAL_SMOKE_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"],
            "rows": rows,
            "registered_rows_exercised": len(rows),
            "registered_task_count": len(spec["tasks"]),
            "final_rows_structural_only": 0,
            "final_rows_synthetic_pipeline": sum(
                bool(row["result"].get("synthetic_final_pipeline")) for row in rows
            ),
            "all_registered_rows_real_semantics": all(
                row["result"].get("semantic_fixture_executed") is True
                and row["result"].get("generic_fallback") is False
                and not bool(row["result"].get("structural_only"))
                for row in rows
            ),
            "semantic_coverage_registry": dict(sorted(LOCAL_SEMANTIC_COVERAGE.items())),
            "semantic_coverage_registry_sha256": with_content_hash({
                "contract": "HCWDL_REPRESENTATION_LOCAL_SEMANTIC_COVERAGE/v1",
                "schema_version": 1,
                "coverage": dict(sorted(LOCAL_SEMANTIC_COVERAGE.items())),
            })["content_hash"],
            "scientific_full_loss_probe_path": str(probe_output),
            "scientific_full_loss_probe_sha256": scientific_probe["content_hash"],
            "scientific_primary_executions": scientific_probe["primary_count"],
            "scientific_control_executions": scientific_probe["control_count"],
            "final_role_accessed": False,
            "authorizes_tigris_or_pilot": False,
        }
    )
    publish(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
