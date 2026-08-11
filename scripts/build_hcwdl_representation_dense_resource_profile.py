#!/usr/bin/env python3
"""Build the genuine four-class resource profile for the dense-only DAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import load_json_mapping, publish
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.scouting.hcwdl_representation_recipe import (
    validate_representation_recipe,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference, build_dense_compatible_measured_profile,
    build_dense_measured_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit")
    parser.add_argument("--ordinary-worker", type=Path)
    parser.add_argument("--deterministic-worker", type=Path)
    parser.add_argument("--measurement-registry", type=Path)
    parser.add_argument("--array-concurrency-limits", type=Path)
    parser.add_argument("--base-profile", type=Path)
    parser.add_argument("--campaign-source-commit")
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--representation-recipe", type=Path)
    parser.add_argument("--collector-recovery-authorization", type=Path)
    parser.add_argument("--collector-recovery-ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compatibility_values = (
        args.base_profile, args.campaign_source_commit, args.project_dir,
        args.representation_recipe, args.collector_recovery_authorization,
        args.collector_recovery_ledger,
    )
    if any(value is not None for value in compatibility_values):
        if not all(value is not None for value in compatibility_values):
            parser.error("compatible reuse requires all compatibility inputs")
        if any(value is not None for value in (
            args.source_commit, args.ordinary_worker,
            args.deterministic_worker, args.measurement_registry,
            args.array_concurrency_limits,
        )):
            parser.error("compatible reuse cannot rebuild the measured profile")
        base_profile = load_json(args.base_profile)
        recipe = load_json(args.representation_recipe)
        recipe_sha256 = validate_representation_recipe(recipe)
        result = build_dense_compatible_measured_profile(
            base_profile=base_profile,
            base_profile_reference=artifact_reference(args.base_profile),
            project_dir=args.project_dir,
            campaign_source_commit=args.campaign_source_commit,
            representation_recipe_sha256=recipe_sha256,
            recipe_producer_source_sha256=recipe["parents"]["producer_source"],
            recovery_authorization=load_json(
                args.collector_recovery_authorization
            ),
            recovery_authorization_reference=artifact_reference(
                args.collector_recovery_authorization
            ),
            recovery_ledger=load_json(args.collector_recovery_ledger),
            recovery_ledger_reference=artifact_reference(
                args.collector_recovery_ledger
            ),
        )
        publish(args.output, result)
        return 0
    if any(value is None for value in (
        args.source_commit, args.ordinary_worker,
        args.deterministic_worker, args.measurement_registry,
    )):
        parser.error("exact-source profile requires source, workers, and measurements")
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
