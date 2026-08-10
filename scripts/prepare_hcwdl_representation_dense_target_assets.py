#!/usr/bin/env python3
"""Publish all pre-campaign target authorities for the dense four-track DAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, load_json_mapping, publish
from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.scouting.hcwdl_representation_target_planning import (
    build_dense_target_planning_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-campaign-spec", type=Path, required=True)
    parser.add_argument("--dense-teacher-import", type=Path, required=True)
    parser.add_argument("--representation-graph", type=Path, required=True)
    parser.add_argument("--representation-recipe", type=Path, required=True)
    parser.add_argument("--tap-schema", type=Path, required=True)
    parser.add_argument("--surface-parity", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-row-selection", type=Path, required=True)
    parser.add_argument("--train-assignment-manifest", type=Path, required=True)
    parser.add_argument("--gpu-target-runtime-measurement", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--base-static-inputs", type=Path, required=True)
    parser.add_argument("--base-artifact-content-hashes", type=Path, required=True)
    parser.add_argument("--static-inputs-output", type=Path, required=True)
    parser.add_argument("--artifact-content-hashes-output", type=Path, required=True)
    parser.add_argument("--target-generations-output", type=Path, required=True)
    args = parser.parse_args()
    assets = build_dense_target_planning_assets(
        planning_campaign_spec=artifact(args.planning_campaign_spec),
        dense_teacher_import=artifact(args.dense_teacher_import),
        representation_graph=artifact(args.representation_graph),
        representation_recipe=artifact(args.representation_recipe),
        tap_schema=artifact(args.tap_schema),
        surface_parity=artifact(args.surface_parity),
        source_manifest=artifact(args.source_manifest),
        split_manifest=artifact(args.split_manifest),
        train_row_selection=artifact(args.train_row_selection),
        train_assignment_manifest=artifact(args.train_assignment_manifest),
        gpu_target_runtime_measurement=artifact(
            args.gpu_target_runtime_measurement
        ),
        project_dir=args.project_dir,
    )
    static_inputs = load_json_mapping(args.base_static_inputs)
    content_hashes = load_json_mapping(args.base_artifact_content_hashes)
    root = args.representation_root.resolve()
    for bank, logical in sorted(assets["logical_banks"].items()):
        path = root / "targets" / bank / "logical_bank.json"
        publish(path, logical)
        key = f"${{logical_bank:{bank}}}"
        if key in static_inputs or key in content_hashes:
            raise PermissionError(f"base target registry already contains {key}")
        static_inputs[key] = {"path": str(path), "sha256": sha256_file(path)}
        content_hashes[key] = logical["content_hash"]
    for key, registry in sorted(assets["consumer_registries"].items()):
        bank, purpose = key.split(":", 1)
        generation = assets["target_generations"][key]["generation_id"]
        directory = root / "targets" / bank / "generations" / generation
        registry_path = directory / "consumer_registry.json"
        forward_path = directory / "target_forward_spec.json"
        publish(registry_path, registry)
        publish(forward_path, assets["forward_specs"][key])
        registry_key = f"${{target_consumer_registry:{bank}:{purpose}}}"
        forward_key = f"${{target_forward_spec:{bank}:{purpose}}}"
        if any(item in static_inputs or item in content_hashes for item in (
            registry_key, forward_key,
        )):
            raise PermissionError(f"base target registry already contains {key}")
        static_inputs[registry_key] = {
            "path": str(registry_path), "sha256": sha256_file(registry_path),
        }
        content_hashes[registry_key] = registry["content_hash"]
        static_inputs[forward_key] = {
            "path": str(forward_path), "sha256": sha256_file(forward_path),
        }
        content_hashes[forward_key] = assets["forward_specs"][key]["content_hash"]
    publish(args.static_inputs_output, static_inputs)
    publish(args.artifact_content_hashes_output, content_hashes)
    publish(args.target_generations_output, assets["target_generations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
