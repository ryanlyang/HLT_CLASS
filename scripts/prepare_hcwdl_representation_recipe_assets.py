#!/usr/bin/env python3
"""Build the dense RKD graph, control registry, and v4 overlay recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_numerical_acceptance import (
    build_numerical_acceptance,
)
from hlt_classification.scouting.hcwdl_recipe import (
    PRIMARY_RECIPE_PROFILE,
    validate_recipe,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    ascent_graph_artifact,
    control_registry_artifact,
    validate_ascent_graph_artifact,
    validate_control_registry_artifact,
)
from hlt_classification.scouting.hcwdl_representation_kernels import (
    generate_spectral_resource_bundle,
    spectral_resource_logical_hashes,
)
from hlt_classification.scouting.hcwdl_representation_dense_teacher import (
    validate_dense_teacher_import,
)
from hlt_classification.scouting.hcwdl_representation_recipe import (
    build_representation_recipe,
    derive_recipe_producer_source_sha256,
    derive_representation_recipe_evidence,
    validate_representation_recipe,
)
from hlt_classification.scouting.hcwdl_representation_smoke import (
    measure_zero_coefficient_parity,
)
from hlt_classification.scouting.hcwdl_representation_reporting import (
    build_dense_training_disposition,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--dense-teacher-import", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument(
        "--project-dir", type=Path, required=True,
        help="clean Git checkout whose measured source snapshot produces the recipe",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.representation_root.is_absolute():
        raise ValueError("representation root must be absolute")
    if not args.project_dir.is_absolute():
        raise ValueError("project directory must be absolute")

    teacher_import = artifact(args.dense_teacher_import)
    teacher_import_sha256 = validate_dense_teacher_import(teacher_import)
    parent_recipe = artifact(args.parent_recipe)
    parent_recipe_sha256 = validate_recipe(
        parent_recipe, require_authorized=True,
        expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    if teacher_import["parents"]["historical_recipe"] != parent_recipe_sha256:
        raise ValueError("parent recipe differs from the dense teacher import")
    assignment_manifest = artifact(args.assignment_manifest)
    from hlt_classification.scouting.highcov_cache import validate_assignment_manifest
    assignment_manifest_sha256 = validate_assignment_manifest(assignment_manifest)

    graph = ascent_graph_artifact(parents={
        "parent_graph": teacher_import["payload"]["historical_parent_graph_sha256"],
        "parent_import": teacher_import_sha256,
    })
    graph_sha256 = validate_ascent_graph_artifact(graph)
    controls = control_registry_artifact(
        ascent_graph_artifact_sha256=graph_sha256,
    )
    controls_sha256 = validate_control_registry_artifact(
        controls, ascent_graph_artifact_sha256=graph_sha256,
    )
    kernels = generate_spectral_resource_bundle()
    numerical = build_numerical_acceptance()
    zero_measurements = measure_zero_coefficient_parity(device=args.device)
    evidence = derive_representation_recipe_evidence(
        numerical_acceptance=numerical,
        zero_coefficient_measurements=zero_measurements,
    )
    parents = {
        "assignment_manifest": assignment_manifest_sha256,
        "dense_teacher_import": teacher_import_sha256,
        "historical_parent_graph": teacher_import["payload"][
            "historical_parent_graph_sha256"
        ],
        "kernel_resources": kernels.content_hash,
        "parent_recipe": parent_recipe_sha256,
        "producer_source": derive_recipe_producer_source_sha256(
            args.project_dir,
        ),
        "representation_ascent_graph": graph_sha256,
        "representation_control_registry": controls_sha256,
        "row_selection": teacher_import["parents"]["historical_row_selection"],
        "source_manifest": teacher_import["parents"]["historical_source_manifest"],
        "split_manifest": teacher_import["parents"]["historical_split_manifest"],
    }
    recipe = build_representation_recipe(
        parents=parents,
        kernel_array_logical_hashes=spectral_resource_logical_hashes(kernels),
        evidence=evidence,
    )
    validate_representation_recipe(recipe, expected_parents=parents)
    disposition = build_dense_training_disposition(
        dense_teacher_import_sha256=teacher_import_sha256,
        representation_recipe_sha256=recipe["content_hash"],
        graph_sha256=graph_sha256,
    )

    root = args.representation_root
    publish(root / "graph" / "ascent_graph.json", graph)
    publish(root / "controls" / "registry.json", controls)
    publish(root / "recipes" / "representation_recipe.json", recipe)
    publish(root / "import" / "dense_training_disposition.json", disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
