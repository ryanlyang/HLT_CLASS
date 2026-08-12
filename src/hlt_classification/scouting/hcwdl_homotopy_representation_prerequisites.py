"""One-shot authenticated prerequisites for the HCWDL-U-RKD campaign.

The representation recipe is a reusable v5 scientific overlay.  This module
materializes it, its installed-Weaver architecture evidence, the numerical
acceptance, and the committed spectral kernels from one already authenticated
historical HCWDL campaign.  It performs no training and opens no final-test
role.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.models import hcwdl_surfaces
from hlt_classification.models import scouting_particle_transformer
from hlt_classification.models.hcwdl_surfaces import (
    HCWDL_PARENT_ARCHITECTURE_NODES,
    build_architecture_attestation_from_files,
    tap_schema,
    validate_architecture_attestation,
)
from hlt_classification.provenance import capture_source_snapshot

from .hcwdl_homotopy_representation_campaign import (
    authenticate_parent, build_integration_attestation,
)
from .hcwdl_homotopy_representation_contracts import (
    PREREQUISITE_BUNDLE_CONTRACT, RECIPE_COMPATIBILITY_CONTRACT,
)
from .hcwdl_numerical_acceptance import build_numerical_acceptance
from .hcwdl_recipe import PRIMARY_RECIPE_PROFILE, validate_recipe
from .hcwdl_representation_dense_teacher import (
    build_dense_teacher_import_from_files, validate_dense_teacher_import,
)
from .hcwdl_representation_graph import (
    ascent_graph_artifact, control_registry_artifact,
    validate_ascent_graph_artifact, validate_control_registry_artifact,
)
from .hcwdl_representation_kernels import (
    generate_spectral_resource_bundle, publish_spectral_resources,
    spectral_resource_logical_hashes,
)
from .hcwdl_representation_recipe import (
    build_representation_recipe, derive_recipe_producer_source_sha256,
    derive_representation_recipe_evidence, validate_representation_recipe,
)
from .hcwdl_representation_runtime_adapters import (
    build_installed_weaver_surface_parity_artifact,
)
from .hcwdl_representation_smoke import measure_zero_coefficient_parity
from .highcov_cache import DenseAssignmentStore


_EXECUTION_POLICY_FIELDS = (
    "recipe_profile", "purpose", "repair_family", "training_passes",
    "validation_every_passes", "batching", "optimizer", "schedule",
    "coefficient_schedule", "single_teacher_coefficients",
    "dual_teacher_coefficients", "controls", "single_privileged_temperature",
    "predecessor_temperature", "privileged_temperature",
    "dual_teacher_peak_learning_rate", "amp_dtype", "class_weights",
)


def _execution_policy(recipe: Mapping[str, Any]) -> dict[str, Any]:
    validate_recipe(
        recipe, require_authorized=True, expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    weighting = recipe["class_weighting"]
    return {
        **{name: recipe[name] for name in _EXECUTION_POLICY_FIELDS},
        "class_weighting_policy": weighting["policy"],
    }


def _without_allowed_lineage(recipe: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(recipe))
    result.pop("content_hash", None)
    result.pop("evidence", None)
    weighting = result.get("class_weighting")
    if not isinstance(weighting, dict):
        raise ValueError("HCWDL-U-RKD recipe class-weight lineage differs")
    weighting.pop("train_class_counts", None)
    weighting.pop("train_row_selection_sha256", None)
    return result


def build_recipe_compatibility(
    *, execution_recipe_path: str | Path, donor_recipe_path: str | Path,
) -> dict[str, Any]:
    execution_path = Path(execution_recipe_path).resolve()
    donor_path = Path(donor_recipe_path).resolve()
    execution = load_json(execution_path); donor = load_json(donor_path)
    execution_hash = validate_recipe(
        execution, require_authorized=True, expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    donor_hash = validate_recipe(
        donor, require_authorized=True, expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    execution_policy = _execution_policy(execution)
    donor_policy = _execution_policy(donor)
    if execution_policy != donor_policy:
        raise ValueError("U/J and representation-donor execution policies differ")
    if _without_allowed_lineage(execution) != _without_allowed_lineage(donor):
        raise ValueError(
            "U/J and representation-donor recipes differ outside authorized lineage"
        )
    return with_content_hash({
        "contract": RECIPE_COMPATIBILITY_CONTRACT, "schema_version": 1,
        "parents": {
            "execution_recipe": execution_hash, "donor_recipe": donor_hash,
        },
        "recipe_paths": {
            "execution_recipe": str(execution_path), "donor_recipe": str(donor_path),
        },
        "recipe_byte_sha256": {
            "execution_recipe": sha256_file(execution_path),
            "donor_recipe": sha256_file(donor_path),
        },
        "execution_policy": execution_policy,
        "allowed_lineage_differences": [
            "content_hash", "evidence", "class_weighting.train_class_counts",
            "class_weighting.train_row_selection_sha256",
        ],
        "policy_equivalent": True, "final_test_accessed": False,
    })


def validate_recipe_compatibility(
    value: Mapping[str, Any], *, execution_recipe: Mapping[str, Any] | None = None,
    representation_recipe: Mapping[str, Any] | None = None,
    verify_files: bool = True,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RECIPE_COMPATIBILITY_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("policy_equivalent") is not True
        or value.get("final_test_accessed") is not False
        or value.get("allowed_lineage_differences") != [
            "content_hash", "evidence", "class_weighting.train_class_counts",
            "class_weighting.train_row_selection_sha256",
        ]
    ):
        raise ValueError("HCWDL-U-RKD recipe compatibility semantics differ")
    paths = value.get("recipe_paths"); byte_hashes = value.get("recipe_byte_sha256")
    parents = value.get("parents")
    if not all(isinstance(item, Mapping) for item in (paths, byte_hashes, parents)):
        raise ValueError("HCWDL-U-RKD recipe compatibility lineage differs")
    if verify_files:
        rebuilt = build_recipe_compatibility(
            execution_recipe_path=str(paths["execution_recipe"]),
            donor_recipe_path=str(paths["donor_recipe"]),
        )
        if dict(value) != rebuilt:
            raise ValueError("HCWDL-U-RKD recipe compatibility files changed")
    if execution_recipe is not None:
        execution_hash = validate_recipe(
            execution_recipe, require_authorized=True,
            expected_profile=PRIMARY_RECIPE_PROFILE,
        )
        if parents.get("execution_recipe") != execution_hash:
            raise ValueError("HCWDL-U-RKD execution recipe compatibility differs")
    if representation_recipe is not None:
        donor_hash = representation_recipe.get("parents", {}).get("parent_recipe")
        if parents.get("donor_recipe") != donor_hash:
            raise ValueError("HCWDL-U-RKD donor recipe compatibility differs")
    if value.get("execution_policy") != _execution_policy(
        execution_recipe if execution_recipe is not None
        else load_json(str(paths["execution_recipe"]))
    ):
        raise ValueError("HCWDL-U-RKD compatible execution policy changed")
    return digest


def _historical_file(root: Path, spec: Mapping[str, Any], field: str,
                     fallback: str) -> Path:
    raw = spec.get(field)
    path = Path(str(raw)) if isinstance(raw, str) and raw else root / fallback
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"historical {field} is absent: {path}")
    return path


def _report_registry(root: Path) -> dict[str, Path]:
    reports = {
        node_id: (root / "training" / node_id / "hcwdl_training_report.json").resolve()
        for node_id in sorted(HCWDL_PARENT_ARCHITECTURE_NODES)
    }
    missing = [str(path) for path in reports.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "historical architecture report registry is incomplete: "
            + ", ".join(missing[:3])
        )
    return reports


def prepare_prerequisites(
    *, parent_homotopy_spec: str | Path, historical_campaign_root: str | Path,
    historical_project_dir: str | Path, project_dir: str | Path,
    output_root: str | Path, source_commit: str, device: str = "cpu",
) -> dict[str, Any]:
    """Publish all non-training prerequisites and return their manifest."""

    parent = authenticate_parent(parent_homotopy_spec)
    historical_root = Path(historical_campaign_root).resolve()
    historical_project = Path(historical_project_dir).resolve()
    project = Path(project_dir).resolve()
    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-U-RKD prerequisite root is not empty")
    if device not in {"cpu", "cuda"}:
        raise ValueError("HCWDL-U-RKD prerequisite device differs")
    source_snapshot = capture_source_snapshot(project, require_clean=True)
    if (
        source_snapshot.get("git_commit") != source_commit
        or source_snapshot.get("worktree_clean") is not True
    ):
        raise PermissionError("HCWDL-U-RKD prerequisite source checkout differs")

    historical_spec_path = historical_root / "campaign_spec.json"
    if not historical_spec_path.is_file():
        raise FileNotFoundError("historical HCWDL campaign specification is absent")
    historical_spec = load_json(historical_spec_path)
    historical_spec_hash = validate_content_hash(
        historical_spec, expected_contract=str(historical_spec["contract"]),
        expected_schema_version=int(historical_spec["schema_version"]),
    )
    historical_recipe_path = _historical_file(
        historical_root, historical_spec, "recipe_path", "recipe.json",
    )
    historical_recipe = load_json(historical_recipe_path)
    historical_recipe_hash = validate_recipe(
        historical_recipe, require_authorized=True,
        expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    current_recipe = load_json(parent["spec"]["recipe_path"])
    validate_recipe(
        current_recipe, require_authorized=True,
        expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    compatibility = build_recipe_compatibility(
        execution_recipe_path=parent["spec"]["recipe_path"],
        donor_recipe_path=historical_recipe_path,
    )

    source_manifest_path = _historical_file(
        historical_root, historical_spec, "source_manifest_path",
        "data/source_manifest.json",
    )
    split_manifest_path = _historical_file(
        historical_root, historical_spec, "split_manifest_path",
        "data/splits/split_manifest.json",
    )
    selection_path = _historical_file(
        historical_root, historical_spec, "selection_manifest_path",
        "source/row_selection.json",
    )
    toff_wrapper = historical_root / "training/TOFF/hcwdl_training_report.json"
    d0w_engine = historical_root / "training/D0w/training_report.json"
    for name, path in (("TOFF wrapper", toff_wrapper), ("D0w engine", d0w_engine)):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"historical {name} is absent: {path}")

    tap_path = root / "architecture/tap.json"
    parity_path = root / "architecture/surface_parity.json"
    architecture_path = root / "architecture/architecture_attestation.json"
    numerical_path = root / "acceptance/numerical_acceptance.json"
    zero_path = root / "acceptance/zero_coefficient_parity.json"
    teacher_path = root / "lineage/dense_teacher_import.json"
    graph_path = root / "lineage/representation_graph.json"
    controls_path = root / "lineage/representation_controls.json"
    recipe_path = root / "recipe/representation_recipe_v5.json"
    integration_path = root / "integration/integration_attestation.json"
    compatibility_path = root / "integration/recipe_compatibility.json"
    kernel_reference_path = root / "kernels/kernel_reference.json"

    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(compatibility_path, compatibility)
    write_immutable_json(tap_path, tap_schema())
    parity = build_installed_weaver_surface_parity_artifact(device=device)
    write_immutable_json(parity_path, parity)

    architecture = build_architecture_attestation_from_files(
        tap_schema_path=tap_path, surface_parity_path=parity_path,
        parent_reports=_report_registry(historical_root),
        model_source_paths={
            "D0w": d0w_engine,
            "hcwdl_surfaces": Path(hcwdl_surfaces.__file__).resolve(),
            "scouting_particle_transformer": Path(
                scouting_particle_transformer.__file__
            ).resolve(),
        },
    )
    architecture_hash = validate_architecture_attestation(
        architecture, require_authorized=True,
    )
    write_immutable_json(architecture_path, architecture)

    teacher = build_dense_teacher_import_from_files(
        authority_files={
            "campaign_spec": historical_spec_path,
            "recipe": historical_recipe_path,
            "source_manifest": source_manifest_path,
            "split_manifest": split_manifest_path,
            "row_selection": selection_path,
            "surface_parity": parity_path,
            "toff_wrapper_report": toff_wrapper,
        },
        historical_project_dir=historical_project,
    )
    teacher_hash = validate_dense_teacher_import(teacher)
    write_immutable_json(teacher_path, teacher)

    graph = ascent_graph_artifact(parents={
        "parent_graph": teacher["payload"]["historical_parent_graph_sha256"],
        "parent_import": teacher_hash,
    })
    graph_hash = validate_ascent_graph_artifact(graph)
    controls = control_registry_artifact(ascent_graph_artifact_sha256=graph_hash)
    controls_hash = validate_control_registry_artifact(
        controls, ascent_graph_artifact_sha256=graph_hash,
    )
    write_immutable_json(graph_path, graph)
    write_immutable_json(controls_path, controls)

    numerical = build_numerical_acceptance()
    zero = measure_zero_coefficient_parity(device=device)
    write_immutable_json(numerical_path, numerical)
    write_immutable_json(zero_path, zero)
    evidence = derive_representation_recipe_evidence(
        numerical_acceptance=numerical,
        zero_coefficient_measurements=zero,
    )
    kernels = generate_spectral_resource_bundle()
    train_assignment = DenseAssignmentStore(
        historical_root / "matcher/train_assignment_manifest.json"
    )
    parents = {
        "assignment_manifest": train_assignment.manifest["content_hash"],
        "dense_teacher_import": teacher_hash,
        "historical_parent_graph": teacher["payload"][
            "historical_parent_graph_sha256"
        ],
        "kernel_resources": kernels.content_hash,
        "parent_recipe": historical_recipe_hash,
        "producer_source": derive_recipe_producer_source_sha256(project),
        "representation_ascent_graph": graph_hash,
        "representation_control_registry": controls_hash,
        "row_selection": teacher["parents"]["historical_row_selection"],
        "source_manifest": teacher["parents"]["historical_source_manifest"],
        "split_manifest": teacher["parents"]["historical_split_manifest"],
    }
    recipe = build_representation_recipe(
        parents=parents,
        kernel_array_logical_hashes=spectral_resource_logical_hashes(kernels),
        evidence=evidence,
    )
    recipe_hash = validate_representation_recipe(recipe, expected_parents=parents)
    write_immutable_json(recipe_path, recipe)

    published = publish_spectral_resources(
        kernels, root=root / "kernels/envelope",
        producer_task_id="prepare_hcwdl_u_rkd_prerequisites",
        immutable_parent_hashes={
            "architecture_attestation": architecture_hash,
            "numerical_acceptance": numerical["content_hash"],
            "representation_recipe": recipe_hash,
        },
        registered_output_row={
            "task_key": "prerequisite_kernel_resources",
            "registered_output": "kernels/envelope/committed/${envelope_id}",
        },
        campaign_or_recovery_owner={
            "parent_homotopy_spec": parent["spec_sha256"],
            "source_commit": source_commit,
        },
    )
    kernel_reference = {
        "committed_directory": str(published.envelope.directory.resolve())
    }
    write_immutable_json(kernel_reference_path, kernel_reference)

    integration = build_integration_attestation(
        repository=project, source_commit=source_commit,
        architecture_attestation=architecture,
        numerical_acceptance=numerical,
        recipe_compatibility=compatibility,
    )
    write_immutable_json(integration_path, integration)
    bundle = with_content_hash({
        "contract": PREREQUISITE_BUNDLE_CONTRACT, "schema_version": 1,
        "parent_homotopy_spec_sha256": parent["spec_sha256"],
        "historical_campaign_spec_sha256": historical_spec_hash,
        "source_commit": source_commit,
        "paths": {
            "architecture_attestation": str(architecture_path.resolve()),
            "integration_attestation": str(integration_path.resolve()),
            "kernel_reference": str(kernel_reference_path.resolve()),
            "numerical_acceptance": str(numerical_path.resolve()),
            "recipe_compatibility": str(compatibility_path.resolve()),
            "representation_recipe": str(recipe_path.resolve()),
        },
        "hashes": {
            "architecture_attestation": architecture_hash,
            "integration_attestation": integration["content_hash"],
            "kernel_resources": kernels.content_hash,
            "numerical_acceptance": numerical["content_hash"],
            "recipe_compatibility": compatibility["content_hash"],
            "representation_recipe": recipe_hash,
        },
        "training_jobs_run": 0, "final_test_accessed": False,
    })
    write_immutable_json(root / "prerequisite_bundle.json", bundle)
    return bundle


__all__ = [
    "PREREQUISITE_BUNDLE_CONTRACT", "RECIPE_COMPATIBILITY_CONTRACT",
    "build_recipe_compatibility", "prepare_prerequisites",
    "validate_recipe_compatibility",
]
