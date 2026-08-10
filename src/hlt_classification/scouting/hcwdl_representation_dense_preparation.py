"""One-shot, non-submitting preparation of the dense HCWDL-RKD smoke candidate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import (
    load_json,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.provenance import (
    capture_source_snapshot,
    validate_source_snapshot_payload,
)

from .hcwdl_representation_campaign import (
    DENSE_TRAINING_DISPOSITION,
    build_command_plan,
    create_campaign_spec,
    validate_campaign_spec,
)
from .hcwdl_representation_candidate import build_executable_candidate_audit
from .hcwdl_representation_dense_teacher import validate_dense_teacher_import
from .hcwdl_representation_graph import validate_ascent_graph_artifact
from .hcwdl_representation_recipe import validate_representation_recipe
from .hcwdl_representation_reporting import validate_dense_training_disposition
from .hcwdl_representation_resources import (
    DENSE_RESOURCE_CLASSES,
    artifact_reference,
    dense_resource_measurement_source_commit,
    dense_resource_peak_rss_bytes,
    validate_dense_measured_profile,
    validate_dense_storage_availability,
    validate_dense_storage_estimate,
    validate_dense_storage_template,
)
from .hcwdl_representation_runtime_adapters import _directory_inventory
from .hcwdl_representation_runtime_binding import (
    build_runtime_binding,
    runtime_campaign_identity,
)
from .hcwdl_representation_runtime_rows import (
    build_runtime_dry_run_audit,
    build_runtime_prerequisites,
    build_runtime_task_rows,
)
from .hcwdl_representation_target_planning import (
    build_dense_target_planning_assets,
)
from .hcwdl_representation_worker_runtime import (
    build_row_runtime_signature,
    validate_live_worker_runtime,
    validate_worker_runtime_measurement,
)
from .selective_assignment import validate_row_selection
from .splits import validate_split_manifest


def _artifact(path: str | Path) -> dict[str, Any]:
    value = load_json(Path(path))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    write_immutable_json(path, value)


def _json_reference(path: Path) -> tuple[dict[str, str], str]:
    value = _artifact(path)
    return artifact_reference(path), require_sha256(
        value.get("content_hash"), name=f"{path.name} content hash",
    )


def _project_runtime_registry(
    *, project_dir: Path, current_source_commit: str,
    compatible_profile: Mapping[str, Any], measurement_root: Path,
    signature_root: Path, data_root: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    """Project measured package/device facts across the reviewed source-only diff."""

    snapshot = capture_source_snapshot(project_dir, require_clean=True)
    snapshot_sha256 = validate_source_snapshot_payload(snapshot)
    if snapshot.get("git_commit") != current_source_commit:
        raise PermissionError("dense preparation checkout differs from requested source")
    measured_source = dense_resource_measurement_source_commit(compatible_profile)
    signatures: dict[str, str] = {}
    signature_values: dict[str, dict[str, Any]] = {}
    shared_conda: str | None = None
    shared_weaver: str | None = None
    for resource_class in DENSE_RESOURCE_CLASSES:
        measurement_path = (
            measurement_root / resource_class / "worker_runtime_measurement.json"
        )
        measurement = _artifact(measurement_path)
        validate_worker_runtime_measurement(measurement)
        old_live = measurement["live_worker_runtime"]
        request = compatible_profile["requests"][resource_class]
        if (
            measurement.get("resource_class") != resource_class
            or measurement.get("resource_request") != request
            or old_live.get("source_commit") != measured_source
            or measurement.get("runtime_facts", {}).get("data_root")
            != str(data_root.resolve())
        ):
            raise PermissionError(
                f"dense runtime measurement lineage differs: {resource_class}"
            )
        live = dict(old_live)
        live.update({
            "project_dir": str(project_dir),
            "source_commit": current_source_commit,
            "source_snapshot_sha256": snapshot_sha256,
        })
        live.pop("content_hash", None)
        live = with_content_hash(live)
        validate_live_worker_runtime(live)
        signature = build_row_runtime_signature(live)
        signature_path = signature_root / f"{resource_class}.json"
        _publish(signature_path, signature)
        signatures[resource_class] = signature["content_hash"]
        signature_values[resource_class] = signature
        conda = str(live["conda"]["environment"])
        weaver = str(live["weaver_runtime_sha256"])
        if shared_conda not in (None, conda) or shared_weaver not in (None, weaver):
            raise PermissionError("dense runtime measurements do not share one environment")
        shared_conda, shared_weaver = conda, weaver
    assert shared_conda is not None and shared_weaver is not None
    facts = {
        "conda_environment": shared_conda,
        "data_root": str(data_root.resolve()),
        "device": "cuda",
        "project_dir": str(project_dir),
        "python_no_user_site": True,
        "source_snapshot_sha256": snapshot_sha256,
        "weaver_runtime_sha256": shared_weaver,
    }
    return facts, signatures, signature_values


def _publish_target_assets(
    *, root: Path, assets: Mapping[str, Any],
    base_static_inputs: Mapping[str, Any],
    base_content_hashes: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    static_inputs = dict(base_static_inputs)
    content_hashes = dict(base_content_hashes)
    for bank, logical in sorted(assets["logical_banks"].items()):
        path = root / "targets" / bank / "logical_bank.json"
        _publish(path, logical)
        key = f"${{logical_bank:{bank}}}"
        static_inputs[key] = artifact_reference(path)
        content_hashes[key] = logical["content_hash"]
    for key, registry in sorted(assets["consumer_registries"].items()):
        bank, purpose = key.split(":", 1)
        generation = assets["target_generations"][key]["generation_id"]
        directory = root / "targets" / bank / "generations" / generation
        registry_path = directory / "consumer_registry.json"
        forward_path = directory / "target_forward_spec.json"
        _publish(registry_path, registry)
        _publish(forward_path, assets["forward_specs"][key])
        registry_key = f"${{target_consumer_registry:{bank}:{purpose}}}"
        forward_key = f"${{target_forward_spec:{bank}:{purpose}}}"
        static_inputs[registry_key] = artifact_reference(registry_path)
        content_hashes[registry_key] = registry["content_hash"]
        static_inputs[forward_key] = artifact_reference(forward_path)
        content_hashes[forward_key] = assets["forward_specs"][key]["content_hash"]
    return static_inputs, content_hashes


def prepare_dense_smoke_candidate(
    *, representation_root: str | Path, project_dir: str | Path,
    compatible_resource_profile_path: str | Path,
    storage_estimate_path: str | Path, storage_template_path: str | Path,
    dense_teacher_import_path: str | Path, representation_graph_path: str | Path,
    representation_recipe_path: str | Path, dense_disposition_path: str | Path,
    tap_schema_path: str | Path, surface_parity_path: str | Path,
    source_manifest_path: str | Path, split_manifest_path: str | Path,
    train_row_selection_path: str | Path,
    train_assignment_manifest_path: str | Path,
    validation_assignment_manifest_path: str | Path,
    historical_campaign_spec_path: str | Path,
    historical_recipe_path: str | Path, historical_project_dir: str | Path,
    runtime_measurement_root: str | Path, data_root: str | Path,
) -> dict[str, Any]:
    """Publish the exact 261-task smoke candidate without invoking Slurm."""

    root = Path(representation_root).resolve()
    project = Path(project_dir).resolve()
    data = Path(data_root).resolve()
    historical_project = Path(historical_project_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("dense representation root must already exist")
    if not project.is_dir() or project.is_symlink() or not data.is_dir():
        raise ValueError("dense preparation project/data root differs")
    source_commit = capture_source_snapshot(project, require_clean=True)["git_commit"]

    profile = _artifact(compatible_resource_profile_path)
    recipe = _artifact(representation_recipe_path)
    recipe_sha256 = validate_representation_recipe(recipe)
    profile_sha256 = validate_dense_measured_profile(
        profile, expected_source_commit=source_commit,
        expected_recipe_sha256=recipe_sha256,
    )
    teacher = _artifact(dense_teacher_import_path)
    teacher_sha256 = validate_dense_teacher_import(teacher)
    graph = _artifact(representation_graph_path)
    graph_sha256 = validate_ascent_graph_artifact(
        graph,
        expected_parents={
            "parent_graph": teacher["payload"]["historical_parent_graph_sha256"],
            "parent_import": teacher_sha256,
        },
    )
    disposition = _artifact(dense_disposition_path)
    disposition_sha256 = validate_dense_training_disposition(disposition)
    source = _artifact(source_manifest_path)
    source_sha256 = require_sha256(source.get("content_hash"), name="source manifest")
    split = _artifact(split_manifest_path)
    split_sha256 = validate_split_manifest(split, source_manifest_sha256=source_sha256)
    selection = _artifact(train_row_selection_path)
    validate_row_selection(
        selection, split_manifest_sha256=split_sha256,
    )
    template = _artifact(storage_template_path)
    measured_source = dense_resource_measurement_source_commit(profile)
    template_sha256 = validate_dense_storage_template(
        template, expected_source_commit=measured_source,
        expected_recipe_sha256=recipe_sha256, expected_graph_sha256=graph_sha256,
        expected_dense_teacher_import_sha256=teacher_sha256,
    )
    storage = _artifact(storage_estimate_path)
    storage_sha256 = validate_dense_storage_estimate(
        storage, storage_template=artifact_reference(storage_template_path),
        expected_source_commit=measured_source,
        expected_recipe_sha256=recipe_sha256, expected_graph_sha256=graph_sha256,
        expected_dense_teacher_import_sha256=teacher_sha256,
    )
    free_bytes = validate_dense_storage_availability(storage, campaign_root=root)

    signature_root = root / "operator" / "runtime_signatures"
    runtime_facts, runtime_signatures, _ = _project_runtime_registry(
        project_dir=project, current_source_commit=source_commit,
        compatible_profile=profile,
        measurement_root=Path(runtime_measurement_root).resolve(),
        signature_root=signature_root, data_root=data,
    )
    runtime_facts_path = root / "operator" / "runtime_facts.json"
    runtime_signatures_path = root / "operator" / "runtime_signatures.json"
    _publish(runtime_facts_path, runtime_facts)
    _publish(runtime_signatures_path, runtime_signatures)

    artifact_paths = {
        "source_manifest": Path(source_manifest_path).resolve(),
        "split_manifest": Path(split_manifest_path).resolve(),
        "parent_import": Path(dense_teacher_import_path).resolve(),
        "representation_graph": Path(representation_graph_path).resolve(),
        "representation_recipe": Path(representation_recipe_path).resolve(),
        "final_disposition": Path(dense_disposition_path).resolve(),
        "runtime_binding": root / "runtime" / "runtime_binding.json",
    }
    common_spec = {
        "mode": "smoke", "campaign_root": root,
        "checkpoint_namespace": root / "checkpoints", "project_dir": project,
        "source_commit": source_commit, "source_manifest_sha256": source_sha256,
        "split_manifest_sha256": split_sha256,
        "parent_import_sha256": teacher_sha256,
        "representation_recipe_sha256": recipe_sha256,
        "graph_sha256": graph_sha256,
        "disposition_sha256": disposition_sha256,
        "disposition": DENSE_TRAINING_DISPOSITION,
        "role_counts": {"train": 512, "validation": 256, "final_test": 0},
        "final_source_partitions": 0, "combined_finalist_count": 0,
        "planning_only": True, "resource_profile": profile,
        "storage_estimate": storage,
        "fixed_size_inventory": artifact_reference(storage_template_path),
        "tigris_acceptance": None, "artifact_paths": artifact_paths,
    }
    draft = create_campaign_spec(**common_spec, runtime_binding_sha256=None)
    if len(draft["tasks"]) != 261:
        raise PermissionError("dense smoke topology is not the exact 261-task graph")
    draft_path = root / "planning" / "runtime_draft.json"
    _publish(draft_path, draft)

    producer_signature_path = signature_root / "gpu_representation.json"
    base_paths: dict[str, Path] = {
        "${assignment_manifest:train}": Path(train_assignment_manifest_path).resolve(),
        "${assignment_manifest:validation}": Path(validation_assignment_manifest_path).resolve(),
        "${dense_teacher_campaign_spec}": Path(historical_campaign_spec_path).resolve(),
        "${dense_teacher_recipe}": Path(historical_recipe_path).resolve(),
        "${dense_teacher_row_selection}": Path(train_row_selection_path).resolve(),
        "${dense_teacher_source_manifest}": Path(source_manifest_path).resolve(),
        "${dense_teacher_split_manifest}": Path(split_manifest_path).resolve(),
        "${dense_teacher_toff_report}": Path(teacher["payload"]["wrapper_report_path"]),
        "${final_disposition}": Path(dense_disposition_path).resolve(),
        "${parent_recipe}": Path(historical_recipe_path).resolve(),
        "${prebuilt_parent_import}": Path(dense_teacher_import_path).resolve(),
        "${prebuilt_representation_recipe}": Path(representation_recipe_path).resolve(),
        "${producer_runtime_signature}": producer_signature_path,
        "${representation_graph}": Path(representation_graph_path).resolve(),
        "${resource_profile}": Path(compatible_resource_profile_path).resolve(),
        "${split_manifest}": Path(split_manifest_path).resolve(),
        "${storage_estimate}": Path(storage_estimate_path).resolve(),
        "${teacher_report:TOFF}": Path(teacher["payload"]["wrapper_report_path"]),
        "${train_row_selection}": Path(train_row_selection_path).resolve(),
        "${train_validation_row_selection}": Path(train_row_selection_path).resolve(),
    }
    base_static: dict[str, Any] = {}
    base_hashes: dict[str, str] = {}
    for logical, path in base_paths.items():
        reference, content_hash = _json_reference(path)
        base_static[logical] = reference
        base_hashes[logical] = content_hash
    project_inventory = _directory_inventory(historical_project)
    base_static["${dense_teacher_project}"] = {
        "path": str(historical_project),
        "sha256": project_inventory["inventory_sha256"],
    }
    base_hashes["${dense_teacher_project}"] = project_inventory["inventory_sha256"]
    _publish(root / "operator" / "base_static_inputs.json", base_static)
    _publish(root / "operator" / "base_artifact_content_hashes.json", base_hashes)

    gpu_target_measurement = _artifact(
        Path(runtime_measurement_root).resolve()
        / "gpu_target" / "worker_runtime_measurement.json"
    )
    assets = build_dense_target_planning_assets(
        planning_campaign_spec=draft, dense_teacher_import=teacher,
        representation_graph=graph, representation_recipe=recipe,
        tap_schema=_artifact(tap_schema_path),
        surface_parity=_artifact(surface_parity_path), source_manifest=source,
        split_manifest=split, train_row_selection=selection,
        train_assignment_manifest_path=train_assignment_manifest_path,
        gpu_target_runtime_measurement=gpu_target_measurement,
        project_dir=project,
    )
    static_inputs, content_hashes = _publish_target_assets(
        root=root, assets=assets, base_static_inputs=base_static,
        base_content_hashes=base_hashes,
    )
    static_path = root / "operator" / "static_inputs.json"
    content_path = root / "operator" / "artifact_content_hashes.json"
    generations_path = root / "operator" / "target_generations.json"
    _publish(static_path, static_inputs)
    _publish(content_path, content_hashes)
    _publish(generations_path, assets["target_generations"])

    target_memory = int(str(draft["resources"]["gpu_target"]["memory"])[0:-1]) * 1024**3
    settings = {
        "kernel_parent_hashes": None,
        "shuffle_parent_hashes": None,
        "target_budgets": {
            "target_storage_cap_bytes": int(storage["estimated_campaign_peak_durable_bytes"]),
            "container_overhead_bytes": int(storage["campaign_fixed_reserve_bytes"]),
            "staging_recovery_reserve_bytes": int(storage["peak_target_staging_plus_committed_bytes"]),
            "quarantine_reserve_bytes": int(storage["peak_target_staging_plus_committed_bytes"]),
            "filesystem_headroom_bytes": int(storage["filesystem_headroom_bytes"]),
            "peak_runtime_bytes": dense_resource_peak_rss_bytes(
                profile, resource_class="gpu_target",
            ),
            "slurm_mem_per_node_bytes": target_memory,
            "filesystem_available_bytes": free_bytes,
        },
        "target_runtime_environment": None,
        "miniature_row_limit": None, "view_cache_max_gib": None,
        "synthetic_passes": None, "training_mode": None,
    }
    settings_path = root / "operator" / "settings.json"
    bundle_path = root / "operator" / "bundle_members.json"
    _publish(settings_path, settings)
    _publish(bundle_path, {})
    prerequisites = build_runtime_prerequisites(
        draft, runtime_facts=runtime_facts, runtime_signatures=runtime_signatures,
        static_inputs=static_inputs, artifact_content_hashes=content_hashes,
        target_generations=assets["target_generations"], bundle_members={},
        settings=settings, split_manifest=split, final_authorities={},
        target_forward_specs=assets["forward_specs"],
        representation_recipe=recipe,
    )
    prerequisites_path = root / "runtime" / "runtime_prerequisites.json"
    _publish(prerequisites_path, prerequisites)
    task_rows = build_runtime_task_rows(draft, prerequisites)
    task_rows_path = root / "runtime" / "task_rows.json"
    _publish(task_rows_path, task_rows)
    binding = build_runtime_binding(
        spec=draft, runtime_facts=runtime_facts, task_rows=task_rows,
    )
    binding_path = root / "runtime" / "runtime_binding.json"
    _publish(binding_path, binding)

    planning = create_campaign_spec(
        **common_spec, runtime_binding_sha256=binding["content_hash"],
    )
    if runtime_campaign_identity(planning) != runtime_campaign_identity(draft):
        raise PermissionError("final dense planning identity differs from runtime draft")
    planning_path = root / "planning" / "campaign_spec.json"
    _publish(planning_path, planning)
    validate_campaign_spec(planning, executable=False)
    command_plan = build_command_plan(planning)
    command_plan_path = root / "command_plan.json"
    _publish(command_plan_path, command_plan)
    dry_run = build_runtime_dry_run_audit(planning, binding, command_plan)
    dry_run_path = root / "runtime" / "dry_run_audit.json"
    _publish(dry_run_path, dry_run)
    candidate = build_executable_candidate_audit(
        planning_spec_path=planning_path, command_plan_path=command_plan_path,
        runtime_binding_path=binding_path,
    )
    candidate_path = root / "acceptance" / "executable_candidate_audit.json"
    _publish(candidate_path, candidate)
    return {
        "source_commit": source_commit,
        "resource_profile_sha256": profile_sha256,
        "storage_template_sha256": template_sha256,
        "storage_estimate_sha256": storage_sha256,
        "planning_spec_sha256": planning["content_hash"],
        "runtime_binding_sha256": binding["content_hash"],
        "command_plan_sha256": command_plan["content_hash"],
        "candidate_sha256": candidate["content_hash"],
        "tasks": len(planning["tasks"]),
        "final_role_rows": planning["role_counts"]["final_test"],
        "scheduler_mutated": False,
    }


__all__ = ["prepare_dense_smoke_candidate"]
