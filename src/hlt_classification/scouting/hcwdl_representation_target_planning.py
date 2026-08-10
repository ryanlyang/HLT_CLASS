"""Pre-campaign publication of exact dense-descent target authorities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
)
from hlt_classification.provenance import (
    capture_source_snapshot, validate_source_snapshot_payload,
)

from .engine import validate_pmard_training_report
from .hcwdl_representation_campaign import (
    DENSE_TRAINING_DISPOSITION, validate_campaign_spec,
)
from .hcwdl_representation_dense_teacher import validate_dense_teacher_import
from .hcwdl_representation_contracts import (
    DENSE_COMPATIBLE_RESOURCE_PROFILE_CONTRACT,
)
from .hcwdl_representation_graph import (
    NODE_REGISTRY, validate_ascent_graph_artifact,
)
from .hcwdl_representation_kernels import (
    generate_spectral_resource_bundle, spectral_resource_logical_hashes,
)
from .hcwdl_representation_recipe import validate_representation_recipe
from .hcwdl_representation_targets import (
    build_logical_target_bank, build_target_consumer_registry,
    build_target_consumer_row, build_target_forward_spec,
    derive_target_generation_id, validate_logical_target_bank,
    validate_target_consumer_registry, validate_target_forward_spec,
)
from .hcwdl_representation_worker_runtime import (
    target_forward_runtime_commitment_from_measurement,
    validate_worker_runtime_measurement,
)
from .hcwdl_assignment import validate_train_assignment_authority
from .hcwdl_representation_resources import (
    validate_dense_profile_runtime_measurement,
)
from .selective_assignment import validate_row_selection
from .splits import role_records, validate_split_manifest


def _implementation_signature(project_dir: Path, *, toff: bool) -> dict[str, Any]:
    sources = {
        "input_decoding_sha256": project_dir / "src/hlt_classification/scouting/dataset.py",
        "feature_layout_sha256": project_dir / "src/hlt_classification/models/hcwdl_surfaces.py",
        "trimmer_sha256": project_dir / "src/hlt_classification/scouting/dataset.py",
        "family_code_sha256": project_dir / "src/hlt_classification/scouting/hcwdl_representation_targets.py",
        "surface_capture_sha256": project_dir / "src/hlt_classification/models/hcwdl_surfaces.py",
        "sketch_arithmetic_sha256": project_dir / "src/hlt_classification/scouting/hcwdl_representation_target_runtime.py",
    }
    result = {}
    for name, path in sources.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"dense target implementation source is absent: {path}")
        result[name] = canonical_sha256({
            "surface": name, "source_file_sha256": sha256_file(path),
        })
    result["teacher_input_fields"] = sorted(
        (
            "charged_features", "charged_mask", "charged_vectors",
            "charged_visible_indices", "neutral_features", "neutral_mask",
            "neutral_vectors", "neutral_visible_indices",
        )
        if toff else
        ("family_codes", "features", "mask", "vectors", "visible_indices")
    )
    return result


def _source_partitions(split: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = tuple(role_records(split, "train"))
    if not records:
        raise ValueError("dense target planning has no train source partitions")
    return {
        f"source_{index:03d}": {
            "source_path": record.path, "source_file_id": index,
        }
        for index, record in enumerate(records)
    }


def _artifact_reference(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"dense target artifact is not an absolute regular file: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _validated_tap_sha256(value: Mapping[str, Any]) -> str:
    from hlt_classification.models.hcwdl_surfaces import (
        tap_schema as canonical_tap_schema, tap_schema_sha256,
    )

    if dict(value) != canonical_tap_schema():
        raise ValueError("target planning tap schema differs")
    return tap_schema_sha256()


def _refresh_compatible_target_producer(
    runtime: Mapping[str, Any], *, project: Path, campaign_source_commit: str,
) -> dict[str, Any]:
    source_snapshot = capture_source_snapshot(project, require_clean=True)
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("git_commit") != campaign_source_commit:
        raise PermissionError("target compatibility checkout differs from campaign")
    return {
        **dict(runtime),
        "producer": {
            **dict(runtime["producer"]),
            "source_commit": source_snapshot["git_commit"],
            "source_snapshot_sha256": source_snapshot[
                "source_snapshot_sha256"
            ],
        },
    }


def build_dense_target_planning_assets(
    *, planning_campaign_spec: Mapping[str, Any], dense_teacher_import: Mapping[str, Any],
    representation_graph: Mapping[str, Any], representation_recipe: Mapping[str, Any],
    tap_schema: Mapping[str, Any], surface_parity: Mapping[str, Any],
    source_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any],
    train_row_selection: Mapping[str, Any],
    train_assignment_manifest_path: str | Path,
    gpu_target_runtime_measurement: Mapping[str, Any], project_dir: str | Path,
) -> dict[str, Any]:
    """Build every logical bank/consumer/forward record in DAG order."""

    spec_sha256 = validate_campaign_spec(planning_campaign_spec, executable=False)
    if planning_campaign_spec.get("disposition") != DENSE_TRAINING_DISPOSITION:
        raise PermissionError("target planning requires the dense-only campaign")
    teacher_import_sha256 = validate_dense_teacher_import(dense_teacher_import)
    if teacher_import_sha256 != planning_campaign_spec.get("parent_import_sha256"):
        raise PermissionError("target planning dense teacher differs from campaign")
    graph_sha256 = validate_ascent_graph_artifact(
        representation_graph,
        expected_parents={
            "parent_graph": dense_teacher_import["payload"][
                "historical_parent_graph_sha256"
            ],
            "parent_import": teacher_import_sha256,
        },
    )
    recipe_sha256 = validate_representation_recipe(representation_recipe)
    if (
        graph_sha256 != planning_campaign_spec.get("graph_sha256")
        or recipe_sha256 != planning_campaign_spec.get(
            "representation_recipe_sha256"
        )
    ):
        raise PermissionError("target planning graph/recipe differs from campaign")
    from hlt_classification.models.hcwdl_surfaces import validate_surface_parity_report
    tap_sha256 = _validated_tap_sha256(tap_schema)
    parity_sha256 = validate_surface_parity_report(surface_parity)
    if surface_parity.get("authorization_capable") is not True:
        raise PermissionError("target planning requires installed-Weaver parity")
    source_sha256 = require_sha256(source_manifest.get("content_hash"), name="source manifest")
    split_sha256 = validate_split_manifest(
        split_manifest, source_manifest_sha256=source_sha256,
    )
    selection_sha256 = validate_row_selection(
        train_row_selection, split_manifest_sha256=split_sha256,
    )
    train_role = train_row_selection.get("roles", {}).get("train")
    if not isinstance(train_role, Mapping):
        raise ValueError("dense target planning train selection differs")
    assignment_sha256 = validate_train_assignment_authority(
        train_assignment_manifest_path,
        split_manifest_sha256=split_sha256,
        row_selection_sha256=selection_sha256,
        expected_mapped_jets=int(train_role.get("rows", 0)),
    )
    parents = representation_recipe["parents"]
    if (
        parents.get("dense_teacher_import") != teacher_import_sha256
        or parents.get("representation_ascent_graph") != graph_sha256
        or parents.get("source_manifest") != source_sha256
        or parents.get("split_manifest") != split_sha256
        or parents.get("row_selection") != selection_sha256
        or parents.get("assignment_manifest") != assignment_sha256
    ):
        raise PermissionError("target planning recipe/data lineage differs")
    project = Path(project_dir).resolve()
    if not project.is_dir() or project.is_symlink():
        raise ValueError("dense target project is not an absolute worktree")
    validate_worker_runtime_measurement(gpu_target_runtime_measurement)
    profile = planning_campaign_spec.get("resource_profile")
    compatible_source = (
        isinstance(profile, Mapping)
        and profile.get("contract") == DENSE_COMPATIBLE_RESOURCE_PROFILE_CONTRACT
    )
    if gpu_target_runtime_measurement.get("campaign_spec_sha256") != spec_sha256:
        if not compatible_source:
            raise PermissionError("target runtime measurement belongs to another campaign")
        validate_dense_profile_runtime_measurement(
            profile, resource_class="gpu_target",
            measurement=gpu_target_runtime_measurement,
        )
    runtime = target_forward_runtime_commitment_from_measurement(
        gpu_target_runtime_measurement,
    )
    if compatible_source:
        runtime = _refresh_compatible_target_producer(
            runtime, project=project,
            campaign_source_commit=str(planning_campaign_spec["source_commit"]),
        )
    elif runtime["producer"]["source_commit"] != planning_campaign_spec.get(
        "source_commit"
    ):
        raise PermissionError("target runtime measurement uses another source")
    kernel_hashes = spectral_resource_logical_hashes(
        generate_spectral_resource_bundle()
    )
    global_parents = {
        "source": source_sha256,
        "split": split_sha256,
        "train_row_selection": selection_sha256,
        "graph": graph_sha256,
        "assignment": assignment_sha256,
        "repair": assignment_sha256,
        "surface_parity": parity_sha256,
        "parent_recipe": parents["parent_recipe"],
        "representation_recipe": recipe_sha256,
        "kernel_resources": parents["kernel_resources"],
        "parent_import": teacher_import_sha256,
    }
    source_partitions = _source_partitions(split_manifest)
    partition_ids = sorted(source_partitions)
    common_forward = {
        **runtime,
        "batching": {
            "batch_size": 256,
            "order": "source_file_id_then_source_entry_v1",
            "cross_source_batches": False,
            "final_short_batch_per_source": True,
            "padding": False,
            "row_duplication": False,
        },
        "source_partitions": partition_ids,
    }
    engine_path = Path(str(dense_teacher_import["payload"]["engine_report_path"]))
    engine = load_json(engine_path)
    validate_pmard_training_report(engine)
    toff_teacher = {
        "source_kind": "imported_checkpoint", "node_id": "TOFF",
        "domain": "toff", "track": "shared",
        "selected_report_sha256": dense_teacher_import["parents"][
            "toff_wrapper_report"
        ],
        "checkpoint_byte_sha256": dense_teacher_import["parents"][
            "toff_selected_checkpoint"
        ],
        "tap_sha256": tap_sha256,
        "installed_weaver_signature_sha256": surface_parity[
            "runtime_signature_sha256"
        ],
    }
    execution_by_node: dict[str, str] = {}
    logical_banks: dict[str, dict[str, Any]] = {}
    consumer_registries: dict[str, dict[str, Any]] = {}
    forward_specs: dict[str, dict[str, Any]] = {}
    target_generations: dict[str, dict[str, Any]] = {}
    tasks = planning_campaign_spec["tasks"]
    for target_task in (row for row in tasks if row["kind"] == "target_build"):
        bank = str(target_task["logical_bank"])
        purpose = str(target_task["target_purpose"])
        if bank not in logical_banks:
            if bank == "TOFF":
                teacher = toff_teacher
            else:
                node = NODE_REGISTRY.get(bank)
                if node is None or bank not in execution_by_node:
                    raise ValueError(f"dense target teacher is not ready in DAG order: {bank}")
                teacher = {
                    "source_kind": "campaign_execution", "node_id": bank,
                    "domain": node.student_domain, "track": node.track,
                    "registered_execution_id": execution_by_node[bank],
                    "tap_sha256": tap_sha256,
                }
            logical_banks[bank] = build_logical_target_bank(
                bank_id=bank, teacher=teacher, parents=global_parents,
            )
            validate_logical_target_bank(logical_banks[bank], expected_bank_id=bank)
        logical = logical_banks[bank]
        consumer_tasks = [
            row for row in tasks
            if row.get("logical_bank") == bank
            and row.get("target_purpose") == purpose
            and row.get("kind") in {"train_node", "confirmation"}
        ]
        consumers = []
        for row in consumer_tasks:
            seeds = (11, 22, 33, 44, 55) if row["kind"] == "confirmation" else (1337,)
            for seed in seeds:
                consumers.append(build_target_consumer_row(
                    logical, purpose=purpose, campaign_sha256=spec_sha256,
                    recipe_sha256=recipe_sha256, node_id=str(row["graph_node"]),
                    seed=seed,
                ))
        registry = build_target_consumer_registry(
            logical, purpose=purpose, consumers=consumers,
            generation_parent_sha256=spec_sha256,
        )
        validate_target_consumer_registry(registry, logical_bank=logical)
        key = f"{bank}:{purpose}"
        consumer_registries[key] = registry
        teacher_payload = logical["payload"]["teacher"]
        forward_teacher = {
            "source_kind": teacher_payload["source_kind"],
            "surface_parity_sha256": parity_sha256,
            "tap_sha256": tap_sha256,
            "kernel_resources_sha256": parents["kernel_resources"],
            "kernel_array_logical_hashes": kernel_hashes,
        }
        if bank == "TOFF":
            forward_teacher.update({
                "checkpoint_byte_sha256": teacher_payload[
                    "checkpoint_byte_sha256"
                ],
                "model_config_sha256": engine["execution_config_sha256"],
            })
        else:
            forward_teacher["registered_execution_id"] = teacher_payload[
                "registered_execution_id"
            ]
        forward = build_target_forward_spec(
            parents={"logical_bank": logical["content_hash"]},
            payload={
                **common_forward,
                "teacher": forward_teacher,
                "implementation": _implementation_signature(
                    project, toff=bank == "TOFF",
                ),
            },
        )
        validate_target_forward_spec(
            forward, expected_parents={"logical_bank": logical["content_hash"]},
        )
        forward_specs[key] = forward
        generation_id = derive_target_generation_id(
            logical["content_hash"], registry["content_hash"],
            purpose=purpose, generation_parent_sha256=spec_sha256,
        )
        execution_ids = {
            f"{row['node_id']}:{row['seed']}": row["execution_id"]
            for row in consumers
        }
        target_generations[key] = {
            "generation_id": generation_id,
            "generation_parent_sha256": spec_sha256,
            "source_partitions": source_partitions,
            "execution_ids": execution_ids,
            "execution_campaign_sha256": spec_sha256,
        }
        if purpose == "screen":
            execution_by_node.update({
                str(row["node_id"]): str(row["execution_id"])
                for row in consumers
            })
    return {
        "logical_banks": logical_banks,
        "consumer_registries": consumer_registries,
        "forward_specs": forward_specs,
        "target_generations": target_generations,
    }


__all__ = ["build_dense_target_planning_assets"]
