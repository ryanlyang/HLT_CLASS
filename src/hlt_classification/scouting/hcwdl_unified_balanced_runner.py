"""Execute shared and arm-local HCWDL-UB fits with one-time RAM views/targets."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .dataset import iterate_model_batches
from .engine import precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_recovery import task_attestation_path, validate_task_attestation
from .hcwdl_homotopy_stream import (
    iterate_homotopy_batches, iterate_unified_balanced_batches,
)
from .hcwdl_training import train_hcwdl_node
from .hcwdl_unified_balanced_cache import BalancedCouplingStore
from .hcwdl_unified_balanced_contracts import (
    TRAINING_REPORT_CONTRACT, validate_arm_spec, validate_foundation_lock,
    validate_endpoint_lock, validate_foundation_spec,
)
from .hcwdl_unified_balanced_graph import (
    META_GRAPH_SHA256, SHARED_ARM, arm_registry, shared_registry,
    shared_training_registry, training_registry_for_arm,
)
from .hcwdl_unified_balanced_targets import (
    DurableUnifiedBalancedTargets, validate_target_lock,
)
from .hcwdl_unified_balanced_targets import (
    publish_target_manifest, publish_target_shard,
)
from .hcwdl_upper_cache import ResidualCouplingStore
from .highcov_cache import DenseAssignmentStore
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .selective_assignment import RowSelection
from .splits import role_records
from .targets import EphemeralTeacherTargets
from .training import GenerationalLossConfiguration, derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


RUNTIME_CONTRACT = "HCWDL_UNIFIED_BALANCED_NODE_RUNTIME/v1"
DOMAINS = {"hlt": {"input": "hlt"}, "privileged": {"input": "privileged"}}


def arm_node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def shared_node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def _view_workers() -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    requested = int(os.environ.get("HCWDL_UB_VIEW_BUILD_WORKERS", str(allocated)))
    if allocated <= 0 or requested <= 0 or requested > allocated:
        raise ValueError("HCWDL-UB view workers exceed the allocated CPUs")
    return requested


def _memory_limit_bytes(configured_gib: float) -> int:
    configured = int(configured_gib * 1024**3)
    slurm = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm:
        configured = min(configured, int(slurm) * 1024**2 * 3 // 4)
    return configured


def _load_common(spec: Mapping[str, Any]):
    foundation_root = Path(spec["campaign_root"])
    paths = spec["artifact_paths"]
    split = load_json(paths["split_manifest"])
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection_raw = load_json(paths["selection_manifest"])
    selection_hash = validate_content_hash(
        selection_raw, expected_contract=str(selection_raw["contract"]),
        expected_schema_version=int(selection_raw["schema_version"]),
    )
    selections = {
        role: RowSelection(selection_raw, role=role, split_manifest_sha256=split_hash)
        for role in ("train", "validation")
    }
    assignments = {
        role: DenseAssignmentStore(paths[f"{role}_assignment_manifest"])
        for role in ("train", "validation")
    }
    balanced = {
        role: BalancedCouplingStore(
            foundation_root / f"balanced/{role}_manifest.json"
        ) for role in ("train", "validation")
    }
    return split, split_hash, selection_hash, selections, assignments, balanced


def _stream(
    *, foundation_spec: Mapping[str, Any], split: Mapping[str, Any],
    selections, assignments, balanced, role: str, behavior: str,
    coordinate, batch_size: int, sampler_seed: int, repair_seed: int,
    legacy: bool = False, epoch: int = 0,
):
    if behavior == "hlt":
        return iterate_model_batches(
            split, data_root=foundation_spec["data_root"], role=role,
            input_mode="hlt", epoch=epoch, batch_size=batch_size,
            sampler_seed=sampler_seed, row_selection=selections[role],
        )
    if behavior == "p0":
        # The exact P0 corner is independent of switch coordinates, but the
        # balanced stream proves it against the same selected identities.
        return iterate_unified_balanced_batches(
            split, data_root=foundation_spec["data_root"], role=role,
            assignment_store=assignments[role], coupling_store=balanced[role],
            row_selection=selections[role], coordinate=coordinate,
            repair_seed=repair_seed, batch_size=batch_size,
            workers=_view_workers(), output_key="privileged",
        )
    if legacy:
        legacy_store = ResidualCouplingStore(
            foundation_spec["artifact_paths"][f"legacy_{role}_manifest"]
        )
        return iterate_homotopy_batches(
            split, data_root=foundation_spec["data_root"], role=role,
            assignment_store=assignments[role], coupling_store=legacy_store,
            row_selection=selections[role], coordinate=coordinate,
            repair_seed=repair_seed, batch_size=batch_size,
            workers=_view_workers(), output_key="privileged",
        )
    return iterate_unified_balanced_batches(
        split, data_root=foundation_spec["data_root"], role=role,
        assignment_store=assignments[role], coupling_store=balanced[role],
        row_selection=selections[role], coordinate=coordinate,
        repair_seed=repair_seed, batch_size=batch_size,
        workers=_view_workers(), output_key="privileged",
    )


def _cache_student_views(
    *, foundation_spec, split, selections, assignments, balanced,
    behavior: str, coordinate, batch_size: int, sampler_seed: int,
    repair_seed: int, memory_gib: float,
):
    caches = {}; remaining = _memory_limit_bytes(memory_gib)
    input_key = "hlt" if behavior == "hlt" else "privileged"
    for role in ("train", "validation"):
        started = time.monotonic()
        stream = _stream(
            foundation_spec=foundation_spec, split=split,
            selections=selections, assignments=assignments, balanced=balanced,
            role=role, behavior=behavior, coordinate=coordinate,
            batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, epoch=0,
            legacy=behavior in {"legacycdf_uniform", "balanced_legacywarp"},
        )
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            stream, expected_rows=selections[role].rows, records=records,
            role=role,
            expected_source_rows=expected_cache_source_rows(
                records, row_selection=selections[role],
            ),
            view_keys=(input_key,), max_gib=remaining / 1024**3,
            lineage={
                "foundation_spec_sha256": foundation_spec["content_hash"],
                "behavior": behavior, "coordinate": coordinate.payload(),
                "student_view_built_once": True,
                "durable_repaired_dataset": False,
            },
        )
        caches[role] = cache; remaining -= int(cache.header["array_bytes"])
        if remaining <= 0:
            raise MemoryError("HCWDL-UB train/validation caches exceed the memory cap")
        print(
            f"HCWDL-UB phase=student_view_cache role={role} behavior={behavior} "
            f"rows={selections[role].rows} seconds={time.monotonic()-started:.3f}",
            flush=True,
        )
    return caches, input_key


def _teacher_location(
    canonical_id: str, *, foundation_root: Path, arm_root: Path,
) -> tuple[Path, object]:
    owner, node_id = canonical_id.split("/", 1)
    if owner == SHARED_ARM:
        node = shared_registry()[node_id]
        return shared_node_output_dir(foundation_root, node_id), node
    node = arm_registry(owner)[node_id]
    return arm_node_output_dir(arm_root, node_id), node


def _target_attestation_context(
    *, teacher_id: str, arm_root: Path, arm_spec_sha256: str,
    recovery_context: Mapping[str, Any] | None,
) -> tuple[Path, str]:
    task_id = f"train_{teacher_id}"
    if recovery_context is not None and task_id in recovery_context["task_ids"]:
        return Path(recovery_context["root"]), str(recovery_context["spec_sha256"])
    return arm_root, arm_spec_sha256


def _teacher_targets(
    *, canonical_id: str, foundation_spec, foundation_root: Path,
    arm_root: Path, split, split_hash: str, selections, assignments, balanced,
    batch_size: int, sampler_seed: int, repair_seed: int, device: str,
    recovery_context: Mapping[str, Any] | None = None,
) -> tuple[EphemeralTeacherTargets, str]:
    output, node = _teacher_location(
        canonical_id, foundation_root=foundation_root, arm_root=arm_root,
    )
    engine_path = output / "training_report.json"
    report = load_json(engine_path); report_hash = validate_pmard_training_report(report)
    checkpoint = output / str(report["selected_checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
        raise ValueError("HCWDL-UB teacher selected checkpoint differs")
    if canonical_id == "shared/U000":
        durable = DurableUnifiedBalancedTargets(
            foundation_root / "targets/u000_train/manifest.json",
            teacher_id=canonical_id,
        )
        expected = {
            "foundation_spec_sha256": foundation_spec["content_hash"],
            "split_manifest_sha256": split_hash,
            "teacher_report_sha256": report_hash,
            "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
        if any(durable.manifest.get("parents", {}).get(name) != value for name, value in expected.items()):
            raise ValueError("HCWDL-UB shared target manifest lineage differs")
        foundation_lock = load_json(foundation_root / "locks/foundation.json")
        validate_foundation_lock(foundation_lock)
        target_lock = load_json(foundation_root / "targets/u000_train/lock.json")
        target_lock_hash = validate_target_lock(target_lock)
        if (
            durable.manifest["content_hash"] != foundation_lock["u000_target_manifest_sha256"]
            or report_hash != foundation_lock["u000_report_sha256"]
            or report["selected_checkpoint_sha256"]
            != foundation_lock["u000_checkpoint_sha256"]
            or target_lock_hash != foundation_lock["parents"]["target_lock_sha256"]
            or target_lock["manifest_sha256"] != durable.manifest["content_hash"]
        ):
            raise ValueError("HCWDL-UB shared target manifest is not foundation-locked")
        targets = durable.as_ephemeral(
            teacher_report_sha256=report_hash,
            split_manifest_sha256=split_hash,
        )
        return targets, report_hash
    local_manifest = output / "targets/manifest.json"
    if local_manifest.is_file():
        consumers = _teacher_consumers(canonical_id)
        durable = DurableUnifiedBalancedTargets(
            local_manifest, teacher_id=canonical_id, consumers=consumers,
        )
        expected = {
            "foundation_spec_sha256": foundation_spec["content_hash"],
            "split_manifest_sha256": split_hash,
            "teacher_report_sha256": report_hash,
            "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
        if any(durable.manifest.get("parents", {}).get(name) != value for name, value in expected.items()):
            raise ValueError("HCWDL-UB arm target manifest lineage differs")
        arm_spec = load_json(arm_root / "arm_spec.json")
        arm_spec_hash = validate_arm_spec(arm_spec)
        owner, teacher_id = canonical_id.split("/", 1)
        if owner != arm_spec.get("arm_id"):
            raise ValueError("HCWDL-UB local target belongs to another arm")
        attestation_root, attestation_scope = _target_attestation_context(
            teacher_id=teacher_id, arm_root=arm_root,
            arm_spec_sha256=arm_spec_hash,
            recovery_context=recovery_context,
        )
        attestation_path = task_attestation_path(
            attestation_root, f"train_{teacher_id}", None,
        )
        attestation = load_json(attestation_path)
        validate_task_attestation(
            attestation, campaign_spec_sha256=attestation_scope,
            task_id=f"train_{teacher_id}", array_index=None,
        )
        manifest_path = str(local_manifest.resolve())
        matching = [
            row for row in attestation["outputs"]
            if str(Path(row["path"]).resolve()) == manifest_path
        ]
        if (
            len(matching) != 1
            or matching[0].get("content_hash") != durable.manifest["content_hash"]
        ):
            raise ValueError("HCWDL-UB local target manifest lacks its producer attestation")
        targets = durable.as_ephemeral(
            teacher_report_sha256=report_hash,
            split_manifest_sha256=split_hash,
        )
        return targets, report_hash
    model, loaded = load_pmard_model(
        engine_path, model_factory=scouting_model_factory_for_report(report),
        device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-UB teacher report changed during load")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    stream = _stream(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train",
        behavior=behavior, coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        legacy=node.behavior in {"legacycdf_uniform", "balanced_legacywarp"},
    )
    targets = precompute_teacher_targets(
        model, stream, input_key="hlt" if behavior == "hlt" else "privileged",
        device=device, teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    del model; gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    return targets, report_hash


def _teacher_consumers(canonical_id: str) -> tuple[str, ...]:
    owner, _ = canonical_id.split("/", 1)
    consumers = []
    registries = (
        [arm_registry(owner)] if owner != SHARED_ARM
        else [arm_registry(arm) for arm in (
            "C25P75", "C10P90", "C05P95", "C10P75G15", "C05P80G15", "C00P100",
        )]
    )
    for registry in registries:
        for node in registry.values():
            if canonical_id in node.teachers:
                consumers.append(node.canonical_id)
    return tuple(sorted(consumers))


def _publish_teacher_targets(
    *, canonical_id: str, output: Path, node, foundation_spec,
    split, split_hash: str, selections, assignments, balanced,
    batch_size: int, sampler_seed: int, repair_seed: int, device: str,
    target_root_override: str | Path | None = None,
    producer_commit: str | None = None,
) -> Path | None:
    """Publish one compact cache only when a selected teacher has >1 consumers."""

    consumers = _teacher_consumers(canonical_id)
    if len(consumers) < 2:
        return None
    report_path = output / "training_report.json"
    report = load_json(report_path); report_hash = validate_pmard_training_report(report)
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-UB selected teacher changed before target publication")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    stream = _stream(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train",
        behavior=behavior, coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        legacy=node.behavior in {"legacycdf_uniform", "balanced_legacywarp"},
    )
    targets = precompute_teacher_targets(
        model, stream, input_key="hlt" if behavior == "hlt" else "privileged",
        device=device, teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    del model; gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    target_root = (
        output / "targets" if target_root_override is None
        else Path(target_root_override)
    )
    by_source: dict[str, list[int]] = {
        record.path: [] for record in role_records(split, "train")
    }
    for index, identity in enumerate(targets.identities):
        source = str(identity).rsplit("::tree::", 1)[0]
        if source not in by_source:
            raise ValueError("HCWDL-UB teacher target identity has an unknown source")
        by_source[source].append(index)
    shard_paths = []
    parents = {
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "split_manifest_sha256": split_hash,
        "teacher_report_sha256": report_hash,
        "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
    }
    for source_index, (source, indexes) in enumerate(by_source.items()):
        base = target_root / f"shard_{source_index:04d}"
        _, metadata = publish_target_shard(
            base, identities=[targets.identities[index] for index in indexes],
            logits=targets.logits[indexes], source_path=source, parents=parents,
            producer_commit=str(producer_commit or foundation_spec["source_commit"]),
            teacher_id=canonical_id,
        )
        shard_paths.append(metadata)
    manifest = publish_target_manifest(
        target_root / "manifest.json", shard_paths=shard_paths,
        expected_sources=list(by_source), expected_rows=selections["train"].rows,
        parents=parents, teacher_id=canonical_id, consumers=consumers,
    )
    return target_root / "manifest.json"


def run_shared_node(
    *, foundation_spec: Mapping[str, Any], node_id: str,
    device: str = "cuda", view_cache_max_gib: float = 80.0,
) -> dict[str, Any]:
    validate_foundation_spec(foundation_spec)
    if node_id not in shared_registry():
        raise ValueError("unknown HCWDL-UB shared node")
    root = Path(foundation_spec["campaign_root"]); node = shared_registry()[node_id]
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation_spec,
    )
    recipe = load_json(foundation_spec["artifact_paths"]["recipe"])
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(int(foundation_spec["replicate_seed"]), f"ub/sampler/{node.seed_alias}")
    repair_seed = derive_seed(int(foundation_spec["replicate_seed"]), "ub/repair/v1")
    behavior = "hlt" if node.input_domain == "hlt" else "p0"
    caches, _ = _cache_student_views(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=view_cache_max_gib,
    )
    output = shared_node_output_dir(root, node_id)
    parents = {
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "foundation_gate_sha256": validate_endpoint_lock(
            load_json(root / "locks/endpoint.json")
        ),
    }
    return train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(foundation_spec["replicate_seed"]),
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        class_weights=np.ones(15, np.float32), output_dir=output,
        parents=parents, device=device, registry=shared_training_registry(),
        domains=DOMAINS, graph_sha256=META_GRAPH_SHA256,
        report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label="HCWDL-UB-FOUNDATION", seed_node_id=node.seed_alias,
        node_contract="HCWDL_UNIFIED_BALANCED_NODE_SPEC/v1",
        scientific_config_extra={
            "canonical_node_id": node.canonical_id,
            "behavior": behavior, "final_test_accessed": False,
            "student_view_built_once": True,
        },
    )


def run_arm_node(
    *, arm_spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    view_cache_max_gib: float = 80.0, producer_commit: str | None = None,
    recovery_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_arm_spec(arm_spec)
    arm_id = str(arm_spec["arm_id"]); registry = arm_registry(arm_id)
    if node_id not in registry:
        raise ValueError("unknown HCWDL-UB arm node")
    node = registry[node_id]; arm_root = Path(arm_spec["campaign_root"])
    foundation_lock_path = Path(arm_spec["foundation_lock_path"])
    foundation_root = foundation_lock_path.parent.parent
    foundation_lock = load_json(foundation_lock_path)
    lock_hash = validate_foundation_lock(foundation_lock)
    if lock_hash != arm_spec["foundation_lock_sha256"]:
        raise ValueError("HCWDL-UB arm foundation lock differs")
    foundation_spec = load_json(foundation_root / "foundation_spec.json")
    validate_foundation_spec(foundation_spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation_spec,
    )
    recipe = load_json(foundation_spec["artifact_paths"]["recipe"])
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(int(foundation_spec["replicate_seed"]), f"ub/sampler/{node.seed_alias}")
    repair_seed = derive_seed(int(foundation_spec["replicate_seed"]), "ub/repair/v1")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    caches, input_key = _cache_student_views(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=view_cache_max_gib,
    )
    parent_targets = grandparent_targets = None
    parent_hash = grandparent_hash = None
    if node.parent_id is not None:
        parent_targets, parent_hash = _teacher_targets(
            canonical_id=node.parent_id, foundation_spec=foundation_spec,
            foundation_root=foundation_root, arm_root=arm_root, split=split,
            split_hash=split_hash, selections=selections, assignments=assignments,
            balanced=balanced, batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, device=device,
            recovery_context=recovery_context,
        )
    if node.grandparent_kd_weight:
        if node.grandparent_id is None:
            raise ValueError("HCWDL-UB grandparent weight lacks a teacher")
        grandparent_targets, grandparent_hash = _teacher_targets(
            canonical_id=node.grandparent_id, foundation_spec=foundation_spec,
            foundation_root=foundation_root, arm_root=arm_root, split=split,
            split_hash=split_hash, selections=selections, assignments=assignments,
            balanced=balanced, batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, device=device,
            recovery_context=recovery_context,
        )
    parents = {
        "arm_spec_sha256": arm_spec["content_hash"],
        "foundation_lock_sha256": lock_hash,
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
    }
    if parent_hash is not None:
        parents["parent_teacher_report_sha256"] = parent_hash
    if grandparent_hash is not None:
        parents["grandparent_teacher_report_sha256"] = grandparent_hash
    loss = GenerationalLossConfiguration(
        arm=f"HCWDL_UB_{arm_id}_{node_id}", ce=node.ce_weight,
        parent_kd=node.parent_kd_weight,
        grandparent_kd=node.grandparent_kd_weight,
        parent_temperature=node.parent_temperature,
        grandparent_temperature=node.grandparent_temperature,
    )
    child_lr = float(recipe["optimizer"]["peak_learning_rates"]["cold_child"])
    output = arm_node_output_dir(arm_root, node_id)
    result = train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(foundation_spec["replicate_seed"]),
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        class_weights=np.ones(15, np.float32), output_dir=output,
        parents=parents, device=device, registry=training_registry_for_arm(arm_id),
        domains=DOMAINS, graph_sha256=META_GRAPH_SHA256,
        report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label=f"HCWDL-UB-{arm_id}", seed_node_id=node.seed_alias,
        node_contract="HCWDL_UNIFIED_BALANCED_NODE_SPEC/v1",
        explicit_loss=loss, recipe_overlay_sha256=arm_spec["recipe_arm_sha256"],
        parent_teacher_targets=parent_targets,
        grandparent_teacher_targets=grandparent_targets,
        peak_learning_rate_override=child_lr,
        scientific_config_extra={
            "canonical_node_id": node.canonical_id,
            "behavior": behavior, "input_key": input_key,
            "parent_id": node.parent_id, "grandparent_id": node.grandparent_id,
            "final_test_accessed": False,
            "student_view_built_once": True,
            "parent_targets_built_once": parent_targets is not None,
            "grandparent_targets_built_once": grandparent_targets is not None,
        },
    )
    _publish_teacher_targets(
        canonical_id=node.canonical_id, output=output, node=node,
        foundation_spec=foundation_spec, split=split, split_hash=split_hash,
        selections=selections, assignments=assignments, balanced=balanced,
        batch_size=batch_size, sampler_seed=sampler_seed,
        repair_seed=repair_seed, device=device,
        producer_commit=producer_commit,
    )
    return result


__all__ = [
    "RUNTIME_CONTRACT", "arm_node_output_dir", "run_arm_node",
    "run_shared_node", "shared_node_output_dir",
]
