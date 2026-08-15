"""Train HCWDL-UB-FULL3 roots and factorized arm nodes."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, validate_content_hash,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .engine import precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_training import train_hcwdl_node
from .hcwdl_unified_balanced_full_campaign import (
    validate_arm_campaign, validate_foundation_campaign,
)
from .hcwdl_unified_balanced_full_contracts import (
    TRAINING_REPORT_CONTRACT, validate_foundation_lock,
)
from .hcwdl_unified_balanced_full_graph import (
    ARM_IDS, CAMPAIGN_LABEL, META_GRAPH_SHA256, SHARED_ARM,
    arm_registry, shared_registry, shared_training_registry,
    training_registry_for_arm,
)
from .hcwdl_unified_balanced_runner import (
    DOMAINS, _cache_student_views, _load_common, _stream,
)
from .hcwdl_unified_balanced_targets import (
    DurableUnifiedBalancedTargets, publish_target_manifest,
    publish_target_shard, validate_target_manifest,
)
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .splits import role_records
from .targets import EphemeralTeacherTargets
from .training import GenerationalLossConfiguration, derive_seed


def shared_node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def arm_node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def _teacher_location(
    canonical_id: str, *, arm_id: str, foundation_root: Path, arm_root: Path,
):
    owner, node_id = canonical_id.split("/", 1)
    if owner == SHARED_ARM:
        return shared_node_output_dir(foundation_root, node_id), shared_registry()[node_id]
    if owner != arm_id:
        raise PermissionError("HCWDL-UB-FULL3 teacher crosses arms")
    return arm_node_output_dir(arm_root, node_id), arm_registry(arm_id)[node_id]


def _teacher_targets(
    *, canonical_id: str, arm_id: str, foundation_spec: Mapping[str, Any],
    foundation_root: Path, arm_root: Path, split: Mapping[str, Any],
    split_hash: str, selections, assignments, balanced, batch_size: int,
    sampler_seed: int, repair_seed: int, device: str,
) -> tuple[EphemeralTeacherTargets, str]:
    output, node = _teacher_location(
        canonical_id, arm_id=arm_id, foundation_root=foundation_root,
        arm_root=arm_root,
    )
    report_path = output / "training_report.json"
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    checkpoint = output / str(report["selected_checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
        raise ValueError("HCWDL-UB-FULL3 teacher checkpoint differs")
    if canonical_id == "shared/U000":
        manifest_path = foundation_root / "targets/u000_train/manifest.json"
        durable = DurableUnifiedBalancedTargets(
            manifest_path, teacher_id=canonical_id,
        )
        if (
            durable.manifest.get("parents", {}).get("foundation_spec_sha256")
            != foundation_spec["content_hash"]
            or durable.manifest.get("parents", {}).get("teacher_report_sha256")
            != report_hash
            or durable.manifest.get("parents", {}).get("teacher_checkpoint_sha256")
            != report["selected_checkpoint_sha256"]
            or int(durable.manifest.get("rows", -1)) != selections["train"].rows
        ):
            raise ValueError("HCWDL-UB-FULL3 U000 target lineage differs")
        return durable.as_ephemeral(
            teacher_report_sha256=report_hash,
            split_manifest_sha256=split_hash,
        ), report_hash
    print(
        f"HCWDL-UB-FULL3 phase=teacher_targets teacher={canonical_id} status=started",
        flush=True,
    )
    started = time.monotonic()
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report),
        device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-UB-FULL3 teacher report changed during load")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    stream = _stream(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train",
        behavior=behavior, coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
    )
    targets = precompute_teacher_targets(
        model, stream, input_key="hlt" if behavior == "hlt" else "privileged",
        device=device, teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(
        f"HCWDL-UB-FULL3 phase=teacher_targets teacher={canonical_id} "
        f"status=complete seconds={time.monotonic()-started:.3f}",
        flush=True,
    )
    return targets, report_hash


def run_shared_node(
    *, foundation_spec: Mapping[str, Any], node_id: str,
    device: str = "cuda", view_cache_max_gib: float = 224.0,
) -> dict[str, Any]:
    validate_foundation_campaign(foundation_spec, executable=False)
    if node_id not in shared_registry():
        raise ValueError("unknown HCWDL-UB-FULL3 shared node")
    root = Path(foundation_spec["campaign_root"])
    node = shared_registry()[node_id]
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation_spec,
    )
    recipe = load_json(root / "recipe.json")
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(
        int(foundation_spec["replicate_seed"]), f"ub_full/sampler/{node.seed_alias}",
    )
    repair_seed = derive_seed(int(foundation_spec["replicate_seed"]), "ub_full/repair/v1")
    behavior = "hlt" if node.input_domain == "hlt" else "p0"
    caches, _ = _cache_student_views(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=view_cache_max_gib,
    )
    print(
        f"HCWDL-UB-FULL3 phase=optimizer_training node={node_id} status=started",
        flush=True,
    )
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
        class_weights=np.ones(15, np.float32),
        output_dir=shared_node_output_dir(root, node_id),
        parents={
            "foundation_spec_sha256": foundation_spec["content_hash"],
            "split_manifest_sha256": split_hash,
            "selection_manifest_sha256": selection_hash,
            "endpoint_lock_sha256": load_json(root / "locks/endpoint.json")["content_hash"],
        },
        device=device, registry=shared_training_registry(), domains=DOMAINS,
        graph_sha256=META_GRAPH_SHA256,
        report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label=f"{CAMPAIGN_LABEL}-FOUNDATION",
        seed_node_id=node.seed_alias,
        node_contract="HCWDL_UNIFIED_BALANCED_FULL_NODE_SPEC/v1",
        recipe_overlay_sha256=foundation_spec["parents"]["recipe_overlay_sha256"],
        explicit_loss=GenerationalLossConfiguration(
            arm=f"HCWDL_UB_FULL_SHARED_{node_id}", ce=1.0,
            parent_kd=0.0, grandparent_kd=0.0,
            parent_temperature=1.0, grandparent_temperature=1.0,
        ),
        scientific_config_extra={
            "canonical_node_id": node.canonical_id, "behavior": behavior,
            "population_policy": foundation_spec["population_policy"],
            "final_test_accessed": False, "student_view_built_once": True,
        },
    )


def publish_u000_targets(
    *, foundation_spec: Mapping[str, Any], device: str = "cuda",
) -> Path:
    validate_foundation_campaign(foundation_spec, executable=False)
    root = Path(foundation_spec["campaign_root"])
    split, split_hash, _, selections, assignments, balanced = _load_common(foundation_spec)
    recipe = load_json(root / "recipe.json")
    node = shared_registry()["U000"]
    output = shared_node_output_dir(root, "U000")
    report_path = output / "training_report.json"
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-UB-FULL3 U000 report changed before target publication")
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(
        int(foundation_spec["replicate_seed"]), f"ub_full/sampler/{node.seed_alias}",
    )
    repair_seed = derive_seed(int(foundation_spec["replicate_seed"]), "ub_full/repair/v1")
    stream = _stream(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train", behavior="p0",
        coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
    )
    targets = precompute_teacher_targets(
        model, stream, input_key="privileged", device=device,
        teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
    )
    del model
    target_root = root / "targets/u000_train"
    by_source = {record.path: [] for record in role_records(split, "train")}
    for index, identity in enumerate(targets.identities):
        source = str(identity).rsplit("::tree::", 1)[0]
        if source not in by_source:
            raise ValueError("HCWDL-UB-FULL3 U000 identity source differs")
        by_source[source].append(index)
    parents = {
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "split_manifest_sha256": split_hash,
        "teacher_report_sha256": report_hash,
        "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
    }
    shards = []
    for source_index, (source, indexes) in enumerate(by_source.items()):
        _, metadata = publish_target_shard(
            target_root / f"shard_{source_index:04d}",
            identities=[targets.identities[index] for index in indexes],
            logits=targets.logits[indexes], source_path=source, parents=parents,
            producer_commit=foundation_spec["source_commit"],
            teacher_id="shared/U000",
        )
        shards.append(metadata)
    publish_target_manifest(
        target_root / "manifest.json", shard_paths=shards,
        expected_sources=list(by_source), expected_rows=selections["train"].rows,
        parents=parents, teacher_id="shared/U000",
        consumers=tuple(
            sorted(f"{arm}/{node.node_id}" for arm in ARM_IDS
                   for node in arm_registry(arm).values()
                   if "shared/U000" in node.teachers)
        ),
    )
    return target_root / "manifest.json"


def run_arm_node(
    *, arm_spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    view_cache_max_gib: float = 224.0,
) -> dict[str, Any]:
    validate_arm_campaign(arm_spec, executable=False)
    arm_id = str(arm_spec["arm_id"])
    registry = arm_registry(arm_id)
    if node_id not in registry:
        raise ValueError("unknown HCWDL-UB-FULL3 arm node")
    node = registry[node_id]
    arm_root = Path(arm_spec["campaign_root"])
    lock_path = Path(arm_spec["foundation_lock_path"])
    foundation_root = lock_path.parent.parent
    lock = load_json(lock_path)
    if validate_foundation_lock(lock) != arm_spec["foundation_lock_sha256"]:
        raise ValueError("HCWDL-UB-FULL3 arm foundation lock differs")
    foundation_spec = load_json(foundation_root / "foundation_spec.json")
    validate_foundation_campaign(foundation_spec, executable=False)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation_spec,
    )
    recipe = load_json(foundation_root / "recipe.json")
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(
        int(foundation_spec["replicate_seed"]), f"ub_full/sampler/{node.seed_alias}",
    )
    repair_seed = derive_seed(int(foundation_spec["replicate_seed"]), "ub_full/repair/v1")
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
            canonical_id=node.parent_id, arm_id=arm_id,
            foundation_spec=foundation_spec, foundation_root=foundation_root,
            arm_root=arm_root, split=split, split_hash=split_hash,
            selections=selections, assignments=assignments, balanced=balanced,
            batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, device=device,
        )
    if node.grandparent_kd_weight:
        if node.grandparent_id is None:
            raise ValueError("HCWDL-UB-FULL3 grandparent target is absent")
        grandparent_targets, grandparent_hash = _teacher_targets(
            canonical_id=node.grandparent_id, arm_id=arm_id,
            foundation_spec=foundation_spec, foundation_root=foundation_root,
            arm_root=arm_root, split=split, split_hash=split_hash,
            selections=selections, assignments=assignments, balanced=balanced,
            batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, device=device,
        )
    parents = {
        "arm_spec_sha256": arm_spec["content_hash"],
        "foundation_lock_sha256": lock["content_hash"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
    }
    if parent_hash is not None:
        parents["parent_teacher_report_sha256"] = parent_hash
    if grandparent_hash is not None:
        parents["grandparent_teacher_report_sha256"] = grandparent_hash
    loss = GenerationalLossConfiguration(
        arm=f"HCWDL_UB_FULL_{arm_id}_{node_id}", ce=node.ce_weight,
        parent_kd=node.parent_kd_weight,
        grandparent_kd=node.grandparent_kd_weight,
        parent_temperature=node.parent_temperature,
        grandparent_temperature=node.grandparent_temperature,
    )
    print(
        f"HCWDL-UB-FULL3 phase=optimizer_training node={node.canonical_id} status=started",
        flush=True,
    )
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
        class_weights=np.ones(15, np.float32),
        output_dir=arm_node_output_dir(arm_root, node_id), parents=parents,
        device=device, registry=training_registry_for_arm(arm_id), domains=DOMAINS,
        graph_sha256=META_GRAPH_SHA256,
        report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label=f"{CAMPAIGN_LABEL}-{arm_id}",
        seed_node_id=node.seed_alias,
        node_contract="HCWDL_UNIFIED_BALANCED_FULL_NODE_SPEC/v1",
        explicit_loss=loss, recipe_overlay_sha256=arm_spec["arm_recipe_sha256"],
        parent_teacher_targets=parent_targets,
        grandparent_teacher_targets=grandparent_targets,
        peak_learning_rate_override=float(
            recipe["optimizer"]["peak_learning_rates"]["cold_child"]
        ),
        scientific_config_extra={
            "canonical_node_id": node.canonical_id,
            "behavior": behavior, "input_key": input_key,
            "parent_id": node.parent_id, "grandparent_id": node.grandparent_id,
            "population_policy": foundation_spec["population_policy"],
            "final_test_accessed": False, "student_view_built_once": True,
            "parent_targets_built_once": parent_targets is not None,
            "grandparent_targets_built_once": grandparent_targets is not None,
        },
    )


__all__ = [
    "arm_node_output_dir", "publish_u000_targets", "run_arm_node",
    "run_shared_node", "shared_node_output_dir",
]
