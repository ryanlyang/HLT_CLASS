"""Train one source-pinned HCWDL-UB-FULLCOARSE3 arm node."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .engine import precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_training import train_hcwdl_node
from .hcwdl_unified_balanced_coarse_campaign import validate_arm_campaign
from .hcwdl_unified_balanced_coarse_contracts import (
    TRAINING_REPORT_CONTRACT,
    validate_foundation_reuse_lock,
)
from .hcwdl_unified_balanced_coarse_graph import (
    CAMPAIGN_LABEL,
    META_GRAPH_SHA256,
    arm_registry,
    training_registry_for_arm,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_full_contracts import validate_foundation_lock
from .hcwdl_unified_balanced_runner import (
    DOMAINS,
    _cache_student_views,
    _load_common,
    _stream,
)
from .hcwdl_unified_balanced_targets import (
    DurableUnifiedBalancedTargets,
    validate_target_manifest,
)
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .targets import EphemeralTeacherTargets
from .training import GenerationalLossConfiguration, derive_seed


def arm_node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def _foundation_context(arm_spec: Mapping[str, Any]):
    reuse = load_json(arm_spec["reuse_lock_path"])
    if validate_foundation_reuse_lock(reuse) != arm_spec["reuse_lock_sha256"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 foundation reuse lock differs")
    lock_path = Path(reuse["foundation_lock_path"])
    lock = load_json(lock_path)
    if validate_foundation_lock(lock) != reuse["foundation_lock_sha256"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 imported foundation lock differs")
    foundation_root = lock_path.parent.parent
    foundation = load_json(foundation_root / "foundation_spec.json")
    if validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    ) != reuse["foundation_spec_sha256"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 imported foundation spec differs")
    return reuse, lock, foundation_root, foundation


def _teacher_location(
    canonical_id: str, *, arm_id: str, foundation_root: Path, arm_root: Path,
) -> tuple[Path, Any | None]:
    owner, node_id = canonical_id.split("/", 1)
    if owner == "shared":
        if node_id != "U000":
            raise PermissionError("HCWDL-UB-FULLCOARSE3 unknown shared teacher")
        return foundation_root / "training/U000", None
    if owner != arm_id:
        raise PermissionError("HCWDL-UB-FULLCOARSE3 teacher crosses arms")
    node = arm_registry(arm_id)[node_id]
    return arm_node_output_dir(arm_root, node_id), node


def _teacher_targets(
    *, canonical_id: str, arm_id: str, reuse: Mapping[str, Any],
    foundation_spec: Mapping[str, Any], foundation_root: Path, arm_root: Path,
    split: Mapping[str, Any], split_hash: str, selections, assignments,
    balanced, batch_size: int, sampler_seed: int, repair_seed: int,
    device: str, consumer_id: str,
) -> tuple[EphemeralTeacherTargets, str]:
    output, node = _teacher_location(
        canonical_id, arm_id=arm_id, foundation_root=foundation_root,
        arm_root=arm_root,
    )
    report_path = output / "training_report.json"
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    checkpoint = output / str(report["selected_checkpoint"])
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 teacher checkpoint differs")
    if canonical_id == "shared/U000":
        if (
            report_hash != reuse["parents"]["u000_report_sha256"]
            or report["selected_checkpoint_sha256"]
            != reuse["parents"]["u000_checkpoint_sha256"]
            or consumer_id not in reuse["u000_target_consumers"]
        ):
            raise ValueError("HCWDL-UB-FULLCOARSE3 U000 teacher reuse differs")
        manifest_path = foundation_root / "targets/u000_train/manifest.json"
        manifest = load_json(manifest_path)
        if (
            validate_target_manifest(manifest, teacher_id=canonical_id)
            != reuse["parents"]["u000_target_manifest_sha256"]
            or int(manifest.get("rows", -1)) != selections["train"].rows
        ):
            raise ValueError("HCWDL-UB-FULLCOARSE3 U000 target reuse differs")
        durable = DurableUnifiedBalancedTargets(
            manifest_path, teacher_id=canonical_id,
        )
        return durable.as_ephemeral(
            teacher_report_sha256=report_hash,
            split_manifest_sha256=split_hash,
        ), report_hash

    print(
        f"HCWDL-UB-FULLCOARSE3 phase=teacher_targets "
        f"teacher={canonical_id} status=started",
        flush=True,
    )
    started = time.monotonic()
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report),
        device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-UB-FULLCOARSE3 teacher report changed during load")
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
        f"HCWDL-UB-FULLCOARSE3 phase=teacher_targets "
        f"teacher={canonical_id} status=complete "
        f"seconds={time.monotonic() - started:.3f}",
        flush=True,
    )
    return targets, report_hash


def run_arm_node(
    *, arm_spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    view_cache_max_gib: float = 224.0,
) -> dict[str, Any]:
    validate_arm_campaign(arm_spec, executable=False)
    arm_id = str(arm_spec["arm_id"])
    registry = arm_registry(arm_id)
    if node_id not in registry:
        raise ValueError("unknown HCWDL-UB-FULLCOARSE3 arm node")
    node = registry[node_id]
    arm_root = Path(arm_spec["campaign_root"])
    reuse, _, foundation_root, foundation = _foundation_context(arm_spec)
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    recipe = load_json(foundation_root / "recipe.json")
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(
        int(foundation["replicate_seed"]),
        f"ub_full_coarse/sampler/{node.seed_alias}",
    )
    # Deliberately reuse the FULL3 field-switch realization.  Only rung density
    # and loss routing change in this comparison.
    repair_seed = derive_seed(int(foundation["replicate_seed"]), "ub_full/repair/v1")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=view_cache_max_gib,
    )
    parent_targets = grandparent_targets = None
    parent_hash = grandparent_hash = None
    if node.parent_id is None:
        raise ValueError("HCWDL-UB-FULLCOARSE3 parent is absent")
    parent_targets, parent_hash = _teacher_targets(
        canonical_id=node.parent_id, arm_id=arm_id, reuse=reuse,
        foundation_spec=foundation, foundation_root=foundation_root,
        arm_root=arm_root, split=split, split_hash=split_hash,
        selections=selections, assignments=assignments, balanced=balanced,
        batch_size=batch_size, sampler_seed=sampler_seed,
        repair_seed=repair_seed, device=device, consumer_id=node.canonical_id,
    )
    if node.grandparent_kd_weight:
        if node.grandparent_id is None:
            raise ValueError("HCWDL-UB-FULLCOARSE3 grandparent is absent")
        grandparent_targets, grandparent_hash = _teacher_targets(
            canonical_id=node.grandparent_id, arm_id=arm_id, reuse=reuse,
            foundation_spec=foundation, foundation_root=foundation_root,
            arm_root=arm_root, split=split, split_hash=split_hash,
            selections=selections, assignments=assignments, balanced=balanced,
            batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, device=device, consumer_id=node.canonical_id,
        )
    parents = {
        "arm_spec_sha256": arm_spec["content_hash"],
        "foundation_reuse_lock_sha256": reuse["content_hash"],
        "foundation_spec_sha256": foundation["content_hash"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "parent_teacher_report_sha256": parent_hash,
    }
    if grandparent_hash is not None:
        parents["grandparent_teacher_report_sha256"] = grandparent_hash
    loss = GenerationalLossConfiguration(
        arm=f"HCWDL_UB_FULLCOARSE3_{arm_id}_{node_id}",
        ce=node.ce_weight, parent_kd=node.parent_kd_weight,
        grandparent_kd=node.grandparent_kd_weight,
        parent_temperature=node.parent_temperature,
        grandparent_temperature=node.grandparent_temperature,
    )
    return train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(foundation["replicate_seed"]),
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        class_weights=np.ones(15, np.float32),
        output_dir=arm_node_output_dir(arm_root, node_id), parents=parents,
        device=device, registry=training_registry_for_arm(arm_id),
        domains=DOMAINS, graph_sha256=META_GRAPH_SHA256,
        report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label=f"{CAMPAIGN_LABEL}-{arm_id}",
        seed_node_id=node.seed_alias,
        node_contract="HCWDL_UNIFIED_BALANCED_FULL_COARSE_NODE_SPEC/v1",
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
            "coordinate_exact": node.coordinate.payload(),
            "population_policy": foundation["population_policy"],
            "final_test_accessed": False,
            "student_view_built_once": True,
            "parent_targets_built_once": True,
            "grandparent_targets_built_once": grandparent_targets is not None,
        },
    )


__all__ = ["arm_node_output_dir", "run_arm_node"]
