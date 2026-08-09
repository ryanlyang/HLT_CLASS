"""Runtime for one node of the supplemental dense cold 300k HCWDL ladder."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, validate_content_hash,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .dataset import iterate_model_batches
from .engine import precompute_teacher_targets
from .hcwdl_dense import (
    DENSE_DOMAINS, DENSE_GRAPH_SHA256, DENSE_NODE_CONTRACT,
    DENSE_NODE_REGISTRY, DENSE_REPAIR_RNG_POLICY,
    DENSE_TRAINING_REPORT_CONTRACT, DenseNodeSpec,
)
from .hcwdl_recipe import validate_recipe, validate_recipe_class_weight_lineage
from .hcwdl_training import train_hcwdl_node
from .highcov_cache import DenseAssignmentStore
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .pmard_stream import iterate_pmard_batches
from .selective_assignment import RowSelection
from .selective_assignment import ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION
from .splits import role_records
from .training import derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


def dense_shared_repair_seed(replicate_seed: int) -> int:
    """One repair coordinate system for every dense alpha domain."""

    return derive_seed(replicate_seed, "hcwdl_dense/repair/shared_v1")


def run_dense_node(
    *, node_id: str, recipe_path: str | Path,
    split_manifest_path: str | Path, selection_manifest_path: str | Path,
    data_root: str | Path, assignment_manifests: Mapping[str, str | Path],
    output_dir: str | Path, teacher_report_path: str | Path,
    replicate_seed: int, source_snapshot_sha256: str,
    assignment_lock_sha256: str, qualification_lock_sha256: str,
    parent_campaign_spec_sha256: str,
    expected_recipe_sha256: str, expected_split_manifest_sha256: str,
    expected_selection_manifest_sha256: str,
    expected_assignment_manifest_sha256: Mapping[str, str],
    expected_teacher_report_sha256: str,
    device: str = "cuda",
    view_cache_max_gib: float = 320.0,
    registry: Mapping[str, DenseNodeSpec] = DENSE_NODE_REGISTRY,
    domains: Mapping[str, Mapping[str, object]] = DENSE_DOMAINS,
    graph_sha256: str = DENSE_GRAPH_SHA256,
    training_report_contract: str = DENSE_TRAINING_REPORT_CONTRACT,
    node_contract: str = DENSE_NODE_CONTRACT,
    campaign_label: str = "HCWDL_DENSE_COLD_300K",
    rung_step: int = 10,
) -> dict[str, Any]:
    if node_id not in registry:
        raise ValueError("unknown dense cold HCWDL node")
    node = registry[node_id]
    recipe = load_json(recipe_path)
    recipe_hash = validate_recipe(recipe, require_authorized=True)
    if recipe_hash != require_sha256(expected_recipe_sha256, name="dense recipe SHA-256"):
        raise ValueError("dense cold recipe hash differs from campaign spec")
    split = load_json(split_manifest_path)
    split_hash = validate_content_hash(
        split, expected_contract=str(split.get("contract")),
        expected_schema_version=int(split.get("schema_version")),
    )
    if split_hash != require_sha256(
        expected_split_manifest_sha256, name="dense split manifest SHA-256",
    ):
        raise ValueError("dense cold split hash differs from campaign spec")
    selection_raw = load_json(selection_manifest_path)
    selection_hash = validate_content_hash(
        selection_raw, expected_contract=ROW_SELECTION_CONTRACT,
        expected_schema_version=ROW_SELECTION_VERSION,
    )
    if selection_hash != require_sha256(
        expected_selection_manifest_sha256, name="dense selection SHA-256",
    ):
        raise ValueError("dense cold row selection differs from campaign spec")
    validate_recipe_class_weight_lineage(recipe, selection_raw)
    selections = {
        role: RowSelection(
            selection_raw, role=role,
            split_manifest_sha256=split["content_hash"],
        )
        for role in ("train", "validation")
    }
    stores = {
        role: DenseAssignmentStore(path) for role, path in assignment_manifests.items()
    }
    if set(stores) != {"train", "validation"}:
        raise ValueError("dense cold node requires train and validation assignments")
    if set(expected_assignment_manifest_sha256) != {"train", "validation"}:
        raise ValueError("dense cold expected assignment hash set differs")
    for role, store in stores.items():
        if store.manifest["content_hash"] != require_sha256(
            expected_assignment_manifest_sha256[role],
            name=f"dense {role} assignment manifest SHA-256",
        ):
            raise ValueError(f"dense cold {role} assignment differs from campaign spec")

    batch_size = int(recipe["batching"]["effective_batch_size"])
    # Preserve the parent's exact sampler trajectory. D100offkd additionally
    # aliases D100's initialization/training seed for a paired top-KD control.
    sampler_seed = derive_seed(replicate_seed, "hcwdl/sampler")
    # One coordinate system is shared across all alpha domains. This makes every
    # discrete identity/validity switch nested as privilege decreases.
    shared_repair_seed = dense_shared_repair_seed(replicate_seed)

    def online_stream(domain: str, role: str, epoch: int = 0):
        if domain == "hlt":
            return iterate_model_batches(
                split, data_root=data_root, role=role, input_mode="hlt",
                epoch=epoch, batch_size=batch_size, sampler_seed=sampler_seed,
                row_selection=selections[role],
            )
        if domain == "toff":
            return iterate_model_batches(
                split, data_root=data_root, role=role, input_mode="toff",
                epoch=epoch, batch_size=batch_size, sampler_seed=sampler_seed,
                row_selection=selections[role],
            )
        alpha = domains[domain]["alpha"]
        if alpha is None:
            raise ValueError("dense cold privileged domain lacks alpha")
        return iterate_pmard_batches(
            split, data_root=data_root, role=role, matcher_model=None,
            alpha=float(alpha), repair_family="HIGHCOV_SHELL_EXACT/v1",
            matcher_variant="highcov_empirical_lexicographic_dr0p30_v1",
            threshold=0.0, epoch=epoch, batch_size=batch_size,
            sampler_seed=sampler_seed, assignment_store=stores[role],
            row_selection=selections[role], repair_seed=shared_repair_seed,
        )

    student_domain = node.student_domain
    student_input_key = str(domains[student_domain]["input"])
    view_caches: dict[str, EphemeralPmardViewCache] = {}
    remaining = float(view_cache_max_gib)
    for role in ("train", "validation"):
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            online_stream(student_domain, role),
            expected_rows=selections[role].rows,
            records=records, role=role,
            expected_source_rows=expected_cache_source_rows(
                records, row_selection=selections[role],
            ),
            view_keys=(student_input_key,), max_gib=remaining,
            lineage={
                "split_manifest_sha256": split["content_hash"],
                "row_selection_sha256": selection_raw["content_hash"],
                "assignment_manifest_sha256": stores[role].manifest["content_hash"],
                "repair_family": (
                    "HIGHCOV_SHELL_EXACT/v1" if student_domain != "hlt"
                    else "not_applicable"
                ),
                "alpha": domains[student_domain]["alpha"],
                "repair_rng_policy": DENSE_REPAIR_RNG_POLICY,
            },
        )
        view_caches[role] = cache
        remaining -= float(cache.header["array_bytes"]) / 1024**3

    def student_batches(role: str, epoch: int = 0):
        return view_caches[role].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        )

    teacher_spec = node.teachers[0]
    teacher_raw = load_json(teacher_report_path)
    teacher_model, teacher_report = load_pmard_model(
        teacher_report_path,
        model_factory=scouting_model_factory_for_report(teacher_raw),
        device=device,
    )
    if teacher_report["content_hash"] != require_sha256(
        expected_teacher_report_sha256, name="dense teacher report SHA-256",
    ):
        raise ValueError("dense cold teacher report differs from declared lineage")
    teacher_input_key = str(domains[teacher_spec.domain]["input"])
    teacher_targets = precompute_teacher_targets(
        teacher_model, online_stream(teacher_spec.domain, "train", 0),
        input_key=teacher_input_key, device=device,
        teacher_report_sha256=teacher_report["content_hash"],
        split_manifest_sha256=split["content_hash"],
    )
    del teacher_model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scientific_extra = {
        "repair_rng_policy": DENSE_REPAIR_RNG_POLICY,
        "seed_identity": "D100" if node_id == "D100offkd" else node_id,
        "coarse_ascent_disabled": True,
        "final_test_accessed": False,
    }
    if rung_step == 5:
        scientific_extra["rung_step"] = 5
    return train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=replicate_seed,
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: student_batches("train", epoch),
        validation_batches=lambda: student_batches("validation", 0),
        class_weights=np.asarray(recipe["class_weights"], np.float32),
        output_dir=output_dir,
        parents={
            "split_manifest_sha256": split["content_hash"],
            "source_snapshot_sha256": source_snapshot_sha256,
            "assignment_lock_sha256": assignment_lock_sha256,
            "qualification_lock_sha256": qualification_lock_sha256,
            "parent_campaign_spec_sha256": parent_campaign_spec_sha256,
            "teacher_sole_report_sha256": teacher_report["content_hash"],
        },
        device=device,
        hlt_teacher_targets=(
            teacher_targets if teacher_spec.domain == "hlt" else None
        ),
        privileged_teacher_targets=(
            teacher_targets if teacher_spec.domain != "hlt" else None
        ),
        registry=registry, domains=domains,
        graph_sha256=graph_sha256,
        report_contract=training_report_contract,
        campaign_label=campaign_label,
        node_contract=node_contract,
        seed_node_id="D100" if node_id == "D100offkd" else node_id,
        scientific_config_extra=scientific_extra,
    )


__all__ = ["dense_shared_repair_seed", "run_dense_node"]
