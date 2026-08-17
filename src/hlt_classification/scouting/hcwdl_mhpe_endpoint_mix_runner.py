"""Target construction and paired exact-HLT training for endpoint mixtures."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash, write_immutable_json
from hlt_classification.models.scouting_particle_transformer import build_scouting_particle_transformer

from .engine import precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_ladder import NodeSpec, TeacherSpec
from .hcwdl_mhpe_endpoint_mix import (
    GRAPH_SHA256, NODES, NODE_CONTRACT, SOURCE_ENDPOINT,
    TARGET_BUILD_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, validate_campaign,
)
from .hcwdl_mhpe_endpoint_mix_targets import (
    ephemeral_from_manifest, fp32_softmax, mix_probabilities, publish_lock,
    publish_manifest, publish_target, validate_bundle,
)
from .hcwdl_mhpe_runner import _context as source_context
from .hcwdl_mhpe_targets import DurableProbabilityTargets, validate_probability_bundle
from .hcwdl_training import train_hcwdl_node
from .hcwdl_unified_balanced_runner import DOMAINS, _cache_student_views
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .training import GenerationalLossConfiguration, derive_seed


def _source(spec: Mapping[str, Any]):
    validate_campaign(spec, verify_source_tree=False)
    source_spec = load_json(spec["source"]["source_spec_path"])
    context = source_context(source_spec, verify_source_tree=False)
    return source_spec, context


def _registry() -> Mapping[str, NodeSpec]:
    return MappingProxyType({
        node_id: NodeSpec(
            node_id=node_id, track="mhpe_endpoint_mix", stage="compression",
            student_domain="hlt", initialization="fresh",
            initialization_parent=None,
            teachers=(TeacherSpec(f"teacher_{node_id}", "hlt", "parent"),),
            loss_kind="ce_kd", deployable=True,
        ) for node_id in NODES
    })


def build_targets(*, spec: Mapping[str, Any], device: str = "cuda") -> dict[str, Any]:
    validate_campaign(spec, verify_source_tree=False)
    source_spec, context = _source(spec)
    (reuse, foundation_root, foundation, split, split_hash, selection_hash,
     selections, assignments, balanced, recipe) = context
    if int(recipe.get("training_passes", -1)) != 60:
        raise ValueError("endpoint-mixture source executable recipe is not 60-pass")
    source_profile = spec["source"]["source_profile"]
    endpoint_root = Path(spec["source"]["endpoint_target_root"])
    validate_probability_bundle(
        endpoint_root, ensemble_id=SOURCE_ENDPOINT, temperature=1.0,
        consumers=["M1"], profile=source_profile,
    )
    report_path = Path(spec["source"]["m0paired_report_path"])
    report = load_json(report_path); report_hash = validate_pmard_training_report(report)
    if report_hash != spec["source"]["m0paired_report_sha256"]:
        raise ValueError("endpoint-mixture M0paired report changed")
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("endpoint-mixture M0paired load lineage differs")
    sampler_seed = derive_seed(int(foundation["replicate_seed"]), "mhpe_endpoint_mix/targets")
    repair_seed = derive_seed(int(foundation["replicate_seed"]), "ub/repair/v1")
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=HomotopyCoordinate(1, 1, 1, 1),
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=72.0,
    )
    root = Path(spec["campaign_root"]); target_root = root / "targets"
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "source_campaign_spec_sha256": spec["source"]["source_spec_sha256"],
        "source_completion_sha256": spec["source"]["source_completion_sha256"],
        "source_endpoint_target_lock_sha256": spec["source"]["endpoint_target_lock_sha256"],
        "foundation_reuse_lock_sha256": spec["source"]["foundation_reuse_lock_sha256"],
        "m0paired_report_sha256": report_hash,
        "m0paired_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
    }
    manifest_hashes: dict[str, dict[str, str]] = {name: {} for name in NODES}
    role_rows = {}
    for role in ("train", "validation"):
        m0_targets = precompute_teacher_targets(
            model, caches[role].iterate_batches(
                epoch=0, sampler_seed=sampler_seed,
                batch_size=int(recipe["batching"]["effective_batch_size"]),
            ), input_key=input_key, device=device,
            teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
        )
        m0_probability = fp32_softmax(m0_targets.logits)
        m0_lookup = {key: index for index, key in enumerate(m0_targets.identities)}
        endpoint_targets = DurableProbabilityTargets(endpoint_root / f"{role}_manifest.json")
        try:
            indexes = [m0_lookup[key] for key in endpoint_targets.identities]
        except KeyError as error:
            raise KeyError(f"endpoint-mixture {SOURCE_ENDPOINT}/M0paired identity join is incomplete") from error
        aligned_m0 = m0_probability[indexes]
        if len(indexes) != len(m0_lookup) or len(set(indexes)) != len(indexes):
            raise ValueError(f"endpoint-mixture {SOURCE_ENDPOINT}/M0paired identity coverage differs")
        role_rows[role] = len(endpoint_targets.identities)
        if len(endpoint_targets.identities) != int(spec["role_counts"][role]):
            raise ValueError("endpoint-mixture target role population differs")
        component_lineage = {
            SOURCE_ENDPOINT: spec["source"]["endpoint_manifest_sha256"][role],
            "M0paired": report_hash,
        }
        for node_id, node in NODES.items():
            probabilities = mix_probabilities(
                endpoint_targets.probabilities, aligned_m0,
                numerator=node.endpoint_weight_numerator,
                denominator=node.endpoint_weight_denominator,
            )
            directory = target_root / node_id
            metadata = publish_target(
                directory / f"{role}_all", node_id=node_id, role=role,
                identities=endpoint_targets.identities, probabilities=probabilities,
                component_lineage=component_lineage, parents=parents,
                producer_commit=spec["source_commit"],
            )
            manifest = publish_manifest(
                directory / f"{role}_manifest.json", node_id=node_id, role=role,
                target_metadata=directory / f"{role}_all.json",
                expected_rows=len(endpoint_targets.identities), parents=parents,
            )
            manifest_hashes[node_id][role] = manifest["content_hash"]
    del model; gc.collect()
    lock = publish_lock(target_root / "lock.json", manifests=manifest_hashes, parents=parents)
    output = with_content_hash({
        "contract": TARGET_BUILD_REPORT_CONTRACT,
        "schema_version": 2, "campaign_spec_sha256": spec["content_hash"],
        "target_lock_sha256": lock["content_hash"], "role_rows": role_rows,
        "m0paired_forward_passes_per_role": 1,
        "source_endpoint": SOURCE_ENDPOINT,
        "endpoint_reused_without_model_inference": True,
        "labels_not_used_in_target_values": True,
        "final_test_accessed": False,
    })
    write_immutable_json(root / "reports/target_build.json", output); return output


def train_node(*, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
               recovery_spec_sha256: str | None = None) -> dict[str, Any]:
    if node_id not in NODES:
        raise ValueError("unknown endpoint-mixture node")
    validate_campaign(spec, verify_source_tree=recovery_spec_sha256 is None)
    _, context = _source(spec)
    (_, _, foundation, split, split_hash, selection_hash,
     selections, assignments, balanced, recipe) = context
    target_root = Path(spec["campaign_root"]) / "targets"
    lock_hash, manifests = validate_bundle(target_root)
    node = NODES[node_id]
    sampler_seed = derive_seed(int(foundation["replicate_seed"]), node.payload()["seed_alias"])
    repair_seed = derive_seed(int(foundation["replicate_seed"]), "ub/repair/v1")
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=HomotopyCoordinate(1, 1, 1, 1),
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=72.0,
    )
    target_path = target_root / node_id / "train_manifest.json"
    probability = ephemeral_from_manifest(target_path, split_manifest_sha256=split_hash)
    loss = endpoint_mix_loss(node_id)
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "source_campaign_spec_sha256": spec["source"]["source_spec_sha256"],
        "target_lock_sha256": lock_hash,
        "teacher_target_manifest_sha256": manifests[node_id]["train"]["content_hash"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
    }
    result = train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(foundation["replicate_seed"]),
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed,
            batch_size=int(recipe["batching"]["effective_batch_size"]),
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed,
            batch_size=int(recipe["batching"]["effective_batch_size"]),
        ),
        class_weights=np.ones(15, np.float32),
        output_dir=Path(spec["campaign_root"]) / "training" / node_id,
        parents=parents, device=device, registry=_registry(), domains=DOMAINS,
        graph_sha256=GRAPH_SHA256, report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label="HCWDL-MHPE-D000E-ENDPOINT-MIX-300K60",
        seed_node_id=node.payload()["seed_alias"],
        node_contract=NODE_CONTRACT,
        explicit_loss=loss, recipe_overlay_sha256=spec["recipe_sha256"],
        parent_probability_targets=probability,
        peak_learning_rate_override=float(recipe["optimizer"]["peak_learning_rates"]["cold_child"]),
        scientific_config_extra={
            "input_key": input_key, "teacher_mixture": node.payload(),
            "paired_comparison": True, "student_view_built_once": True,
            "teacher_targets_built_once": True, "final_test_accessed": False,
        },
    )
    returned = dict(result)
    returned["_cache_array_bytes"] = {role: int(cache.header["array_bytes"]) for role, cache in caches.items()}
    return returned


def endpoint_mix_loss(node_id: str) -> GenerationalLossConfiguration:
    """Return the registered endpoint-mixture loss under the UB namespace."""
    if node_id not in NODES:
        raise ValueError("unknown endpoint-mixture node")
    return GenerationalLossConfiguration(
        arm=f"HCWDL_UB_MHPE_ENDPOINT_MIX_{node_id}",
        ce=.10, parent_kd=.90, grandparent_kd=0,
        parent_temperature=1.0, grandparent_temperature=1.0,
    )


__all__ = ["build_targets", "endpoint_mix_loss", "train_node"]
