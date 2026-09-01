"""Fit and target-bank workers for attention-reoptimized four spines."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, write_immutable_json,
)

from .evaluation import classification_metrics
from .hcwdl_attention_reoptimization import DEFAULT_ATTENTION_RECIPE
from .hcwdl_fullcard_bottleneck_foundation_campaign import validate_foundation
from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend, _infer_cache
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, load_tri60_model,
    train_tri60_node,
)
from .hcwdl_tri100_spine4_attention_campaign import validate_campaign
from .hcwdl_tri100_spine4_attention_contracts import (
    FINAL_CHECKPOINT_CONTRACT, RECIPE_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_attention_execution import (
    validate_attention_execution_acceptance, validate_parameter_lock,
)
from .hcwdl_tri100_spine4_attention_graph import (
    ANCHOR_NODE_ID, GRAPH_SHA256, LR_SCHEDULE, NODE_REGISTRY,
    PROBABILITY_COMPONENTS, RELATIONAL_CARRIERS,
)
from .hcwdl_tri100_spine4_bottleneck_probability import (
    BottleneckProbabilityTargets, load_probability_role,
    publish_probability_lock, publish_probability_role,
    validate_probability_lock,
)
from .hcwdl_tri100_spine4_bottleneck_source import validate_source_lock
from .hcwdl_tri100_spine4_persistent_support import validate_support_audit
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


def node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def distribution_output_dir(root: str | Path, distribution_id: str) -> Path:
    return Path(root) / "probabilities" / distribution_id


def training_authority(node_id: str) -> Tri60TrainingAuthority:
    authority = Tri60TrainingAuthority(
        node=NODE_REGISTRY[node_id], graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
        allowed_training_passes=(NODE_REGISTRY[node_id].training_passes,),
    )
    authority.validate()
    return authority


def _foundation(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation(value)
    return value


def _runtime(spec: Mapping[str, Any], *, node_id: str) -> Tri60TrainingRuntime:
    recipe = load_json(spec["artifact_paths"]["recipe"])
    validate_artifact(recipe, contract=RECIPE_CONTRACT)
    training = recipe[
        "anchor_training" if node_id == ANCHOR_NODE_ID else "downstream_training"
    ]
    passes = int(training.get("maximum_passes", training.get("passes")))
    return Tri60TrainingRuntime(
        passes=passes, batch_size=int(training["effective_batch_size"]),
        peak_learning_rate=float(training["peak_learning_rate"]),
        weight_decay=float(training["weight_decay"]), warmup_fraction=0.05,
        minimum_lr_fraction=float(
            training.get("stage0_learning_rate_schedule", {}).get(
                "minimum_lr_fraction",
                training.get("learning_rate_floor_fraction", 0.05),
            )
        ),
        amp_dtype=str(training["forward_precision"]),
    )


def _behavior(coordinate_name: str) -> str:
    return "hlt" if coordinate_name == "D000" else "balanced_uniform"


def _caches(
    spec: Mapping[str, Any], *, node, roles=("train", "validation"),
    memory_gib: float = 240.0,
):
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation,
    )
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "tri60/repair/shared_v1",
    )
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced,
        behavior=_behavior(node.coordinate_name), coordinate=node.coordinate,
        batch_size=256, sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=memory_gib, include_hcwdl_metadata=True,
        support_policy=PERSISTENT_HLT_SUPPORT_POLICY, roles=tuple(roles),
    )
    return foundation, split_hash, selection_hash, caches, input_key


def _probability_targets(
    spec: Mapping[str, Any], distribution_id: str, *, consumer_id: str,
):
    root = distribution_output_dir(spec["campaign_root"], distribution_id)
    lock, _ = validate_probability_lock(
        root / "lock.json", distribution_id=distribution_id,
    )
    if (
        consumer_id not in lock["consumers"]
        or lock.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or lock.get("parents", {}).get("foundation") != spec["parents"]["foundation"]
        or lock.get("parents", {}).get("assignment_lock")
        != spec["parents"]["assignment_lock"]
    ):
        raise PermissionError("attention probability-bank lineage differs")
    return BottleneckProbabilityTargets.load(
        root / "train_manifest.json", distribution_id=distribution_id,
    ), lock


def _model(spec: Mapping[str, Any], node_id: str, *, device: str):
    path = node_output_dir(spec["campaign_root"], node_id) / "training_report.json"
    return load_tri60_model(
        path, device=device, authority=training_authority(node_id),
    )


def _validated_execution(spec: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    acceptance_hash = validate_attention_execution_acceptance(
        acceptance, spec=spec,
    )
    parameter = load_json(spec["artifact_paths"]["parameter_lock"])
    validate_parameter_lock(parameter, spec=spec)
    return acceptance_hash, parameter["registry"]


def _common_parents(
    spec: Mapping[str, Any], *, split_hash: str, selection_hash: str,
    execution_acceptance: str, parameter_lock: str,
) -> dict[str, str]:
    support_hash = validate_support_audit(
        load_json(spec["artifact_paths"]["support_audit"]), spec=spec,
    )
    return {
        "campaign_spec": spec["content_hash"],
        "source_campaign": spec["parents"]["source_campaign"],
        "source_lock": spec["parents"]["source_lock"],
        "foundation": spec["parents"]["foundation"],
        "assignment_lock": spec["parents"]["assignment_lock"],
        "matcher_spec": spec["parents"]["matcher_spec"],
        "graph": GRAPH_SHA256, "recipe": spec["parents"]["recipe"],
        "split_manifest": split_hash, "selection_manifest": selection_hash,
        "execution_acceptance": execution_acceptance,
        "attention_parameter_lock": parameter_lock,
        "support_audit": support_hash,
    }


def run_fit(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec)
    _configure_deterministic_backend()
    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown attention four-spine fit")
    node = NODE_REGISTRY[node_id]
    acceptance_hash, registry = _validated_execution(spec)
    parameter_lock = load_json(spec["artifact_paths"]["parameter_lock"])[
        "content_hash"
    ]
    targets = target_lock = None
    relational_teacher = None
    parent_caches = {}
    relational_input_key = None
    if node_id != ANCHOR_NODE_ID:
        targets, target_lock = _probability_targets(
            spec, node.distribution_teacher_id, consumer_id=node_id,
        )
    started = time.monotonic()
    _, split_hash, selection_hash, caches, input_key = _caches(spec, node=node)
    cache_seconds = time.monotonic() - started
    parents = _common_parents(
        spec, split_hash=split_hash, selection_hash=selection_hash,
        execution_acceptance=acceptance_hash, parameter_lock=parameter_lock,
    )
    if node_id == ANCHOR_NODE_ID:
        source = load_json(spec["artifact_paths"]["source_lock"])
        validate_source_lock(source)
        parents["pure_offline_oracle_report"] = source["u000"]["report_sha256"]
        parents["pure_offline_oracle_checkpoint"] = source["u000"][
            "selected_checkpoint_sha256"
        ]
    else:
        parent_id = RELATIONAL_CARRIERS[node_id]
        if parent_id != node.parent_node_id:
            raise ValueError("attention relational carrier differs")
        relational_teacher, parent_report = _model(
            spec, parent_id, device=device,
        )
        parents["teacher_report"] = parent_report["content_hash"]
        parents["teacher_checkpoint"] = parent_report[
            "selected_checkpoint_sha256"
        ]
        parents["relational_carrier_report"] = parent_report["content_hash"]
        parents["relational_carrier_checkpoint"] = parent_report[
            "selected_checkpoint_sha256"
        ]
        parents["probability_lock"] = target_lock["content_hash"]
        _, _, _, parent_caches, relational_input_key = _caches(
            spec, node=NODE_REGISTRY[parent_id], roles=("train",),
            memory_gib=200.0,
        )
    if recovery_spec_sha256 is not None:
        parents["recovery_spec"] = recovery_spec_sha256
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=targets,
            output_dir=node_output_dir(spec["campaign_root"], node_id),
            parents=parents, campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=(
                execution_source_commit or spec["source_commit"]
            ),
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec, node_id=node_id), execution_mode="scientific",
            preparation_metrics={
                "student_view_cache_seconds": cache_seconds,
                "pre_training_total_seconds": time.monotonic() - started,
            },
            authority=training_authority(node_id),
            learning_rate_schedule=(
                None if node_id == ANCHOR_NODE_ID else dict(LR_SCHEDULE)
            ),
            early_stopping=None,
            attention_reoptimization=(
                None if node_id == ANCHOR_NODE_ID
                else DEFAULT_ATTENTION_RECIPE.payload()
            ),
            attention_parameter_registry=(
                None if node_id == ANCHOR_NODE_ID else registry
            ),
            relational_teacher_model=relational_teacher,
            relational_train_cache=(
                None if node_id == ANCHOR_NODE_ID else parent_caches["train"]
            ),
            relational_input_key=relational_input_key,
        )
    finally:
        caches.clear()
        parent_caches.clear()
        del relational_teacher, targets
        gc.collect()


def run_reducer(
    *, spec: Mapping[str, Any], distribution_id: str, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec)
    _configure_deterministic_backend()
    components = PROBABILITY_COMPONENTS.get(distribution_id)
    if components is None or len(components) != 1:
        raise KeyError("unknown attention four-spine reducer")
    component = components[0]
    node = NODE_REGISTRY[component]
    _, split_hash, selection_hash, caches, input_key = _caches(spec, node=node)
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), node.seed_alias + "/sampler",
    )
    acceptance_hash, _ = _validated_execution(spec)
    parameter_lock = load_json(spec["artifact_paths"]["parameter_lock"])[
        "content_hash"
    ]
    started = time.monotonic()
    role_state = {}
    try:
        model, report = _model(spec, component, device=device)
        for role in ("train", "validation"):
            identities, logits, labels = _infer_cache(
                model, caches[role], input_key=input_key,
                sampler_seed=sampler_seed, device=device,
            )
            role_state[role] = {
                "identities": identities, "logits": logits, "labels": labels,
            }
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        parents = _common_parents(
            spec, split_hash=split_hash, selection_hash=selection_hash,
            execution_acceptance=acceptance_hash, parameter_lock=parameter_lock,
        )
        parents.update({
            "component_report": report["content_hash"],
            "component_checkpoint": report["selected_checkpoint_sha256"],
        })
        if recovery_spec_sha256 is not None:
            parents["recovery_spec"] = recovery_spec_sha256
        root = distribution_output_dir(spec["campaign_root"], distribution_id)
        manifests = {}
        for role in ("train", "validation"):
            state = role_state[role]
            lineage = {component: {
                "report_sha256": report["content_hash"],
                "checkpoint_sha256": report["selected_checkpoint_sha256"],
                "logits_sha256": array_sha256("logits", state["logits"]),
            }}
            manifests[role] = publish_probability_role(
                root, distribution_id=distribution_id, role=role,
                identity_digests=state["identities"],
                component_logits={component: state["logits"]},
                component_lineage=lineage, parents=parents,
                producer_commit=execution_source_commit or spec["source_commit"],
            )
        lock = publish_probability_lock(
            root / "lock.json", distribution_id=distribution_id,
            train_manifest=manifests["train"],
            validation_manifest=manifests["validation"], parents=parents,
        )
        _, _, probabilities = load_probability_role(
            root / "validation_manifest.json",
            expected_distribution_id=distribution_id,
            expected_role="validation",
        )
        metrics = classification_metrics(
            np.log(np.maximum(probabilities, 1e-30)),
            role_state["validation"]["labels"],
        )
        report_out = artifact({
            "parents": {**parents, "probability_lock": lock["content_hash"]},
            "distribution_id": distribution_id,
            "component_order": [component],
            "single_component_selected_checkpoint": True,
            "temperature": {"train": 2.0, "validation": 1.0},
            "validation_metrics": metrics,
            "runtime_seconds": time.monotonic() - started,
            "durable_output_bytes": sum(
                path.stat().st_size for path in root.rglob("*") if path.is_file()
            ),
            "durable_particle_views": False,
            "durable_hidden_states": False,
            "durable_attention_targets": False,
            "source_campaign_outputs_mutated": False,
            "poor_metrics_do_not_control_graph": True,
            "final_test_accessed": False,
        }, contract=STAGE_REPORT_CONTRACT)
        write_immutable_json(
            Path(spec["campaign_root"]) / "reports/stages"
            / f"{distribution_id}.json", report_out,
        )
        return report_out
    finally:
        caches.clear()


__all__ = [
    "distribution_output_dir", "node_output_dir", "run_fit", "run_reducer",
    "training_authority",
]
