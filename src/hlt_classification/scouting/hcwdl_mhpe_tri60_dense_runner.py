"""Fit and reducer execution for the isolated TRI60 dense extension."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import array_sha256, load_json, write_immutable_json

from .evaluation import classification_metrics
from .hcwdl_mhpe_runner import _diversity
from .hcwdl_mhpe_targets import uniform_probability_ensemble
from .hcwdl_mhpe_tri60_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_tri60_contracts import (
    EPHEMERAL_REP_AUDIT_CONTRACT, validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_dense_campaign import validate_campaign
from .hcwdl_mhpe_tri60_dense_contracts import (
    FINAL_CHECKPOINT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
)
from .hcwdl_mhpe_tri60_dense_graph import (
    COORDINATES, ENSEMBLE_COMPONENTS, GRAPH_SHA256, NODE_REGISTRY,
    SOURCE_DISTRIBUTIONS, component_origin,
)
from .hcwdl_mhpe_tri60_dense_probability import (
    DenseProbabilityTargets, load_probability_role, publish_probability_lock,
    publish_probability_role, validate_probability_lock,
)
from .hcwdl_mhpe_tri60_dense_source import (
    source_node_lineage, validate_source_gate, validate_source_lock,
)
from .hcwdl_mhpe_tri60_ephemeral import EphemeralRepresentationTargetBank
from .hcwdl_mhpe_tri60_graph import NODE_REGISTRY as SOURCE_NODE_REGISTRY
from .hcwdl_mhpe_tri60_probability import (
    Tri60ProbabilityTargets, validate_probability_lock as validate_source_probability_lock,
)
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import (
    _carrier_source_partitions, _configure_deterministic_backend, _infer_cache,
    _target_batch, _teacher_forward,
)
from .hcwdl_mhpe_tri60_training import (
    TRI60_PREFETCH_DEPTH, Tri60TrainingAuthority, Tri60TrainingRuntime,
    _BatchPrefetcher, load_tri60_model, train_tri60_node,
)
from .hcwdl_representation_kernels import generate_spectral_resource_bundle
from .hcwdl_representation_target_runtime import (
    TargetForwardBatch, prepare_target_generation_in_memory,
)
from .hcwdl_representation_targets import ORDINARY_BANK
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common, _stream
from .splits import role_records
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
    )
    authority.validate()
    return authority


def _source_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    source = load_json(spec["artifact_paths"]["source_campaign_spec"])
    validate_source_campaign(source, executable=False, verify_source_tree=False)
    return source


def _foundation(spec: Mapping[str, Any]) -> dict[str, Any]:
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    return foundation


def _source_lock(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(value)
    return value


def _source_gate(spec: Mapping[str, Any], *, required: bool) -> dict[str, Any] | None:
    path = Path(spec["artifact_paths"]["source_gate"])
    if not path.is_file():
        if required:
            raise FileNotFoundError("dense source gate is absent")
        return None
    value = load_json(path)
    validate_source_gate(value, source_lock=_source_lock(spec))
    return value


def _runtime(spec: Mapping[str, Any]) -> Tri60TrainingRuntime:
    recipe = load_json(spec["artifact_paths"]["recipe"])
    validate_recipe(recipe)
    training = recipe["training"]
    return Tri60TrainingRuntime(
        passes=int(training["passes"]),
        batch_size=int(training["effective_batch_size"]),
        peak_learning_rate=float(training["peak_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_fraction=float(training["warmup_fraction"]),
        minimum_lr_fraction=float(training["learning_rate_floor_fraction"]),
        amp_dtype=str(training["forward_precision"]),
    )


def _behavior(coordinate_name: str) -> str:
    return "hlt" if coordinate_name == "D000" else "p0" if coordinate_name == "U000" else "balanced_uniform"


def _student_caches(spec: Mapping[str, Any], *, node) -> tuple[Any, ...]:
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(foundation)
    sampler_seed = derive_seed(int(spec["replicate_seed"]), node.seed_alias + "/sampler")
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    requested = 360.0 if node.track in {"RSET", "RREL"} else 240.0
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced,
        behavior=_behavior(node.coordinate_name), coordinate=node.coordinate,
        batch_size=256, sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=requested, include_hcwdl_metadata=True,
    )
    return (
        foundation, split, split_hash, selection_hash, selections,
        assignments, balanced, caches, input_key,
    )


def _node(spec: Mapping[str, Any], node_id: str):
    if node_id in NODE_REGISTRY:
        return NODE_REGISTRY[node_id]
    try:
        return SOURCE_NODE_REGISTRY[node_id]
    except KeyError as error:
        raise KeyError(f"unknown dense/source node: {node_id}") from error


def _model(spec: Mapping[str, Any], node_id: str, *, device: str):
    if node_id in NODE_REGISTRY:
        path = node_output_dir(spec["campaign_root"], node_id) / "training_report.json"
        return load_tri60_model(
            path, device=device, authority=training_authority(node_id),
        )
    lock = _source_lock(spec)
    gate = _source_gate(spec, required=node_id not in lock["early_nodes"])
    lineage = source_node_lineage(
        source_lock=lock, source_gate=gate, node_id=node_id,
    )
    return load_tri60_model(lineage["report_path"], device=device)


def _probability_targets(
    spec: Mapping[str, Any], distribution_id: str, *, consumer_id: str,
):
    if distribution_id in SOURCE_DISTRIBUTIONS:
        lock = _source_lock(spec)
        if consumer_id not in lock["authorized_dense_probability_consumers"].get(
            distribution_id, ()
        ):
            raise PermissionError("dense probability consumer is not authorized")
        row = lock["early_distributions"][distribution_id]
        source_lock, _ = validate_source_probability_lock(
            row["lock_path"], distribution_id=distribution_id,
        )
        if source_lock["content_hash"] != row["lock_sha256"]:
            raise ValueError("dense imported probability lock changed")
        target = Tri60ProbabilityTargets.load(
            Path(row["root"]) / "train_manifest.json",
            distribution_id=distribution_id,
        )
        return target, source_lock
    directory = distribution_output_dir(spec["campaign_root"], distribution_id)
    lock, _ = validate_probability_lock(
        directory / "lock.json", distribution_id=distribution_id,
    )
    return DenseProbabilityTargets.load(
        directory / "train_manifest.json", distribution_id=distribution_id,
    ), lock


def _carrier_targets(
    spec: Mapping[str, Any], *, node_id: str, foundation, split,
    selections, assignments, balanced, device: str,
) -> tuple[EphemeralRepresentationTargetBank, dict[str, Any]]:
    node = NODE_REGISTRY[node_id]
    carrier_id = str(node.representation_carrier_id)
    model, carrier_report = _model(spec, carrier_id, device=device)
    carrier_node = _node(spec, carrier_id)
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), carrier_node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    factories: dict[str, Callable[[], Iterable[TargetForwardBatch]]] = {}
    partition_specs: dict[str, dict[str, int]] = {}
    records = role_records(split, "train")
    view_key = "hlt" if _behavior(carrier_node.coordinate_name) == "hlt" else "privileged"
    partitions = _carrier_source_partitions(records, selection=selections["train"])
    for source in partitions:
        source_index = int(source["source_index"])
        partition = str(source["partition"])

        def unprefetched_factory(
            *, source_index=source_index, partition=partition, view_key=view_key,
        ):
            stream = _stream(
                foundation_spec=foundation, split=split, selections=selections,
                assignments=assignments, balanced=balanced, role="train",
                behavior=_behavior(carrier_node.coordinate_name),
                coordinate=carrier_node.coordinate, batch_size=256,
                sampler_seed=sampler_seed, repair_seed=repair_seed,
                include_hcwdl_metadata=True, source_index=source_index,
            )
            for batch in stream:
                yield _target_batch(
                    batch, partition=partition, source_file_id=source_index,
                    view_key=view_key,
                )

        def factory(*, unprefetched_factory=unprefetched_factory):
            with _BatchPrefetcher(
                unprefetched_factory(), depth=TRI60_PREFETCH_DEPTH,
            ) as prefetched:
                yield from prefetched

        factories[partition] = factory
        partition_specs[partition] = {
            "rows": int(source["rows"]),
            "source_file_id": int(source["source_file_id"]),
        }
    bundle = generate_spectral_resource_bundle()
    endpoint = load_json(spec["artifact_paths"]["endpoint_resource_lock"])
    if (
        endpoint.get("spectral_resource_sha256") != bundle.content_hash
        or endpoint.get("token_resource_sha256") != bundle.token.content_hash
        or endpoint.get("relation_resource_sha256") != bundle.relation.content_hash
    ):
        raise ValueError("dense spectral resources differ")
    prepared = prepare_target_generation_in_memory(
        bank_kind=ORDINARY_BANK, partition_batches=factories,
        partition_specs=partition_specs,
        teacher_forward=_teacher_forward(model, device=device),
        token_resources=bundle.token, relation_resources=bundle.relation,
        teacher_model=model,
        allowed_input_fields=(
            "family_codes", "features", "mask", "vectors", "visible_indices",
        ),
    )
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    bank = EphemeralRepresentationTargetBank.from_prepared(
        prepared, strategy=node.track, carrier_node_id=carrier_id,
        carrier_report_sha256=carrier_report["content_hash"],
        carrier_checkpoint_sha256=carrier_report["selected_checkpoint_sha256"],
        campaign_spec_sha256=spec["content_hash"], graph_sha256=GRAPH_SHA256,
        recipe_sha256=spec["parents"]["source_recipe"],
    )
    return bank, {"bundle": bundle, "carrier_report": carrier_report}


def run_fit(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec)
    _configure_deterministic_backend()
    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown dense fit")
    node = NODE_REGISTRY[node_id]
    probability_targets, probability_lock = _probability_targets(
        spec, str(node.distribution_teacher_id), consumer_id=node_id,
    )
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(foundation)
    rep_bank = rep_context = None
    started = time.monotonic()
    if node.auxiliary != "none":
        target_started = time.monotonic()
        rep_bank, rep_context = _carrier_targets(
            spec, node_id=node_id, foundation=foundation, split=split,
            selections=selections, assignments=assignments, balanced=balanced,
            device=device,
        )
        target_seconds = time.monotonic() - target_started
    else:
        target_seconds = 0.0
    cache_started = time.monotonic()
    _, _, _, _, _, _, _, caches, input_key = _student_caches(spec, node=node)
    cache_seconds = time.monotonic() - cache_started
    parents = {
        "campaign_spec": spec["content_hash"],
        "source_campaign": spec["parents"]["source_campaign"],
        "source_lock": spec["parents"]["source_lock"],
        "foundation": spec["parents"]["foundation"],
        "graph": GRAPH_SHA256, "recipe": spec["parents"]["source_recipe"],
        "split_manifest": split_hash, "selection_manifest": selection_hash,
        "probability_lock": probability_lock["content_hash"],
    }
    if recovery_spec_sha256 is not None:
        parents["recovery_spec"] = recovery_spec_sha256
    audit_hash = None
    token_resources = relation_resources = None
    if rep_bank is not None:
        import resource
        try:
            import torch
            cuda_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except ImportError:
            cuda_peak = 0
        raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        audit = rep_bank.audit(
            peak_rss_bytes=raw_rss * 1024, peak_cuda_bytes=cuda_peak,
        )
        output = node_output_dir(spec["campaign_root"], node_id)
        write_immutable_json(output / "ephemeral_representation_audit.json", audit)
        audit_hash = validate_source_artifact(
            audit, contract=EPHEMERAL_REP_AUDIT_CONTRACT,
        )
        parents["ephemeral_representation_audit"] = audit_hash
        parents["representation_carrier_report"] = rep_context["carrier_report"]["content_hash"]
        parents["representation_carrier_checkpoint"] = rep_context["carrier_report"]["selected_checkpoint_sha256"]
        token_resources = rep_context["bundle"].token
        relation_resources = rep_context["bundle"].relation
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=probability_targets,
            representation_targets=rep_bank,
            representation_audit_sha256=audit_hash,
            token_resources=token_resources, relation_resources=relation_resources,
            output_dir=node_output_dir(spec["campaign_root"], node_id),
            parents=parents, campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["source_recipe"],
            execution_source_commit=execution_source_commit or spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec), execution_mode="scientific",
            preparation_metrics={
                "carrier_representation_target_seconds": target_seconds,
                "student_view_cache_seconds": cache_seconds,
                "pre_training_total_seconds": time.monotonic() - started,
            }, authority=training_authority(node_id),
        )
    finally:
        caches.clear()
        if rep_bank is not None:
            rep_bank.release()


def run_reducer(
    *, spec: Mapping[str, Any], distribution_id: str, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec)
    _configure_deterministic_backend()
    components = ENSEMBLE_COMPONENTS.get(distribution_id)
    if not components:
        raise KeyError("unknown dense reducer")
    representative_node = _node(spec, components[0])
    _, _, _, _, _, _, _, caches, input_key = _student_caches(
        spec, node=representative_node,
    )
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), representative_node.seed_alias + "/sampler",
    )
    role_state = {
        role: {"identities": None, "labels": None, "logits": {}, "lineage": {}}
        for role in ("train", "validation")
    }
    started = time.monotonic()
    try:
        for component in components:
            model, report = _model(spec, component, device=device)
            for role in ("train", "validation"):
                identities, logits, labels = _infer_cache(
                    model, caches[role], input_key=input_key,
                    sampler_seed=sampler_seed, device=device,
                )
                state = role_state[role]
                if state["identities"] is None:
                    state["identities"], state["labels"] = identities, labels
                elif not np.array_equal(state["identities"], identities):
                    raise ValueError("dense reducer component identities reorder")
                elif not np.array_equal(state["labels"], labels):
                    raise ValueError("dense reducer component labels differ")
                state["logits"][component] = logits
                state["lineage"][component] = {
                    "report_sha256": report["content_hash"],
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                    "logits_sha256": array_sha256("logits", logits),
                }
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        parents = {
            "campaign_spec": spec["content_hash"],
            "source_campaign": spec["parents"]["source_campaign"],
            "source_lock": spec["parents"]["source_lock"],
            "foundation": spec["parents"]["foundation"],
            "graph": GRAPH_SHA256, "recipe": spec["parents"]["source_recipe"],
        }
        if any(component_origin(component) == "source" and component not in _source_lock(spec)["early_nodes"] for component in components):
            gate = _source_gate(spec, required=True)
            parents["source_gate"] = gate["content_hash"]
        if recovery_spec_sha256 is not None:
            parents["recovery_spec"] = recovery_spec_sha256
        root = distribution_output_dir(spec["campaign_root"], distribution_id)
        manifests = {}
        for role in ("train", "validation"):
            state = role_state[role]
            manifests[role] = publish_probability_role(
                root, distribution_id=distribution_id, role=role,
                identity_digests=state["identities"],
                component_logits=state["logits"],
                component_lineage=state["lineage"], parents=parents,
                producer_commit=execution_source_commit or spec["source_commit"],
            )
        lock = publish_probability_lock(
            root / "lock.json", distribution_id=distribution_id,
            train_manifest=manifests["train"],
            validation_manifest=manifests["validation"], parents=parents,
        )
        _, _, probability = load_probability_role(
            root / "validation_manifest.json",
            expected_distribution_id=distribution_id,
            expected_role="validation",
        )
        labels = role_state["validation"]["labels"]
        metrics = classification_metrics(np.log(np.maximum(probability, 1e-30)), labels)
        component_metrics = {
            name: classification_metrics(logits, labels)
            for name, logits in role_state["validation"]["logits"].items()
        }
        leave_one_out = {}
        for omitted in components:
            reduced = {
                name: value for name, value in role_state["validation"]["logits"].items()
                if name != omitted
            }
            if reduced:
                value = uniform_probability_ensemble(reduced, temperature=1.0)
                leave_one_out[omitted] = classification_metrics(
                    np.log(np.maximum(value, 1e-30)), labels,
                )
        aucs = [float(row["macro_ovr_auc"]) for row in component_metrics.values()]
        best = max(component_metrics, key=lambda name: float(component_metrics[name]["macro_ovr_auc"]))
        report = artifact({
            "parents": {**parents, "probability_lock": lock["content_hash"]},
            "distribution_id": distribution_id,
            "component_order": list(components),
            "component_origin": {name: component_origin(name) for name in components},
            "uniform_probability_weights": [1, len(components)],
            "component_metrics": component_metrics, "ensemble_metrics": metrics,
            "ensemble_minus_mean_component_auc": float(metrics["macro_ovr_auc"]) - float(np.mean(aucs)),
            "ensemble_minus_best_component_auc": float(metrics["macro_ovr_auc"]) - max(aucs),
            "best_component_id": best, "leave_one_out": leave_one_out,
            "diversity": _diversity(role_state["validation"]["logits"], 1.0, labels),
            "runtime_seconds": time.monotonic() - started,
            "source_campaign_outputs_mutated": False,
            "poor_metrics_do_not_control_graph": True,
            "final_test_accessed": False,
        }, contract=STAGE_REPORT_CONTRACT)
        write_immutable_json(
            Path(spec["campaign_root"]) / "reports/stages" / f"{distribution_id}.json",
            report,
        )
        return report
    finally:
        caches.clear()


__all__ = [
    "distribution_output_dir", "node_output_dir", "run_fit", "run_reducer",
    "training_authority",
]
