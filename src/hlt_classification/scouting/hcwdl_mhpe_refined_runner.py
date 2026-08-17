"""Execution and reporting for the R-augmented MHPE continuation."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .engine import (
    classification_metrics, precompute_teacher_targets,
    validate_pmard_training_report,
)
from .hcwdl_ladder import NodeSpec, TeacherSpec
from .hcwdl_mhpe_contracts import stage_report_contract
from .hcwdl_mhpe_graph import node_registry
from .hcwdl_mhpe_refined import (
    AGGREGATE_CONTRACT, AUGMENTED_ENSEMBLES, COMPLETION_CONTRACT,
    GRAPH_SHA256, NODES, NODE_CONTRACT, RUNTIME_CONTRACT, SOURCE_PROFILE,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT,
    campaign_tasks, ephemeral_augmented_target, publish_augmented_bundle,
    publish_augmented_target, validate_augmented_bundle, validate_campaign,
)
from .hcwdl_mhpe_runner import (
    _context as source_context, _predict_components as source_predict_components,
    _stream,
)
from .hcwdl_mhpe_targets import (
    DurableProbabilityTargets, uniform_probability_ensemble,
)
from .hcwdl_training import train_hcwdl_node
from .hcwdl_unified_balanced_runner import DOMAINS, _cache_student_views
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .targets import EphemeralProbabilityTargets
from .training import GenerationalLossConfiguration, derive_seed


def _source(spec: Mapping[str, Any]):
    validate_campaign(spec, verify_source_tree=False)
    source_spec = load_json(spec["source"]["source_spec_path"])
    return source_spec, source_context(source_spec, verify_source_tree=False)


def _training_registry() -> Mapping[str, NodeSpec]:
    rows = {}
    for node in NODES.values():
        rows[node.node_id] = NodeSpec(
            node_id=node.node_id, track="mhpe_refined_continuation",
            stage="refiner" if node.ce_weight == .10 else "projection",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh", initialization_parent=None,
            teachers=(TeacherSpec(
                node.teacher_id,
                "hlt" if node.teacher_id == "D000Eplus" else "privileged",
                "parent",
            ),),
            loss_kind="ce_kd", deployable=node.input_domain == "hlt",
        )
    return MappingProxyType(rows)


TRAINING_REGISTRY = _training_registry()


def _loss(node_id: str) -> GenerationalLossConfiguration:
    node = NODES[node_id]
    return GenerationalLossConfiguration(
        arm=f"HCWDL_MHPE_REFINED_{node_id}", ce=node.ce_weight,
        parent_kd=node.kd_weight, grandparent_kd=0,
        parent_temperature=node.temperature, grandparent_temperature=1.0,
    )


def _source_probability(spec: Mapping[str, Any], ensemble_id: str, role: str,
                        *, split_hash: str) -> EphemeralProbabilityTargets:
    bundle = spec["source"]["bundles"][ensemble_id]
    manifest_path = Path(bundle["root"]) / f"{role}_manifest.json"
    durable = DurableProbabilityTargets(manifest_path)
    if durable.manifest["content_hash"] != bundle["manifests"][role]:
        raise ValueError("refined-continuation source probability lineage differs")
    return durable.as_ephemeral(split_manifest_sha256=split_hash)


def _model_teacher_targets(*, spec: Mapping[str, Any], teacher_id: str,
                           foundation: Mapping[str, Any], split: Mapping[str, Any],
                           split_hash: str, selections, assignments, balanced,
                           recipe: Mapping[str, Any], sampler_seed: int,
                           repair_seed: int, device: str):
    teacher_node = NODES[teacher_id]
    report_path = Path(spec["campaign_root"]) / "training" / teacher_id / "training_report.json"
    report = load_json(report_path); report_hash = validate_pmard_training_report(report)
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("refined-continuation teacher changed during load")
    behavior = "hlt" if teacher_node.input_domain == "hlt" else "balanced_uniform"
    input_key = "hlt" if behavior == "hlt" else "privileged"
    stream = _stream(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train",
        behavior=behavior, coordinate=teacher_node.coordinate,
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed,
    )
    targets = precompute_teacher_targets(
        model, stream, input_key=input_key, device=device,
        teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
    )
    del model; gc.collect()
    return targets, report_hash, report["selected_checkpoint_sha256"]


def train_node(*, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
               recovery_spec_sha256: str | None = None) -> dict[str, Any]:
    if node_id not in NODES:
        raise ValueError("unknown refined-continuation node")
    validate_campaign(spec, verify_source_tree=recovery_spec_sha256 is None)
    _, context = _source(spec)
    (_, _, foundation, split, split_hash, selection_hash,
     selections, assignments, balanced, recipe) = context
    node = NODES[node_id]
    sampler_seed = derive_seed(
        int(foundation["replicate_seed"]), f"mhpe/sampler/{node.seed_alias}",
    )
    repair_seed = derive_seed(int(foundation["replicate_seed"]), "ub/repair/v1")
    behavior = "hlt" if node.input_domain == "hlt" else "balanced_uniform"
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=node.coordinate,
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=72.0,
    )
    probability = logits = None
    if node.teacher_kind == "source_probabilities":
        probability = _source_probability(
            spec, node.teacher_id, "train", split_hash=split_hash,
        )
        teacher_hash = probability.header["target_manifest_sha256"]
    elif node.teacher_kind == "augmented_probabilities":
        target_root = Path(spec["campaign_root"]) / "targets" / node.teacher_id / "T1"
        lock_hash, manifests = validate_augmented_bundle(target_root, ensemble_id=node.teacher_id)
        expected_target_parents = {
            "campaign_spec_sha256": spec["content_hash"],
            "source_campaign_spec_sha256": spec["source"]["source_spec_sha256"],
            "split_manifest_sha256": split_hash,
            "selection_manifest_sha256": selection_hash,
        }
        if any(
            manifests["train"]["parents"].get(name) != value
            for name, value in expected_target_parents.items()
        ):
            raise ValueError("refined-continuation target parent lineage differs")
        probability = ephemeral_augmented_target(
            target_root / "train_manifest.json", split_manifest_sha256=split_hash,
        )
        teacher_hash = manifests["train"]["content_hash"]
        if probability.header["target_manifest_sha256"] != teacher_hash:
            raise ValueError("refined-continuation probability target differs")
    else:
        logits, teacher_hash, _ = _model_teacher_targets(
            spec=spec, teacher_id=node.teacher_id, foundation=foundation,
            split=split, split_hash=split_hash, selections=selections,
            assignments=assignments, balanced=balanced, recipe=recipe,
            sampler_seed=sampler_seed, repair_seed=repair_seed, device=device,
        )
        lock_hash = None
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "source_campaign_spec_sha256": spec["source"]["source_spec_sha256"],
        "foundation_reuse_lock_sha256": spec["source"]["foundation_reuse_lock_sha256"],
        "teacher_target_or_report_sha256": teacher_hash,
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
    }
    if node.teacher_kind == "augmented_probabilities":
        parents["teacher_target_lock_sha256"] = lock_hash
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
        parents=parents, device=device, registry=TRAINING_REGISTRY, domains=DOMAINS,
        graph_sha256=GRAPH_SHA256, report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label="HCWDL-MHPE-REFINED-CONTINUATION-300K60",
        seed_node_id=node.seed_alias, node_contract=NODE_CONTRACT,
        explicit_loss=_loss(node_id), recipe_overlay_sha256=spec["recipe_sha256"],
        parent_teacher_targets=logits, parent_probability_targets=probability,
        peak_learning_rate_override=float(recipe["optimizer"]["peak_learning_rates"]["cold_child"]),
        scientific_config_extra={
            "coordinate_exact": node.coordinate.payload(), "input_key": input_key,
            "teacher_id": node.teacher_id, "teacher_kind": node.teacher_kind,
            "refiner": node.ce_weight == .10,
            "paired_source_coordinate_seed": node_id in {
                "D066_from_U100R", "D033_from_D066R", "D000_from_D033R", "M1R",
            },
            "student_view_built_once": True, "teacher_targets_built_once": True,
            "final_test_accessed": False,
        },
    )
    returned = dict(result)
    returned["_cache_array_bytes"] = {
        role: int(cache.header["array_bytes"]) for role, cache in caches.items()
    }
    return returned


def _labels(cache, *, sampler_seed: int, batch_size: int) -> tuple[tuple[str, ...], np.ndarray]:
    identities = []; labels = []
    for batch in cache.iterate_batches(epoch=0, sampler_seed=sampler_seed, batch_size=batch_size):
        identities.extend(map(str, batch["identity_keys"]))
        labels.append(np.asarray(batch["labels"], np.int64))
    return tuple(identities), np.concatenate(labels)


def run_ensemble(*, spec: Mapping[str, Any], ensemble_id: str,
                 device: str = "cuda", recovery_spec_sha256: str | None = None) -> dict[str, Any]:
    if ensemble_id not in AUGMENTED_ENSEMBLES:
        raise ValueError("unknown refined-continuation ensemble")
    validate_campaign(spec, verify_source_tree=recovery_spec_sha256 is None)
    _, context = _source(spec)
    (_, _, foundation, split, split_hash, selection_hash,
     selections, assignments, balanced, recipe) = context
    config = AUGMENTED_ENSEMBLES[ensemble_id]
    new_id = config["new_component"]
    report_path = Path(spec["campaign_root"]) / "training" / new_id / "training_report.json"
    report = load_json(report_path); report_hash = validate_pmard_training_report(report)
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("refined-continuation reducer component changed")
    coordinate = NODES[new_id].coordinate
    sampler_seed = derive_seed(int(foundation["replicate_seed"]), f"mhpe/refined/reducer/{ensemble_id}")
    repair_seed = derive_seed(int(foundation["replicate_seed"]), "ub/repair/v1")
    behavior = "hlt" if config["coordinate_name"] == "D000" else "balanced_uniform"
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=coordinate,
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=72.0,
    )
    target_root = Path(spec["campaign_root"]) / "targets" / ensemble_id / "T1"
    source_bundle = spec["source"]["bundles"][config["source_ensemble"]]
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "source_campaign_spec_sha256": spec["source"]["source_spec_sha256"],
        "source_ensemble_target_lock_sha256": source_bundle["lock_sha256"],
        "new_component_report_sha256": report_hash,
        "new_component_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
    }
    role_metadata = {}; metrics = None; source_metrics = None
    source_spec = load_json(spec["source"]["source_spec_path"])
    source_role_data, _, _ = source_predict_components(
        spec=source_spec, ensemble_id=config["source_ensemble"], device=device,
        recovery_spec_sha256=spec["source"]["source_completion_sha256"],
    )
    for role in ("train", "validation"):
        targets = precompute_teacher_targets(
            model, caches[role].iterate_batches(
                epoch=0, sampler_seed=sampler_seed,
                batch_size=int(recipe["batching"]["effective_batch_size"]),
            ), input_key=input_key, device=device,
            teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
        )
        source_probability = _source_probability(
            spec, config["source_ensemble"], role, split_hash=split_hash,
        )
        source_identities, source_logits, _, _, _, _ = source_role_data[role]
        lookup = {key: index for index, key in enumerate(source_identities)}
        try:
            source_indexes = [lookup[key] for key in targets.identities]
        except KeyError as error:
            raise KeyError("refined-continuation reducer identity join is incomplete") from error
        if len(source_indexes) != len(lookup) or len(set(source_indexes)) != len(source_indexes):
            raise ValueError("refined-continuation reducer identity coverage differs")
        aligned_source_logits = {
            name: values[source_indexes].astype(np.float32, copy=False)
            for name, values in source_logits.items()
        }
        old_recomputed = uniform_probability_ensemble(
            aligned_source_logits, temperature=1.0,
        )
        stored_lookup = {
            key: index for index, key in enumerate(source_probability.identities)
        }
        try:
            stored_indexes = [stored_lookup[key] for key in targets.identities]
        except KeyError as error:
            raise KeyError("refined-continuation stored source identity join is incomplete") from error
        old_aligned = source_probability.probabilities[stored_indexes]
        if not np.array_equal(old_recomputed, old_aligned):
            raise ValueError("refined-continuation source ensemble recomputation differs")
        all_logits = {**aligned_source_logits, new_id: targets.logits}
        if len(all_logits) != config["source_component_count"] + 1:
            raise ValueError("refined-continuation effective component count differs")
        augmented = uniform_probability_ensemble(all_logits, temperature=1.0)
        metadata = publish_augmented_target(
            target_root / f"{role}_all", ensemble_id=ensemble_id, role=role,
            identities=targets.identities, probabilities=augmented,
            new_logits_sha256=array_sha256("logits", targets.logits),
            new_report_sha256=report_hash,
            new_checkpoint_sha256=report["selected_checkpoint_sha256"],
            parents=parents, producer_commit=spec["source_commit"],
        )
        role_metadata[role] = metadata
        if role == "validation":
            label_ids, labels = _labels(
                caches[role], sampler_seed=sampler_seed,
                batch_size=int(recipe["batching"]["effective_batch_size"]),
            )
            if label_ids != targets.identities:
                raise ValueError("refined-continuation reducer label order differs")
            metrics = classification_metrics(np.log(np.maximum(augmented, np.float32(1e-30))), labels)
            source_metrics = classification_metrics(np.log(np.maximum(old_aligned, np.float32(1e-30))), labels)
    del model; gc.collect()
    lock = publish_augmented_bundle(
        target_root, ensemble_id=ensemble_id, role_metadata=role_metadata,
        parents=parents,
    )
    stage = with_content_hash({
        "contract": STAGE_REPORT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "ensemble_id": ensemble_id,
        "source_ensemble": config["source_ensemble"], "new_component": new_id,
        "source_component_count": config["source_component_count"],
        "effective_component_count": config["source_component_count"] + 1,
        "effective_component_weights_are_uniform": True,
        "target_lock_sha256": lock["content_hash"],
        "ensemble_metrics": metrics, "source_ensemble_recomputed_metrics": source_metrics,
        "new_component_metrics": report["validation"],
        "final_test_accessed": False,
    })
    write_immutable_json(Path(spec["campaign_root"]) / "reports" / f"{ensemble_id}_stage.json", stage)
    return stage


def _source_stage(spec: Mapping[str, Any], ensemble_id: str) -> dict[str, Any]:
    path = Path(spec["source"]["source_root"]) / "reports" / f"{ensemble_id}_stage.json"
    value = load_json(path)
    validate_content_hash(
        value, expected_contract=stage_report_contract(SOURCE_PROFILE), expected_schema_version=1,
    )
    if value["content_hash"] != spec["source"]["stage_report_sha256"][ensemble_id]:
        raise ValueError("refined-continuation source stage changed")
    return value


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, verify_source_tree=False)
    root = Path(spec["campaign_root"]); rows = []
    for node_id in NODES:
        directory = root / "training" / node_id
        report = load_json(directory / "training_report.json")
        outer = load_json(directory / "hcwdl_training_report.json")
        runtime = load_json(root / "reports/runtime" / f"{node_id}.json")
        report_hash = validate_pmard_training_report(report)
        validate_content_hash(
            outer, expected_contract=TRAINING_REPORT_CONTRACT,
            expected_schema_version=1,
        )
        validate_content_hash(
            runtime, expected_contract=RUNTIME_CONTRACT,
            expected_schema_version=1,
        )
        if (outer.get("node_id") != node_id
                or outer.get("graph_sha256") != GRAPH_SHA256
                or outer.get("recipe_overlay_sha256") != spec["recipe_sha256"]
                or outer.get("pmard_engine_report_sha256") != report_hash
                or runtime.get("node_id") != node_id
                or runtime.get("task_id") != f"train_{node_id}"):
            raise ValueError("refined-continuation training/runtime lineage differs")
        rows.append({
            "node_id": node_id, "kind": "model", "teacher_id": NODES[node_id].teacher_id,
            "report_sha256": report_hash,
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "metrics": report["validation"], "runtime": runtime,
        })
    for ensemble_id in AUGMENTED_ENSEMBLES:
        stage = load_json(root / "reports" / f"{ensemble_id}_stage.json")
        stage_hash = validate_content_hash(stage, expected_contract=STAGE_REPORT_CONTRACT, expected_schema_version=1)
        lock_hash, _ = validate_augmented_bundle(
            root / "targets" / ensemble_id / "T1", ensemble_id=ensemble_id,
        )
        config = AUGMENTED_ENSEMBLES[ensemble_id]
        if (stage.get("campaign_spec_sha256") != spec["content_hash"]
                or stage.get("source_ensemble") != config["source_ensemble"]
                or stage.get("new_component") != config["new_component"]
                or stage.get("target_lock_sha256") != lock_hash
                or stage.get("effective_component_weights_are_uniform") is not True):
            raise ValueError("refined-continuation stage/target lineage differs")
        rows.append({
            "node_id": ensemble_id, "kind": "ensemble", "teacher_id": None,
            "report_sha256": stage_hash, "selected_checkpoint_sha256": stage_hash,
            "metrics": stage["ensemble_metrics"],
        })
    source_rows = []
    for node_id in ("D066_from_U100E", "D033_from_D066E", "D000_from_D033E", "M1"):
        record = spec["source"]["reports"][node_id]
        report = load_json(record["path"])
        if validate_pmard_training_report(report) != record["report_sha256"]:
            raise ValueError("refined-continuation comparison report changed")
        source_rows.append({"node_id": node_id, "kind": "source_model", "metrics": report["validation"]})
    for ensemble_id in ("D066E", "D033E", "D000E"):
        stage = _source_stage(spec, ensemble_id)
        source_rows.append({"node_id": ensemble_id, "kind": "source_ensemble", "metrics": stage["ensemble_metrics"]})
    metrics_by_id = {row["node_id"]: row["metrics"] for row in [*rows, *source_rows]}
    pairs = (
        ("D066_from_U100R", "D066_from_U100E"), ("D066Eplus", "D066E"),
        ("D033_from_D066R", "D033_from_D066E"), ("D033Eplus", "D033E"),
        ("D000_from_D033R", "D000_from_D033E"), ("D000Eplus", "D000E"),
        ("M1R", "M1"),
    )
    comparisons = []
    for candidate, reference in pairs:
        left, right = metrics_by_id[candidate], metrics_by_id[reference]
        comparisons.append({
            "candidate": candidate, "reference": reference,
            "delta_macro_ovr_auc": float(left["macro_ovr_auc"]) - float(right["macro_ovr_auc"]),
            "delta_cross_entropy": float(left["cross_entropy"]) - float(right["cross_entropy"]),
            "delta_macro_mean_log_qcd_rejection_at_50pct_signal": (
                float(left["macro_mean_log_qcd_rejection_at_50pct_signal"])
                - float(right["macro_mean_log_qcd_rejection_at_50pct_signal"])
            ),
        })
    return with_content_hash({
        "contract": AGGREGATE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "rows": rows,
        "source_context": source_rows, "paired_comparisons": comparisons,
        "primary_comparison": "M1R_minus_M1", "fresh_fit_count": 7,
        "reducer_count": 3, "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    })


class RefinedContinuationWorkflow:
    def __init__(self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None) -> None:
        validate_campaign(spec, verify_source_tree=recovery_spec_sha256 is None)
        self.spec = spec; self.root = Path(spec["campaign_root"])
        self.recovery_spec_sha256 = recovery_spec_sha256

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        task = next((row for row in campaign_tasks() if row["task_id"] == task_id), None)
        if task is None:
            raise ValueError("unknown refined-continuation task")
        if task["kind"] == "train":
            started = time.monotonic()
            result = train_node(
                spec=self.spec, node_id=task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
            )
            runtime = with_content_hash({
                "contract": RUNTIME_CONTRACT, "schema_version": 1,
                "task_id": task_id, "node_id": task["node_id"],
                "elapsed_seconds": time.monotonic() - started,
                "cache_array_bytes": result.pop("_cache_array_bytes"),
                "final_test_accessed": False,
            })
            write_immutable_json(self.root / "reports/runtime" / f"{task['node_id']}.json", runtime)
            return result
        if task["kind"] == "ensemble":
            return run_ensemble(
                spec=self.spec, ensemble_id=task["ensemble_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
            )
        if task["kind"] == "aggregate":
            result = build_aggregate(self.spec)
            write_immutable_json(self.root / "reports/validation_aggregate.json", result)
            return result
        aggregate = load_json(self.root / "reports/validation_aggregate.json")
        aggregate_hash = validate_content_hash(
            aggregate, expected_contract=AGGREGATE_CONTRACT, expected_schema_version=1,
        )
        if aggregate.get("campaign_spec_sha256") != self.spec["content_hash"]:
            raise ValueError("refined-continuation aggregate differs")
        if (aggregate.get("fresh_fit_count") != 7
                or aggregate.get("reducer_count") != 3
                or len(aggregate.get("rows", ())) != 10
                or len(aggregate.get("paired_comparisons", ())) != 7
                or aggregate.get("final_test_accessed") is not False):
            raise ValueError("refined-continuation aggregate closure differs")
        result = with_content_hash({
            "contract": COMPLETION_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": self.spec["content_hash"],
            "aggregate_sha256": aggregate_hash, "fresh_fit_count": 7,
            "reducer_count": 3, "scientific_result_does_not_control_completion": True,
            "final_test_accessed": False,
        })
        write_immutable_json(self.root / "reports/campaign_complete.json", result)
        return result


__all__ = [
    "RefinedContinuationWorkflow", "TRAINING_REGISTRY", "build_aggregate",
    "run_ensemble", "train_node",
]
