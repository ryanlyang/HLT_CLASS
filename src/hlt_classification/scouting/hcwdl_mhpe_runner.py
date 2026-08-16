"""Training and probability-reduction workers for HCWDL-MHPE profiles."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import build_scouting_particle_transformer

from .engine import classification_metrics, precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_mhpe_campaign import validate_campaign
from .hcwdl_mhpe_contracts import (
    STAGE_REPORT_CONTRACT, campaign_profile, training_report_contract,
)
from .hcwdl_mhpe_graph import (
    COORDINATES, ENSEMBLE_COMPONENTS, PROFILE_C10P90,
    PROFILE_C10P90_300K60, PROFILE_C25P75_300K60, campaign_label,
    graph_sha256, node_registry, training_registry,
)
from .hcwdl_mhpe_targets import (
    DurableProbabilityTargets, publish_probability_manifest,
    publish_probability_shard, target_lock_payload, uniform_probability_ensemble,
    validate_probability_bundle,
)
from .hcwdl_training import train_hcwdl_node
from .hcwdl_unified_balanced_runner import DOMAINS, _cache_student_views, _load_common, _stream
from .hcwdl_unified_balanced_targets import DurableUnifiedBalancedTargets
from .hcwdl_unified_balanced_targets import (
    publish_target_manifest as publish_logit_manifest,
    publish_target_shard as publish_logit_shard,
)
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .training import GenerationalLossConfiguration, derive_seed


def _is_300k60(profile: str) -> bool:
    return profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}


def _runtime_parameters(profile: str) -> tuple[str, float]:
    return ("ub/repair/v1", 72.0) if _is_300k60(profile) else (
        "ub_full/repair/v1", 224.0,
    )


def _training_recipe(
    *, profile: str, foundation_root: Path, foundation: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the executable recipe used by the imported foundation.

    The all-mapped full-data foundation publishes its executable HCWDL recipe
    at ``foundation_root/recipe.json``.  The 300k unified-balanced foundation
    instead publishes a small scientific overlay at that location and binds
    the executable HCWDL recipe (batching, optimizer, and schedule included)
    through ``foundation_spec.artifact_paths.recipe``.  MHPE must follow the
    same recipe indirection as the foundation training worker.
    """
    if not _is_300k60(profile):
        return load_json(foundation_root / "recipe.json")
    artifact_paths = foundation.get("artifact_paths")
    if not isinstance(artifact_paths, Mapping) or "recipe" not in artifact_paths:
        raise ValueError("HCWDL-MHPE 300k executable recipe path is absent")
    return load_json(Path(str(artifact_paths["recipe"])))


def _context(spec: Mapping[str, Any], *, verify_source_tree: bool = True):
    validate_campaign(spec, executable=False, verify_source_tree=verify_source_tree)
    profile = campaign_profile(spec)
    reuse = load_json(spec["reuse_lock_path"])
    foundation_root = Path(reuse["foundation_spec_path"]).parent
    foundation = load_json(reuse["foundation_spec_path"])
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(foundation)
    recipe = _training_recipe(
        profile=profile, foundation_root=foundation_root, foundation=foundation,
    )
    return reuse, foundation_root, foundation, split, split_hash, selection_hash, selections, assignments, balanced, recipe


def _teacher_checkpoint(spec: Mapping[str, Any], teacher_id: str, foundation_root: Path) -> tuple[Path, dict[str, Any], str]:
    root = Path(spec["campaign_root"])
    canonical = "U050_from_U000" if teacher_id == "U050" else teacher_id
    output = foundation_root / "training/U000" if canonical == "U000" else root / "training" / canonical
    report = load_json(output / "training_report.json")
    report_hash = validate_pmard_training_report(report)
    checkpoint = output / str(report["selected_checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
        raise ValueError("HCWDL-MHPE teacher checkpoint differs")
    return output, report, report_hash


def _model_targets(
    *, spec, teacher_id, foundation_root, foundation, split, split_hash,
    selections, assignments, balanced, recipe, sampler_seed, repair_seed,
    selection_hash,
    device, registry,
):
    output, report, report_hash = _teacher_checkpoint(spec, teacher_id, foundation_root)
    if teacher_id == "U000":
        durable = DurableUnifiedBalancedTargets(
            foundation_root / "targets/u000_train/manifest.json", teacher_id="shared/U000",
        )
        return durable.as_ephemeral(
            teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
        ), report_hash
    if teacher_id == "U050":
        consumers = tuple(sorted(
            candidate.node_id for candidate in registry.values()
            if candidate.teacher_id == "U050"
        ))
        manifest_path = Path(spec["campaign_root"]) / "targets/U050/train_manifest.json"
        durable = DurableUnifiedBalancedTargets(
            manifest_path, teacher_id="MHPE/U050", consumers=consumers,
        )
        expected_parents = {
            "campaign_spec_sha256": spec["content_hash"],
            "foundation_reuse_lock_sha256": spec["reuse_lock_sha256"],
            "split_manifest_sha256": split_hash,
            "selection_manifest_sha256": selection_hash,
            "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "teacher_report_sha256": report_hash,
        }
        if durable.manifest["parents"] != expected_parents:
            raise ValueError("HCWDL-MHPE U050 target lineage differs")
        return durable.as_ephemeral(
            teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
        ), report_hash
    teacher_node = registry["U050_from_U000" if teacher_id == "U050" else teacher_id]
    behavior = "hlt" if teacher_node.input_domain == "hlt" else "balanced_uniform"
    input_key = "hlt" if behavior == "hlt" else "privileged"
    stream = _stream(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train",
        behavior=behavior, coordinate=teacher_node.coordinate,
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed,
    )
    model, loaded = load_pmard_model(
        output / "training_report.json", model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-MHPE teacher changed during load")
    targets = precompute_teacher_targets(
        model, stream, input_key=input_key, device=device,
        teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
    )
    del model; gc.collect()
    return targets, report_hash


def run_specialist(*, spec: Mapping[str, Any], node_id: str, device: str = "cuda", recovery_spec_sha256: str | None = None) -> dict[str, Any]:
    profile = campaign_profile(spec)
    registry = node_registry(profile)
    if node_id not in registry:
        raise ValueError("unknown HCWDL-MHPE specialist")
    node = registry[node_id]
    (reuse, foundation_root, foundation, split, split_hash, selection_hash,
     selections, assignments, balanced, recipe) = _context(spec, verify_source_tree=recovery_spec_sha256 is None)
    sampler_seed = derive_seed(int(foundation["replicate_seed"]), f"mhpe/sampler/{node.seed_alias}")
    repair_domain, memory_gib = _runtime_parameters(profile)
    repair_seed = derive_seed(int(foundation["replicate_seed"]), repair_domain)
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced,
        behavior="hlt" if node.input_domain == "hlt" else "balanced_uniform",
        coordinate=node.coordinate, batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=memory_gib,
    )
    teacher_logits = teacher_probability = None
    if node.teacher_kind == "probabilities":
        temperature_dir = "T1" if node.temperature == 1 else "T2"
        target_directory = Path(spec["campaign_root"]) / "targets" / node.teacher_id / temperature_dir
        expected_consumers = sorted(
            candidate.node_id for candidate in registry.values()
            if candidate.teacher_id == node.teacher_id
            and candidate.temperature == node.temperature
        )
        lock_hash, manifests = validate_probability_bundle(
            target_directory, ensemble_id=node.teacher_id,
            temperature=node.temperature, consumers=expected_consumers,
        )
        manifest_path = target_directory / "train_manifest.json"
        durable = DurableProbabilityTargets(manifest_path)
        if (manifests["train"]["content_hash"] != durable.manifest["content_hash"]
                or node_id not in expected_consumers):
            raise ValueError("HCWDL-MHPE probability target is not authorized for this child")
        teacher_probability = durable.as_ephemeral(split_manifest_sha256=split_hash)
        if teacher_probability.temperature != node.temperature:
            raise ValueError("HCWDL-MHPE probability temperature differs")
        teacher_hash = durable.manifest["content_hash"]
    else:
        teacher_logits, teacher_hash = _model_targets(
            spec=spec, teacher_id=node.teacher_id, foundation_root=foundation_root,
            foundation=foundation, split=split, split_hash=split_hash,
            selections=selections, assignments=assignments, balanced=balanced,
            recipe=recipe, sampler_seed=sampler_seed, repair_seed=repair_seed, device=device,
            selection_hash=selection_hash, registry=registry,
        )
    loss = GenerationalLossConfiguration(
        arm=f"HCWDL_UB_MHPE_{node_id}", ce=node.ce_weight,
        parent_kd=node.kd_weight, grandparent_kd=0,
        parent_temperature=node.temperature, grandparent_temperature=node.temperature,
    )
    parents = {
        "campaign_spec_sha256": spec["content_hash"], "foundation_reuse_lock_sha256": reuse["content_hash"],
        "split_manifest_sha256": split_hash, "selection_manifest_sha256": selection_hash,
        "teacher_target_sha256": teacher_hash,
    }
    if node.teacher_kind == "probabilities":
        parents["teacher_target_lock_sha256"] = lock_hash
    result = train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(foundation["replicate_seed"]), model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=int(recipe["batching"]["effective_batch_size"]),
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=int(recipe["batching"]["effective_batch_size"]),
        ),
        class_weights=np.ones(15, np.float32),
        output_dir=Path(spec["campaign_root"]) / "training" / node_id,
        parents=parents, device=device, registry=training_registry(profile), domains=DOMAINS,
        graph_sha256=graph_sha256(profile),
        report_contract=training_report_contract(profile),
        campaign_label=campaign_label(profile), seed_node_id=node.seed_alias,
        node_contract=node.contract, explicit_loss=loss,
        recipe_overlay_sha256=spec["recipe_sha256"],
        parent_teacher_targets=teacher_logits,
        parent_probability_targets=teacher_probability,
        peak_learning_rate_override=float(recipe["optimizer"]["peak_learning_rates"]["cold_child"]),
        scientific_config_extra={
            "coordinate_exact": node.coordinate.payload(), "teacher_id": node.teacher_id,
            "teacher_kind": node.teacher_kind, "input_key": input_key,
            "population_policy": foundation.get(
                "population_policy", "authenticated_selected_300k_rows_v1",
            ), "final_test_accessed": False,
            "student_view_built_once": True, "teacher_targets_built_once": True,
            **({"recipe_profile": profile} if profile != "C25P75" else {}),
            **({"population_profile": "pilot_300k_60pass"}
               if _is_300k60(profile) else {}),
        },
    )
    if node_id == "U050_from_U000":
        _publish_u050_targets(
            spec=spec, result=result, caches=caches, input_key=input_key,
            sampler_seed=sampler_seed, batch_size=int(recipe["batching"]["effective_batch_size"]),
            split_hash=split_hash, selection_hash=selection_hash, device=device,
            registry=registry,
        )
    returned = dict(result)
    returned["_runtime_cache_array_bytes"] = {
        role: int(cache.header["array_bytes"]) for role, cache in caches.items()
    }
    return returned


def _publish_u050_targets(
    *, spec: Mapping[str, Any], result: Mapping[str, Any], caches,
    input_key: str, sampler_seed: int, batch_size: int, split_hash: str,
    selection_hash: str, device: str, registry,
) -> None:
    """Publish the selected U050 logits once for its four direct consumers."""
    root = Path(spec["campaign_root"])
    report_path = root / "training/U050_from_U000/training_report.json"
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    if result.get("pmard_engine_report_sha256") != report_hash:
        raise ValueError("HCWDL-MHPE U050 selected report changed")
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-MHPE U050 changed during target load")
    targets = precompute_teacher_targets(
        model, caches["train"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=batch_size,
        ), input_key=input_key, device=device, teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    del model; gc.collect()
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "foundation_reuse_lock_sha256": spec["reuse_lock_sha256"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "teacher_report_sha256": report_hash,
    }
    directory = root / "targets/U050"
    _, metadata = publish_logit_shard(
        directory / "train_all", identities=targets.identities,
        logits=targets.logits, source_path="__all_mapped_train__",
        parents=parents, producer_commit=spec["source_commit"],
        teacher_id="MHPE/U050",
    )
    consumers = tuple(sorted(
        candidate.node_id for candidate in registry.values()
        if candidate.teacher_id == "U050"
    ))
    publish_logit_manifest(
        directory / "train_manifest.json", shard_paths=[metadata],
        expected_sources=["__all_mapped_train__"],
        expected_rows=len(targets.identities), parents=parents,
        teacher_id="MHPE/U050", consumers=consumers,
    )


def _predict_components(*, spec, ensemble_id, device, recovery_spec_sha256=None):
    profile = campaign_profile(spec)
    registry = node_registry(profile)
    (_, foundation_root, foundation, split, split_hash, selection_hash, selections,
     assignments, balanced, recipe) = _context(spec, verify_source_tree=recovery_spec_sha256 is None)
    stage = ensemble_id[:-1]
    sampler_seed = derive_seed(int(foundation["replicate_seed"]), f"mhpe/ensemble/{ensemble_id}")
    repair_domain, memory_gib = _runtime_parameters(profile)
    repair_seed = derive_seed(int(foundation["replicate_seed"]), repair_domain)
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced,
        behavior="hlt" if stage == "D000" else "balanced_uniform",
        coordinate=COORDINATES[stage], batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=memory_gib,
    )
    role_state = {
        role: {"identities": None, "logits": {}, "lineage": {}, "labels": None}
        for role in ("train", "validation")
    }
    for component in ENSEMBLE_COMPONENTS[ensemble_id]:
        output, report, report_hash = _teacher_checkpoint(spec, component, foundation_root)
        model, _ = load_pmard_model(output / "training_report.json", model_factory=scouting_model_factory_for_report(report), device=device)
        for role in ("train", "validation"):
            target = precompute_teacher_targets(
                model, caches[role].iterate_batches(
                    epoch=0, sampler_seed=sampler_seed,
                    batch_size=int(recipe["batching"]["effective_batch_size"]),
                ), input_key=input_key, device=device,
                teacher_report_sha256=report_hash,
                split_manifest_sha256=split_hash,
            )
            state = role_state[role]
            if state["identities"] is None:
                state["identities"] = target.identities
            elif state["identities"] != target.identities:
                raise ValueError("HCWDL-MHPE component identity order differs")
            state["logits"][component] = target.logits.astype(np.float32, copy=False)
            state["lineage"][component] = {
                "report_sha256": report_hash,
                "checkpoint_sha256": report["selected_checkpoint_sha256"],
                "logits_sha256": array_sha256("logits", state["logits"][component]),
            }
        del model; gc.collect()
    # Labels are read only for validation diagnostics, never for target construction.
    label_parts = []; identity_parts = []
    for batch in caches["validation"].iterate_batches(
        epoch=0, sampler_seed=sampler_seed,
        batch_size=int(recipe["batching"]["effective_batch_size"]),
    ):
        label_parts.append(np.asarray(batch["labels"], dtype=np.int64))
        identity_parts.extend(map(str, batch["identity_keys"]))
    if tuple(identity_parts) != role_state["validation"]["identities"]:
        raise ValueError("HCWDL-MHPE validation label/identity order differs")
    role_state["validation"]["labels"] = np.concatenate(label_parts)
    role_data = {
        role: (
            state["identities"], state["logits"], state["lineage"],
            state["labels"], split_hash, selection_hash,
        )
        for role, state in role_state.items()
    }
    cache_bytes = {
        role: int(cache.header["array_bytes"]) for role, cache in caches.items()
    }
    return role_data, cache_bytes, registry


def _diversity(component_logits: Mapping[str, np.ndarray], temperature: float, labels: np.ndarray) -> dict[str, Any]:
    names = sorted(component_logits); rows = []
    probabilities = {
        name: uniform_probability_ensemble({name: component_logits[name]}, temperature=temperature)
        for name in names
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            p, q = probabilities[left].astype(np.float64), probabilities[right].astype(np.float64)
            midpoint = .5 * (p + q)
            js = .5 * np.sum(p * np.log(np.maximum(p, 1e-30) / midpoint), axis=1)
            js += .5 * np.sum(q * np.log(np.maximum(q, 1e-30) / midpoint), axis=1)
            disagreement = p.argmax(1) != q.argmax(1)
            left_flat = p.ravel(); right_flat = q.ravel()
            correlation = None
            if left_flat.std() > 0 and right_flat.std() > 0:
                candidate = float(np.corrcoef(left_flat, right_flat)[0, 1])
                correlation = candidate if np.isfinite(candidate) else None
            rows.append({
                "left": left, "right": right,
                "probability_correlation": correlation,
                "mean_jensen_shannon": float(js.mean()),
                "prediction_disagreement": float(np.mean(disagreement)),
                "classwise_prediction_disagreement": [
                    None if not np.any(labels == index) else float(np.mean(disagreement[labels == index]))
                    for index in range(15)
                ],
            })
    return {"pairs": rows}


def run_ensemble(*, spec: Mapping[str, Any], ensemble_id: str, device: str = "cuda", recovery_spec_sha256: str | None = None) -> dict[str, Any]:
    if ensemble_id not in ENSEMBLE_COMPONENTS:
        raise ValueError("unknown HCWDL-MHPE ensemble")
    profile = campaign_profile(spec)
    started = time.monotonic()
    root = Path(spec["campaign_root"]); primary_t = 1.0 if ensemble_id == "D000E" else 2.0
    role_data, cache_bytes, registry = _predict_components(
        spec=spec, ensemble_id=ensemble_id, device=device,
        recovery_spec_sha256=recovery_spec_sha256,
    )
    manifests = {}; manifest_paths = {}; target_locks = {}
    for temperature in (1.0, 2.0):
        label = f"T{int(temperature)}"; directory = root / "targets" / ensemble_id / label
        target_parents = {
            "campaign_spec_sha256": spec["content_hash"],
            "foundation_reuse_lock_sha256": spec["reuse_lock_sha256"],
            "graph_sha256": spec["graph_sha256"],
            "recipe_sha256": spec["recipe_sha256"],
        }
        for role in ("train", "validation"):
            identities, logits, lineage, _, split_hash, selection_hash = role_data[role]
            role_parents = {
                **target_parents, "split_manifest_sha256": split_hash,
                "selection_manifest_sha256": selection_hash,
            }
            _, metadata = publish_probability_shard(
                directory / f"{role}_all", ensemble_id=ensemble_id, role=role,
                identities=identities, component_logits=logits, component_lineage=lineage,
                temperature=temperature, source_path=f"__all_mapped_{role}__",
                parents=role_parents,
                producer_commit=spec["source_commit"],
            )
            consumers = sorted(node.node_id for node in registry.values() if node.teacher_id == ensemble_id and node.temperature == temperature)
            manifest_path = directory / f"{role}_manifest.json"
            manifest = publish_probability_manifest(
                manifest_path, ensemble_id=ensemble_id, role=role, shard_paths=[metadata],
                expected_sources=[f"__all_mapped_{role}__"], expected_rows=len(identities),
                temperature=temperature, consumers=consumers,
                parents=role_parents,
            )
            manifests[f"{label}_{role}"] = manifest["content_hash"]
            manifest_paths[f"{label}_{role}"] = str(manifest_path.resolve())
        lock = target_lock_payload(
            manifests={role: manifests[f"{label}_{role}"] for role in ("train", "validation")},
            ensemble_id=ensemble_id,
            consumers=sorted(node.node_id for node in registry.values() if node.teacher_id == ensemble_id and node.temperature == temperature),
            parents={
                **target_parents,
                "split_manifest_sha256": role_data["train"][4],
                "selection_manifest_sha256": role_data["train"][5],
            },
        )
        write_immutable_json(directory / "lock.json", lock)
        target_locks[label] = lock["content_hash"]
    _, validation_logits, validation_lineage, labels, _, _ = role_data["validation"]
    ensemble_probability = uniform_probability_ensemble(validation_logits, temperature=1)
    ensemble_metrics = classification_metrics(np.log(np.maximum(ensemble_probability, 1e-30)), labels)
    component_metrics = {
        name: classification_metrics(value, labels) for name, value in validation_logits.items()
    }
    leave_one_out = {}
    for omitted in sorted(validation_logits):
        reduced = {name: value for name, value in validation_logits.items() if name != omitted}
        if reduced:
            probability = uniform_probability_ensemble(reduced, temperature=1)
            leave_one_out[omitted] = classification_metrics(np.log(np.maximum(probability, 1e-30)), labels)
    aucs = [row["macro_ovr_auc"] for row in component_metrics.values()]
    scalar_metrics = sorted(
        name for name, value in ensemble_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    mean_metrics = {
        name: float(np.mean([float(row[name]) for row in component_metrics.values()]))
        for name in scalar_metrics
    }
    best_component = max(
        component_metrics,
        key=lambda name: float(component_metrics[name]["macro_ovr_auc"]),
    )
    ensemble_minus_mean = {
        name: float(ensemble_metrics[name]) - mean_metrics[name]
        for name in scalar_metrics
    }
    ensemble_minus_best = {
        name: float(ensemble_metrics[name]) - float(component_metrics[best_component][name])
        for name in scalar_metrics
    }
    local_teacher = {
        "U100E": "U050", "D066E": "U100E",
        "D033E": "D066E", "D000E": "D033E",
    }[ensemble_id]
    local_component = f"{ensemble_id[:-1]}_from_{local_teacher}"
    local_vs_skips = {
        name: {
            metric: float(component_metrics[local_component][metric]) - float(component_metrics[name][metric])
            for metric in scalar_metrics
        }
        for name in ENSEMBLE_COMPONENTS[ensemble_id] if name != local_component
    }
    stage_payload = {
        "contract": STAGE_REPORT_CONTRACT, "schema_version": 1,
        "ensemble_id": ensemble_id, "component_order": list(ENSEMBLE_COMPONENTS[ensemble_id]),
        "component_metrics": component_metrics, "ensemble_metrics": ensemble_metrics,
        "component_lineage": validation_lineage,
        "ensemble_minus_mean_component_auc": ensemble_metrics["macro_ovr_auc"] - float(np.mean(aucs)),
        "ensemble_minus_best_component_auc": ensemble_metrics["macro_ovr_auc"] - max(aucs),
        "ensemble_minus_mean_component": ensemble_minus_mean,
        "ensemble_minus_best_auc_component_id": best_component,
        "ensemble_minus_best_component": ensemble_minus_best,
        "local_predecessor_component_id": local_component,
        "local_predecessor_minus_skip_specialists": local_vs_skips,
        "leave_one_out": leave_one_out, "diversity": _diversity(validation_logits, 1.0, labels),
        "manifest_paths": manifest_paths, "manifest_sha256": manifests,
        "target_lock_sha256": target_locks,
        "primary_target_temperature": primary_t, "final_test_accessed": False,
        "runtime": {
            "target_build_seconds": time.monotonic() - started,
            "cache_array_bytes": cache_bytes,
        },
    }
    report = with_content_hash(stage_payload)
    write_immutable_json(root / "reports" / f"{ensemble_id}_stage.json", report)
    return report


__all__ = ["run_ensemble", "run_specialist"]
