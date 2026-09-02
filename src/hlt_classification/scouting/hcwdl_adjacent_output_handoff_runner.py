"""Production execution for adjacent-view output-fusion handoff."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, write_immutable_json,
)

from .evaluation import classification_metrics, softmax
from .hcwdl_adjacent_output_handoff_campaign import validate_campaign
from .hcwdl_adjacent_output_handoff_contracts import (
    ENSEMBLE_REPORT_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
)
from .hcwdl_adjacent_output_handoff_execution import validate_execution_acceptance
from .hcwdl_adjacent_output_handoff_fusion import (
    PROBABILITY_FLOOR, evaluate_mixture_curve, mix_probabilities,
)
from .hcwdl_adjacent_output_handoff_graph import (
    EARLY_STOPPING, ENSEMBLE_IDS, GRAPH_SHA256, LR_SCHEDULE, NODE_REGISTRY,
    SELECTION_IDS, distribution_consumers, ensemble_components,
    node_distribution, selection_components,
)
from .hcwdl_adjacent_output_handoff_partition import (
    PARTITION_NAMES, load_partition, publish_partition,
)
from .hcwdl_adjacent_output_handoff_probability import (
    HandoffProbabilityTargets, ROLES, load_probability_role,
    publish_probability_lock, publish_probability_role, validate_probability_lock,
)
from .hcwdl_adjacent_output_handoff_source import validate_source_lock
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend, _infer_cache
from .hcwdl_mhpe_tri60_runner import _student_caches as tri60_student_caches
from .hcwdl_mhpe_tri60_ce_control import load_control_model
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, load_tri60_model,
    train_tri60_node,
)
from .hcwdl_tri100_spine4_bottleneck_graph import NODE_REGISTRY as SOURCE_NODES
from .hcwdl_tri100_spine4_bottleneck_runner import (
    _student_caches as fullcard_student_caches,
    training_authority as source_training_authority,
)
from .training import derive_seed


SOURCE_DISTRIBUTION = "SOURCE_U100"


class _ValidationSubset:
    """RAM-only exact validation subset accepted by TRI60 training."""

    def __init__(self, cache, identity_hexes: Sequence[str], *, content_hash: str):
        self.cache = cache; self.identity_hexes = tuple(identity_hexes)
        fraction = len(self.identity_hexes) / int(cache.header["rows"])
        self.header = {
            **dict(cache.header), "rows": len(self.identity_hexes),
            "array_bytes": int(cache.header["array_bytes"] * fraction),
            "content_hash": content_hash,
        }

    def iterate_batches(self, *, batch_size: int, **_: Any):
        return self.cache.iterate_identity_digest_batches(
            self.identity_hexes, batch_size=batch_size,
        )


def probability_dir(spec: Mapping[str, Any], distribution_id: str) -> Path:
    return Path(spec["campaign_root"]) / "probabilities" / distribution_id


def training_dir(spec: Mapping[str, Any], node_id: str) -> Path:
    return Path(spec["campaign_root"]) / "training" / node_id


def training_authority(node_id: str) -> Tri60TrainingAuthority:
    node = NODE_REGISTRY[node_id]
    authority = Tri60TrainingAuthority(
        node=node, graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
        allowed_training_passes=(100,),
    )
    authority.validate(); return authority


def _runtime() -> Tri60TrainingRuntime:
    return Tri60TrainingRuntime(
        passes=100, batch_size=256, peak_learning_rate=3e-4,
        weight_decay=.01, warmup_fraction=.05,
        minimum_lr_fraction=.05, amp_dtype="bfloat16",
    )


def _source(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(value); return value


def _source_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return load_json(_source(spec)["source_campaign_spec_path"])


def _caches(spec: Mapping[str, Any], node):
    # The established full-cardinality runner is the sole view constructor.
    # It reads only the new spec's authenticated foundation and support policy.
    return fullcard_student_caches(spec, node=node)


def _partition_lookup(spec: Mapping[str, Any]):
    report, arrays = load_partition(spec["artifact_paths"]["validation_partition"])
    return report, {bytes(row): int(code) for row, code in zip(
        arrays["identity_digest"], arrays["partition"], strict=True,
    )}


def _split_validation(
    identities: np.ndarray, probabilities: np.ndarray, labels: np.ndarray,
    lookup: Mapping[bytes, int],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        codes = np.asarray([lookup[bytes(row)] for row in identities], dtype=np.uint8)
    except KeyError as error:
        raise KeyError("validation partition does not cover inference identities") from error
    result = {}
    for code, role in enumerate(PARTITION_NAMES):
        selected = codes == code
        result[role] = (
            np.ascontiguousarray(identities[selected]),
            np.ascontiguousarray(probabilities[selected]),
            np.ascontiguousarray(labels[selected]),
        )
    return result


def run_partition(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec); source = _source(spec); source_spec = _source_spec(spec)
    node = SOURCE_NODES[source["u100_node_id"]]
    _, split_hash, selection_hash, caches, _ = fullcard_student_caches(source_spec, node=node)
    try:
        cache = caches["validation"]
        if cache.identity_digests is None:
            raise ValueError("validation cache lacks identity digests")
        labels = []
        for batch in cache.iterate_canonical_batches(batch_size=4096):
            labels.append(np.asarray(batch["labels"], dtype=np.int16))
        return publish_partition(
            spec["artifact_paths"]["validation_partition"],
            identity_digests=cache.identity_digests,
            labels=np.concatenate(labels),
            parents={
                "campaign_spec": spec["content_hash"], "source_lock": source["content_hash"],
                "split_manifest": split_hash, "selection_manifest": selection_hash,
            }, source_commit=spec["source_commit"],
        )
    finally:
        caches.clear()


def _load_new_model(spec: Mapping[str, Any], node_id: str, *, device: str):
    return load_tri60_model(
        training_dir(spec, node_id) / "training_report.json",
        device=device, authority=training_authority(node_id),
    )


def _load_source_model(spec: Mapping[str, Any], *, device: str):
    source = _source(spec); source_spec = _source_spec(spec)
    return load_tri60_model(
        source["u100_report_path"], device=device,
        authority=source_training_authority(source["u100_node_id"]),
    ), source_spec, SOURCE_NODES[source["u100_node_id"]]


def _targets(spec: Mapping[str, Any], distribution_id: str, *, consumer: str):
    root = probability_dir(spec, distribution_id)
    lock, _ = validate_probability_lock(root / "lock.json", distribution_id=distribution_id)
    if consumer not in lock["consumers"]:
        raise PermissionError("output-handoff probability consumer is unauthorized")
    return HandoffProbabilityTargets.load(
        root / "train_manifest.json", distribution_id=distribution_id,
    ), lock


def run_fit(
    spec: Mapping[str, Any], node_id: str, *, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec); _configure_deterministic_backend()
    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown output-handoff fit")
    node = NODE_REGISTRY[node_id]
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    acceptance_hash = validate_execution_acceptance(
        acceptance, spec=spec,
    )
    target_lock = None; targets = None
    if node.kd_weight:
        targets, target_lock = _targets(
            spec, node.teacher_distribution_id, consumer=node_id,
        )
    started = time.monotonic()
    _, split_hash, selection_hash, caches, input_key = _caches(spec, node)
    partition, arrays = load_partition(spec["artifact_paths"]["validation_partition"])
    checkpoint_hexes = tuple(
        bytes(row).hex() for row in arrays["identity_digest"][arrays["partition"] == 0]
    )
    validation = _ValidationSubset(
        caches["validation"], checkpoint_hexes, content_hash=partition["content_hash"],
    )
    parents = {
        "campaign_spec": spec["content_hash"], "source_lock": spec["parents"]["source_lock"],
        "foundation": spec["parents"]["foundation"], "graph": GRAPH_SHA256,
        "recipe": spec["parents"]["recipe"], "split_manifest": split_hash,
        "selection_manifest": selection_hash, "validation_partition": partition["content_hash"],
        "execution_acceptance": acceptance_hash,
    }
    if target_lock is not None:
        parents["probability_lock"] = target_lock["content_hash"]
    if recovery_spec_sha256 is not None:
        parents["recovery_spec"] = recovery_spec_sha256
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"], validation_cache=validation,
            input_key=input_key, probability_targets=targets,
            output_dir=training_dir(spec, node_id), parents=parents,
            campaign_spec_sha256=spec["content_hash"], recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=execution_source_commit or spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(), execution_mode="scientific",
            preparation_metrics={
                "student_view_cache_seconds": time.monotonic() - started,
                "pre_training_total_seconds": time.monotonic() - started,
            }, authority=training_authority(node_id),
            learning_rate_schedule=dict(LR_SCHEDULE), early_stopping=dict(EARLY_STOPPING),
        )
    finally:
        caches.clear()


def _publish_roles(
    spec: Mapping[str, Any], *, distribution_id: str,
    role_values: Mapping[str, tuple[np.ndarray, np.ndarray]],
    component_order: Sequence[str], component_lineage: Mapping[str, Mapping[str, str]],
    parents: Mapping[str, str], target_temperature: float,
    execution_source_commit: str | None,
) -> dict[str, Any]:
    consumers = distribution_consumers(distribution_id)
    manifests = {}
    for role in ROLES:
        identities, probabilities = role_values[role]
        manifests[role] = publish_probability_role(
            probability_dir(spec, distribution_id), distribution_id=distribution_id,
            role=role, identity_digests=identities, probabilities=probabilities,
            component_order=component_order, component_lineage=component_lineage,
            consumers=consumers, parents=parents,
            producer_commit=execution_source_commit or spec["source_commit"],
            target_temperature=(target_temperature if role == "train" else 1.0),
        )
    return publish_probability_lock(
        probability_dir(spec, distribution_id) / "lock.json",
        distribution_id=distribution_id, manifests=manifests,
        consumers=consumers, parents=parents,
    )


def run_model_reducer(
    spec: Mapping[str, Any], node_id: str, *, device: str = "cuda",
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec); _configure_deterministic_backend()
    partition, lookup = _partition_lookup(spec)
    if node_id == "SOURCE_U100":
        (model, report), cache_spec, node = _load_source_model(spec, device=device)
        distribution_id = SOURCE_DISTRIBUTION
        source = _source(spec)
        report_hash = report["content_hash"]; checkpoint_hash = source["u100_checkpoint_sha256"]
    else:
        model, report = _load_new_model(spec, node_id, device=device)
        node = NODE_REGISTRY[node_id]; cache_spec = spec
        distribution_id = node_distribution(node_id)
        report_hash = report["content_hash"]; checkpoint_hash = report["selected_checkpoint_sha256"]
    _, split_hash, selection_hash, caches, input_key = fullcard_student_caches(cache_spec, node=node)
    sampler_seed = derive_seed(int(spec["replicate_seed"]), node.seed_alias + "/sampler")
    started = time.monotonic()
    try:
        train_ids, train_logits, _ = _infer_cache(
            model, caches["train"], input_key=input_key, sampler_seed=sampler_seed, device=device,
        )
        val_ids, val_logits, val_labels = _infer_cache(
            model, caches["validation"], input_key=input_key, sampler_seed=sampler_seed, device=device,
        )
        train_model = softmax(train_logits).astype(np.float32)
        # Persist canonical T=1 output probabilities once.  Consumers derive
        # their registered KD temperature in RAM during the identity join.
        target_temperature = 2.0
        val_model = softmax(val_logits).astype(np.float32)
        split = _split_validation(val_ids, val_model, val_labels, lookup)
        role_values = {
            "train": (train_ids, train_model),
            **{role: (rows[0], rows[1]) for role, rows in split.items()},
        }
        parents = {
            "campaign_spec": spec["content_hash"], "source_lock": spec["parents"]["source_lock"],
            "graph": GRAPH_SHA256, "recipe": spec["parents"]["recipe"],
            "validation_partition": partition["content_hash"], "split_manifest": split_hash,
            "selection_manifest": selection_hash, "component_report": report_hash,
            "component_checkpoint": checkpoint_hash,
        }
        lineage = {node_id: {
            "report": report_hash, "checkpoint": checkpoint_hash,
            "train_logits": array_sha256("logits", train_logits),
            "validation_logits": array_sha256("logits", val_logits),
        }}
        lock = _publish_roles(
            spec, distribution_id=distribution_id, role_values=role_values,
            component_order=(node_id,), component_lineage=lineage, parents=parents,
            target_temperature=target_temperature,
            execution_source_commit=execution_source_commit,
        )
        metrics = classification_metrics(
            np.log(np.maximum(split["V_report"][1], PROBABILITY_FLOOR)), split["V_report"][2],
        )
        output = artifact({
            "parents": {**parents, "probability_lock": lock["content_hash"]},
            "distribution_id": distribution_id, "component_order": [node_id],
            "report_role": "V_report", "validation_metrics": metrics,
            "runtime_seconds": time.monotonic() - started,
            "durable_particle_views": False, "durable_hidden_states": False,
            "poor_metrics_do_not_control_graph": True, "final_test_accessed": False,
        }, contract=STAGE_REPORT_CONTRACT)
        path = Path(spec["campaign_root"]) / "reports/stages" / f"{distribution_id}.json"
        write_immutable_json(path, output); return output
    finally:
        caches.clear(); del model; gc.collect()


def run_control_reducer(
    spec: Mapping[str, Any], control_id: str, *, device: str = "cuda",
) -> dict[str, Any]:
    """Reevaluate an imported reporting control on the untouched V_report."""

    validate_campaign(spec); _configure_deterministic_backend()
    if control_id not in {"M0CE60", "U000"}:
        raise KeyError("unknown output-handoff reporting control")
    controls = load_json(spec["artifact_paths"]["control_lock"])
    partition, lookup = _partition_lookup(spec)
    if control_id == "M0CE60":
        report_path = Path(controls["m0ce60_report_path"])
        model, report = load_control_model(report_path, device=device)
        _, split_hash, selection_hash, caches, input_key = _caches(
            spec, NODE_REGISTRY["CE_D000_S1"],
        )
    else:
        report_path = Path(controls["pure_offline_u000_report_path"])
        model, report = load_tri60_model(report_path, device=device)
        control_spec = load_json(controls["pure_offline_u000_campaign_spec_path"])
        (
            _, _, split_hash, selection_hash, _, _, _, caches, input_key,
        ) = tri60_student_caches(control_spec, node_id="U000")
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), f"output-handoff/control/{control_id}/report",
    )
    try:
        identities, logits, labels = _infer_cache(
            model, caches["validation"], input_key=input_key,
            sampler_seed=sampler_seed, device=device,
        )
        probabilities = softmax(logits).astype(np.float32)
        split = _split_validation(identities, probabilities, labels, lookup)
        metrics = classification_metrics(
            np.log(np.maximum(split["V_report"][1], PROBABILITY_FLOOR)),
            split["V_report"][2],
        )
        parents = {
            "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
            "controls": spec["parents"]["controls"],
            "validation_partition": partition["content_hash"],
            "split_manifest": split_hash, "selection_manifest": selection_hash,
            "control_report": report["content_hash"],
            "control_checkpoint": report["selected_checkpoint_sha256"],
        }
        output = artifact({
            "parents": parents, "distribution_id": f"CONTROL_{control_id}",
            "component_order": [control_id], "report_role": "V_report",
            "validation_metrics": metrics, "reporting_only": True,
            "training_teacher_use": False, "final_test_accessed": False,
        }, contract=STAGE_REPORT_CONTRACT)
        write_immutable_json(
            Path(spec["campaign_root"]) / "reports/stages" / f"CONTROL_{control_id}.json",
            output,
        )
        return output
    finally:
        caches.clear(); del model; gc.collect()


def _load_role(spec: Mapping[str, Any], distribution_id: str, role: str):
    return load_probability_role(
        probability_dir(spec, distribution_id) / f"{role}_manifest.json",
        distribution_id=distribution_id, role=role,
    )


def run_selection(
    spec: Mapping[str, Any], selection_id: str, *, execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec)
    if selection_id not in SELECTION_IDS:
        raise KeyError("unknown output-handoff selection")
    rich_id, poor_id = selection_components(selection_id)
    rich_lock, _ = validate_probability_lock(
        probability_dir(spec, rich_id) / "lock.json", distribution_id=rich_id,
    )
    poor_lock, _ = validate_probability_lock(
        probability_dir(spec, poor_id) / "lock.json", distribution_id=poor_id,
    )
    partition, arrays = load_partition(spec["artifact_paths"]["validation_partition"])
    _, rich_ids, rich_blend = _load_role(spec, rich_id, "V_blend")
    _, poor_ids, poor_blend = _load_role(spec, poor_id, "V_blend")
    if not np.array_equal(rich_ids, poor_ids):
        raise ValueError("output-handoff selection identities differ")
    labels_by_id = {bytes(row): int(label) for row, label in zip(
        arrays["identity_digest"], arrays["label"], strict=True,
    )}
    labels = np.asarray([labels_by_id[bytes(row)] for row in rich_ids], dtype=np.int64)
    parents = {
        "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
        "recipe": spec["parents"]["recipe"], "validation_partition": partition["content_hash"],
        "rich_probability_lock": rich_lock["content_hash"],
        "poor_probability_lock": poor_lock["content_hash"],
    }
    curve, selected, _ = evaluate_mixture_curve(
        rich_probabilities=rich_blend, poor_probabilities=poor_blend, labels=labels,
        rich_id=rich_id, poor_id=poor_id, transition_id=selection_id,
        parents=parents, bootstrap_seed=derive_seed(int(spec["replicate_seed"]), selection_id + "/bootstrap"),
        bootstrap_samples=int(spec["bootstrap_samples"]),
    )
    report_root = Path(spec["campaign_root"]) / "reports/mixtures" / selection_id
    write_immutable_json(
        report_root / "temperature_rich.json",
        curve["temperature_calibrations"]["rich"],
    )
    write_immutable_json(
        report_root / "temperature_poor.json",
        curve["temperature_calibrations"]["poor"],
    )
    write_immutable_json(report_root / "bootstrap.json", curve["bootstrap_report"])
    write_immutable_json(report_root / "curve.json", curve)
    write_immutable_json(report_root / "selected.json", selected)
    family = selected["selected_family"]; alpha = selected["selected_alpha_numerator"] / 40
    role_values = {}
    for role in ROLES:
        _, rich_role_ids, rich_values = _load_role(spec, rich_id, role)
        _, poor_role_ids, poor_values = _load_role(spec, poor_id, role)
        if not np.array_equal(rich_role_ids, poor_role_ids):
            raise ValueError("output-handoff mixture role identities differ")
        mixed = mix_probabilities(
            rich_values, poor_values, alpha=alpha, family=family,
            rich_temperature=selected["selected_rich_temperature"],
            poor_temperature=selected["selected_poor_temperature"],
        )
        role_values[role] = (rich_role_ids, mixed)
    lineage = {
        rich_id: {"probability_lock": rich_lock["content_hash"]},
        poor_id: {"probability_lock": poor_lock["content_hash"]},
    }
    lock = _publish_roles(
        spec, distribution_id=selection_id, role_values=role_values,
        component_order=(rich_id, poor_id), component_lineage=lineage,
        parents={**parents, "mixture_curve": curve["content_hash"], "selection": selected["content_hash"]},
        target_temperature=2.0, execution_source_commit=execution_source_commit,
    )
    report_ids, report_probabilities = role_values["V_report"]
    report_labels = np.asarray(
        [labels_by_id[bytes(row)] for row in report_ids], dtype=np.int64,
    )
    report_metrics = classification_metrics(
        np.log(np.maximum(report_probabilities, PROBABILITY_FLOOR)), report_labels,
    )
    output = artifact({
        "parents": {
            **parents, "mixture_curve": curve["content_hash"],
            "selection": selected["content_hash"],
            "probability_lock": lock["content_hash"],
        },
        "distribution_id": selection_id,
        "selected_family": selected["selected_family"],
        "selected_alpha_numerator": selected["selected_alpha_numerator"],
        "selected_alpha_denominator": selected["selected_alpha_denominator"],
        "selection_role": "V_blend", "report_role": "V_report",
        "validation_metrics": report_metrics,
        "poor_metrics_do_not_control_graph": True,
        "final_test_accessed": False,
    }, contract=STAGE_REPORT_CONTRACT)
    write_immutable_json(
        Path(spec["campaign_root"]) / "reports/stages" / f"{selection_id}.json",
        output,
    )
    return output


def run_ensemble(
    spec: Mapping[str, Any], ensemble_id: str, *, execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec)
    if ensemble_id not in ENSEMBLE_IDS:
        raise KeyError("unknown output-handoff ensemble")
    components = ensemble_components(ensemble_id)
    locks = {}
    role_values = {}
    for role in ROLES:
        values = []; identities = None
        for component in components:
            lock, _ = validate_probability_lock(
                probability_dir(spec, component) / "lock.json", distribution_id=component,
            ); locks[component] = lock
            _, current_ids, probability = _load_role(spec, component, role)
            if identities is None: identities = current_ids
            elif not np.array_equal(identities, current_ids):
                raise ValueError("output-handoff ensemble identities differ")
            values.append(probability.astype(np.float64))
        role_values[role] = (identities, np.ascontiguousarray(np.mean(values, axis=0), dtype=np.float32))
    parents = {
        "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
        "recipe": spec["parents"]["recipe"],
        **{f"component/{name}": locks[name]["content_hash"] for name in components},
    }
    lock = _publish_roles(
        spec, distribution_id=ensemble_id, role_values=role_values,
        component_order=components,
        component_lineage={name: {"probability_lock": locks[name]["content_hash"]} for name in components},
        parents=parents, target_temperature=1.0,
        execution_source_commit=execution_source_commit,
    )
    _, _, report_probs = _load_role(spec, ensemble_id, "V_report")
    _, partition_arrays = load_partition(spec["artifact_paths"]["validation_partition"])
    report_ids = role_values["V_report"][0]
    labels = {bytes(i): int(y) for i, y in zip(
        partition_arrays["identity_digest"], partition_arrays["label"], strict=True,
    )}
    metrics = classification_metrics(
        np.log(np.maximum(report_probs, PROBABILITY_FLOOR)),
        np.asarray([labels[bytes(row)] for row in report_ids]),
    )
    output = artifact({
        "parents": {**parents, "probability_lock": lock["content_hash"]},
        "ensemble_id": ensemble_id, "component_order": list(components),
        "weights": [1 / len(components)] * len(components),
        "prefix_uniform_no_subset_search": True, "validation_metrics": metrics,
        "report_role": "V_report", "final_test_accessed": False,
    }, contract=ENSEMBLE_REPORT_CONTRACT)
    write_immutable_json(
        Path(spec["campaign_root"]) / "reports/ensembles" / f"{ensemble_id}.json", output,
    )
    return output


__all__ = [
    "SOURCE_DISTRIBUTION", "probability_dir", "run_control_reducer",
    "run_ensemble", "run_fit",
    "run_model_reducer", "run_partition", "run_selection", "training_authority",
    "training_dir",
]
