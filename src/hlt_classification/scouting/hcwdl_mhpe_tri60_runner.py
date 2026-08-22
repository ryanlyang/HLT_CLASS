"""Production fit and reducer runners for the TRI60 campaign."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, sha256_file, write_immutable_json,
)

from .evaluation import classification_metrics
from .hcwdl_mhpe_runner import _diversity
from .hcwdl_mhpe_tri60_campaign import validate_campaign
from .hcwdl_mhpe_tri60_contracts import (
    EPHEMERAL_REP_AUDIT_CONTRACT, STAGE_REPORT_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_ephemeral import EphemeralRepresentationTargetBank
from .hcwdl_mhpe_tri60_graph import (
    ENSEMBLE_COMPONENTS, GRAPH_SHA256, NODE_REGISTRY,
)
from .hcwdl_mhpe_tri60_probability import (
    Tri60ProbabilityTargets, load_probability_role,
    publish_probability_lock, publish_probability_role,
    uniform_probability_ensemble, validate_probability_lock,
)
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_training import (
    TRI60_PREFETCH_DEPTH, Tri60TrainingRuntime, _BatchPrefetcher,
    load_tri60_model, train_tri60_node,
)
from .hcwdl_representation_data import (
    HCWDLParticleInputs, canonical_identity_digests,
)
from .hcwdl_representation_kernels import generate_spectral_resource_bundle
from .hcwdl_representation_target_runtime import (
    TargetForwardBatch, prepare_target_generation_in_memory,
)
from .hcwdl_representation_targets import ORDINARY_BANK
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import (
    _cache_student_views, _load_common, _stream,
)
from .splits import role_records
from .training import derive_seed


def node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def distribution_output_dir(root: str | Path, distribution_id: str) -> Path:
    return Path(root) / "probabilities" / distribution_id


def _foundation(spec: Mapping[str, Any]):
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    return foundation


def _configure_deterministic_backend() -> None:
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("TRI60 deterministic CUBLAS workspace is absent")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _behavior(node_id: str) -> str:
    coordinate = NODE_REGISTRY[node_id].coordinate_name
    if coordinate == "D000":
        return "hlt"
    if coordinate == "U000":
        return "p0"
    return "balanced_uniform"


def _runtime(spec: Mapping[str, Any], node_id: str) -> Tri60TrainingRuntime:
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


def _student_caches(
    spec: Mapping[str, Any], *, node_id: str,
):
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation,
    )
    node = NODE_REGISTRY[node_id]
    sampler_seed = derive_seed(int(spec["replicate_seed"]), node.seed_alias + "/sampler")
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    requested = 360.0 if node.track in {"RSET", "RREL"} else 240.0
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced,
        behavior=_behavior(node_id), coordinate=node.coordinate,
        batch_size=256, sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=requested,
        # The shared TRI60 normalization boundary authenticates canonical
        # identity digests, visible indexes, and particle-family codes for
        # every track.  LOGIT-only models do not consume representation heads,
        # but their cache batches still require this metadata for identity-
        # bound probability joins and the common strict batch contract.
        include_hcwdl_metadata=True,
    )
    return (
        foundation, split, split_hash, selection_hash, selections,
        assignments, balanced, caches, input_key,
    )


def _target_batch(
    batch: Mapping[str, Any], *, partition: str, source_file_id: int,
    view_key: str,
) -> TargetForwardBatch:
    keys = np.asarray(batch.get("identity_keys"))
    labels = np.ascontiguousarray(np.asarray(batch.get("labels"), dtype=np.uint8))
    view = batch.get(view_key)
    if not isinstance(view, HCWDLParticleInputs):
        raise TypeError("TRI60 carrier stream lacks strict HCWDL metadata")
    identities = np.ascontiguousarray(batch.get("identity_digests"))
    expected = canonical_identity_digests(tuple(map(str, keys.tolist())))
    if not np.array_equal(identities, expected):
        raise ValueError("TRI60 carrier identity digests differ")
    try:
        entries = np.asarray(
            [int(str(value).rsplit("::tree::", 1)[1]) for value in keys],
            dtype="<u8",
        )
    except (IndexError, ValueError) as error:
        raise ValueError("TRI60 canonical carrier identities differ") from error
    if len(entries) and np.any(entries[1:] <= entries[:-1]):
        raise ValueError("TRI60 carrier source entries reorder")
    return TargetForwardBatch(
        source_partition=partition,
        source_file_id=np.full(len(keys), source_file_id, dtype="<u4"),
        source_entry=entries, identity_digest=identities, label=labels,
        teacher_inputs={
            "features": np.ascontiguousarray(view.features, dtype=np.float32),
            "vectors": np.ascontiguousarray(view.vectors, dtype=np.float32),
            "mask": np.ascontiguousarray(view.mask, dtype=np.bool_),
            "visible_indices": np.ascontiguousarray(view.visible_indices),
            "family_codes": np.ascontiguousarray(view.family_codes),
        },
    )


def _teacher_forward(model, *, device: str):
    import torch

    order = ("features", "vectors", "mask", "visible_indices", "family_codes")

    def forward(inputs):
        if set(inputs.arrays) != set(order):
            raise PermissionError("TRI60 carrier input fields differ")
        tensors = []
        for name in order:
            value = np.asarray(inputs.arrays[name])
            tensor = torch.as_tensor(value, device=device)
            if value.dtype.kind == "f":
                tensor = tensor.float()
            tensors.append(tensor)
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type, enabled=False,
        ):
            return model.forward_hcwdl_surfaces(*tensors)

    return forward


def _carrier_source_partitions(
    records: Iterable[Any], *, selection: Any,
) -> tuple[dict[str, Any], ...]:
    """Resolve the authenticated nonempty carrier partitions.

    ``RowSelection.source_rows`` uses ``-1`` as the canonical sentinel for
    "all mapped rows in this source". Full-population TRI60 campaigns use
    that sentinel, while an explicit selection may legitimately contribute
    zero rows from a source that remains present in the authenticated split.
    Neither case is corruption: translate the sentinel through the split's
    authenticated mapped-row count and omit only exact zero-row partitions.
    """

    partitions = []
    for source_index, record in enumerate(records):
        selected_rows = int(selection.source_rows(record.path))
        mapped_rows = int(record.mapped_entries)
        if selected_rows == -1:
            selected_rows = mapped_rows
        elif selected_rows < -1:
            raise ValueError("TRI60 carrier source-row sentinel differs")
        if selected_rows > mapped_rows:
            raise ValueError("TRI60 carrier source selection exceeds mapped rows")
        if selected_rows == 0:
            continue
        partitions.append({
            "partition": f"source_{source_index:04d}",
            "source_index": source_index,
            "source_file_id": source_index,
            "source_path": str(record.path),
            "rows": selected_rows,
        })
    expected_rows = int(selection.rows)
    observed_rows = sum(int(row["rows"]) for row in partitions)
    if not partitions or observed_rows != expected_rows:
        raise ValueError(
            "TRI60 carrier partition coverage differs: "
            f"expected {expected_rows}, observed {observed_rows}"
        )
    return tuple(partitions)


def _carrier_targets(
    spec: Mapping[str, Any], *, node_id: str, foundation, split,
    selections, assignments, balanced, device: str,
) -> tuple[EphemeralRepresentationTargetBank, dict[str, Any]]:
    node = NODE_REGISTRY[node_id]
    carrier_id = str(node.representation_carrier_id)
    carrier_report_path = node_output_dir(spec["campaign_root"], carrier_id) / "training_report.json"
    model, carrier_report = load_tri60_model(carrier_report_path, device=device)
    carrier_checkpoint = str(carrier_report["selected_checkpoint_sha256"])
    carrier_node = NODE_REGISTRY[carrier_id]
    sampler_seed = derive_seed(int(spec["replicate_seed"]), carrier_node.seed_alias + "/sampler")
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    factories: dict[str, Callable[[], Iterable[TargetForwardBatch]]] = {}
    partition_specs: dict[str, dict[str, int]] = {}
    records = role_records(split, "train")
    view_key = "hlt" if _behavior(carrier_id) == "hlt" else "privileged"
    partitions = _carrier_source_partitions(
        records, selection=selections["train"],
    )
    for source in partitions:
        source_index = int(source["source_index"])
        partition = str(source["partition"])

        def unprefetched_factory(
            *, source_index=source_index, partition=partition,
            view_key=view_key,
        ):
            stream = _stream(
                foundation_spec=foundation, split=split,
                selections=selections, assignments=assignments, balanced=balanced,
                role="train", behavior=_behavior(carrier_id),
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
            # The target runtime already retains current/following batches to
            # identify the final short batch.  This one additional bounded
            # producer slot overlaps construction of the next ROOT/view batch
            # with teacher inference and spectral sketching without changing
            # any yielded bytes or order.
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
        raise ValueError("TRI60 regenerated spectral resources differ")
    prepared = prepare_target_generation_in_memory(
        bank_kind=ORDINARY_BANK,
        partition_batches=factories,
        partition_specs=partition_specs,
        teacher_forward=_teacher_forward(model, device=device),
        token_resources=bundle.token,
        relation_resources=bundle.relation,
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
        carrier_checkpoint_sha256=carrier_checkpoint,
        campaign_spec_sha256=spec["content_hash"], graph_sha256=GRAPH_SHA256,
        recipe_sha256=spec["parents"]["recipe"],
    )
    return bank, {
        "bundle": bundle, "carrier_report": carrier_report,
        "carrier_report_path": carrier_report_path,
    }


def run_fit(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    fit_started = time.monotonic()
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown TRI60 fit")
    node = NODE_REGISTRY[node_id]
    probability_targets = None
    probability_lock = None
    if node.distribution_teacher_id is not None:
        distribution_id = str(node.distribution_teacher_id)
        directory = distribution_output_dir(spec["campaign_root"], distribution_id)
        probability_lock, _ = validate_probability_lock(
            directory / "lock.json", distribution_id=distribution_id,
        )
        probability_targets = Tri60ProbabilityTargets.load(
            directory / "train_manifest.json", distribution_id=distribution_id,
        )
    rep_bank = None
    rep_context = None
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation,
    )
    output = node_output_dir(spec["campaign_root"], node_id)
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
    # Build the student views only after the carrier and transient teacher
    # chunks are gone.  This ordering is a storage and peak-RAM invariant.
    cache_started = time.monotonic()
    _, _, _, _, _, _, _, caches, input_key = _student_caches(
        spec, node_id=node_id,
    )
    cache_seconds = time.monotonic() - cache_started
    parents = {
        "campaign_spec": spec["content_hash"],
        "foundation": spec["parents"]["foundation"],
        "integration": spec["parents"]["integration"],
        "endpoint_resources": spec["parents"]["endpoint_resources"],
        "graph": spec["parents"]["graph"],
        "recipe": spec["parents"]["recipe"],
        "split_manifest": split_hash,
        "selection_manifest": selection_hash,
    }
    if probability_lock is not None:
        parents["probability_lock"] = probability_lock["content_hash"]
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
        rss = raw_rss * 1024
        audit = rep_bank.audit(peak_rss_bytes=rss, peak_cuda_bytes=cuda_peak)
        write_immutable_json(output / "ephemeral_representation_audit.json", audit)
        audit_hash = validate_artifact(audit, contract=EPHEMERAL_REP_AUDIT_CONTRACT)
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
            token_resources=token_resources,
            relation_resources=relation_resources,
            output_dir=output, parents=parents,
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=execution_source_commit or spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec, node_id), execution_mode="scientific",
            preparation_metrics={
                "carrier_representation_target_seconds": target_seconds,
                "student_view_cache_seconds": cache_seconds,
                "pre_training_total_seconds": time.monotonic() - fit_started,
            },
        )
    finally:
        if rep_bank is not None:
            rep_bank.release()


def _infer_cache(model, cache, *, input_key: str, sampler_seed: int, device: str):
    import torch

    logits = []
    identities = []
    labels = []
    model.eval()
    with torch.inference_mode():
        batches = cache.iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=256,
        )
        with _BatchPrefetcher(
            batches, depth=TRI60_PREFETCH_DEPTH,
        ) as prefetched:
            for batch in prefetched:
                view = batch[input_key]
                value = model(
                    torch.as_tensor(view.features, device=device).float(),
                    torch.as_tensor(view.vectors, device=device).float(),
                    torch.as_tensor(view.mask, device=device).bool(),
                ).float().cpu().numpy()
                logits.append(np.ascontiguousarray(value, dtype=np.float32))
                identities.append(np.ascontiguousarray(
                    batch["identity_digests"], dtype=np.uint8,
                ))
                labels.append(np.ascontiguousarray(batch["labels"], dtype=np.int64))
    return (
        np.concatenate(identities), np.concatenate(logits), np.concatenate(labels),
    )


def run_reducer(
    *, spec: Mapping[str, Any], distribution_id: str, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    components = ("U000",) if distribution_id == "U000" else ENSEMBLE_COMPONENTS.get(distribution_id)
    if not components:
        raise KeyError("unknown TRI60 probability reducer")
    representative = components[0]
    _, _, _, _, _, _, _, caches, input_key = _student_caches(
        spec, node_id=representative,
    )
    seed_alias = NODE_REGISTRY[representative].seed_alias
    sampler_seed = derive_seed(int(spec["replicate_seed"]), seed_alias + "/sampler")
    role_state = {
        role: {"identities": None, "labels": None, "logits": {}, "lineage": {}}
        for role in ("train", "validation")
    }
    started = time.monotonic()
    for component in components:
        report_path = node_output_dir(spec["campaign_root"], component) / "training_report.json"
        model, report = load_tri60_model(report_path, device=device)
        for role in ("train", "validation"):
            identities, logits, labels = _infer_cache(
                model, caches[role], input_key=input_key,
                sampler_seed=sampler_seed, device=device,
            )
            state = role_state[role]
            if state["identities"] is None:
                state["identities"], state["labels"] = identities, labels
            elif not np.array_equal(state["identities"], identities):
                raise ValueError("TRI60 reducer component identity order differs")
            elif not np.array_equal(state["labels"], labels):
                raise ValueError("TRI60 reducer component labels differ")
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
    root = distribution_output_dir(spec["campaign_root"], distribution_id)
    parents = {
        "campaign_spec": spec["content_hash"],
        "foundation": spec["parents"]["foundation"],
        "graph": spec["parents"]["graph"],
        "recipe": spec["parents"]["recipe"],
    }
    if recovery_spec_sha256 is not None:
        parents["recovery_spec"] = recovery_spec_sha256
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
    _, _, validation_probability = load_probability_role(
        root / "validation_manifest.json", expected_distribution_id=distribution_id,
        expected_role="validation",
    )
    labels = role_state["validation"]["labels"]
    metrics = classification_metrics(
        np.log(np.maximum(validation_probability, 1e-30)), labels,
    )
    component_metrics = {
        name: classification_metrics(logits, labels)
        for name, logits in role_state["validation"]["logits"].items()
    }
    leave_one_out = {}
    for omitted in sorted(components):
        reduced = {
            name: value for name, value in role_state["validation"]["logits"].items()
            if name != omitted
        }
        if reduced:
            probabilities = uniform_probability_ensemble(reduced, temperature=1.0)
            leave_one_out[omitted] = classification_metrics(
                np.log(np.maximum(probabilities, 1e-30)), labels,
            )
    auc_values = [float(row["macro_ovr_auc"]) for row in component_metrics.values()]
    best = max(component_metrics, key=lambda name: float(component_metrics[name]["macro_ovr_auc"]))
    report = artifact({
        "parents": {**parents, "probability_lock": lock["content_hash"]},
        "distribution_id": distribution_id,
        "component_order": list(components),
        "component_metrics": component_metrics,
        "ensemble_metrics": metrics,
        "ensemble_minus_mean_component_auc": (
            float(metrics["macro_ovr_auc"]) - float(np.mean(auc_values))
        ),
        "ensemble_minus_best_component_auc": (
            float(metrics["macro_ovr_auc"]) - max(auc_values)
        ),
        "best_component_id": best,
        "leave_one_out": leave_one_out,
        "diversity": _diversity(
            role_state["validation"]["logits"], 1.0, labels,
        ),
        "train_probability_bytes": int(
            role_state["train"]["identities"].nbytes
            + manifests["train"]["rows"] * 15 * 4
        ),
        "validation_probability_bytes": int(
            role_state["validation"]["identities"].nbytes
            + manifests["validation"]["rows"] * 15 * 4
        ),
        "runtime_seconds": time.monotonic() - started,
        "throughput_optimizations": {
            "cpu_batch_prefetch_depth": TRI60_PREFETCH_DEPTH,
            "prefetch_changes_batch_order": False,
        },
        "poor_metrics_do_not_control_graph": True,
        "final_test_accessed": False,
    }, contract=STAGE_REPORT_CONTRACT)
    write_immutable_json(Path(spec["campaign_root"]) / "reports/stages" / f"{distribution_id}.json", report)
    return report


__all__ = [
    "distribution_output_dir", "node_output_dir", "run_fit", "run_reducer",
]
