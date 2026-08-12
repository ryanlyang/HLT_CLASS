"""Execution adapters for representation KD on the authenticated U/D path."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import gc
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.models.hcwdl_representation import (
    load_hcwdl_deployable_checkpoint,
)

from .dataset import iterate_model_batches
from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_homotopy_graph import DOMAINS
from .hcwdl_homotopy_representation_contracts import (
    CALIBRATION_CONTRACT, DEPLOYABLE_EXTRACTION_CONTRACT,
    RESUME_STATE_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    SCHEMA_VERSION, TRAINING_REPORT_CONTRACT, build_artifact, validate_artifact,
)
from .hcwdl_homotopy_representation_graph import (
    GRAPH_SHA256, NODE_REGISTRY,
)
from .hcwdl_homotopy_representation_recipe import validate_recipe
from .hcwdl_homotopy_representation_targets import (
    HomotopyRepresentationTargetBank, build_target_spec,
    publish_prepared_targets, validate_target_manifest, validate_target_spec,
)
from .hcwdl_homotopy_stream import iterate_homotopy_batches
from .hcwdl_representation_data import training_batch_from_parent
from .hcwdl_representation_target_runtime import prepare_target_generation_in_memory
from .hcwdl_representation_training import (
    paired_rng_streams, train_hcwdl_representation_node,
    validate_representation_training_report,
)
from .highcov_cache import DenseAssignmentStore
from .selective_assignment import RowSelection
from .splits import role_records
from .training import derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows
from .hcwdl_upper_cache import ResidualCouplingStore


def coordinate_for_domain(domain: str) -> HomotopyCoordinate:
    row = DOMAINS.get(domain)
    if not isinstance(row, Mapping) or row.get("s") is None or row.get("f") is None:
        raise ValueError(f"HCWDL-U-RKD domain has no coordinate: {domain!r}")
    return HomotopyCoordinate(
        round(float(row["s"]) * 20), 20,
        round(float(row["f"]) * 20), 20,
    )


def node_output_dir(root: str | Path, node_id: str) -> Path:
    node = NODE_REGISTRY[node_id]
    return Path(root) / "training" / node.strategy / node_id


def target_output_dir(root: str | Path, bank_id: str) -> Path:
    return Path(root) / "targets" / bank_id


def _kernel_bundle(reference: Mapping[str, Any]):
    from .hcwdl_representation_production import _load_kernel_bundle

    return _load_kernel_bundle(reference)


def _target_forward_batch(*args, **kwargs):
    from .hcwdl_representation_production import _target_forward_batch as implementation

    return implementation(*args, **kwargs)


def _teacher_surface_forward(*args, **kwargs):
    from .hcwdl_representation_production import _teacher_surface_forward as implementation

    return implementation(*args, **kwargs)


def _configure_target_backend() -> None:
    from .hcwdl_representation_production import _configure_target_backend

    _configure_target_backend()


def _load_parent(spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    base = load_json(spec["base_recipe_path"])
    representation = load_json(spec["representation_recipe_path"])
    combined = load_json(Path(spec["campaign_root"]) / "combined_recipe.json")
    validate_recipe(combined, base_recipe=base, representation_recipe=representation)
    return base, representation


def _stores(spec: Mapping[str, Any], role: str):
    parent_root = Path(spec["parent_homotopy_root"])
    return (
        DenseAssignmentStore(spec["assignment_manifests"][role]),
        ResidualCouplingStore(parent_root / f"coupling/{role}_manifest.json"),
        RowSelection(
            load_json(spec["selection_manifest_path"]), role=role,
            split_manifest_sha256=spec["split_manifest_sha256"],
        ),
    )


def _homotopy_stream(
    spec: Mapping[str, Any], *, domain: str, role: str,
    batch_size: int, source_index: int | None = None,
):
    assignment, coupling, selection = _stores(spec, role)
    return iterate_homotopy_batches(
        load_json(spec["split_manifest_path"]), data_root=spec["data_root"],
        role=role, assignment_store=assignment, coupling_store=coupling,
        row_selection=selection, coordinate=coordinate_for_domain(domain),
        repair_seed=derive_seed(
            int(spec["replicate_seed"]), "hcwdl_uj/repair/shared_v1",
        ),
        batch_size=batch_size, source_index=source_index,
        output_key="privileged",
    )


def _source_partitions(spec: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    split = load_json(spec["split_manifest_path"])
    _, _, selection = _stores(spec, "train")
    rows = []
    for index, record in enumerate(role_records(split, "train")):
        selected = selection.source_rows(record.path)
        if selected <= 0:
            raise ValueError("HCWDL-U-RKD target source has no selected rows")
        rows.append({
            "partition": f"source_{index:04d}", "source_index": index,
            "source_path": record.path, "source_file_id": index, "rows": selected,
        })
    if sum(row["rows"] for row in rows) != int(spec["role_counts"]["train"]):
        raise ValueError("HCWDL-U-RKD target source coverage differs")
    return tuple(rows)


def _target_factories(
    spec: Mapping[str, Any], *, teacher_domain: str,
) -> tuple[dict[str, Callable[[], Iterable[Any]]], dict[str, dict[str, int]], str]:
    partitions = _source_partitions(spec)
    split = load_json(spec["split_manifest_path"])
    _, _, selection = _stores(spec, "train")
    bank_kind = "toff" if teacher_domain == "toff" else "ordinary"
    factories = {}
    specifications = {}
    for row in partitions:
        partition = row["partition"]
        source_index = int(row["source_index"])
        source_file_id = int(row["source_file_id"])

        def factory(
            *, partition=partition, source_index=source_index,
            source_file_id=source_file_id,
        ):
            if teacher_domain == "toff":
                stream = iterate_model_batches(
                    split, data_root=spec["data_root"], role="train",
                    input_mode="paired", rank=source_index,
                    world_size=len(partitions), epoch=0, sampler_seed=1337,
                    batch_size=256, shuffle_buffer_rows=256,
                    interleave_source_files=1, row_selection=selection,
                    include_hcwdl_metadata=True, canonical_order=True,
                    shuffle_within_chunk=False,
                )
                view = "toff"
            else:
                stream = _homotopy_stream(
                    spec, domain=teacher_domain, role="train", batch_size=256,
                    source_index=source_index,
                )
                view = teacher_domain
            for batch in stream:
                yield _target_forward_batch(
                    batch, partition=partition, source_file_id=source_file_id,
                    bank_kind=bank_kind, teacher_view=view,
                )

        factories[partition] = factory
        specifications[partition] = {
            "rows": int(row["rows"]), "source_file_id": source_file_id,
        }
    return factories, specifications, bank_kind


def _load_teacher(spec: Mapping[str, Any], bank_id: str, *, device: str):
    if bank_id == "TOFF":
        from .loaders import load_pmard_model, scouting_model_factory_for_report

        record = spec["imported_controls"]["TOFF"]
        report = load_json(record["report_path"])
        model, validated = load_pmard_model(
            record["report_path"],
            model_factory=scouting_model_factory_for_report(report), device=device,
        )
        if validated["content_hash"] != record["report_sha256"]:
            raise ValueError("HCWDL-U-RKD imported TOFF report differs")
        checkpoint_sha256 = record["checkpoint_sha256"]
        report_sha256 = record["report_sha256"]
        domain = "toff"
    else:
        output = node_output_dir(spec["campaign_root"], bank_id)
        report = load_json(output / "training_report.json")
        validate_representation_training_report(
            report, expected_execution_id=bank_id,
            expected_recipe_sha256=spec["representation_recipe_sha256"],
        )
        extraction = report["deployable_extraction"]
        model = load_hcwdl_deployable_checkpoint(
            extraction["checkpoint_path"],
            expected_sha256=extraction["checkpoint_sha256"],
        )
        checkpoint_sha256 = extraction["checkpoint_sha256"]
        report_sha256 = report["content_hash"]
        domain = NODE_REGISTRY[bank_id].student_domain
    model.to(device).float().eval()
    return model, report_sha256, checkpoint_sha256, domain


def _teacher_evidence(spec: Mapping[str, Any], bank_id: str) -> tuple[str, str, str]:
    if bank_id == "TOFF":
        record = spec["imported_controls"]["TOFF"]
        return record["report_sha256"], record["checkpoint_sha256"], "toff"
    output = node_output_dir(spec["campaign_root"], bank_id)
    report = load_json(output / "training_report.json")
    validate_representation_training_report(
        report, expected_execution_id=bank_id,
        expected_recipe_sha256=spec["representation_recipe_sha256"],
    )
    extraction = report["deployable_extraction"]
    return report["content_hash"], extraction["checkpoint_sha256"], NODE_REGISTRY[bank_id].student_domain


def _expected_target_spec(spec: Mapping[str, Any], bank_id: str) -> Mapping[str, Any]:
    report_hash, checkpoint_hash, domain = _teacher_evidence(spec, bank_id)
    return build_target_spec(
        bank_id=bank_id, teacher_report_sha256=report_hash,
        teacher_checkpoint_sha256=checkpoint_hash, teacher_domain=domain,
        combined_recipe_sha256=spec["combined_recipe_sha256"],
        parent_campaign_sha256=spec["parent_homotopy_spec_sha256"],
        split_manifest_sha256=spec["split_manifest_sha256"],
        selection_manifest_sha256=spec["selection_manifest_sha256"],
        coupling_lock_sha256=spec["coupling_lock_sha256"],
        coordinate_sha256=spec["coordinate_sha256"],
        endpoint_lock_sha256=spec["endpoint_lock_sha256"],
        representation_recipe_sha256=spec["representation_recipe_sha256"],
        kernel_resources_sha256=spec["kernel_resources_sha256"],
        architecture_attestation_sha256=spec["architecture_attestation_sha256"],
    )


def build_target_bank(
    spec: Mapping[str, Any], *, bank_id: str, device: str = "cuda",
    producer_commit: str | None = None,
) -> Mapping[str, Any]:
    root = target_output_dir(spec["campaign_root"], bank_id)
    if (root / "manifest.json").is_file():
        expected = _expected_target_spec(spec, bank_id)
        stored = load_json(root / "target_spec.json")
        validate_target_spec(stored)
        if stored != expected:
            raise ValueError("HCWDL-U-RKD reused target specification differs")
        manifest = load_json(root / "manifest.json")
        validate_target_manifest(
            manifest, expected_spec_sha256=expected["content_hash"],
        )
        return manifest
    model, report_hash, checkpoint_hash, domain = _load_teacher(
        spec, bank_id, device=device,
    )
    factories, partition_specs, bank_kind = _target_factories(
        spec, teacher_domain=domain,
    )
    bundle = _kernel_bundle(spec["kernel_envelope"])
    forward, input_fields = _teacher_surface_forward(
        model, device=device, bank_kind=bank_kind,
    )
    _configure_target_backend()
    prepared = prepare_target_generation_in_memory(
        bank_kind=bank_kind, partition_batches=factories,
        partition_specs=partition_specs, teacher_forward=forward,
        token_resources=bundle.token, relation_resources=bundle.relation,
        teacher_model=model, allowed_input_fields=input_fields,
    )
    target_spec = _expected_target_spec(spec, bank_id)
    write_immutable_json(root / "target_spec.json", target_spec)
    manifest = publish_prepared_targets(
        prepared, target_spec=target_spec, output_dir=root,
        producer_commit=str(producer_commit or spec["source_commit"]),
    )
    del model, prepared
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return manifest


def _build_view_caches(
    spec: Mapping[str, Any], *, domain: str, batch_size: int,
    max_gib: float,
) -> tuple[dict[str, EphemeralPmardViewCache], float]:
    started = time.perf_counter()
    split = load_json(spec["split_manifest_path"])
    caches = {}
    remaining = float(max_gib)
    for role in ("train", "validation"):
        _, _, selection = _stores(spec, role)
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            _homotopy_stream(
                spec, domain=domain, role=role, batch_size=batch_size,
            ),
            expected_rows=int(spec["role_counts"][role]), records=records,
            role=role,
            expected_source_rows=expected_cache_source_rows(
                records, row_selection=selection,
            ),
            view_keys=("privileged",), max_gib=remaining,
            lineage={
                "campaign_spec_sha256": spec["content_hash"],
                "parent_homotopy_spec_sha256": spec["parent_homotopy_spec_sha256"],
                "coupling_lock_sha256": spec["coupling_lock_sha256"],
                "endpoint_lock_sha256": spec["endpoint_lock_sha256"],
                "coordinate_sha256": spec["coordinate_sha256"],
                "domain": domain, "training_metadata_only": True,
                "durable_repaired_dataset": False,
            },
        )
        caches[role] = cache
        remaining -= int(cache.header["array_bytes"]) / 1024**3
        if remaining <= 0:
            raise MemoryError("HCWDL-U-RKD simultaneous role caches exceed cap")
    return caches, time.perf_counter() - started


def _take_training(batch: Mapping[str, Any], indexes: np.ndarray) -> dict[str, Any]:
    from .hcwdl_representation_production import _training_take

    return _training_take(batch, indexes)


def _runtime_signature(
    spec: Mapping[str, Any], *, producer_commit: str | None = None,
    recovery_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "unavailable"
    return with_content_hash({
        "contract": "HCWDL_HOMOTOPY_REPRESENTATION_RUNTIME_SIGNATURE/v2",
        "schema_version": SCHEMA_VERSION,
        # Scientific resume identity remains the immutable campaign source.
        # Corrected execution source is bound separately by the recovery and
        # task-attestation contracts so an existing rolling checkpoint stays
        # exactly resumable.
        "source_commit": spec["source_commit"],
        "python": platform.python_version(), "torch": torch_version,
        "graph_sha256": GRAPH_SHA256, "final_test_accessed": False,
    })


def train_node(
    spec: Mapping[str, Any], *, node_id: str, device: str = "cuda",
    view_cache_max_gib: float = 72.0,
    preemption_requested=None,
    producer_commit: str | None = None,
    recovery_sha256: str | None = None,
) -> Mapping[str, Any]:
    node = NODE_REGISTRY[node_id]
    output = node_output_dir(spec["campaign_root"], node_id)
    report_path = output / "training_report.json"
    if report_path.is_file():
        report = load_json(report_path)
        validate_representation_training_report(
            report, expected_execution_id=node_id,
            expected_recipe_sha256=spec["representation_recipe_sha256"],
        )
        wrapper_path = output / "combined_training_report.json"
        if not wrapper_path.is_file():
            target_manifest = load_json(
                target_output_dir(
                    spec["campaign_root"], node.target_bank_identity,
                ) / "manifest.json"
            )
            publish_training_wrappers(
                spec, node_id=node_id, engine_report=report,
                target_manifest=target_manifest,
            )
        return report
    base_recipe, representation_recipe = _load_parent(spec)
    batch_size = int(base_recipe["batching"]["effective_batch_size"])
    caches, cache_seconds = _build_view_caches(
        spec, domain=node.student_domain, batch_size=batch_size,
        max_gib=view_cache_max_gib,
    )
    target = HomotopyRepresentationTargetBank.load(
        target_output_dir(spec["campaign_root"], node.target_bank_identity) / "manifest.json",
        strategy=node.strategy,
    )
    target_manifest = target.manifest
    expected_target_spec = _expected_target_spec(spec, node.target_bank_identity)
    if target_manifest["parents"]["target_spec"] != expected_target_spec["content_hash"]:
        raise ValueError("HCWDL-U-RKD training target lineage differs")
    target_generation = load_json(
        target_output_dir(spec["campaign_root"], node.target_bank_identity) / "generation.json"
    )
    bundle = _kernel_bundle(spec["kernel_envelope"])
    rng = paired_rng_streams(node_id, int(spec["replicate_seed"]))
    sampler_seed = int(rng["streams"]["sampler"])

    def batches(role: str, epoch: int):
        for raw in caches[role].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        ):
            yield training_batch_from_parent(raw, student_view="privileged")

    calibration_rows = min(4096, int(spec["role_counts"]["train"]))
    from .hcwdl_representation_production import _calibration_population

    calibration, calibration_selection = _calibration_population(
        caches["train"], sampler_seed=sampler_seed, rows=calibration_rows,
        campaign_sha256=spec["content_hash"],
        parent_logit_counterpart_node_id=node.parent_counterpart,
        student_view="privileged",
    )

    def calibration_batches(_phase: str):
        for start in range(0, calibration_rows, 256):
            yield _take_training(
                calibration, np.arange(start, min(start + 256, calibration_rows)),
            )

    def diagnostic_batches():
        yield _take_training(calibration, np.arange(min(256, calibration_rows)))

    runtime = _runtime_signature(
        spec, producer_commit=producer_commit, recovery_sha256=recovery_sha256,
    )
    registered_execution = canonical_sha256({
        "campaign": spec["content_hash"], "node_id": node_id,
        "strategy": node.strategy, "seed": spec["replicate_seed"],
    })
    lineage = {
        "ascent_graph": GRAPH_SHA256,
        "execution": registered_execution,
        "producer_runtime_signature": runtime["content_hash"],
        "representation_recipe": spec["representation_recipe_sha256"],
        "target_generation": target_manifest["parents"]["target_generation"],
        "target_logical": target_manifest["payload"]["logical_target_sha256"],
    }
    engine_report = train_hcwdl_representation_node(
        execution_id=node_id, parent_recipe=base_recipe,
        representation_recipe=representation_recipe,
        campaign_sha256=spec["content_hash"],
        train_rows=int(spec["role_counts"]["train"]),
        replicate_seed=int(spec["replicate_seed"]),
        train_batches=lambda epoch, start: (
            batch for index, batch in enumerate(batches("train", epoch))
            if index >= start
        ),
        validation_batches=lambda: batches("validation", 0),
        target_bank=target,
        target_cache_diagnostics={
            "construction_seconds": float(target_generation["construction_seconds"]),
            "load_seconds": 0.0,
            "hlt_view_cache_construction_seconds": cache_seconds,
            "generation_sha256": lineage["target_generation"],
            "logical_sha256": lineage["target_logical"],
            "manifest_sha256": target_manifest["content_hash"],
            "source": "authenticated_compact_target_and_process_local_homotopy_views",
        },
        token_resources=bundle.token, relation_resources=bundle.relation,
        output_dir=output, resume_lineage=lineage,
        producer_runtime_signature=runtime,
        architecture_attestation_sha256=spec["architecture_attestation_sha256"],
        device=device, mode="smoke" if spec["mode"] == "smoke" else "scientific",
        synthetic_passes=1, calibration_batches=calibration_batches,
        calibration_selection=calibration_selection,
        calibration_expected_batches=math.ceil(calibration_rows / 256),
        calibration_minimum_valid_batches=min(12, math.ceil(calibration_rows / 256)),
        diagnostic_batches=diagnostic_batches,
        registered_output_row={
            "task_key": f"train_{node_id}", "node_id": node_id,
            "registered_execution_id": registered_execution,
        },
        publication_owner={
            "campaign_spec_sha256": spec["content_hash"],
            "combined_recipe_sha256": spec["combined_recipe_sha256"],
        },
        preemption_requested=preemption_requested,
    )
    publish_training_wrappers(
        spec, node_id=node_id, engine_report=engine_report,
        target_manifest=target_manifest,
    )
    return engine_report


def publish_training_wrappers(
    spec: Mapping[str, Any], *, node_id: str,
    engine_report: Mapping[str, Any], target_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_representation_training_report(
        engine_report, expected_execution_id=node_id,
        expected_recipe_sha256=spec["representation_recipe_sha256"],
    )
    validate_target_manifest(target_manifest)
    output = node_output_dir(spec["campaign_root"], node_id)
    parents = {
        "campaign_spec": spec["content_hash"],
        "graph": GRAPH_SHA256, "combined_recipe": spec["combined_recipe_sha256"],
        "engine_report": engine_report["content_hash"],
        "target_manifest": target_manifest["content_hash"],
    }
    calibration = build_artifact(
        CALIBRATION_CONTRACT, parents=parents,
        node_id=node_id, payload=engine_report["calibration"],
    )
    selection = build_artifact(
        SELECTED_CHECKPOINT_CONTRACT, parents=parents,
        node_id=node_id,
        selected_checkpoint_id=engine_report["selected_checkpoint_id"],
        selected_training_checkpoint_sha256=engine_report[
            "selected_training_checkpoint_sha256"
        ],
        selector=["macro_ovr_auc", "cross_entropy", "logR50", "earliest_update"],
    )
    extraction_row = engine_report["deployable_extraction"]
    extraction = build_artifact(
        DEPLOYABLE_EXTRACTION_CONTRACT,
        parents={**parents, "selection": selection["content_hash"],
                 "inner_extraction": extraction_row["report_sha256"]},
        node_id=node_id, student_domain=NODE_REGISTRY[node_id].student_domain,
        hlt_only=NODE_REGISTRY[node_id].deployable,
        checkpoint_path=extraction_row["checkpoint_path"],
        checkpoint_sha256=extraction_row["checkpoint_sha256"],
        training_only_heads_excluded=True,
    )
    resume = build_artifact(
        RESUME_STATE_CONTRACT, parents=parents, node_id=node_id,
        resume_audit=engine_report["resume_audit"], exact_resume=True,
    )
    wrapper = build_artifact(
        TRAINING_REPORT_CONTRACT,
        parents={
            **parents, "calibration": calibration["content_hash"],
            "selection": selection["content_hash"],
            "extraction": extraction["content_hash"],
            "resume": resume["content_hash"],
        },
        node_id=node_id, strategy=NODE_REGISTRY[node_id].strategy,
        student_domain=NODE_REGISTRY[node_id].student_domain,
        teacher_node_id=NODE_REGISTRY[node_id].teacher.node_id,
        engine_report_path=str((output / "training_report.json").resolve()),
        validation=engine_report["validation"],
        completed_optimizer_updates=engine_report["completed_optimizer_updates"],
        completed_passes=engine_report["completed_natural_population_passes"],
        finite_poor_results_retained=True,
    )
    for name, value in (
        ("combined_calibration.json", calibration),
        ("combined_checkpoint_selection.json", selection),
        ("combined_deployable_extraction.json", extraction),
        ("combined_resume_state.json", resume),
        ("combined_training_report.json", wrapper),
    ):
        write_immutable_json(output / name, value)
    return wrapper


__all__ = [
    "build_target_bank", "coordinate_for_domain", "node_output_dir",
    "publish_training_wrappers", "target_output_dir", "train_node",
]
