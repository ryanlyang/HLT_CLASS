"""Runtime for the paired direct native-offline-to-HLT KD ablation."""

from __future__ import annotations

from collections.abc import Mapping
import gc
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_native_offline_particle_transformer, build_scouting_particle_transformer,
)

from .dataset import iterate_model_batches
from .hcwdl_direct_offline_kd_graph import (
    BASE_NODE_REGISTRY, CAMPAIGN_LABEL, GRAPH_SHA256, HLT_SEED_ALIAS,
    REPRESENTATION_NODE_REGISTRY, TOFF_SEED_ALIAS,
)
from .hcwdl_direct_offline_kd_targets import (
    TARGET_MANIFEST_CONTRACT, DirectTargetBank, as_ephemeral_logit_targets, build_target_spec,
    publish_prepared_targets, validate_target_manifest,
)
from .hcwdl_ladder import DOMAINS
from .hcwdl_representation_data import training_batch_from_parent
from .hcwdl_representation_target_runtime import prepare_target_generation_in_memory
from .hcwdl_representation_training import (
    paired_rng_streams, train_hcwdl_representation_node,
    validate_representation_training_report,
)
from .engine import validate_pmard_training_report
from .hcwdl_training import train_hcwdl_node
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .selective_assignment import RowSelection
from .splits import role_records
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


BASE_REPORT_CONTRACT = "HCWDL_DIRECT_OFFLINE_KD_BASE_REPORT/v1"
REPRESENTATION_REPORT_CONTRACT = "HCWDL_DIRECT_OFFLINE_KD_REPRESENTATION_REPORT/v1"
RUNTIME_CONTRACT = "HCWDL_DIRECT_OFFLINE_KD_RUNTIME/v1"


def node_output_dir(root: str | Path, node_id: str) -> Path:
    if node_id not in {*BASE_NODE_REGISTRY, *REPRESENTATION_NODE_REGISTRY}:
        raise ValueError("unknown direct offline-KD node")
    return Path(root) / "training" / node_id


def target_output_dir(root: str | Path) -> Path:
    return Path(root) / "targets" / "TOFF_CE"


def _selections(spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, RowSelection]]:
    split = load_json(spec["split_manifest_path"])
    selection = load_json(spec["selection_manifest_path"])
    rows = {
        role: RowSelection(selection, role=role, split_manifest_sha256=spec["split_manifest_sha256"])
        for role in ("train", "validation")
    }
    return split, rows


def _input_stream(
    spec: Mapping[str, Any], *, domain: str, role: str,
    selection: RowSelection, batch_size: int, sampler_seed: int,
    epoch: int = 0, paired_metadata: bool = False,
):
    return iterate_model_batches(
        load_json(spec["split_manifest_path"]), data_root=spec["data_root"],
        role=role, input_mode="paired" if paired_metadata else domain,
        epoch=epoch, batch_size=batch_size, sampler_seed=sampler_seed,
        row_selection=selection,
        include_hcwdl_metadata=paired_metadata,
        canonical_order=paired_metadata,
        shuffle_within_chunk=not paired_metadata,
        shuffle_buffer_rows=batch_size,
        interleave_source_files=1 if paired_metadata else 4,
    )


def _view_caches(
    spec: Mapping[str, Any], *, domain: str, batch_size: int,
    sampler_seed: int, max_gib: float = 72.0,
    require_hcwdl_metadata: bool = False,
) -> tuple[dict[str, EphemeralPmardViewCache], float]:
    if require_hcwdl_metadata and domain != "hlt":
        raise ValueError("registered HCWDL metadata is only defined for the HLT view")
    started = time.perf_counter(); split, selections = _selections(spec)
    caches = {}; remaining = float(max_gib)
    for role in ("train", "validation"):
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            _input_stream(
                spec, domain=domain, role=role, selection=selections[role],
                batch_size=batch_size, sampler_seed=sampler_seed,
                paired_metadata=require_hcwdl_metadata,
            ),
            expected_rows=int(spec["role_counts"][role]), records=records, role=role,
            expected_source_rows=expected_cache_source_rows(
                records, row_selection=selections[role],
            ),
            view_keys=(domain,), max_gib=remaining,
            lineage={
                "campaign_spec_sha256": spec["content_hash"],
                "domain": domain, "exact_observed_view": True,
                "view_built_once": True, "durable_dataset": False,
            },
        )
        caches[role] = cache
        remaining -= int(cache.header["array_bytes"]) / 1024**3
        if remaining <= 0:
            raise MemoryError("direct KD simultaneous role caches exceed cap")
    return caches, time.perf_counter() - started


def _runtime_signature(spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "unavailable"
    return with_content_hash({
        "contract": RUNTIME_CONTRACT, "schema_version": 1,
        "source_commit": spec["source_commit"], "python": platform.python_version(),
        "torch": torch_version, "graph_sha256": GRAPH_SHA256,
        "final_test_accessed": False,
    })


def _base_training_parents(spec: Mapping[str, Any]) -> dict[str, str]:
    """Build the exact parent namespace consumed by the PMARD engine."""
    return {
        "campaign_spec": str(spec["content_hash"]),
        "graph": GRAPH_SHA256,
        # The PMARD engine authenticates RAM teacher targets against this
        # exact parent key. Keep the artifact name explicit; the former
        # ``split_manifest`` alias made a correct target bank fail closed.
        "split_manifest_sha256": str(spec["split_manifest_sha256"]),
        "selection_manifest": str(spec["selection_manifest_sha256"]),
    }


def _base_wrapper(
    spec: Mapping[str, Any], *, node_id: str, engine: Mapping[str, Any],
) -> dict[str, Any]:
    value = with_content_hash({
        "contract": BASE_REPORT_CONTRACT, "schema_version": 1,
        "parents": {
            "campaign_spec": spec["content_hash"],
            "graph": GRAPH_SHA256, "engine_report": engine["content_hash"],
            "base_recipe": spec["base_recipe_sha256"],
        },
        "node_id": node_id, "validation": engine["validation"],
        "selected_checkpoint": engine["selected_checkpoint"],
        "selected_checkpoint_sha256": engine["selected_checkpoint_sha256"],
        "training_passes": 60, "validation_every_passes": 1,
        "complete": True, "finite_poor_results_retained": True,
        "final_test_accessed": False,
    })
    write_immutable_json(node_output_dir(spec["campaign_root"], node_id) / "direct_report.json", value)
    return value


def validate_base_wrapper(
    spec: Mapping[str, Any], *, node_id: str, value: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=BASE_REPORT_CONTRACT, expected_schema_version=1,
    )
    engine = load_json(node_output_dir(spec["campaign_root"], node_id) / "training_report.json")
    engine_hash = validate_pmard_training_report(engine)
    if (
        value.get("node_id") != node_id
        or value.get("parents") != {
            "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
            "engine_report": engine_hash, "base_recipe": spec["base_recipe_sha256"],
        }
        or value.get("selected_checkpoint_sha256") != engine.get("selected_checkpoint_sha256")
        or value.get("validation") != engine.get("validation")
        or value.get("training_passes") != 60
        or value.get("validation_every_passes") != 1
        or value.get("complete") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("direct KD completed base report lineage differs")
    return digest


def validate_representation_wrapper(
    spec: Mapping[str, Any], *, node_id: str, value: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=REPRESENTATION_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    engine = load_json(node_output_dir(spec["campaign_root"], node_id) / "training_report.json")
    engine_hash = validate_representation_training_report(
        engine, expected_execution_id=node_id,
        expected_recipe_sha256=spec["representation_recipe_sha256"],
    )
    parents = value.get("parents", {})
    target_manifest = load_json(target_output_dir(spec["campaign_root"]) / "manifest.json")
    target_hash = validate_content_hash(
        target_manifest,
        expected_contract=TARGET_MANIFEST_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("node_id") != node_id
        or value.get("strategy") != REPRESENTATION_NODE_REGISTRY[node_id].strategy
        or parents.get("campaign_spec") != spec["content_hash"]
        or parents.get("graph") != GRAPH_SHA256
        or parents.get("engine_report") != engine_hash
        or parents.get("target_manifest") != target_hash
        or parents.get("combined_recipe") != spec["combined_recipe_sha256"]
        or value.get("validation") != engine.get("validation")
        or value.get("deployable_extraction") != engine.get("deployable_extraction")
        or value.get("training_passes") != 60
        or value.get("validation_every_passes") != 1
        or value.get("complete") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("direct KD completed representation report lineage differs")
    return digest


def train_base_node(
    spec: Mapping[str, Any], *, node_id: str, device: str = "cuda",
) -> Mapping[str, Any]:
    if node_id not in BASE_NODE_REGISTRY:
        raise ValueError("unknown direct KD base node")
    output = node_output_dir(spec["campaign_root"], node_id)
    completed = output / "direct_report.json"
    if completed.is_file():
        value = load_json(completed)
        validate_base_wrapper(spec, node_id=node_id, value=value)
        return value
    recipe = load_json(spec["base_recipe_path"])
    batch_size = int(recipe["batching"]["effective_batch_size"])
    rng = paired_rng_streams("HLT_RSET", int(spec["replicate_seed"]))
    sampler_seed = int(rng["streams"]["sampler"])
    domain = BASE_NODE_REGISTRY[node_id].student_domain
    caches, cache_seconds = _view_caches(
        spec, domain=domain, batch_size=batch_size, sampler_seed=sampler_seed,
    )

    def batches(role: str, epoch: int = 0):
        return caches[role].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        )

    targets = None
    parents = _base_training_parents(spec)
    if node_id == "HLT_LOGIT":
        bank = DirectTargetBank.load(
            target_output_dir(spec["campaign_root"]) / "manifest.json", strategy="RSET",
        )
        split = load_json(spec["split_manifest_path"])
        teacher = load_json(node_output_dir(spec["campaign_root"], "TOFF_CE") / "training_report.json")
        targets = as_ephemeral_logit_targets(
            bank, source_paths=[record.path for record in role_records(split, "train")],
            teacher_report_sha256=teacher["content_hash"],
            split_manifest_sha256=spec["split_manifest_sha256"],
        )
        parents["teacher_TOFF_CE"] = teacher["content_hash"]
        parents["target_manifest"] = bank.manifest["content_hash"]
    seed_alias = TOFF_SEED_ALIAS if node_id == "TOFF_CE" else HLT_SEED_ALIAS
    train_hcwdl_node(
        node_id=node_id, recipe=recipe,
        train_rows=int(spec["role_counts"]["train"]),
        replicate_seed=int(spec["replicate_seed"]),
        model_factory=(
            build_native_offline_particle_transformer
            if domain == "toff" else build_scouting_particle_transformer
        ),
        train_batches=lambda epoch: batches("train", epoch),
        validation_batches=lambda: batches("validation", 0),
        class_weights=np.ones(15, np.float32), output_dir=output,
        parents=parents, device=device,
        privileged_teacher_targets=targets,
        registry=BASE_NODE_REGISTRY, domains=DOMAINS, graph_sha256=GRAPH_SHA256,
        campaign_label=CAMPAIGN_LABEL, seed_node_id=seed_alias,
        scientific_config_extra={
            "paired_hlt_seed_alias": HLT_SEED_ALIAS,
            "fresh_toff_seed_alias": TOFF_SEED_ALIAS,
            "student_view_built_once": True,
            "cache_construction_seconds": cache_seconds,
            "unweighted_ce": True, "final_test_accessed": False,
        },
    )
    raw = load_json(output / "training_report.json")
    return _base_wrapper(spec, node_id=node_id, engine=raw)


def _source_partitions(spec: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    split, selections = _selections(spec); selection = selections["train"]
    rows = []
    for index, record in enumerate(role_records(split, "train")):
        count = selection.source_rows(record.path)
        if count <= 0:
            raise ValueError("direct KD target source has no selected rows")
        rows.append({
            "partition": f"source_{index:04d}", "source_index": index,
            "source_file_id": index, "rows": count,
        })
    if sum(row["rows"] for row in rows) != int(spec["role_counts"]["train"]):
        raise ValueError("direct KD target coverage differs")
    return tuple(rows)


def _kernel_bundle(spec: Mapping[str, Any]):
    from .hcwdl_homotopy_representation_training import _kernel_bundle
    return _kernel_bundle(spec["kernel_envelope"])


def _target_forward_batch(*args, **kwargs):
    from .hcwdl_representation_production import _target_forward_batch
    return _target_forward_batch(*args, **kwargs)


def build_target_bank(
    spec: Mapping[str, Any], *, device: str = "cuda",
) -> Mapping[str, Any]:
    output = target_output_dir(spec["campaign_root"])
    manifest_path = output / "manifest.json"
    teacher_output = node_output_dir(spec["campaign_root"], "TOFF_CE")
    teacher_report = load_json(teacher_output / "training_report.json")
    target_spec = build_target_spec(
        teacher_report_sha256=teacher_report["content_hash"],
        teacher_checkpoint_sha256=teacher_report["selected_checkpoint_sha256"],
        base_recipe_sha256=spec["base_recipe_sha256"],
        representation_recipe_sha256=spec["representation_recipe_sha256"],
        split_manifest_sha256=spec["split_manifest_sha256"],
        selection_manifest_sha256=spec["selection_manifest_sha256"],
        kernel_resources_sha256=spec["kernel_resources_sha256"],
        architecture_attestation_sha256=spec["architecture_attestation_sha256"],
    )
    if manifest_path.is_file():
        stored = load_json(output / "target_spec.json")
        if stored != target_spec:
            raise ValueError("reused direct target specification differs")
        manifest = load_json(manifest_path)
        validate_target_manifest(manifest, expected_spec_sha256=target_spec["content_hash"])
        return manifest
    model, validated = load_pmard_model(
        teacher_output / "training_report.json",
        model_factory=scouting_model_factory_for_report(teacher_report), device=device,
    )
    if validated["content_hash"] != teacher_report["content_hash"]:
        raise ValueError("fresh TOFF teacher report changed")
    model.to(device).float().eval()
    split, selections = _selections(spec); selection = selections["train"]
    partitions = _source_partitions(spec); factories = {}; partition_specs = {}
    from .hcwdl_representation_production import (
        _configure_target_backend, _teacher_surface_forward,
    )
    for row in partitions:
        partition = row["partition"]; rank = int(row["source_index"])
        source_file_id = int(row["source_file_id"])

        def factory(*, partition=partition, rank=rank, source_file_id=source_file_id):
            stream = iterate_model_batches(
                split, data_root=spec["data_root"], role="train", input_mode="paired",
                rank=rank, world_size=len(partitions), epoch=0, sampler_seed=1337,
                batch_size=256, shuffle_buffer_rows=256, interleave_source_files=1,
                row_selection=selection, include_hcwdl_metadata=True,
                canonical_order=True, shuffle_within_chunk=False,
            )
            for batch in stream:
                yield _target_forward_batch(
                    batch, partition=partition, source_file_id=source_file_id,
                    bank_kind="toff", teacher_view="toff",
                )

        factories[partition] = factory
        partition_specs[partition] = {"rows": int(row["rows"]), "source_file_id": source_file_id}
    bundle = _kernel_bundle(spec)
    forward, input_fields = _teacher_surface_forward(model, device=device, bank_kind="toff")
    _configure_target_backend()
    prepared = prepare_target_generation_in_memory(
        bank_kind="toff", partition_batches=factories,
        partition_specs=partition_specs, teacher_forward=forward,
        token_resources=bundle.token, relation_resources=bundle.relation,
        teacher_model=model, allowed_input_fields=input_fields,
    )
    write_immutable_json(output / "target_spec.json", target_spec)
    manifest = publish_prepared_targets(
        prepared, target_spec=target_spec, output_dir=output,
        producer_commit=spec["source_commit"],
    )
    del model, prepared; gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    return manifest


def _training_take(batch: Mapping[str, Any], indexes: np.ndarray) -> dict[str, Any]:
    from .hcwdl_representation_production import _training_take
    return _training_take(batch, indexes)


def train_representation_node(
    spec: Mapping[str, Any], *, node_id: str, device: str = "cuda",
    preemption_requested=None,
) -> Mapping[str, Any]:
    if node_id not in REPRESENTATION_NODE_REGISTRY:
        raise ValueError("unknown direct representation node")
    output = node_output_dir(spec["campaign_root"], node_id)
    wrapper_path = output / "direct_report.json"
    if wrapper_path.is_file():
        value = load_json(wrapper_path)
        validate_representation_wrapper(spec, node_id=node_id, value=value)
        return value
    base_recipe = load_json(spec["base_recipe_path"])
    representation_recipe = load_json(spec["representation_recipe_path"])
    batch_size = int(base_recipe["batching"]["effective_batch_size"])
    rng = paired_rng_streams(node_id, int(spec["replicate_seed"]))
    sampler_seed = int(rng["streams"]["sampler"])
    caches, cache_seconds = _view_caches(
        spec, domain="hlt", batch_size=batch_size, sampler_seed=sampler_seed,
        require_hcwdl_metadata=True,
    )

    def batches(role: str, epoch: int):
        for raw in caches[role].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        ):
            yield training_batch_from_parent(raw, student_view="hlt")

    target = DirectTargetBank.load(
        target_output_dir(spec["campaign_root"]) / "manifest.json",
        strategy=REPRESENTATION_NODE_REGISTRY[node_id].strategy,
    )
    bundle = _kernel_bundle(spec)
    from .hcwdl_representation_production import _calibration_population
    calibration_rows = min(4096, int(spec["role_counts"]["train"]))
    calibration, selection = _calibration_population(
        caches["train"], sampler_seed=sampler_seed, rows=calibration_rows,
        campaign_sha256=spec["content_hash"],
        parent_logit_counterpart_node_id="HLT_DIRECT_PAIR", student_view="hlt",
    )

    def calibration_batches(_phase: str):
        for start in range(0, calibration_rows, 256):
            yield _training_take(
                calibration, np.arange(start, min(start + 256, calibration_rows)),
            )

    def diagnostic_batches():
        yield _training_take(calibration, np.arange(min(256, calibration_rows)))

    runtime = _runtime_signature(spec)
    execution = canonical_sha256({
        "campaign": spec["content_hash"], "node_id": node_id,
        "replicate_seed": spec["replicate_seed"],
    })
    lineage = {
        "ascent_graph": GRAPH_SHA256, "execution": execution,
        "producer_runtime_signature": runtime["content_hash"],
        "representation_recipe": spec["representation_recipe_sha256"],
        "target_generation": target.manifest["parents"]["target_generation"],
        "target_logical": target.manifest["payload"]["logical_target_sha256"],
    }
    engine = train_hcwdl_representation_node(
        execution_id=node_id, parent_recipe=base_recipe,
        representation_recipe=representation_recipe,
        recipe_compatibility=None,
        campaign_sha256=spec["content_hash"],
        train_rows=int(spec["role_counts"]["train"]),
        replicate_seed=int(spec["replicate_seed"]),
        train_batches=lambda epoch, start: (
            batch for index, batch in enumerate(batches("train", epoch)) if index >= start
        ),
        validation_batches=lambda: batches("validation", 0),
        target_bank=target,
        target_cache_diagnostics={
            "construction_seconds": float(load_json(
                target_output_dir(spec["campaign_root"]) / "generation.json"
            )["construction_seconds"]),
            "load_seconds": 0.0, "hlt_view_cache_construction_seconds": cache_seconds,
            "generation_sha256": lineage["target_generation"],
            "logical_sha256": lineage["target_logical"],
            "manifest_sha256": target.manifest["content_hash"],
            "source": "fresh_toff_one_forward_targets_and_exact_hlt_ram_views",
        },
        token_resources=bundle.token, relation_resources=bundle.relation,
        output_dir=output, resume_lineage=lineage,
        producer_runtime_signature=runtime,
        architecture_attestation_sha256=spec["architecture_attestation_sha256"],
        device=device, mode="scientific", calibration_batches=calibration_batches,
        calibration_selection=selection,
        calibration_expected_batches=math.ceil(calibration_rows / 256),
        calibration_minimum_valid_batches=min(12, math.ceil(calibration_rows / 256)),
        diagnostic_batches=diagnostic_batches,
        registered_output_row={
            "task_key": f"train_{node_id}", "node_id": node_id,
            "registered_execution_id": execution,
        },
        publication_owner={
            "campaign_spec_sha256": spec["content_hash"],
            "combined_recipe_sha256": spec["combined_recipe_sha256"],
        },
        preemption_requested=preemption_requested,
    )
    validate_representation_training_report(
        engine, expected_execution_id=node_id,
        expected_recipe_sha256=spec["representation_recipe_sha256"],
    )
    wrapper = with_content_hash({
        "contract": REPRESENTATION_REPORT_CONTRACT, "schema_version": 1,
        "parents": {
            "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
            "engine_report": engine["content_hash"],
            "target_manifest": target.manifest["content_hash"],
            "combined_recipe": spec["combined_recipe_sha256"],
        },
        "node_id": node_id,
        "strategy": REPRESENTATION_NODE_REGISTRY[node_id].strategy,
        "validation": engine["validation"],
        "deployable_extraction": engine["deployable_extraction"],
        "training_passes": 60, "validation_every_passes": 1,
        "complete": True, "finite_poor_results_retained": True,
        "final_test_accessed": False,
    })
    write_immutable_json(wrapper_path, wrapper)
    return wrapper


__all__ = [
    "BASE_REPORT_CONTRACT", "REPRESENTATION_REPORT_CONTRACT", "RUNTIME_CONTRACT",
    "build_target_bank", "node_output_dir", "target_output_dir", "train_base_node",
    "train_representation_node", "validate_base_wrapper",
    "validate_representation_wrapper",
]
