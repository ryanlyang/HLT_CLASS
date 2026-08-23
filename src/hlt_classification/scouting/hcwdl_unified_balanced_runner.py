"""Execute shared and arm-local HCWDL-UB fits with one-time RAM views/targets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .dataset import iterate_model_batches
from .engine import precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_recovery import task_attestation_path, validate_task_attestation
from .hcwdl_homotopy_stream import (
    iterate_homotopy_batches, iterate_unified_balanced_batches,
)
from .hcwdl_training import train_hcwdl_node
from .hcwdl_unified_balanced_cache import BalancedCouplingStore
from .hcwdl_unified_balanced_contracts import (
    TRAINING_REPORT_CONTRACT, validate_arm_spec, validate_foundation_lock,
    validate_endpoint_lock, validate_foundation_spec,
)
from .hcwdl_unified_balanced_graph import (
    META_GRAPH_SHA256, SHARED_ARM, arm_registry, shared_registry,
    shared_training_registry, training_registry_for_arm,
)
from .hcwdl_unified_balanced_targets import (
    DurableUnifiedBalancedTargets, validate_target_lock,
    validate_target_manifest,
)
from .hcwdl_unified_balanced_targets import (
    publish_target_manifest, publish_target_shard,
)
from .hcwdl_upper_cache import ResidualCouplingStore
from .highcov_cache import DenseAssignmentStore
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .selective_assignment import RowSelection
from .splits import role_records
from .targets import EphemeralTeacherTargets
from .training import GenerationalLossConfiguration, derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


RUNTIME_CONTRACT = "HCWDL_UNIFIED_BALANCED_NODE_RUNTIME/v1"
TARGET_DIGEST_SHADOW_EVIDENCE_CONTRACT = (
    "HCWDL_UNIFIED_BALANCED_TARGET_DIGEST_SHADOW_EVIDENCE/v1"
)
TARGET_DIGEST_SHADOW_REPAIR = "target_manifest_digest_shadow_execution_repair_v1"
DOMAINS = {"hlt": {"input": "hlt"}, "privileged": {"input": "privileged"}}


def arm_node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def shared_node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def _view_workers() -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    requested = int(os.environ.get("HCWDL_UB_VIEW_BUILD_WORKERS", str(allocated)))
    if allocated <= 0 or requested <= 0 or requested > allocated:
        raise ValueError("HCWDL-UB view workers exceed the allocated CPUs")
    return requested


def _view_worker_plan(
    source_count: int, *, environ: Mapping[str, str] | None = None,
) -> tuple[int, int, int]:
    """Partition the allocated CPU envelope across sources and transforms."""

    env = os.environ if environ is None else environ
    allocated = int(env.get("SLURM_CPUS_PER_TASK", "1"))
    total = int(env.get("HCWDL_UB_VIEW_BUILD_WORKERS", str(allocated)))
    if allocated <= 0 or total <= 0 or total > allocated or source_count < 0:
        raise ValueError("HCWDL-UB view workers exceed the allocated CPUs")
    if source_count == 0:
        return total, 0, 0
    default_sources = max(1, total // 4)
    source_workers = int(env.get(
        "HCWDL_UB_VIEW_SOURCE_WORKERS", str(min(source_count, default_sources)),
    ))
    if source_workers <= 0 or source_workers > source_count or source_workers > total:
        raise ValueError("HCWDL-UB source workers exceed the worker/source budget")
    transform_workers = max(1, (total - source_workers) // source_workers)
    return total, source_workers, transform_workers


def _memory_limit_bytes(configured_gib: float) -> int:
    configured = int(configured_gib * 1024**3)
    slurm = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm:
        configured = min(configured, int(slurm) * 1024**2 * 3 // 4)
    return configured


def _load_common(spec: Mapping[str, Any]):
    foundation_root = Path(spec["campaign_root"])
    paths = spec["artifact_paths"]
    split = load_json(paths["split_manifest"])
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection_raw = load_json(paths["selection_manifest"])
    selection_hash = validate_content_hash(
        selection_raw, expected_contract=str(selection_raw["contract"]),
        expected_schema_version=int(selection_raw["schema_version"]),
    )
    selections = {
        role: RowSelection(selection_raw, role=role, split_manifest_sha256=split_hash)
        for role in ("train", "validation")
    }
    assignments = {
        role: DenseAssignmentStore(paths[f"{role}_assignment_manifest"])
        for role in ("train", "validation")
    }
    balanced = {
        role: BalancedCouplingStore(
            foundation_root / f"balanced/{role}_manifest.json"
        ) for role in ("train", "validation")
    }
    return split, split_hash, selection_hash, selections, assignments, balanced


def _stream(
    *, foundation_spec: Mapping[str, Any], split: Mapping[str, Any],
    selections, assignments, balanced, role: str, behavior: str,
    coordinate, batch_size: int, sampler_seed: int, repair_seed: int,
    legacy: bool = False, epoch: int = 0,
    include_hcwdl_metadata: bool = False,
    source_index: int | None = None,
    view_workers: int | None = None,
):
    workers = _view_workers() if view_workers is None else int(view_workers)
    if workers <= 0:
        raise ValueError("HCWDL-UB stream workers must be positive")
    if behavior == "hlt":
        records = role_records(split, role)
        if source_index is not None:
            if source_index < 0 or source_index >= len(records):
                raise IndexError("HCWDL-UB HLT stream source index is out of range")
        return iterate_model_batches(
            split, data_root=foundation_spec["data_root"], role=role,
            input_mode="hlt", epoch=epoch, batch_size=batch_size,
            sampler_seed=sampler_seed, row_selection=selections[role],
            include_hcwdl_metadata=include_hcwdl_metadata,
            rank=0 if source_index is None else source_index,
            world_size=1 if source_index is None else len(records),
            canonical_order=source_index is not None,
            shuffle_within_chunk=source_index is None,
            interleave_source_files=(
                4 if source_index is None else 1
            ),
            shuffle_buffer_rows=(
                8192 if source_index is None else batch_size
            ),
        )
    if behavior == "p0":
        # The exact P0 corner is independent of switch coordinates, but the
        # balanced stream proves it against the same selected identities.
        return iterate_unified_balanced_batches(
            split, data_root=foundation_spec["data_root"], role=role,
            assignment_store=assignments[role], coupling_store=balanced[role],
            row_selection=selections[role], coordinate=coordinate,
            repair_seed=repair_seed, batch_size=batch_size,
            workers=workers, output_key="privileged",
            include_training_metadata=include_hcwdl_metadata,
            source_index=source_index,
        )
    if legacy:
        legacy_store = ResidualCouplingStore(
            foundation_spec["artifact_paths"][f"legacy_{role}_manifest"]
        )
        return iterate_homotopy_batches(
            split, data_root=foundation_spec["data_root"], role=role,
            assignment_store=assignments[role], coupling_store=legacy_store,
            row_selection=selections[role], coordinate=coordinate,
            repair_seed=repair_seed, batch_size=batch_size,
            workers=workers, output_key="privileged",
        )
    return iterate_unified_balanced_batches(
        split, data_root=foundation_spec["data_root"], role=role,
        assignment_store=assignments[role], coupling_store=balanced[role],
        row_selection=selections[role], coordinate=coordinate,
        repair_seed=repair_seed, batch_size=batch_size,
        workers=workers, output_key="privileged",
        include_training_metadata=include_hcwdl_metadata,
        source_index=source_index,
    )


def _parallel_source_streams(
    *, foundation_spec, split, selections, assignments, balanced,
    role: str, behavior: str, coordinate, batch_size: int,
    sampler_seed: int, repair_seed: int, include_hcwdl_metadata: bool,
    records, expected_source_rows: Mapping[str, int],
    source_workers: int, transform_workers: int,
) -> Iterable[tuple[str, Mapping[str, object]]]:
    """Build independent source streams concurrently with bounded buffering."""

    selected = [
        (index, record) for index, record in enumerate(records)
        if int(expected_source_rows[record.path]) > 0
    ]
    if source_workers <= 1:
        for source_index, record in selected:
            yield from (
                (record.path, batch) for batch in _stream(
                    foundation_spec=foundation_spec, split=split,
                    selections=selections, assignments=assignments,
                    balanced=balanced, role=role, behavior=behavior,
                    coordinate=coordinate, batch_size=batch_size,
                    sampler_seed=sampler_seed, repair_seed=repair_seed,
                    epoch=0,
                    legacy=behavior in {"legacycdf_uniform", "balanced_legacywarp"},
                    include_hcwdl_metadata=include_hcwdl_metadata,
                    source_index=source_index, view_workers=transform_workers,
                )
            )
        return

    messages: Queue[tuple[str, str, object]] = Queue(maxsize=2 * source_workers)
    stop = threading.Event()

    def publish(message: tuple[str, str, object]) -> bool:
        while not stop.is_set():
            try:
                messages.put(message, timeout=0.2)
                return True
            except Full:
                continue
        return False

    def produce(source_index: int, source_path: str) -> None:
        stream = None
        try:
            stream = _stream(
                foundation_spec=foundation_spec, split=split,
                selections=selections, assignments=assignments,
                balanced=balanced, role=role, behavior=behavior,
                coordinate=coordinate, batch_size=batch_size,
                sampler_seed=sampler_seed, repair_seed=repair_seed,
                epoch=0,
                legacy=behavior in {"legacycdf_uniform", "balanced_legacywarp"},
                include_hcwdl_metadata=include_hcwdl_metadata,
                source_index=source_index, view_workers=transform_workers,
            )
            for batch in stream:
                if not publish(("batch", source_path, batch)):
                    return
        except BaseException as error:
            publish(("error", source_path, error))
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            publish(("done", source_path, None))

    executor = ThreadPoolExecutor(
        max_workers=source_workers, thread_name_prefix="hcwdl-ub-source",
    )
    futures = [
        executor.submit(produce, source_index, record.path)
        for source_index, record in selected
    ]
    completed = 0
    try:
        while completed < len(selected):
            try:
                kind, source_path, payload = messages.get(timeout=0.5)
            except Empty:
                if all(future.done() for future in futures):
                    for future in futures:
                        future.result()
                    raise RuntimeError("HCWDL-UB source workers ended without completion")
                continue
            if kind == "batch":
                yield source_path, payload
            elif kind == "error":
                raise RuntimeError(
                    f"HCWDL-UB source preprocessing failed for {source_path!r}"
                ) from payload
            elif kind == "done":
                completed += 1
            else:
                raise RuntimeError("HCWDL-UB source worker emitted an invalid message")
        for future in futures:
            future.result()
    finally:
        stop.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _cache_student_views(
    *, foundation_spec, split, selections, assignments, balanced,
    behavior: str, coordinate, batch_size: int, sampler_seed: int,
    repair_seed: int, memory_gib: float,
    include_hcwdl_metadata: bool = False,
):
    caches = {}; remaining = _memory_limit_bytes(memory_gib)
    input_key = "hlt" if behavior == "hlt" else "privileged"
    for role in ("train", "validation"):
        started = time.monotonic()
        records = role_records(split, role)
        source_rows = expected_cache_source_rows(
            records, row_selection=selections[role],
        )
        nonempty_sources = sum(count > 0 for count in source_rows.values())
        total_workers, source_workers, transform_workers = _view_worker_plan(
            nonempty_sources,
        )
        if behavior in {"legacycdf_uniform", "balanced_legacywarp"}:
            source_workers = 1
            transform_workers = total_workers
        partitioned = source_workers > 1
        stream = (
            _parallel_source_streams(
                foundation_spec=foundation_spec, split=split,
                selections=selections, assignments=assignments,
                balanced=balanced, role=role, behavior=behavior,
                coordinate=coordinate, batch_size=batch_size,
                sampler_seed=sampler_seed, repair_seed=repair_seed,
                include_hcwdl_metadata=include_hcwdl_metadata,
                records=records, expected_source_rows=source_rows,
                source_workers=source_workers,
                transform_workers=transform_workers,
            )
            if partitioned else _stream(
                foundation_spec=foundation_spec, split=split,
                selections=selections, assignments=assignments,
                balanced=balanced, role=role, behavior=behavior,
                coordinate=coordinate, batch_size=batch_size,
                sampler_seed=sampler_seed, repair_seed=repair_seed, epoch=0,
                legacy=behavior in {"legacycdf_uniform", "balanced_legacywarp"},
                include_hcwdl_metadata=include_hcwdl_metadata,
                view_workers=transform_workers,
            )
        )
        cache = EphemeralPmardViewCache.build(
            stream, expected_rows=selections[role].rows, records=records,
            role=role,
            expected_source_rows=source_rows,
            view_keys=(input_key,), max_gib=remaining / 1024**3,
            partitioned_sources=partitioned,
            lineage={
                "foundation_spec_sha256": foundation_spec["content_hash"],
                "behavior": behavior, "coordinate": coordinate.payload(),
                "student_view_built_once": True,
                "durable_repaired_dataset": False,
                "source_partitioned_preprocessing": partitioned,
                "view_worker_budget": total_workers,
                "source_workers": source_workers,
                "transform_workers_per_source": transform_workers,
            },
        )
        caches[role] = cache; remaining -= int(cache.header["array_bytes"])
        if remaining <= 0:
            raise MemoryError("HCWDL-UB train/validation caches exceed the memory cap")
        print(
            f"HCWDL-UB phase=student_view_cache role={role} behavior={behavior} "
            f"rows={selections[role].rows} workers={total_workers} "
            f"source_workers={source_workers} transform_workers={transform_workers} "
            f"seconds={time.monotonic()-started:.3f}",
            flush=True,
        )
    return caches, input_key


def _teacher_location(
    canonical_id: str, *, foundation_root: Path, arm_root: Path,
) -> tuple[Path, object]:
    owner, node_id = canonical_id.split("/", 1)
    if owner == SHARED_ARM:
        node = shared_registry()[node_id]
        return shared_node_output_dir(foundation_root, node_id), node
    node = arm_registry(owner)[node_id]
    return arm_node_output_dir(arm_root, node_id), node


def _target_attestation_context(
    *, teacher_id: str, arm_root: Path, arm_spec_sha256: str,
    recovery_context: Mapping[str, Any] | None,
) -> tuple[Path, str]:
    task_id = f"train_{teacher_id}"
    if recovery_context is not None and task_id in recovery_context["task_ids"]:
        return Path(recovery_context["root"]), str(recovery_context["spec_sha256"])
    return arm_root, arm_spec_sha256


def inspect_shared_u000_target_lineage(
    *, foundation_spec: Mapping[str, Any], foundation_root: str | Path,
) -> dict[str, Any]:
    """Authenticate U000 targets and classify the one known legacy lock defect.

    This function never authorizes the defect.  It only returns a complete,
    content-hashed observation that a separately authenticated recovery spec
    may bind.  Any mismatch other than the exact historical digest shadow
    fails closed.
    """

    root = Path(foundation_root)
    report_path = root / "training/U000/training_report.json"
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    checkpoint = report_path.parent / str(report["selected_checkpoint"])
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != report["selected_checkpoint_sha256"]:
        raise ValueError("HCWDL-UB U000 selected checkpoint differs")

    manifest = load_json(root / "targets/u000_train/manifest.json")
    manifest_hash = validate_target_manifest(
        manifest, teacher_id="shared/U000",
    )
    expected_manifest_parents = {
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "split_manifest_sha256": foundation_spec["parents"]["split_manifest_sha256"],
        "teacher_report_sha256": report_hash,
        "teacher_checkpoint_sha256": checkpoint_hash,
    }
    if any(
        manifest.get("parents", {}).get(name) != digest
        for name, digest in expected_manifest_parents.items()
    ):
        raise ValueError("HCWDL-UB shared target manifest lineage differs")

    foundation_lock = load_json(root / "locks/foundation.json")
    foundation_lock_hash = validate_foundation_lock(foundation_lock)
    target_lock = load_json(root / "targets/u000_train/lock.json")
    target_lock_hash = validate_target_lock(target_lock)
    expected_target_lock = {
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "teacher_report_sha256": report_hash,
        "teacher_checkpoint_sha256": checkpoint_hash,
        "split_manifest_sha256": foundation_spec["parents"]["split_manifest_sha256"],
        "selection_manifest_sha256": foundation_spec["parents"]["selection_manifest_sha256"],
    }
    if any(target_lock.get(name) != digest for name, digest in expected_target_lock.items()):
        raise ValueError("HCWDL-UB shared target lock lineage differs")
    if (
        foundation_lock.get("foundation_spec_sha256") != foundation_spec["content_hash"]
        or foundation_lock.get("u000_report_sha256") != report_hash
        or foundation_lock.get("u000_checkpoint_sha256") != checkpoint_hash
        or foundation_lock.get("parents", {}).get("target_lock_sha256")
        != target_lock_hash
    ):
        raise ValueError("HCWDL-UB foundation/U000 target lineage differs")

    direct = (
        target_lock.get("manifest_sha256") == manifest_hash
        and foundation_lock.get("u000_target_manifest_sha256") == manifest_hash
    )
    legacy_shadow = (
        manifest_hash != report_hash
        and target_lock.get("manifest_sha256") == report_hash
        and foundation_lock.get("u000_target_manifest_sha256") == report_hash
    )
    if not direct and not legacy_shadow:
        raise ValueError("HCWDL-UB shared target manifest is not foundation-locked")

    return with_content_hash({
        "contract": TARGET_DIGEST_SHADOW_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "classification": "direct" if direct else TARGET_DIGEST_SHADOW_REPAIR,
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "foundation_lock_sha256": foundation_lock_hash,
        "target_lock_sha256": target_lock_hash,
        "actual_target_manifest_sha256": manifest_hash,
        "recorded_target_manifest_sha256": target_lock["manifest_sha256"],
        "u000_report_sha256": report_hash,
        "u000_checkpoint_sha256": checkpoint_hash,
        "final_test_accessed": False,
    })


def validate_shared_u000_target_lineage(
    *, foundation_spec: Mapping[str, Any], foundation_root: str | Path,
    recovery_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = inspect_shared_u000_target_lineage(
        foundation_spec=foundation_spec, foundation_root=foundation_root,
    )
    if evidence["classification"] == "direct":
        return evidence
    authorized = None if recovery_context is None else recovery_context.get(
        "target_digest_shadow_repair"
    )
    if authorized != evidence:
        raise ValueError("HCWDL-UB shared target manifest is not foundation-locked")
    return evidence


def _teacher_targets(
    *, canonical_id: str, foundation_spec, foundation_root: Path,
    arm_root: Path, split, split_hash: str, selections, assignments, balanced,
    batch_size: int, sampler_seed: int, repair_seed: int, device: str,
    recovery_context: Mapping[str, Any] | None = None,
) -> tuple[EphemeralTeacherTargets, str]:
    output, node = _teacher_location(
        canonical_id, foundation_root=foundation_root, arm_root=arm_root,
    )
    engine_path = output / "training_report.json"
    report = load_json(engine_path); report_hash = validate_pmard_training_report(report)
    checkpoint = output / str(report["selected_checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
        raise ValueError("HCWDL-UB teacher selected checkpoint differs")
    if canonical_id == "shared/U000":
        lineage = validate_shared_u000_target_lineage(
            foundation_spec=foundation_spec, foundation_root=foundation_root,
            recovery_context=recovery_context,
        )
        durable = DurableUnifiedBalancedTargets(
            foundation_root / "targets/u000_train/manifest.json",
            teacher_id=canonical_id,
        )
        expected = {
            "foundation_spec_sha256": foundation_spec["content_hash"],
            "split_manifest_sha256": split_hash,
            "teacher_report_sha256": report_hash,
            "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
        if any(durable.manifest.get("parents", {}).get(name) != value for name, value in expected.items()):
            raise ValueError("HCWDL-UB shared target manifest lineage differs")
        if durable.manifest["content_hash"] != lineage["actual_target_manifest_sha256"]:
            raise ValueError("HCWDL-UB shared target manifest changed after preflight")
        targets = durable.as_ephemeral(
            teacher_report_sha256=report_hash,
            split_manifest_sha256=split_hash,
        )
        return targets, report_hash
    local_manifest = output / "targets/manifest.json"
    if local_manifest.is_file():
        consumers = _teacher_consumers(canonical_id)
        durable = DurableUnifiedBalancedTargets(
            local_manifest, teacher_id=canonical_id, consumers=consumers,
        )
        expected = {
            "foundation_spec_sha256": foundation_spec["content_hash"],
            "split_manifest_sha256": split_hash,
            "teacher_report_sha256": report_hash,
            "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
        if any(durable.manifest.get("parents", {}).get(name) != value for name, value in expected.items()):
            raise ValueError("HCWDL-UB arm target manifest lineage differs")
        arm_spec = load_json(arm_root / "arm_spec.json")
        arm_spec_hash = validate_arm_spec(arm_spec)
        owner, teacher_id = canonical_id.split("/", 1)
        if owner != arm_spec.get("arm_id"):
            raise ValueError("HCWDL-UB local target belongs to another arm")
        attestation_root, attestation_scope = _target_attestation_context(
            teacher_id=teacher_id, arm_root=arm_root,
            arm_spec_sha256=arm_spec_hash,
            recovery_context=recovery_context,
        )
        attestation_path = task_attestation_path(
            attestation_root, f"train_{teacher_id}", None,
        )
        attestation = load_json(attestation_path)
        validate_task_attestation(
            attestation, campaign_spec_sha256=attestation_scope,
            task_id=f"train_{teacher_id}", array_index=None,
        )
        manifest_path = str(local_manifest.resolve())
        matching = [
            row for row in attestation["outputs"]
            if str(Path(row["path"]).resolve()) == manifest_path
        ]
        if (
            len(matching) != 1
            or matching[0].get("content_hash") != durable.manifest["content_hash"]
        ):
            raise ValueError("HCWDL-UB local target manifest lacks its producer attestation")
        targets = durable.as_ephemeral(
            teacher_report_sha256=report_hash,
            split_manifest_sha256=split_hash,
        )
        return targets, report_hash
    model, loaded = load_pmard_model(
        engine_path, model_factory=scouting_model_factory_for_report(report),
        device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-UB teacher report changed during load")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    stream = _stream(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train",
        behavior=behavior, coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        legacy=node.behavior in {"legacycdf_uniform", "balanced_legacywarp"},
    )
    targets = precompute_teacher_targets(
        model, stream, input_key="hlt" if behavior == "hlt" else "privileged",
        device=device, teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    del model; gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    return targets, report_hash


def _teacher_consumers(canonical_id: str) -> tuple[str, ...]:
    owner, _ = canonical_id.split("/", 1)
    consumers = []
    registries = (
        [arm_registry(owner)] if owner != SHARED_ARM
        else [arm_registry(arm) for arm in (
            "C25P75", "C10P90", "C05P95", "C10P75G15", "C05P80G15", "C00P100",
        )]
    )
    for registry in registries:
        for node in registry.values():
            if canonical_id in node.teachers:
                consumers.append(node.canonical_id)
    return tuple(sorted(consumers))


def _publish_teacher_targets(
    *, canonical_id: str, output: Path, node, foundation_spec,
    split, split_hash: str, selections, assignments, balanced,
    batch_size: int, sampler_seed: int, repair_seed: int, device: str,
    target_root_override: str | Path | None = None,
    producer_commit: str | None = None,
) -> Path | None:
    """Publish one compact cache only when a selected teacher has >1 consumers."""

    consumers = _teacher_consumers(canonical_id)
    if len(consumers) < 2:
        return None
    report_path = output / "training_report.json"
    report = load_json(report_path); report_hash = validate_pmard_training_report(report)
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("HCWDL-UB selected teacher changed before target publication")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    stream = _stream(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="train",
        behavior=behavior, coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        legacy=node.behavior in {"legacycdf_uniform", "balanced_legacywarp"},
    )
    targets = precompute_teacher_targets(
        model, stream, input_key="hlt" if behavior == "hlt" else "privileged",
        device=device, teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    del model; gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    target_root = (
        output / "targets" if target_root_override is None
        else Path(target_root_override)
    )
    by_source: dict[str, list[int]] = {
        record.path: [] for record in role_records(split, "train")
    }
    for index, identity in enumerate(targets.identities):
        source = str(identity).rsplit("::tree::", 1)[0]
        if source not in by_source:
            raise ValueError("HCWDL-UB teacher target identity has an unknown source")
        by_source[source].append(index)
    shard_paths = []
    parents = {
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "split_manifest_sha256": split_hash,
        "teacher_report_sha256": report_hash,
        "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
    }
    for source_index, (source, indexes) in enumerate(by_source.items()):
        base = target_root / f"shard_{source_index:04d}"
        _, metadata = publish_target_shard(
            base, identities=[targets.identities[index] for index in indexes],
            logits=targets.logits[indexes], source_path=source, parents=parents,
            producer_commit=str(producer_commit or foundation_spec["source_commit"]),
            teacher_id=canonical_id,
        )
        shard_paths.append(metadata)
    manifest = publish_target_manifest(
        target_root / "manifest.json", shard_paths=shard_paths,
        expected_sources=list(by_source), expected_rows=selections["train"].rows,
        parents=parents, teacher_id=canonical_id, consumers=consumers,
    )
    return target_root / "manifest.json"


def run_shared_node(
    *, foundation_spec: Mapping[str, Any], node_id: str,
    device: str = "cuda", view_cache_max_gib: float = 80.0,
) -> dict[str, Any]:
    validate_foundation_spec(foundation_spec)
    if node_id not in shared_registry():
        raise ValueError("unknown HCWDL-UB shared node")
    root = Path(foundation_spec["campaign_root"]); node = shared_registry()[node_id]
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation_spec,
    )
    recipe = load_json(foundation_spec["artifact_paths"]["recipe"])
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(int(foundation_spec["replicate_seed"]), f"ub/sampler/{node.seed_alias}")
    repair_seed = derive_seed(int(foundation_spec["replicate_seed"]), "ub/repair/v1")
    behavior = "hlt" if node.input_domain == "hlt" else "p0"
    caches, _ = _cache_student_views(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=view_cache_max_gib,
    )
    output = shared_node_output_dir(root, node_id)
    parents = {
        "foundation_spec_sha256": foundation_spec["content_hash"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "foundation_gate_sha256": validate_endpoint_lock(
            load_json(root / "locks/endpoint.json")
        ),
    }
    return train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(foundation_spec["replicate_seed"]),
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        class_weights=np.ones(15, np.float32), output_dir=output,
        parents=parents, device=device, registry=shared_training_registry(),
        domains=DOMAINS, graph_sha256=META_GRAPH_SHA256,
        report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label="HCWDL-UB-FOUNDATION", seed_node_id=node.seed_alias,
        node_contract="HCWDL_UNIFIED_BALANCED_NODE_SPEC/v1",
        scientific_config_extra={
            "canonical_node_id": node.canonical_id,
            "behavior": behavior, "final_test_accessed": False,
            "student_view_built_once": True,
        },
    )


def run_arm_node(
    *, arm_spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    view_cache_max_gib: float = 80.0, producer_commit: str | None = None,
    recovery_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_arm_spec(arm_spec)
    arm_id = str(arm_spec["arm_id"]); registry = arm_registry(arm_id)
    if node_id not in registry:
        raise ValueError("unknown HCWDL-UB arm node")
    node = registry[node_id]; arm_root = Path(arm_spec["campaign_root"])
    foundation_lock_path = Path(arm_spec["foundation_lock_path"])
    foundation_root = foundation_lock_path.parent.parent
    foundation_lock = load_json(foundation_lock_path)
    lock_hash = validate_foundation_lock(foundation_lock)
    if lock_hash != arm_spec["foundation_lock_sha256"]:
        raise ValueError("HCWDL-UB arm foundation lock differs")
    foundation_spec = load_json(foundation_root / "foundation_spec.json")
    validate_foundation_spec(foundation_spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation_spec,
    )
    recipe = load_json(foundation_spec["artifact_paths"]["recipe"])
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(int(foundation_spec["replicate_seed"]), f"ub/sampler/{node.seed_alias}")
    repair_seed = derive_seed(int(foundation_spec["replicate_seed"]), "ub/repair/v1")
    behavior = "hlt" if node.input_domain == "hlt" else node.behavior
    shared_teachers = {node.parent_id, node.grandparent_id} - {None}
    shared_lineage = None
    if "shared/U000" in shared_teachers:
        shared_lineage = validate_shared_u000_target_lineage(
            foundation_spec=foundation_spec, foundation_root=foundation_root,
            recovery_context=recovery_context,
        )
    caches, input_key = _cache_student_views(
        foundation_spec=foundation_spec, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior=behavior,
        coordinate=node.coordinate, batch_size=batch_size,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=view_cache_max_gib,
    )
    parent_targets = grandparent_targets = None
    parent_hash = grandparent_hash = None
    if node.parent_id is not None:
        parent_targets, parent_hash = _teacher_targets(
            canonical_id=node.parent_id, foundation_spec=foundation_spec,
            foundation_root=foundation_root, arm_root=arm_root, split=split,
            split_hash=split_hash, selections=selections, assignments=assignments,
            balanced=balanced, batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, device=device,
            recovery_context=recovery_context,
        )
    if node.grandparent_kd_weight:
        if node.grandparent_id is None:
            raise ValueError("HCWDL-UB grandparent weight lacks a teacher")
        grandparent_targets, grandparent_hash = _teacher_targets(
            canonical_id=node.grandparent_id, foundation_spec=foundation_spec,
            foundation_root=foundation_root, arm_root=arm_root, split=split,
            split_hash=split_hash, selections=selections, assignments=assignments,
            balanced=balanced, batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed, device=device,
            recovery_context=recovery_context,
        )
    parents = {
        "arm_spec_sha256": arm_spec["content_hash"],
        "foundation_lock_sha256": lock_hash,
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
    }
    if parent_hash is not None:
        parents["parent_teacher_report_sha256"] = parent_hash
    if grandparent_hash is not None:
        parents["grandparent_teacher_report_sha256"] = grandparent_hash
    if recovery_context is not None:
        parents["recovery_spec_sha256"] = require_sha256(
            recovery_context["spec_sha256"], name="HCWDL-UB recovery spec",
        )
    if shared_lineage is not None:
        parents["shared_u000_target_manifest_sha256"] = shared_lineage[
            "actual_target_manifest_sha256"
        ]
        if shared_lineage["classification"] != "direct":
            parents["target_digest_shadow_evidence_sha256"] = shared_lineage[
                "content_hash"
            ]
    loss = GenerationalLossConfiguration(
        arm=f"HCWDL_UB_{arm_id}_{node_id}", ce=node.ce_weight,
        parent_kd=node.parent_kd_weight,
        grandparent_kd=node.grandparent_kd_weight,
        parent_temperature=node.parent_temperature,
        grandparent_temperature=node.grandparent_temperature,
    )
    child_lr = float(recipe["optimizer"]["peak_learning_rates"]["cold_child"])
    output = arm_node_output_dir(arm_root, node_id)
    result = train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(foundation_spec["replicate_seed"]),
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=batch_size,
        ),
        class_weights=np.ones(15, np.float32), output_dir=output,
        parents=parents, device=device, registry=training_registry_for_arm(arm_id),
        domains=DOMAINS, graph_sha256=META_GRAPH_SHA256,
        report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label=f"HCWDL-UB-{arm_id}", seed_node_id=node.seed_alias,
        node_contract="HCWDL_UNIFIED_BALANCED_NODE_SPEC/v1",
        explicit_loss=loss, recipe_overlay_sha256=arm_spec["recipe_arm_sha256"],
        parent_teacher_targets=parent_targets,
        grandparent_teacher_targets=grandparent_targets,
        peak_learning_rate_override=child_lr,
        scientific_config_extra={
            "canonical_node_id": node.canonical_id,
            "behavior": behavior, "input_key": input_key,
            "parent_id": node.parent_id, "grandparent_id": node.grandparent_id,
            "final_test_accessed": False,
            "student_view_built_once": True,
            "parent_targets_built_once": parent_targets is not None,
            "grandparent_targets_built_once": grandparent_targets is not None,
            "execution_repair": (
                None if shared_lineage is None
                else shared_lineage["classification"]
            ),
        },
    )
    _publish_teacher_targets(
        canonical_id=node.canonical_id, output=output, node=node,
        foundation_spec=foundation_spec, split=split, split_hash=split_hash,
        selections=selections, assignments=assignments, balanced=balanced,
        batch_size=batch_size, sampler_seed=sampler_seed,
        repair_seed=repair_seed, device=device,
        producer_commit=producer_commit,
    )
    return result


__all__ = [
    "RUNTIME_CONTRACT", "TARGET_DIGEST_SHADOW_EVIDENCE_CONTRACT",
    "TARGET_DIGEST_SHADOW_REPAIR", "arm_node_output_dir",
    "inspect_shared_u000_target_lineage", "run_arm_node",
    "run_shared_node", "shared_node_output_dir",
    "validate_shared_u000_target_lineage",
]
