"""Fixed-schema production assembly for HCWDL-RKD data-plane tasks.

This module is the closed bridge between immutable, path-only runtime rows and
the repository-owned target, training, and shared-final APIs.  It deliberately
contains no import-string, callback-string, command, or generic entry-point
facility.  Every callable handed to a lower-level API is assembled here from
one of the frozen data-plane variants below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
import gc
import hashlib
import math
import tempfile
import time
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    write_immutable_json,
)
from .hcwdl_representation_contracts import (
    SHARED_FINAL_ASSIGNMENT_SHARD_CONTRACT,
)


TARGET_ASSEMBLY_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_ASSEMBLY/v3"
TRAINING_ASSEMBLY_CONTRACT: Final = "HCWDL_REPRESENTATION_TRAINING_ASSEMBLY/v3"
FINAL_SELECTION_ASSEMBLY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FINAL_SELECTION_ASSEMBLY/v1"
)
FINAL_ASSIGNMENT_ASSEMBLY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FINAL_ASSIGNMENT_ASSEMBLY/v1"
)
FINAL_PREDICTION_ASSEMBLY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FINAL_PREDICTION_ASSEMBLY/v1"
)
FINAL_JOIN_ASSEMBLY_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_JOIN_ASSEMBLY/v1"
FINAL_EXECUTION_ASSEMBLY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FINAL_EXECUTION_ASSEMBLY/v1"
)
FINAL_AGGREGATE_ASSEMBLY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FINAL_AGGREGATE_ASSEMBLY/v1"
)
VALIDATION_ONLY_ASSEMBLY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_VALIDATION_ONLY_ASSEMBLY/v1"
)
FINAL_ASSIGNMENT_ENVELOPE_CONTRACT: Final = SHARED_FINAL_ASSIGNMENT_SHARD_CONTRACT
VALIDATION_ONLY_AGGREGATE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_VALIDATION_ONLY_AGGREGATE/v1"
)


def _runtime_helpers():
    # Imported lazily to avoid a module-import cycle: runtime_adapters owns the
    # closed dispatcher and imports this module only after defining helpers.
    from .hcwdl_representation_runtime_adapters import (
        ProductionConfigurationError,
        _outputs,
        _parameters,
        _publish_exact_json,
        _published_path_matches_output,
        _require_exact_parameters,
        _validate_registered_outputs,
        resolve_registered_arguments,
    )

    return {
        "error": ProductionConfigurationError,
        "outputs": _outputs,
        "parameters": _parameters,
        "publish": _publish_exact_json,
        "published_matches": _published_path_matches_output,
        "require": _require_exact_parameters,
        "resolve": resolve_registered_arguments,
        "validate_outputs": _validate_registered_outputs,
    }


def _assembly(
    task: Any,
    runtime_row: Mapping[str, Any],
    *,
    contract: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> Mapping[str, Any]:
    helpers = _runtime_helpers()
    parameters = helpers["require"](
        task, runtime_row, required=("assembly",),
    )
    raw_value = parameters["assembly"]
    if not isinstance(raw_value, Mapping):
        raise helpers["error"]("production assembly must be an immutable JSON object")
    value = helpers["resolve"](
        raw_value, runtime_row, location=f"{contract}.assembly",
    )
    if not isinstance(value, Mapping):
        raise helpers["error"]("resolved production assembly is not an object")
    required_fields = {"contract", *required}
    missing = required_fields - set(value)
    extra = set(value) - required_fields - set(optional)
    if missing or extra or value.get("contract") != contract:
        raise helpers["error"](
            f"{contract} schema differs; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def _reference(value: object, *, name: str, json_value: bool = True):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} artifact reference fields differ")
    path = Path(str(value["path"]))
    digest = require_sha256(value["sha256"], name=f"{name} byte SHA-256")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{name} artifact bytes differ")
    return load_json(path) if json_value else path


def _registered_input_path(value: object, *, name: str) -> Path:
    """Require a path to carry the resolver's registered-input provenance."""

    from .hcwdl_representation_runtime_adapters import RegisteredInputPath

    if not isinstance(value, RegisteredInputPath):
        raise PermissionError(
            f"{name} must resolve from registered_path or registered_member"
        )
    return Path(str(value))


def _versioned_reference(value: object, *, name: str) -> Mapping[str, Any]:
    artifact = _reference(value, name=name)
    if not isinstance(artifact, Mapping):
        raise TypeError(f"{name} artifact is not an object")
    contract = artifact.get("contract")
    schema = artifact.get("schema_version")
    if not isinstance(contract, str) or not isinstance(schema, int):
        raise ValueError(f"{name} artifact lacks a versioned contract")
    validate_content_hash(
        artifact, expected_contract=contract, expected_schema_version=schema,
    )
    return artifact


def _exact_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty immutable mapping")
    return value


def _identity_entries(identity_keys: Sequence[object]) -> np.ndarray:
    values = tuple(str(value) for value in identity_keys)
    try:
        entries = [int(value.rsplit("::tree::", 1)[1]) for value in values]
    except (IndexError, ValueError) as error:
        raise ValueError("canonical model identity keys differ") from error
    result = np.asarray(entries, dtype=np.dtype("<u8"))
    if len(result) and np.any(result[1:] <= result[:-1]):
        raise ValueError("canonical source entries reorder")
    return result


def _native_visible_indices(mask: np.ndarray) -> np.ndarray:
    visible = np.asarray(mask, dtype=np.bool_)[:, 0]
    indexes = np.broadcast_to(
        np.arange(visible.shape[1], dtype=np.int64), visible.shape,
    ).copy()
    indexes[~visible] = -1
    return np.ascontiguousarray(indexes)


def _target_forward_batch(
    batch: Mapping[str, object], *, partition: str, source_file_id: int,
    bank_kind: str, teacher_view: str,
):
    from .hcwdl_representation_data import (
        HCWDLParticleInputs, canonical_identity_digests,
    )
    from .hcwdl_representation_target_runtime import TargetForwardBatch
    from .inputs import NativeOfflineInputs

    keys = np.asarray(batch.get("identity_keys"))
    labels = np.ascontiguousarray(np.asarray(batch.get("labels"), dtype=np.uint8))
    if keys.ndim != 1 or labels.shape != (len(keys),):
        raise ValueError("canonical target batch identity/label rows differ")
    identities = canonical_identity_digests(tuple(map(str, keys.tolist())))
    source_ids = np.full(len(keys), source_file_id, dtype=np.dtype("<u4"))
    entries = _identity_entries(keys)
    if bank_kind == "ordinary":
        input_key = "hlt" if teacher_view == "hlt" else "privileged"
        view = batch.get(input_key)
        if not isinstance(view, HCWDLParticleInputs):
            raise TypeError("ordinary target stream lacks strict HCWDL token metadata")
        teacher_inputs = {
            "family_codes": np.ascontiguousarray(view.family_codes),
            "features": np.ascontiguousarray(view.features, dtype=np.float32),
            "mask": np.ascontiguousarray(view.mask, dtype=np.bool_),
            "vectors": np.ascontiguousarray(view.vectors, dtype=np.float32),
            "visible_indices": np.ascontiguousarray(view.visible_indices),
        }
        companion = (None, None, None)
    else:
        native = batch.get("toff")
        hlt = batch.get("hlt")
        if not isinstance(native, NativeOfflineInputs) or not isinstance(
            hlt, HCWDLParticleInputs,
        ):
            raise TypeError("TOFF target stream lacks paired native/HCT inputs")
        charged, neutral = native.charged, native.neutral
        teacher_inputs = {
            "charged_features": np.ascontiguousarray(charged.features, dtype=np.float32),
            "charged_mask": np.ascontiguousarray(charged.mask, dtype=np.bool_),
            "charged_vectors": np.ascontiguousarray(charged.vectors, dtype=np.float32),
            "charged_visible_indices": _native_visible_indices(charged.mask),
            "neutral_features": np.ascontiguousarray(neutral.features, dtype=np.float32),
            "neutral_mask": np.ascontiguousarray(neutral.mask, dtype=np.bool_),
            "neutral_vectors": np.ascontiguousarray(neutral.vectors, dtype=np.float32),
            "neutral_visible_indices": _native_visible_indices(neutral.mask),
        }
        # Channels 1..6 are identity transforms of the validated raw
        # charge/PID branches.  Recovering them here retains the exact raw
        # pre-transform values after canonical HLT trimming without exposing
        # those fields to the teacher callback.
        companion = (
            np.ascontiguousarray(hlt.features[:, 1, :], dtype=np.float32),
            np.ascontiguousarray(
                np.transpose(hlt.features[:, 2:7, :], (0, 2, 1)),
                dtype=np.float32,
            ),
            np.ascontiguousarray(hlt.mask[:, 0, :], dtype=np.bool_),
        )
    return TargetForwardBatch(
        source_partition=partition,
        source_file_id=source_ids,
        source_entry=entries,
        identity_digest=identities,
        label=labels,
        teacher_inputs=teacher_inputs,
        companion_hlt_charge=companion[0],
        companion_hlt_pid_flags=companion[1],
        companion_hlt_visible_mask=companion[2],
    )


class _TargetPartitionAuditor:
    """Hash canonical batch metadata without retaining any particle tensors."""

    def __init__(self, *, partition: str, source_file_id: int) -> None:
        from .hcwdl_representation_targets import TargetPopulationHasher

        self.partition = partition
        self.source_file_id = source_file_id
        self.rows = 0
        self.class_counts = np.zeros(15, dtype=np.int64)
        self.identity_chunks: list[np.ndarray] = []
        self.population = TargetPopulationHasher()
        self.last_entry: int | None = None

    def update(self, batch: Any, *, final: bool) -> None:
        from .hcwdl_representation_targets import TARGET_FORWARD_BATCH_SIZE

        rows = int(batch.rows)
        source_ids = np.asarray(batch.source_file_id)
        entries = np.asarray(batch.source_entry)
        identities = np.asarray(batch.identity_digest)
        labels = np.asarray(batch.label)
        if (
            batch.source_partition != self.partition
            or not 1 <= rows <= TARGET_FORWARD_BATCH_SIZE
            or (not final and rows != TARGET_FORWARD_BATCH_SIZE)
            or source_ids.dtype != np.dtype("<u4") or source_ids.shape != (rows,)
            or entries.dtype != np.dtype("<u8") or entries.shape != (rows,)
            or identities.dtype != np.dtype("u1") or identities.shape != (rows, 32)
            or labels.dtype != np.dtype("u1") or labels.shape != (rows,)
            or np.any(source_ids != self.source_file_id)
            or np.any(entries[1:] <= entries[:-1])
            or (self.last_entry is not None and int(entries[0]) <= self.last_entry)
            or len({bytes(row) for row in identities}) != rows
            or np.any(labels > 14)
        ):
            raise ValueError("target partition audit identity/batch schema differs")
        self.last_entry = int(entries[-1])
        self.rows += rows
        self.class_counts += np.bincount(labels, minlength=15)
        # Identity digests are the only per-row values retained by an audit.
        # Teacher inputs, companion arrays, and raw particle views remain owned
        # by the current yielded batch and become collectible immediately.
        self.identity_chunks.append(np.ascontiguousarray(identities.copy()))
        self.population.update(
            source_file_id=source_ids, source_entry=entries,
            identity_digest=identities, label=labels,
        )

    def finish(self, *, expected_rows: int) -> dict[str, Any]:
        from .hcwdl_representation_targets import (
            identity_order_sha256, identity_set_sha256,
        )

        if self.rows <= 0 or expected_rows not in {-1, self.rows}:
            raise ValueError("target source partition row count differs from selection")
        identities = np.ascontiguousarray(np.concatenate(self.identity_chunks, axis=0))
        if len({bytes(row) for row in identities}) != self.rows:
            raise ValueError("target source partition repeats an identity")
        result = {
            "rows": self.rows,
            "source_file_id": self.source_file_id,
            "class_counts": self.class_counts.astype(int).tolist(),
            "identity_order_sha256": identity_order_sha256(identities),
            "identity_set_sha256": identity_set_sha256(identities),
            "population_rows_sha256": self.population.hexdigest(),
        }
        self.identity_chunks.clear()
        return result


def _audit_target_factory(
    factory: Callable[[], Iterable[Any]], *, partition: str, source_file_id: int,
    expected_rows: int, on_batch: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Consume one lazy factory pass and retain only its authenticated audit."""

    auditor = _TargetPartitionAuditor(
        partition=partition, source_file_id=source_file_id,
    )
    iterator = iter(factory())
    try:
        current = next(iterator)
    except StopIteration as error:
        raise ValueError(f"target source partition is empty: {partition}") from error
    while True:
        try:
            following = next(iterator)
            final = False
        except StopIteration:
            following = None
            final = True
        auditor.update(current, final=final)
        if on_batch is not None:
            on_batch(current)
        current = None
        if final:
            break
        current = following
        following = None
    del iterator, current
    return auditor.finish(expected_rows=expected_rows)


def _bind_audited_target_factory(
    factory: Callable[[], Iterable[Any]], *, partition: str, source_file_id: int,
    expected_rows: int, expected_audit: Mapping[str, Any],
) -> Callable[[], Iterator[Any]]:
    """Return a re-iterable factory that revalidates the first-pass audit."""

    frozen_expected = dict(expected_audit)

    def checked() -> Iterator[Any]:
        auditor = _TargetPartitionAuditor(
            partition=partition, source_file_id=source_file_id,
        )
        iterator = iter(factory())
        try:
            current = next(iterator)
        except StopIteration as error:
            raise ValueError(f"target source partition is empty: {partition}") from error
        while True:
            try:
                following = next(iterator)
                final = False
            except StopIteration:
                following = None
                final = True
            auditor.update(current, final=final)
            yielded = current
            current = None
            yield yielded
            yielded = None
            if final:
                break
            current = following
            following = None
        del iterator, current
        actual = auditor.finish(expected_rows=expected_rows)
        if actual != frozen_expected:
            raise ValueError(
                "target source partition changed between population audit and build"
            )

    return checked


def _take_target_forward_batch(batch: Any, rows: int) -> Any:
    """Take a deterministic row prefix without changing target batch semantics."""

    from .hcwdl_representation_target_runtime import TargetForwardBatch

    if (
        isinstance(rows, bool) or not isinstance(rows, int)
        or not 1 <= rows <= int(batch.rows)
    ):
        raise ValueError("bounded target batch row count differs")

    def take(value: Any, *, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim < 1 or array.shape[0] != int(batch.rows):
            raise ValueError(f"bounded target {name} row axis differs")
        return np.ascontiguousarray(array[:rows])

    teacher_inputs = {
        str(name): take(value, name=f"teacher input {name}")
        for name, value in batch.teacher_inputs.items()
    }
    companions = {}
    for name in (
        "companion_hlt_charge", "companion_hlt_pid_flags",
        "companion_hlt_visible_mask",
    ):
        value = getattr(batch, name)
        companions[name] = None if value is None else take(value, name=name)
    return TargetForwardBatch(
        source_partition=str(batch.source_partition),
        source_file_id=take(batch.source_file_id, name="source_file_id"),
        source_entry=take(batch.source_entry, name="source_entry"),
        identity_digest=take(batch.identity_digest, name="identity_digest"),
        label=take(batch.label, name="label"),
        teacher_inputs=teacher_inputs,
        **companions,
    )


def _bounded_target_factory(
    factory: Callable[[], Iterable[Any]], *, row_limit: int,
) -> Callable[[], Iterator[Any]]:
    """Bound one canonical source stream while preserving its batch partition."""

    if isinstance(row_limit, bool) or not isinstance(row_limit, int) or row_limit <= 0:
        raise ValueError("bounded target source row limit differs")

    def bounded() -> Iterator[Any]:
        remaining = row_limit
        for batch in factory():
            if remaining <= 0:
                break
            rows = int(batch.rows)
            if rows <= 0:
                raise ValueError("bounded target source emitted an empty batch")
            take = min(rows, remaining)
            yield batch if take == rows else _take_target_forward_batch(batch, take)
            remaining -= take
            if remaining == 0:
                break

    return bounded


def _prepare_target_partitions(
    *, split: Mapping[str, Any], selection: Mapping[str, Any],
    data_root: str | Path, teacher_view: str,
    source_partitions: Mapping[str, Any], assignment_manifest: str | Path | None,
    bounded_row_limit: int | None = None,
) -> tuple[
    dict[str, Callable[[], Iterator[Any]]], dict[str, dict[str, int]],
]:
    from .dataset import iterate_model_batches
    from .highcov_cache import DenseAssignmentStore
    from .pmard_stream import iterate_pmard_batches
    from .selective_assignment import RowSelection
    from .splits import role_records

    records = tuple(role_records(split, "train"))
    by_path = {record.path: (index, record) for index, record in enumerate(records)}
    row_selection = RowSelection(
        selection, role="train", split_manifest_sha256=split["content_hash"],
    )
    bank_kind = "toff" if teacher_view == "toff" else "ordinary"
    if teacher_view in {"hlt", "toff"}:
        teacher_level = None
    elif teacher_view.startswith("d") and teacher_view[1:].isdigit():
        teacher_level = int(teacher_view[1:])
        if teacher_level not in range(5, 101, 5):
            raise ValueError("dense target teacher privilege level differs")
    else:
        raise ValueError("target teacher view differs")
    if teacher_view.startswith("d") and teacher_view != "hlt":
        if assignment_manifest is None:
            raise ValueError("repaired target view lacks an assignment manifest")
        # Validate the compact manifest now.  Each factory creates its own lazy
        # store so one source assignment shard is released before the next.
        DenseAssignmentStore(assignment_manifest)
    elif assignment_manifest is not None:
        raise ValueError("unrepaired target view unexpectedly binds assignments")
    normalized_sources: dict[str, tuple[int, Any]] = {}
    for partition, raw in source_partitions.items():
        if not isinstance(raw, Mapping) or set(raw) != {"source_path", "source_file_id"}:
            raise ValueError("target source-partition record fields differ")
        source = str(raw["source_path"])
        source_id = raw["source_file_id"]
        if source not in by_path or isinstance(source_id, bool) or not isinstance(
            source_id, int,
        ) or not 0 <= source_id < 2**32:
            raise ValueError("target source-partition identity differs")
        normalized_sources[str(partition)] = (source_id, by_path[source][1])
    if set(record.path for record in records) != {
        record.path for _, record in normalized_sources.values()
    }:
        raise ValueError("target source partitions do not cover the selected train role")
    partition_limits: dict[str, int] | None = None
    if bounded_row_limit is not None:
        if (
            isinstance(bounded_row_limit, bool)
            or not isinstance(bounded_row_limit, int)
            or not 1 <= bounded_row_limit <= 4096
            or bounded_row_limit < len(normalized_sources)
        ):
            raise ValueError("cache-miniature row bound cannot cover every source partition")
        if not row_selection.all_rows and row_selection.rows <= bounded_row_limit:
            # A bounded non-final acceptance selection is already an exact,
            # authenticated population.  Preserve its per-source allocation;
            # re-slicing it evenly can silently underfill the required 512
            # rows whenever the global class-stratified ranks are uneven.
            partition_limits = {
                partition: row_selection.source_rows(record.path)
                for partition, (_source_id, record) in normalized_sources.items()
            }
            if (
                any(limit <= 0 for limit in partition_limits.values())
                or sum(partition_limits.values()) != row_selection.rows
            ):
                raise ValueError(
                    "bounded target selection does not exactly cover every source"
                )
        else:
            # A generic miniature may start from an unbounded selection.  Keep
            # every source represented and deterministically allocate its cap.
            base, remainder = divmod(bounded_row_limit, len(normalized_sources))
            partition_limits = {
                partition: base + (offset < remainder)
                for offset, partition in enumerate(sorted(normalized_sources))
            }
    factories: dict[str, Callable[[], Iterator[Any]]] = {}
    partition_specs: dict[str, dict[str, int]] = {}
    for partition in sorted(normalized_sources):
        source_id, source_record = normalized_sources[partition]
        rank = by_path[source_record.path][0]
        expected_rows = (
            int(source_record.mapped_entries)
            if row_selection.all_rows
            else row_selection.source_rows(source_record.path)
        )

        def base_factory(
            *, partition=partition, source_id=source_id, rank=rank,
            expected_view=teacher_view,
        ) -> Iterator[Any]:
            common = dict(
                data_root=data_root, role="train", rank=rank,
                world_size=len(records), epoch=0, sampler_seed=1337,
                batch_size=256, shuffle_buffer_rows=256,
                interleave_source_files=1, row_selection=row_selection,
                include_hcwdl_metadata=True, canonical_order=True,
            )
            if expected_view in {"hlt", "toff"}:
                stream = iterate_model_batches(
                    split, input_mode="paired" if expected_view == "toff" else "hlt",
                    shuffle_within_chunk=False, **common,
                )
            else:
                store = DenseAssignmentStore(assignment_manifest)
                stream = iterate_pmard_batches(
                    split, matcher_model=None,
                    alpha=int(expected_view[1:]) / 100.0,
                    matcher_variant="highcov_empirical_lexicographic_dr0p30_v1",
                    threshold=0.0, repair_family="HIGHCOV_SHELL_EXACT/v1",
                    assignment_store=store, repair_seed=1337,
                    **common,
                )
            for raw_batch in stream:
                yield _target_forward_batch(
                    raw_batch, partition=partition, source_file_id=source_id,
                    bank_kind=bank_kind, teacher_view=expected_view,
                )

        if partition_limits is not None:
            source_limit = partition_limits[partition]
            expected_rows = min(source_limit, expected_rows)
            base_factory = _bounded_target_factory(
                base_factory, row_limit=expected_rows,
            )

        partition_specs[partition] = {
            "rows": int(expected_rows), "source_file_id": source_id,
        }
        factories[partition] = base_factory
    return factories, partition_specs


def _committed_coordinates(value: object, *, name: str, parent_name: str = "committed"):
    """Derive a publication root/ID only from an authenticated exact child."""

    directory = _registered_input_path(value, name=name)
    if (
        directory.parent.name != parent_name
        or len(directory.name) != 64
        or any(character not in "0123456789abcdef" for character in directory.name)
    ):
        raise ValueError(f"{name} path differs")
    return directory, directory.parent.parent, directory.name


def _load_kernel_bundle(value: Mapping[str, Any]):
    legacy = {"root", "envelope_id", "expected_parents", "owner_id"}
    committed = {"committed_directory"}
    if set(value) not in (legacy, committed):
        raise ValueError("kernel envelope reference fields differ")
    from .hcwdl_representation_kernels import load_spectral_resources

    if set(value) == committed:
        directory, root, envelope_id = _committed_coordinates(
            value["committed_directory"], name="kernel committed envelope",
        )
    else:
        root = _registered_input_path(value["root"], name="kernel envelope root")
        envelope_id = str(value["envelope_id"])
    if set(value) == committed:
        commit = load_json(root / "committed" / envelope_id / "commit.json")
        expected_parents = commit["parents"]
        owner_id = None
    else:
        expected_parents = _exact_mapping(
            value["expected_parents"], name="kernel parent registry",
        )
        owner_id = None if value["owner_id"] is None else str(value["owner_id"])
    return load_spectral_resources(
        root, envelope_id,
        expected_parents=expected_parents, expected_owner_id=owner_id,
    )


def _pmard_report_chain(
    report_reference: object, *, name: str,
) -> dict[str, Any]:
    """Resolve either a parent HCWDL wrapper or its PMARD engine report."""

    source = _versioned_reference(report_reference, name=name)
    source_path = Path(str(report_reference["path"])).resolve()
    from .engine import validate_pmard_training_report
    from .hcwdl_training import (
        TRAINING_REPORT_CONTRACT as HCWDL_PARENT_REPORT_CONTRACT,
        validate_hcwdl_training_report,
    )

    wrapper = None
    wrapper_path = None
    if source.get("contract") == HCWDL_PARENT_REPORT_CONTRACT:
        validate_hcwdl_training_report(source)
        wrapper = source
        wrapper_path = source_path
        engine_path = source_path.parent / "training_report.json"
        if engine_path == source_path or not engine_path.is_file():
            raise FileNotFoundError(f"{name} PMARD engine report is absent")
        engine = load_json(engine_path)
        engine_sha256 = validate_pmard_training_report(engine)
        if (
            wrapper.get("pmard_engine_report_sha256") != engine_sha256
            or wrapper.get("selected_checkpoint_sha256")
            != engine.get("selected_checkpoint_sha256")
        ):
            raise ValueError(f"{name} wrapper/engine report lineage differs")
    else:
        engine = source
        engine_path = source_path
        engine_sha256 = validate_pmard_training_report(engine)
    selected_name = engine.get("selected_checkpoint")
    if (
        not isinstance(selected_name, str) or not selected_name
        or Path(selected_name).name != selected_name
    ):
        raise ValueError(f"{name} selected checkpoint name differs")
    checkpoint = engine_path.parent / selected_name
    digest = require_sha256(
        engine.get("selected_checkpoint_sha256"), name=f"{name} selected checkpoint",
    )
    if not checkpoint.is_file() or sha256_file(checkpoint) != digest:
        raise ValueError(f"{name} selected checkpoint bytes differ")
    return {
        "source": source, "source_path": source_path,
        "wrapper": wrapper, "wrapper_path": wrapper_path,
        "engine": engine, "engine_path": engine_path.resolve(),
        "engine_sha256": engine_sha256,
        "checkpoint": checkpoint.resolve(), "checkpoint_sha256": digest,
    }


def _load_pmard_teacher(report_reference: object, *, device: str):
    chain = _pmard_report_chain(report_reference, name="teacher report")
    from .loaders import load_pmard_model, scouting_model_factory_for_report

    report = chain["engine"]
    path = chain["engine_path"]
    model, validated = load_pmard_model(
        path, model_factory=scouting_model_factory_for_report(report), device=device,
    )
    model.to(device).float().eval()
    return model, validated


def _validated_parent_import_rows(
    parent_import: Mapping[str, Any], architecture: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Return the exact import/architecture registries after hash binding."""

    from hlt_classification.models.hcwdl_surfaces import (
        validate_architecture_attestation,
    )
    from .hcwdl_representation_locks import validate_parent_import

    validate_parent_import(parent_import)
    architecture_sha256 = validate_architecture_attestation(
        architecture, require_authorized=True, verify_files=False,
    )
    if parent_import["parents"].get("architecture_attestation") != architecture_sha256:
        raise ValueError("parent import architecture attestation differs")
    imported = {
        str(row["node_id"]): row
        for group in (
            parent_import["payload"]["teachers"],
            parent_import["payload"]["logit_controls"],
        )
        for row in group
    }
    architecture_rows = {
        str(row["node_id"]): row for row in architecture["checkpoint_audits"]
    }
    if set(imported) != set(architecture_rows):
        raise ValueError("parent import/architecture registries differ")
    return imported, architecture_rows


def _validate_imported_pmard_source(
    report_reference: Mapping[str, Any], *, node_id: str,
    parent_import: Mapping[str, Any], architecture: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    """Prove a target/training parent source is the exact imported model."""

    if parent_import.get("contract") == (
        "HCWDL_REPRESENTATION_DENSE_TEACHER_IMPORT/v1"
    ):
        from .hcwdl_representation_dense_teacher import validate_dense_teacher_import
        validate_dense_teacher_import(parent_import)
        if node_id != "TOFF":
            raise PermissionError("dense teacher import can authorize only TOFF")
        chain = _pmard_report_chain(report_reference, name=name)
        payload = parent_import["payload"]
        parents = parent_import["parents"]
        if (
            chain["engine_path"] != Path(payload["engine_report_path"]).resolve()
            or chain["engine_sha256"] != parents["toff_engine_report"]
            or chain["checkpoint"] != Path(payload["selected_checkpoint_path"]).resolve()
            or chain["checkpoint_sha256"] != parents["toff_selected_checkpoint"]
        ):
            raise ValueError(f"{name} differs from the dense TOFF import")
        return chain
    imported, architecture_rows = _validated_parent_import_rows(
        parent_import, architecture,
    )
    if node_id not in imported:
        raise ValueError(f"{name} is absent from the parent import")
    row = imported[node_id]
    audit = architecture_rows[node_id]
    if (
        Path(str(row["report_path"])).resolve()
        != Path(str(audit["report_path"])).resolve()
        or row["report_sha256"] != audit["report_sha256"]
        or Path(str(row["checkpoint_path"])).resolve()
        != Path(str(audit["checkpoint_path"])).resolve()
        or row["checkpoint_byte_sha256"] != audit["checkpoint_sha256"]
    ):
        raise ValueError(f"{name} imported architecture lineage differs")
    chain = _pmard_report_chain(report_reference, name=name)
    if chain["wrapper"] is not None:
        if (
            chain["wrapper_path"] != Path(str(row["report_path"])).resolve()
            or chain["wrapper"]["content_hash"] != row["report_sha256"]
        ):
            raise ValueError(f"{name} wrapper report differs from parent import")
    elif (
        chain["engine_path"] != Path(str(audit["engine_report_path"])).resolve()
        or chain["engine_sha256"] != audit["engine_report_sha256"]
    ):
        raise ValueError(f"{name} engine report differs from parent import")
    if (
        chain["engine_path"] != Path(str(audit["engine_report_path"])).resolve()
        or chain["engine_sha256"] != audit["engine_report_sha256"]
        or chain["checkpoint"] != Path(str(row["checkpoint_path"])).resolve()
        or chain["checkpoint_sha256"] != row["checkpoint_byte_sha256"]
    ):
        raise ValueError(f"{name} report/checkpoint lineage differs from parent import")
    return chain


def _configure_target_backend() -> None:
    import torch
    from .hcwdl_representation_worker_runtime import (
        configure_deterministic_worker_backend,
    )

    configure_deterministic_worker_backend(torch)


def _teacher_surface_forward(model: Any, *, device: str, bank_kind: str):
    import torch

    ordinary = (
        "features", "vectors", "mask", "visible_indices", "family_codes",
    )
    toff = (
        "charged_features", "charged_vectors", "charged_mask",
        "neutral_features", "neutral_vectors", "neutral_mask",
        "charged_visible_indices", "neutral_visible_indices",
    )
    order = ordinary if bank_kind == "ordinary" else toff

    def forward(inputs):
        if set(inputs.arrays) != set(order):
            raise PermissionError("teacher input fields differ from the fixed model surface")
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

    return forward, tuple(sorted(order))


def target_build_adapter(spec, task, index, runtime_row):
    del index
    value = _assembly(
        task, runtime_row, contract=TARGET_ASSEMBLY_CONTRACT,
        required=(
            "bank_root", "logical_bank", "consumer_registry", "forward_spec",
            "split_manifest", "row_selection", "data_root", "teacher_view",
            "source_partitions", "assignment_manifest", "teacher_source",
            "parent_import", "architecture_attestation",
            "kernel_envelope", "build_owner", "budgets", "storage_estimate",
            "resource_profile", "runtime_environment",
        ),
    )
    logical = _versioned_reference(value["logical_bank"], name="logical target bank")
    registry = _versioned_reference(
        value["consumer_registry"], name="target consumer registry",
    )
    forward_spec = _versioned_reference(value["forward_spec"], name="target forward spec")
    parent_import = _versioned_reference(
        value["parent_import"], name="target parent import",
    )
    architecture = _versioned_reference(
        value["architecture_attestation"], name="target architecture attestation",
    )
    teacher_node = str(task.logical_bank)
    declared_runtime = _exact_mapping(
        value["runtime_environment"], name="declared target runtime environment",
    )
    frozen_runtime = {
        name: forward_spec["payload"][name]
        for name in ("producer", "device", "precision", "determinism")
    }
    if declared_runtime != frozen_runtime:
        raise PermissionError(
            "declared target runtime environment differs from the frozen forward spec"
        )
    live_worker_runtime = runtime_row.get("_live_worker_runtime")
    if not isinstance(live_worker_runtime, Mapping):
        raise PermissionError("target worker lacks measured live runtime provenance")
    from .hcwdl_representation_worker_runtime import (
        build_live_target_runtime_environment,
    )

    # This happens before any ROOT partition or teacher checkpoint is opened.
    # The post-build attestation receives these observations, never the caller
    # declaration above (which is only an exact frozen-spec consistency gate).
    live_runtime_environment = build_live_target_runtime_environment(
        forward_spec, live_worker_runtime=live_worker_runtime,
    )
    split = _versioned_reference(value["split_manifest"], name="split manifest")
    selection = _versioned_reference(value["row_selection"], name="row selection")
    storage = _versioned_reference(value["storage_estimate"], name="storage estimate")
    profile = _versioned_reference(value["resource_profile"], name="resource profile")
    assignment_path = None
    if value["assignment_manifest"] is not None:
        assignment_path = _reference(
            value["assignment_manifest"], name="assignment manifest", json_value=False,
        )
    source_partitions = _exact_mapping(
        value["source_partitions"], name="target source partitions",
    )
    from .hcwdl_representation_targets import (
        NONFINAL_ACCEPTANCE_TARGET_PURPOSE,
        validate_target_consumer_registry,
    )

    validate_target_consumer_registry(registry, logical_bank=logical)
    purpose = str(registry["payload"]["purpose"])
    if task.target_purpose != purpose:
        raise ValueError("target build task purpose differs from consumer registry")
    bounded_row_limit = None
    if purpose in {"miniature", NONFINAL_ACCEPTANCE_TARGET_PURPOSE}:
        consumers = registry["payload"]["consumers"]
        # The registry validator proves every bounded consumer shares one
        # immutable limit.  Acceptance may have several real training
        # trajectories, but it may never expand the source selection.
        limits = {
            row["execution_identity_payload"]["bounded_row_limit"]
            for row in consumers
        }
        if len(limits) != 1:
            raise PermissionError("bounded target consumers disagree on row limit")
        bounded_row_limit = int(next(iter(limits)))
    partition_factories, partitions = _prepare_target_partitions(
        split=split, selection=selection, data_root=value["data_root"],
        teacher_view=str(value["teacher_view"]),
        source_partitions=source_partitions,
        assignment_manifest=assignment_path,
        bounded_row_limit=bounded_row_limit,
    )
    if list(partitions) != forward_spec["payload"]["source_partitions"]:
        raise ValueError("materialized source partitions differ from forward spec")
    bundle = _load_kernel_bundle(value["kernel_envelope"])
    budgets = value["budgets"]
    budget_fields = {
        "target_storage_cap_bytes", "container_overhead_bytes",
        "staging_recovery_reserve_bytes", "quarantine_reserve_bytes",
        "filesystem_headroom_bytes", "peak_runtime_bytes",
        "slurm_mem_per_node_bytes", "filesystem_available_bytes",
    }
    if not isinstance(budgets, Mapping) or set(budgets) != budget_fields or any(
        isinstance(raw, bool) or not isinstance(raw, int) or raw < 0
        for raw in budgets.values()
    ):
        raise ValueError("target build budget fields differ")
    from .hcwdl_representation_targets import begin_target_generation
    from .hcwdl_representation_target_runtime import (
        build_target_generation_from_prepared,
        build_target_generation_from_teacher,
        prepare_target_generation_in_memory,
    )

    build_owner = _exact_mapping(value["build_owner"], name="target build owner")
    expected = next(
        iter(_runtime_helpers()["outputs"](task, runtime_row).values())
    ).resolve()
    bank_root = Path(value["bank_root"]).resolve()
    if (
        expected.parent.name != "generations"
        or expected.parent.parent.resolve() != bank_root
        or len(expected.name) != 64
    ):
        raise ValueError("target generation directory differs from registered output")

    def begin(*, class_counts, identity_order, identity_set, population_rows):
        context = begin_target_generation(
            bank_root, logical_bank=logical, consumer_registry=registry,
            forward_spec=forward_spec, partitions=partitions,
            expected_class_counts=class_counts,
            expected_identity_order_sha256=identity_order,
            expected_identity_set_sha256=identity_set,
            expected_population_rows_sha256=population_rows,
            build_owner=build_owner, storage_estimate=storage,
            resource_profile=profile, **dict(budgets),
        )
        if context.committed_directory.resolve() != expected:
            raise ValueError("target generation directory differs from registered output")
        return context

    # A successfully committed generation is the one-time teacher result.  A
    # retry validates the current immutable request against its committed build
    # intent and returns without reopening ROOT or loading/forwarding a teacher.
    if expected.is_dir():
        prior_intent = load_json(expected / "build_intent.json")["payload"]
        begin(
            class_counts=prior_intent["expected_class_counts"],
            identity_order=prior_intent["expected_identity_order_sha256"],
            identity_set=prior_intent["expected_identity_set_sha256"],
            population_rows=prior_intent["expected_population_rows_sha256"],
        )
        del partition_factories, bundle
        gc.collect()
        return _runtime_helpers()["validate_outputs"](
            task, runtime_row, operation="target_build",
        )

    teacher = forward_spec["payload"]["teacher"]
    logical_teacher = logical["payload"]["teacher"]
    teacher_source = value["teacher_source"]
    if not isinstance(teacher_source, Mapping):
        raise ValueError("target teacher source fields differ")
    if teacher_source.get("kind") == "pmard":
        if set(teacher_source) != {"kind", "report"}:
            raise ValueError("imported target teacher source fields differ")
        if (
            logical_teacher.get("source_kind") != "imported_checkpoint"
            or teacher.get("source_kind") != "imported_checkpoint"
        ):
            raise ValueError("imported target teacher commitment differs")
        _validate_imported_pmard_source(
            teacher_source["report"], node_id=teacher_node,
            parent_import=parent_import, architecture=architecture,
            name=f"{teacher_node} target teacher",
        )
        model, teacher_report = _load_pmard_teacher(
            teacher_source["report"], device=str(runtime_row["device"]),
        )
        selected_checkpoint = (
            Path(str(teacher_source["report"]["path"])).parent
            / str(teacher_report["selected_checkpoint"])
        )
        if (
            teacher_report["selected_checkpoint_sha256"]
            != teacher["checkpoint_byte_sha256"]
            or sha256_file(selected_checkpoint)
            != teacher["checkpoint_byte_sha256"]
        ):
            raise ValueError("loaded teacher checkpoint differs from target forward spec")
    elif teacher_source.get("kind") == "hcwdl":
        if set(teacher_source) != {"kind", "execution_directory"}:
            raise ValueError("campaign target teacher source fields differ")
        if (
            logical_teacher.get("source_kind") != "campaign_execution"
            or teacher.get("source_kind") != "campaign_execution"
        ):
            raise ValueError("campaign target teacher commitment differs")
        execution_directory = _registered_input_path(
            teacher_source["execution_directory"],
            name=f"{teacher_node} target teacher execution",
        )
        evidence = _hcwdl_source_evidence(
            execution_directory, name=f"{teacher_node} target teacher",
        )
        expected_execution = require_sha256(
            logical_teacher["registered_execution_id"],
            name=f"{teacher_node} target teacher execution",
        )
        if (
            evidence["report"].get("node_id") != teacher_node
            or evidence["registered_execution_id"] != expected_execution
            or teacher["registered_execution_id"] != expected_execution
        ):
            raise ValueError("campaign target teacher execution lineage differs")
        model, _, _ = _load_model_source(
            teacher_source, name=f"{teacher_node} target teacher",
            device=str(runtime_row["device"]),
        )
    else:
        raise ValueError("target teacher source kind differs")
    forward, input_fields = _teacher_surface_forward(
        model, device=str(runtime_row["device"]),
        bank_kind=str(logical["payload"]["bank_kind"]),
    )
    if list(input_fields) != forward_spec["payload"]["implementation"][
        "teacher_input_fields"
    ]:
        raise ValueError("target forward spec teacher fields differ from fixed model API")

    _configure_target_backend()
    staged_parent = bank_root / "staging" / expected.name
    staged_intents = (
        sorted(staged_parent.glob("*/build_intent.json"))
        if staged_parent.is_dir() else []
    )
    if staged_intents:
        if len(staged_intents) != 1:
            raise PermissionError("target generation has multiple staged owners")
        prior_intent = load_json(staged_intents[0])["payload"]
        context = begin(
            class_counts=prior_intent["expected_class_counts"],
            identity_order=prior_intent["expected_identity_order_sha256"],
            identity_set=prior_intent["expected_identity_set_sha256"],
            population_rows=prior_intent["expected_population_rows_sha256"],
        )
        build_target_generation_from_teacher(
            context,
            partition_batches=partition_factories,
            teacher_forward=forward,
            token_resources=bundle.token,
            relation_resources=bundle.relation,
            runtime_environment=live_runtime_environment,
            teacher_model=model,
        )
        del model, partition_factories
        gc.collect()
        return _runtime_helpers()["validate_outputs"](
            task, runtime_row, operation="target_build",
        )

    prepared = prepare_target_generation_in_memory(
        bank_kind=str(logical["payload"]["bank_kind"]),
        partition_batches=partition_factories,
        partition_specs=partitions,
        teacher_forward=forward,
        token_resources=bundle.token,
        relation_resources=bundle.relation,
        teacher_model=model,
        allowed_input_fields=input_fields,
    )

    context = begin(
        class_counts=prepared.class_counts,
        identity_order=prepared.identity_order_sha256,
        identity_set=prepared.identity_set_sha256,
        population_rows=prepared.population_rows_sha256,
    )
    result = build_target_generation_from_prepared(
        context,
        prepared=prepared, token_resources=bundle.token,
        relation_resources=bundle.relation,
        runtime_environment=live_runtime_environment,
    )
    del model, partition_factories, prepared
    gc.collect()
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="target_build",
    )


def _particle_take(view: Any, indexes: np.ndarray):
    values = [
        np.ascontiguousarray(view.features[indexes]),
        np.ascontiguousarray(view.vectors[indexes]),
        np.ascontiguousarray(view.mask[indexes]),
        np.ascontiguousarray(view.raw_lengths[indexes]),
    ]
    for name in ("visible_indices", "family_codes", "family_reason_codes"):
        if hasattr(view, name):
            values.append(np.ascontiguousarray(getattr(view, name)[indexes]))
    return type(view)(*values)


def _particle_concat(views: Sequence[Any]):
    if not views:
        raise ValueError("cannot concatenate an empty particle-view collection")
    values = [
        np.ascontiguousarray(np.concatenate([view.features for view in views], axis=0)),
        np.ascontiguousarray(np.concatenate([view.vectors for view in views], axis=0)),
        np.ascontiguousarray(np.concatenate([view.mask for view in views], axis=0)),
        np.ascontiguousarray(np.concatenate([view.raw_lengths for view in views], axis=0)),
    ]
    for name in ("visible_indices", "family_codes", "family_reason_codes"):
        present = [hasattr(view, name) for view in views]
        if any(present):
            if not all(present):
                raise ValueError("particle metadata topology changes across batches")
            values.append(np.ascontiguousarray(np.concatenate([
                getattr(view, name) for view in views
            ], axis=0)))
    return type(views[0])(*values)


def _training_take(batch: Mapping[str, Any], indexes: np.ndarray) -> dict[str, Any]:
    return {
        "hlt": _particle_take(batch["hlt"], indexes),
        "labels": np.ascontiguousarray(np.asarray(batch["labels"])[indexes]),
        "identity_digests": np.ascontiguousarray(
            np.asarray(batch["identity_digests"])[indexes], dtype=np.uint8,
        ),
    }


def _training_concat(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("cannot concatenate an empty training-batch collection")
    return {
        "hlt": _particle_concat([batch["hlt"] for batch in batches]),
        "labels": np.ascontiguousarray(np.concatenate([
            np.asarray(batch["labels"]) for batch in batches
        ])),
        "identity_digests": np.ascontiguousarray(np.concatenate([
            np.asarray(batch["identity_digests"], dtype=np.uint8) for batch in batches
        ], axis=0)),
    }


def _calibration_population(
    cache: Any,
    *,
    sampler_seed: int,
    rows: int,
    campaign_sha256: str,
    parent_logit_counterpart_node_id: str,
    student_view: str = "hlt",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the frozen campaign/counterpart-hashed calibration population.

    The first scan retains only 32-byte identities.  The second scan copies
    exactly the selected rows, so this does not duplicate the full RAM cache.
    """

    from .hcwdl_representation_data import training_batch_from_parent
    from .hcwdl_representation_calibration import (
        build_calibration_selection_artifact,
        validate_calibration_selection_artifact,
    )

    identity_values: list[bytes] = []
    for raw in cache.iterate_batches(
        epoch=0, sampler_seed=sampler_seed, batch_size=256,
    ):
        batch = training_batch_from_parent(raw, student_view=student_view)
        identity_values.extend(bytes(value) for value in batch["identity_digests"])
    if len(identity_values) != len(set(identity_values)) or len(identity_values) < rows:
        raise ValueError("calibration source identity coverage differs")
    selection = build_calibration_selection_artifact(
        campaign_sha256=campaign_sha256,
        parent_logit_counterpart_node_id=parent_logit_counterpart_node_id,
        identity_sha256s=[value.hex() for value in identity_values],
        limit=rows,
    )
    validate_calibration_selection_artifact(
        selection,
        expected_campaign_sha256=campaign_sha256,
        expected_parent_logit_counterpart_node_id=(
            parent_logit_counterpart_node_id
        ),
    )
    ordered = [bytes.fromhex(value) for value in selection["ordered_identity_sha256s"]]
    selected = set(ordered)
    pieces = []
    for raw in cache.iterate_batches(
        epoch=0, sampler_seed=sampler_seed, batch_size=256,
    ):
        batch = training_batch_from_parent(raw, student_view=student_view)
        indexes = np.asarray([
            index for index, value in enumerate(batch["identity_digests"])
            if bytes(value) in selected
        ], dtype=np.int64)
        if len(indexes):
            pieces.append(_training_take(batch, indexes))
    joined = _training_concat(pieces)
    observed = {
        bytes(value): index
        for index, value in enumerate(joined["identity_digests"])
    }
    if len(observed) != len(joined["identity_digests"]):
        raise ValueError("calibration materialization repeats an identity")
    try:
        positions = np.asarray([observed[value] for value in ordered], dtype=np.int64)
    except KeyError as error:
        raise ValueError("calibration materialization lost a selected identity") from error
    result = _training_take(joined, positions)
    if len(result["labels"]) != rows or [
        bytes(value) for value in result["identity_digests"]
    ] != ordered:
        raise ValueError("calibration population selection/order differs")
    return result, selection


def _build_training_view_caches(
    *, split: Mapping[str, Any], selection: Mapping[str, Any], data_root: str | Path,
    train_rows: int, validation_rows: int, lineage: Mapping[str, Any], max_gib: float,
    student_domain: str,
    assignment_manifests: Mapping[str, Path] | None,
):
    from .dataset import iterate_model_batches
    from .highcov_cache import DenseAssignmentStore
    from .pmard_stream import iterate_pmard_batches
    from .selective_assignment import RowSelection
    from .splits import role_records
    from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows

    if student_domain == "hlt":
        student_view = "hlt"
        if assignment_manifests is not None:
            raise ValueError("HLT student unexpectedly binds repaired-view assignments")
    elif student_domain.startswith("d") and student_domain[1:].isdigit():
        level = int(student_domain[1:])
        if level not in range(5, 101, 5):
            raise ValueError("dense-descent student privilege level differs")
        student_view = "privileged"
        if not isinstance(assignment_manifests, Mapping) or set(
            assignment_manifests
        ) != {"train", "validation"}:
            raise ValueError("repaired-view student lacks exact assignment manifests")
    else:
        raise ValueError("dense-descent student domain differs")

    result = {}
    started = time.perf_counter()
    for role, expected_rows in (("train", train_rows), ("validation", validation_rows)):
        row_selection = RowSelection(
            selection, role=role, split_manifest_sha256=split["content_hash"],
        )
        records = tuple(role_records(split, role))
        expected_sources = expected_cache_source_rows(
            records, row_selection=row_selection,
        )
        if sum(expected_sources.values()) != expected_rows:
            raise ValueError(f"{role} cache rows differ from the immutable assembly")
        common = dict(
            data_root=data_root, role=role, epoch=0, sampler_seed=1337,
            batch_size=256, rank=0, world_size=1,
            shuffle_buffer_rows=256, interleave_source_files=1,
            row_selection=row_selection, include_hcwdl_metadata=True,
            canonical_order=True,
        )
        if student_view == "hlt":
            stream = iterate_model_batches(
                split, input_mode="hlt", shuffle_within_chunk=False, **common,
            )
            assignment_sha256 = None
        else:
            assert assignment_manifests is not None
            store = DenseAssignmentStore(assignment_manifests[role])
            stream = iterate_pmard_batches(
                split, matcher_model=None,
                alpha=int(student_domain[1:]) / 100.0,
                matcher_variant="highcov_empirical_lexicographic_dr0p30_v1",
                threshold=0.0, repair_family="HIGHCOV_SHELL_EXACT/v1",
                assignment_store=store, repair_seed=1337, **common,
            )
            assignment_sha256 = store.manifest["content_hash"]
        result[role] = EphemeralPmardViewCache.build(
            stream, expected_rows=expected_rows, records=records, role=role,
            expected_source_rows=expected_sources, view_keys=(student_view,),
            lineage={
                **dict(lineage), "student_domain": student_domain,
                "student_view": student_view,
                "assignment_manifest_sha256": assignment_sha256,
            }, max_gib=max_gib,
        )
    return result, time.perf_counter() - started


def _hcwdl_source_evidence(
    execution: Path, *, name: str,
) -> dict[str, Any]:
    """Authenticate the complete report/selector/extraction model identity."""

    report_path = execution / "training_report.json"
    selection_path = execution / "checkpoint_selection.json"
    extraction_path = execution / "deployable_extraction.json"
    if any(not path.is_file() for path in (report_path, selection_path, extraction_path)):
        raise FileNotFoundError(f"{name} HCWDL model-source artifacts are incomplete")
    report = load_json(report_path)
    selection = load_json(selection_path)
    extraction = load_json(extraction_path)
    from .hcwdl_representation_contracts import validate_versioned_artifact
    from .hcwdl_representation_training import (
        DEPLOYABLE_EXTRACTION_CONTRACT,
        validate_representation_training_report,
    )
    from .hcwdl_representation_contracts import CHECKPOINT_SELECTION_CONTRACT

    report_sha256 = validate_representation_training_report(report)
    selection_sha256 = validate_versioned_artifact(
        selection, expected_contract=CHECKPOINT_SELECTION_CONTRACT,
        required_payload_keys=(
            "selected_checkpoint_id", "selected_update", "ordering", "selector",
            "validation_records",
        ),
    )
    extraction_sha256 = validate_versioned_artifact(
        extraction, expected_contract=DEPLOYABLE_EXTRACTION_CONTRACT,
        required_payload_keys=(
            "node_id", "registered_execution_id", "selected_envelope_id",
            "selected_training_state_path", "deployable_state_path",
            "deployable_state_sha256", "student_domain",
            "deployment_authorized", "strict_hlt_only",
            "training_only_heads_excluded",
        ),
    )
    registered_execution = require_sha256(
        report["registered_execution_id"], name=f"{name} registered execution",
    )
    if (
        report["selection_sha256"] != selection_sha256
        or selection["parents"].get("execution") != registered_execution
        or extraction["parents"].get("execution") != registered_execution
        or extraction["payload"].get("registered_execution_id") != registered_execution
        or extraction["payload"].get("node_id") != report["node_id"]
        or extraction["payload"].get("student_domain")
        != report.get("student_domain")
        or extraction["payload"].get("deployment_authorized")
        is not report.get("deployment_authorized")
        or extraction["payload"].get("strict_hlt_only")
        is not report.get("deployment_authorized")
        or extraction["payload"].get("training_only_heads_excluded") is not True
        or selection["parents"].get("selected_training_checkpoint")
        != report["selected_training_checkpoint_sha256"]
    ):
        raise ValueError(f"{name} report/selector/extraction lineage differs")
    envelope_id = require_sha256(
        extraction["payload"]["selected_envelope_id"],
        name=f"{name} selected envelope ID",
    )
    envelope_root = execution / "checkpoints" / "selected"
    commit = load_json(envelope_root / "committed" / envelope_id / "commit.json")
    from .hcwdl_representation_artifacts import validate_binary_envelope
    from .hcwdl_representation_training import SELECTED_TRAINING_CHECKPOINT_CONTRACT

    envelope = validate_binary_envelope(
        envelope_root, envelope_id,
        expected_contract=SELECTED_TRAINING_CHECKPOINT_CONTRACT,
        expected_parents=commit["parents"],
    )
    checkpoint = envelope.directory / "deployable_state.pt"
    digest = require_sha256(
        extraction["payload"]["deployable_state_sha256"],
        name=f"{name} deployable checkpoint",
    )
    selected_summary = report["checkpoint_envelopes"]["selected"]
    if (
        extraction["parents"].get("selected_envelope") != envelope.commit["content_hash"]
        or selected_summary.get("envelope_id") != envelope_id
        or selected_summary.get("commit_sha256") != envelope.commit["content_hash"]
        or report["deployable_extraction"].get("checkpoint_sha256") != digest
        or extraction["parents"].get("deployable_state") != digest
        or not checkpoint.is_file() or sha256_file(checkpoint) != digest
    ):
        raise PermissionError(f"{name} selected/deployable envelope lineage differs")
    return {
        "checkpoint": checkpoint, "checkpoint_sha256": digest,
        "report": report, "report_sha256": report_sha256,
        "selection": selection, "selection_sha256": selection_sha256,
        "extraction": extraction, "extraction_sha256": extraction_sha256,
        "registered_execution_id": registered_execution,
    }


def _validate_finalist_model_source_identity(
    source: Mapping[str, Any], finalist: Mapping[str, Any], *, name: str,
    evidence: Mapping[str, Any],
) -> None:
    """Match every frozen finalist identity, never just checkpoint bytes."""

    checkpoint_sha256 = require_sha256(
        finalist.get("checkpoint_sha256"), name=f"{name} finalist checkpoint",
    )
    report_sha256 = require_sha256(
        finalist.get("report_sha256"), name=f"{name} finalist report",
    )
    if evidence["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"{name} checkpoint differs from finalist lock")
    if source.get("kind") == "pmard":
        source_report = evidence["source"]
        if source_report.get("content_hash") != report_sha256:
            raise ValueError(f"{name} report differs from finalist lock")
        if (
            finalist.get("execution_id") is not None
            or finalist.get("checkpoint_selection_sha256") is not None
        ):
            raise ValueError(f"{name} PMARD finalist claims HCWDL execution identities")
        expected_extraction = report_sha256 if finalist.get("deployable") is True else None
        if finalist.get("extraction_sha256") != expected_extraction:
            raise ValueError(f"{name} PMARD extraction identity differs")
        return
    if source.get("kind") != "hcwdl":
        raise ValueError(f"{name} finalist model-source kind differs")
    if (
        evidence["report_sha256"] != report_sha256
        or evidence["selection_sha256"]
        != require_sha256(
            finalist.get("checkpoint_selection_sha256"),
            name=f"{name} finalist checkpoint selection",
        )
        or evidence["extraction_sha256"]
        != require_sha256(
            finalist.get("extraction_sha256"), name=f"{name} finalist extraction",
        )
        or evidence["registered_execution_id"]
        != require_sha256(
            finalist.get("execution_id"), name=f"{name} finalist execution",
        )
    ):
        raise ValueError(f"{name} HCWDL finalist artifact lineage differs")


def _source_checkpoint(
    source: Mapping[str, Any], *, name: str,
    expected_finalist: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    if set(source) not in (
        {"kind", "report"}, {"kind", "checkpoint"},
        {"kind", "execution_directory"},
    ):
        raise ValueError(f"{name} model-source fields differ")
    kind = str(source["kind"])
    if kind == "pmard" and set(source) == {"kind", "report"}:
        evidence = _pmard_report_chain(
            source["report"], name=f"{name} PMARD report",
        )
        checkpoint = evidence["checkpoint"]
        digest = evidence["checkpoint_sha256"]
    elif kind == "hcwdl" and set(source) == {"kind", "checkpoint"}:
        if expected_finalist is not None:
            raise ValueError(
                f"{name} finalist source lacks report/selector/extraction evidence"
            )
        checkpoint = _reference(
            source["checkpoint"], name=f"{name} HCWDL checkpoint", json_value=False,
        )
        digest = require_sha256(
            source["checkpoint"]["sha256"], name=f"{name} HCWDL checkpoint",
        )
    elif kind == "hcwdl" and set(source) == {"kind", "execution_directory"}:
        execution = _registered_input_path(
            source["execution_directory"], name=f"{name} HCWDL execution directory",
        )
        evidence = _hcwdl_source_evidence(execution, name=name)
        checkpoint = evidence["checkpoint"]
        digest = evidence["checkpoint_sha256"]
    else:
        raise ValueError(f"{name} model-source kind differs")
    if not checkpoint.is_file() or sha256_file(checkpoint) != digest:
        raise ValueError(f"{name} checkpoint bytes differ")
    if expected_finalist is not None:
        _validate_finalist_model_source_identity(
            source, expected_finalist, name=name, evidence=evidence,
        )
    return checkpoint, digest


def _load_model_source(
    source: Mapping[str, Any], *, name: str, device: str,
    expected_finalist: Mapping[str, Any] | None = None,
):
    checkpoint, digest = _source_checkpoint(
        source, name=name, expected_finalist=expected_finalist,
    )
    if source["kind"] == "pmard":
        model, _ = _load_pmard_teacher(source["report"], device=device)
    else:
        from hlt_classification.models.hcwdl_representation import (
            load_hcwdl_deployable_checkpoint,
        )
        model = load_hcwdl_deployable_checkpoint(checkpoint, expected_sha256=digest)
        model.to(device).float().eval()
    return model, checkpoint, digest


def _load_shuffle_joiner(value: Mapping[str, Any], *, target_bank: Any):
    legacy = {"root", "envelope_id", "expected_parents", "owner_id"}
    committed = {"committed_directory"}
    if set(value) not in (legacy, committed):
        raise ValueError("shuffle-map envelope reference fields differ")
    from hlt_classification.data.cache_contracts import load_npz_arrays
    from .hcwdl_representation_artifacts import validate_binary_envelope
    from .hcwdl_representation_contracts import SHUFFLE_MAP_CONTRACT, build_versioned_artifact
    from .hcwdl_representation_controls import (
        apply_representation_shuffle, validate_within_class_shuffle_map,
    )

    if set(value) == committed:
        directory, root, envelope_id = _committed_coordinates(
            value["committed_directory"], name="shuffle-map committed envelope",
        )
        commit = load_json(directory / "commit.json")
        parents = _exact_mapping(commit["parents"], name="shuffle-map committed parents")
        owner_id = None
    else:
        root = _registered_input_path(value["root"], name="shuffle-map envelope root")
        envelope_id = str(value["envelope_id"])
        parents = _exact_mapping(value["expected_parents"], name="shuffle-map parents")
        owner_id = None if value["owner_id"] is None else str(value["owner_id"])
    envelope = validate_binary_envelope(
        root, envelope_id,
        expected_contract=SHUFFLE_MAP_CONTRACT, expected_parents=parents,
        expected_owner_id=owner_id,
    )
    payload = dict(envelope.sidecar["payload"])
    source_hash = require_sha256(
        payload.pop("source_shuffle_artifact_sha256"), name="source shuffle artifact",
    )
    artifact = build_versioned_artifact(
        SHUFFLE_MAP_CONTRACT, parents=parents, payload=payload,
    )
    if artifact["content_hash"] != source_hash:
        raise ValueError("shuffle-map source artifact cannot be reconstructed exactly")
    arrays = load_npz_arrays(envelope.directory / "shuffle_map.npz")
    if set(arrays) != {"target_index"}:
        raise ValueError("shuffle-map payload members differ")
    identities = [bytes(row).hex() for row in target_bank.arrays["identity_digest"]]
    labels = np.asarray(target_bank.arrays["label"], dtype=np.int64)
    mapping = np.asarray(arrays["target_index"])
    validate_within_class_shuffle_map(
        artifact, mapping, identity_sha256=identities, labels=labels,
    )
    shuffled = apply_representation_shuffle(target_bank.arrays, mapping)
    lookup = {
        bytes(identity): index
        for index, identity in enumerate(target_bank.arrays["identity_digest"])
    }
    representation_names = set(shuffled) - {
        "logits", "identity_digest", "label", "source_file_id", "source_entry",
    }

    def join(identity_digests: np.ndarray) -> dict[str, np.ndarray]:
        keys = [bytes(row) for row in np.asarray(identity_digests)]
        if len(keys) != len(set(keys)):
            raise ValueError("shuffle join repeats a requested identity")
        try:
            indexes = np.asarray([lookup[key] for key in keys], dtype=np.int64)
        except KeyError as error:
            raise KeyError("shuffle join identity coverage is incomplete") from error
        return {
            name: np.ascontiguousarray(shuffled[name][indexes])
            for name in representation_names
        }

    return join, envelope.sidecar["content_hash"]


def _training_output_directory(task: Any, runtime_row: Mapping[str, Any]) -> Path:
    outputs = _runtime_helpers()["outputs"](task, runtime_row)
    required_names = {
        "training_report.json", "checkpoint_selection.json", "deployable_extraction.json",
    }
    by_name = {path.name: path for path in outputs.values() if path.name in required_names}
    if set(by_name) != required_names or len({path.parent for path in by_name.values()}) != 1:
        raise ValueError("training registered outputs do not bind one canonical directory")
    return next(iter(by_name.values())).parent


def training_adapter(spec, task, index, runtime_row):
    del index
    value = _assembly(
        task, runtime_row, contract=TRAINING_ASSEMBLY_CONTRACT,
        required=(
            "parent_recipe", "representation_recipe", "split_manifest",
            "row_selection", "data_root", "train_rows", "validation_rows",
            "target", "kernel_envelope", "execution_id", "registered_execution_id",
            "replicate_seed", "mode", "synthetic_passes", "resume_lineage",
            "producer_runtime_signature", "architecture_attestation",
            "parent_import", "model_sources", "shuffle_map", "view_cache_max_gib",
            "assignment_manifests",
            "registered_output_row", "publication_owner", "confirmation_registry",
        ),
        optional=(
            "acceptance_full_loss_binding", "acceptance_row_selection_sha256",
        ),
    )
    parent_recipe = _versioned_reference(value["parent_recipe"], name="parent recipe")
    recipe = _versioned_reference(value["representation_recipe"], name="representation recipe")
    split = _versioned_reference(value["split_manifest"], name="split manifest")
    selection = _versioned_reference(value["row_selection"], name="row selection")
    acceptance_selection_sha256 = value.get("acceptance_row_selection_sha256")
    if acceptance_selection_sha256 is not None and (
        require_sha256(
            acceptance_selection_sha256,
            name="acceptance row selection SHA-256",
        ) != selection["content_hash"]
    ):
        raise PermissionError("acceptance row selection lineage differs")
    runtime_signature = _versioned_reference(
        value["producer_runtime_signature"], name="producer runtime signature",
    )
    architecture = _versioned_reference(
        value["architecture_attestation"], name="architecture attestation",
    )
    parent_import = _versioned_reference(
        value["parent_import"], name="training parent import",
    )
    from .hcwdl_representation_training import (
        paired_rng_streams, resolve_node_execution,
        train_hcwdl_representation_node,
    )
    from .hcwdl_representation_targets import RepresentationTargetBank
    from .hcwdl_representation_data import training_batch_from_parent

    execution_id = str(value["execution_id"])
    execution = resolve_node_execution(execution_id)
    raw_assignments = value["assignment_manifests"]
    if execution.student_domain == "hlt":
        if raw_assignments is not None:
            raise ValueError("HLT training assembly unexpectedly binds assignments")
        assignment_manifests = None
        student_view = "hlt"
    else:
        if not isinstance(raw_assignments, Mapping) or set(raw_assignments) != {
            "train", "validation",
        }:
            raise ValueError("repaired-view training assembly lacks assignments")
        assignment_manifests = {
            role: _reference(
                raw_assignments[role], name=f"{role} assignment manifest",
                json_value=False,
            )
            for role in ("train", "validation")
        }
        student_view = "privileged"
    model_sources = _exact_mapping(value["model_sources"], name="training model sources")
    required_sources = {
        source for source in (
            execution.initialization_parent, execution.predecessor_logit_teacher,
        ) if source is not None
    }
    if set(model_sources) != required_sources:
        raise ValueError("training model-source registry differs from graph semantics")
    for source_name in sorted(required_sources):
        source = model_sources[source_name]
        if isinstance(source, Mapping) and source.get("kind") == "pmard":
            _validate_imported_pmard_source(
                source["report"], node_id=source_name,
                parent_import=parent_import, architecture=architecture,
                name=f"{source_name} training parent",
            )
    target = value["target"]
    legacy_target = {"bank_root", "generation_id", "logical_target_sha256"}
    committed_target = {"committed_directory"}
    if not isinstance(target, Mapping) or set(target) not in (
        legacy_target, committed_target,
    ):
        raise ValueError("training target-bank reference fields differ")
    if set(target) == committed_target:
        target_directory, target_root, generation_id = _committed_coordinates(
            target["committed_directory"],
            name="training committed target generation", parent_name="generations",
        )
    else:
        target_root = _registered_input_path(
            target["bank_root"], name="training target-bank root",
        )
        generation_id = str(target["generation_id"])
    strategy = "JET_ONLY" if execution.jet_only else execution.short_strategy
    load_started = time.perf_counter()
    target_bank = RepresentationTargetBank.load(
        target_root, generation_id, strategy=strategy,
        expected_logical_target_sha256=(
            None if "logical_target_sha256" not in target
            else str(target["logical_target_sha256"])
        ),
    )
    derived_target_generation = require_sha256(
        target_bank.manifest["parents"]["target_generation"],
        name="authenticated target generation",
    )
    derived_target_logical = require_sha256(
        target_bank.manifest["payload"]["logical_target_sha256"],
        name="authenticated target logical identity",
    )
    raw_resume_lineage = value["resume_lineage"]
    if not isinstance(raw_resume_lineage, Mapping):
        raise ValueError("training resume lineage differs")
    required_static_lineage = {
        "ascent_graph", "execution", "producer_runtime_signature",
        "representation_recipe",
    }
    if set(raw_resume_lineage) not in (
        required_static_lineage,
        required_static_lineage | {"target_generation", "target_logical"},
    ):
        raise ValueError("training resume lineage fields differ")
    resume_lineage = dict(raw_resume_lineage)
    for name, derived in (
        ("target_generation", derived_target_generation),
        ("target_logical", derived_target_logical),
    ):
        if name in resume_lineage and resume_lineage[name] != derived:
            raise PermissionError(f"training {name} differs from authenticated target")
        resume_lineage[name] = derived
    if str(value["registered_execution_id"]) != resume_lineage.get("execution"):
        raise ValueError("training registered execution differs from resume lineage")
    target_load_seconds = time.perf_counter() - load_started
    caches, cache_seconds = _build_training_view_caches(
        split=split, selection=selection, data_root=value["data_root"],
        train_rows=int(value["train_rows"]),
        validation_rows=int(value["validation_rows"]),
        lineage={
            "split_manifest": split["content_hash"],
            "row_selection": selection["content_hash"],
        },
        max_gib=float(value["view_cache_max_gib"]),
        student_domain=execution.student_domain,
        assignment_manifests=assignment_manifests,
    )
    seed = int(value["replicate_seed"])
    rng_streams = paired_rng_streams(execution_id, seed)
    sampler_seed = int(rng_streams["streams"]["sampler"])
    calibration_rows = min(4096, int(value["train_rows"]))
    if calibration_rows < 256:
        raise ValueError("training cache lacks the fixed 256-row diagnostic boundary")
    calibration, calibration_selection = _calibration_population(
        caches["train"], sampler_seed=sampler_seed, rows=calibration_rows,
        campaign_sha256=str(spec["content_hash"]),
        parent_logit_counterpart_node_id=execution.parent_counterpart,
        student_view=student_view,
    )
    calibration_batches_count = math.ceil(calibration_rows / 256)

    def train_batches(epoch: int, start_batch: int):
        for batch_index, raw in enumerate(caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed,
            batch_size=int(parent_recipe["batching"]["effective_batch_size"]),
        )):
            if batch_index >= start_batch:
                yield training_batch_from_parent(raw, student_view=student_view)

    def validation_batches():
        for raw in caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed,
            batch_size=int(parent_recipe["batching"]["effective_batch_size"]),
        ):
            yield training_batch_from_parent(raw, student_view=student_view)

    def calibration_batches(_phase: str):
        for start in range(0, calibration_rows, 256):
            yield _training_take(
                calibration, np.arange(start, min(start + 256, calibration_rows)),
            )

    def diagnostic_batches():
        yield _training_take(calibration, np.arange(0, 256))

    warm_checkpoint = warm_digest = None
    if execution.initialization_parent is not None:
        warm_checkpoint, warm_digest = _source_checkpoint(
            model_sources[execution.initialization_parent],
            name=execution.initialization_parent,
        )

    def warm_loader(path: str | Path, digest: str):
        name = execution.initialization_parent
        if name is None:
            raise ValueError("cold execution requested a warm loader")
        model, expected_path, expected_digest = _load_model_source(
            model_sources[name], name=name, device=str(runtime_row["device"]),
        )
        if Path(path).resolve() != expected_path.resolve() or digest != expected_digest:
            raise ValueError("warm initialization checkpoint binding differs")
        return model

    def predecessor_loader(name: str):
        if name != execution.predecessor_logit_teacher:
            raise ValueError("predecessor loader requested an unregistered model")
        return _load_model_source(
            model_sources[name], name=name, device=str(runtime_row["device"]),
        )[0]

    shuffle_joiner = None
    shuffle_hash = None
    if execution.shuffled_representation_targets:
        if not isinstance(value["shuffle_map"], Mapping):
            raise ValueError("shuffled control lacks its fixed shuffle-map envelope")
        shuffle_joiner, shuffle_hash = _load_shuffle_joiner(
            value["shuffle_map"], target_bank=target_bank,
        )
    elif value["shuffle_map"] is not None:
        raise ValueError("unshuffled execution unexpectedly binds a shuffle map")
    bundle = _load_kernel_bundle(value["kernel_envelope"])
    output = _training_output_directory(task, runtime_row)
    report = train_hcwdl_representation_node(
        execution_id=execution_id, parent_recipe=parent_recipe,
        representation_recipe=recipe, campaign_sha256=str(spec["content_hash"]),
        train_rows=int(value["train_rows"]),
        replicate_seed=seed, train_batches=train_batches,
        validation_batches=validation_batches, target_bank=target_bank,
        target_cache_diagnostics={
            "construction_seconds": float(
                target_bank.execution_attestation["payload"][
                    "construction_seconds"
                ]
            ),
            "load_seconds": target_load_seconds,
            "hlt_view_cache_construction_seconds": cache_seconds,
            "generation_sha256": resume_lineage["target_generation"],
            "logical_sha256": resume_lineage["target_logical"],
            "manifest_sha256": target_bank.manifest["content_hash"],
            "source": "authenticated_disk_target_then_process_local_ram",
            **(
                {"row_selection_sha256": selection["content_hash"]}
                if acceptance_selection_sha256 is not None
                else {}
            ),
        },
        token_resources=bundle.token, relation_resources=bundle.relation,
        output_dir=output, resume_lineage=resume_lineage,
        producer_runtime_signature=runtime_signature,
        architecture_attestation_sha256=architecture["content_hash"],
        device=str(runtime_row["device"]), mode=str(value["mode"]),
        synthetic_passes=int(value["synthetic_passes"]),
        warm_checkpoint=warm_checkpoint, warm_checkpoint_sha256=warm_digest,
        warm_loader=warm_loader if execution.initialization_parent is not None else None,
        predecessor_model_loader=(
            predecessor_loader if execution.predecessor_logit_teacher is not None else None
        ),
        predecessor_batches=lambda: train_batches(0, 0),
        shuffled_representation_joiner=shuffle_joiner,
        shuffle_map_sha256=shuffle_hash,
        calibration_batches=calibration_batches,
        calibration_selection=calibration_selection,
        calibration_expected_batches=calibration_batches_count,
        calibration_minimum_valid_batches=min(12, calibration_batches_count),
        diagnostic_batches=diagnostic_batches,
        acceptance_full_loss_binding=value.get("acceptance_full_loss_binding"),
        preemption_requested=runtime_row.get("_preemption_requested"),
        preemption_wait_after_update=runtime_row.get(
            "_preemption_wait_after_update"
        ),
        preemption_wait=runtime_row.get("_preemption_wait"),
        registered_output_row=value["registered_output_row"],
        publication_owner=value["publication_owner"],
    )
    if str(value["mode"]) == "scientific" and value["confirmation_registry"] is not None:
        registry = _versioned_reference(
            value["confirmation_registry"], name="confirmation registry",
        )
        from .hcwdl_representation_reporting import build_confirmation_run
        pointer = build_confirmation_run(registry=registry, training_report=report)
        outputs = _runtime_helpers()["outputs"](task, runtime_row)
        matches = [path for logical, path in outputs.items() if "confirmation/runs/" in logical]
        if len(matches) != 1:
            raise ValueError("confirmation row lacks its exact run-pointer output")
        _runtime_helpers()["publish"](matches[0], pointer)
    elif value["confirmation_registry"] is not None:
        raise ValueError("non-scientific training unexpectedly binds confirmation registry")
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation=str(getattr(task, "kind")),
    )


def _final_capability(
    *, claim_reference: object, registry_reference: object, task_id: str,
    execution_lock_reference: object | None,
):
    claim = _versioned_reference(claim_reference, name="final execution claim")
    registry = _versioned_reference(registry_reference, name="final task registry")
    execution = None
    execution_hash = None
    if execution_lock_reference is not None:
        execution = _versioned_reference(
            execution_lock_reference, name="final execution lock",
        )
        execution_hash = execution["content_hash"]
    from .hcwdl_shared_final import issue_role_capability
    capability = issue_role_capability(
        claim=claim, task_registry=registry, task_id=task_id,
        execution_lock_sha256=execution_hash,
    )
    return claim, registry, execution, capability


def _publish_capability_output(
    task: Any, runtime_row: Mapping[str, Any], capability: Mapping[str, Any],
) -> Path:
    outputs = _runtime_helpers()["outputs"](task, runtime_row)
    matches = [
        path for logical, path in outputs.items()
        if "capabilit" in logical.lower()
    ]
    if len(matches) != 1:
        raise ValueError("branch-opening task lacks one registered capability output")
    _runtime_helpers()["publish"](matches[0], capability)
    return matches[0]


def _scan_final_selection(
    *, split: Mapping[str, Any], population: Mapping[str, Any], data_root: str | Path,
    capability: Mapping[str, Any], execution_claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], task_id: str, step_size: int,
):
    from .hcwdl_final_stream import SELECTION_BRANCHES, validate_projected_branches
    from .labels import multiclass_labels
    from .schema import TREE_NAME, baseline_mask
    from .splits import role_records
    from .streaming import iterate_projected_chunks

    branches = validate_projected_branches(path="selection", branches=SELECTION_BRANCHES)
    population_records = {
        (str(row["source_file_sha256"]), int(row["source_entry"])): {
            **dict(row), "identity_digest": digest,
        }
        for row, digest in zip(
            population["identity_records"], population["identity_digests"], strict=True,
        )
    }
    split_records = {record.path: record for record in role_records(split, "final_test")}
    identities: list[str] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    ranks: list[int] = []
    accesses = []
    for source_path in sorted(split_records):
        source = split_records[source_path]
        source_access = []
        for chunk in iterate_projected_chunks(
            (Path(data_root) / source_path,), branches, data_root=data_root,
            role="final_test", shared_final_capability=capability,
            shared_final_claim=execution_claim,
            shared_final_task_registry=task_registry,
            final_population_sha256=population["population_sha256"],
            final_task_id=task_id, final_branch_family="selection",
            shared_reservation_active=True,
            step_size=step_size,
        ):
            target = multiclass_labels(chunk.arrays)
            indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (target >= 0))
            for index in indexes:
                entry = chunk.entry_start + int(index)
                key = (source.sha256, entry)
                if key not in population_records:
                    raise ValueError("selection scan contains a jet outside the registered population")
                row = population_records[key]
                identities.append(str(row["identity_digest"]))
                labels.append(int(target[index]))
                records.append({
                    "identity_digest": row["identity_digest"],
                    "source_path": source_path,
                    "source_file_sha256": source.sha256,
                    "source_entry": entry,
                })
                rank_payload = (
                    f"pmard-row-selection/v1/1337/final_test/{source_path}/{entry}"
                ).encode("utf-8")
                ranks.append(int.from_bytes(hashlib.sha256(rank_payload).digest()[:16], "big"))
            source_access.append({
                "source_path": source_path, "source_file_sha256": source.sha256,
                "tree": TREE_NAME, "entry_start": chunk.entry_start,
                "entry_stop": chunk.entry_stop,
            })
        accesses.extend(source_access)
    if set(identities) != set(population["identity_digests"]) or len(identities) != len(
        population["identity_digests"]
    ):
        raise ValueError("selection scan does not exactly cover the registered population")
    from .hcwdl_final_stream import build_branch_access_record
    branch_access = build_branch_access_record(
        path="selection", capability_sha256=capability["content_hash"],
        branches=branches, source_rows=accesses,
        population_sha256=population["population_sha256"], task_id=task_id,
        execution_lock_sha256=None,
    )
    return identities, np.asarray(labels, dtype=np.int64), records, ranks, branch_access


def final_selection_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_SELECTION_ASSEMBLY_CONTRACT,
        required=(
            "split_manifest", "population", "claim", "task_registry", "task_id",
            "data_root", "rows_per_class", "selection_rule_sha256", "step_size",
            "escrow_root", "producer_task_id", "registered_output_row",
            "publication_owner",
        ),
    )
    split = _versioned_reference(value["split_manifest"], name="final split manifest")
    population = _versioned_reference(value["population"], name="final population")
    execution_claim, task_registry, _, capability = _final_capability(
        claim_reference=value["claim"], registry_reference=value["task_registry"],
        task_id=str(value["task_id"]), execution_lock_reference=None,
    )
    capability_path = _publish_capability_output(task, runtime_row, capability)
    quotas = value["rows_per_class"]
    if not isinstance(quotas, list) or len(quotas) != 15 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in quotas
    ) or sum(quotas) <= 0:
        raise ValueError("final selection class quotas differ")
    identities, labels, records, ranks, branch_access = _scan_final_selection(
        split=split, population=population, data_root=value["data_root"],
        capability=capability, execution_claim=execution_claim,
        task_registry=task_registry, task_id=str(value["task_id"]),
        step_size=int(value["step_size"]),
    )
    from .hcwdl_final_stream import class_stratified_selection, publish_label_escrow
    selection, escrow_arrays = class_stratified_selection(
        identities=identities, labels=labels, rows_per_class=quotas,
        population_sha256=population["population_sha256"],
        selection_rule_sha256=str(value["selection_rule_sha256"]),
        capability=capability, execution_claim=execution_claim,
        task_registry=task_registry, task_id=str(value["task_id"]),
        identity_records=records, selection_ranks=ranks,
        expected_population_identity_digests=population["identity_digests"],
    )
    # Branch evidence is deliberately outside the label escrow and proves the
    # sole final-label projection was the registered selection capability.
    if branch_access["label_free"] is not False:
        raise ValueError("selection branch-access evidence differs")
    outputs = _runtime_helpers()["outputs"](task, runtime_row)
    files = [
        path for logical, path in outputs.items()
        if "row_selection" in logical
    ]
    branch_paths = [
        path for logical, path in outputs.items()
        if "selection/branch_access" in logical
    ]
    directories = [
        path for logical, path in outputs.items()
        if "label_escrow" in logical
    ]
    if (
        len(files) != 1 or len(branch_paths) != 1 or len(directories) != 1
        or capability_path in files + branch_paths + directories
    ):
        raise ValueError("final selection output registry differs")
    _runtime_helpers()["publish"](files[0], selection)
    _runtime_helpers()["publish"](branch_paths[0], branch_access)
    envelope = publish_label_escrow(
        value["escrow_root"], arrays=escrow_arrays,
        selection_sha256=selection["content_hash"],
        population_sha256=population["population_sha256"],
        capability_sha256=capability["content_hash"],
        producer_task_id=str(value["producer_task_id"]),
        registered_output_row=value["registered_output_row"],
        campaign_or_recovery_owner=value["publication_owner"],
    )
    escrow_logical = next(
        logical for logical in getattr(task, "registered_outputs", ())
        if "label_escrow" in logical
    )
    if not _runtime_helpers()["published_matches"](
        task, runtime_row, escrow_logical, envelope.directory,
    ):
        raise ValueError("label escrow envelope differs from registered output")
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="final_selection",
    )


def assignment_shard_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_ASSIGNMENT_ASSEMBLY_CONTRACT,
        required=(
            "split_manifest", "selection", "matcher_resources", "claim",
            "task_registry", "task_id", "data_root", "source_path", "step_size",
            "assignment_spec_sha256",
            "envelope_root", "producer_task_id", "registered_output_row",
            "publication_owner",
        ),
    )
    split = _versioned_reference(value["split_manifest"], name="assignment split")
    selection = _versioned_reference(value["selection"], name="final selection")
    resources = _versioned_reference(value["matcher_resources"], name="matcher resources")
    from .hcwdl_representation_final_policy import build_final_assignment_spec
    from .splits import role_records
    assignment_spec = build_final_assignment_spec(
        matcher_resources=resources,
        source_partitions=[record.path for record in role_records(split, "final_test")],
        step_size=int(value["step_size"]),
    )
    if assignment_spec["content_hash"] != require_sha256(
        value["assignment_spec_sha256"], name="final assignment specification",
    ):
        raise PermissionError("assignment worker specification differs from reservation")
    execution_claim, task_registry, _, capability = _final_capability(
        claim_reference=value["claim"], registry_reference=value["task_registry"],
        task_id=str(value["task_id"]), execution_lock_reference=None,
    )
    capability_path = _publish_capability_output(task, runtime_row, capability)
    from .hcwdl_assignment import build_shared_final_assignment_rows
    rows = build_shared_final_assignment_rows(
        split_manifest=split, selection_manifest=selection,
        resources_report=resources, data_root=value["data_root"],
        source_path=str(value["source_path"]), capability=capability,
        execution_claim=execution_claim, task_registry=task_registry,
        population_sha256=selection["population_sha256"],
        task_id=str(value["task_id"]), step_size=int(value["step_size"]),
    )
    expected_parents = {
        "split_manifest": split["content_hash"],
        "selection": selection["content_hash"],
        "matcher_resources": resources["content_hash"],
        "capability": capability["content_hash"],
        "branch_access": rows["branch_access"]["content_hash"],
    }
    parents = expected_parents
    from .highcov_cache import publish_assignment_shard
    from .hcwdl_representation_artifacts import publish_binary_envelope
    with tempfile.TemporaryDirectory(prefix="hcwdl-final-assignment-") as directory:
        base = Path(directory) / "assignment"
        metadata = publish_assignment_shard(
            base, source_path=rows["source_path"], role="final_test", source_fold=None,
            entries=rows["entries"], hlt_categories=rows["categories"],
            results=rows["results"], parents=parents,
        )
        envelope = publish_binary_envelope(
            value["envelope_root"],
            artifact_contract=FINAL_ASSIGNMENT_ENVELOPE_CONTRACT,
            producer_task_id=str(value["producer_task_id"]),
            schema={
                "metadata": "HIGHCOV_DENSE_ASSIGNMENT_SHARD/v2",
                "arrays": "deterministic_npz",
            },
            immutable_parent_hashes=parents,
            registered_output_row=value["registered_output_row"],
            campaign_or_recovery_owner=value["publication_owner"],
            payloads={
                "assignment.json": base.with_suffix(".json").read_bytes(),
                "assignment.npz": base.with_suffix(".npz").read_bytes(),
            },
            member_metadata={
                "assignment.json": {
                    "logical_sha256": metadata["content_hash"], "dtype": "json",
                    "shape": [int(metadata["rows"])],
                },
                "assignment.npz": {
                    "logical_sha256": canonical_sha256(metadata["array_sha256"]),
                    "dtype": "npz", "shape": [int(metadata["rows"])],
                },
            },
            sidecar_payload={
                "source_path": rows["source_path"],
                "source_file_sha256": rows["source_file_sha256"],
                "assignment_metadata_sha256": metadata["content_hash"],
                "rows": int(metadata["rows"]),
                "branch_access_sha256": rows["branch_access"]["content_hash"],
            },
            branch_access=rows["branch_access"],
        )
    envelope_outputs = [
        path for path in _runtime_helpers()["outputs"](task, runtime_row).values()
        if path != capability_path
    ]
    if len(envelope_outputs) != 1:
        raise ValueError("assignment-shard envelope output registry differs")
    envelope_logical = next(
        logical for logical in getattr(task, "registered_outputs", ())
        if "committed/${envelope_id}" in logical
    )
    if not _runtime_helpers()["published_matches"](
        task, runtime_row, envelope_logical, envelope.directory,
    ):
        raise ValueError("assignment envelope differs from registered output")
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="assignment_shard",
    )


def _assignment_envelope(value: Mapping[str, Any]):
    legacy = {"root", "envelope_id", "expected_parents", "owner_id"}
    exact = {"committed_directory"}
    if set(value) not in (legacy, exact):
        raise ValueError("assignment envelope reference fields differ")
    from .hcwdl_representation_artifacts import validate_binary_envelope
    if set(value) == exact:
        directory, root, envelope_id = _committed_coordinates(
            value["committed_directory"], name="assignment committed envelope",
        )
        commit = load_json(directory / "commit.json")
        parents = _exact_mapping(commit["parents"], name="assignment committed parents")
        owner_id = None
    else:
        root = _registered_input_path(value["root"], name="assignment envelope root")
        envelope_id = str(value["envelope_id"])
        parents = _exact_mapping(
            value["expected_parents"], name="assignment envelope parents",
        )
        owner_id = None if value["owner_id"] is None else str(value["owner_id"])
    return validate_binary_envelope(
        root, envelope_id,
        expected_contract=FINAL_ASSIGNMENT_ENVELOPE_CONTRACT,
        expected_parents=parents, expected_owner_id=owner_id,
    )


def assignment_finalize_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_ASSIGNMENT_ASSEMBLY_CONTRACT,
        required=(
            "split_manifest", "selection", "matcher_resources", "claim",
            "task_registry", "assignment_spec_sha256", "shards",
            "expected_mapped_jets", "require_sub10pct_dustbins",
        ),
    )
    split = _versioned_reference(value["split_manifest"], name="assignment split")
    selection = _versioned_reference(value["selection"], name="final selection")
    resources = _versioned_reference(
        value["matcher_resources"], name="matcher resources",
    )
    claim = _versioned_reference(value["claim"], name="final execution claim")
    registry = _versioned_reference(value["task_registry"], name="final task registry")
    from .splits import role_records
    source_records = role_records(split, "final_test")
    source_partitions = [record.path for record in source_records]
    from .hcwdl_representation_final_policy import build_final_assignment_spec
    assignment_spec = build_final_assignment_spec(
        matcher_resources=resources, source_partitions=source_partitions,
    )
    assignment_spec_sha256 = require_sha256(
        value["assignment_spec_sha256"], name="final assignment specification",
    )
    if assignment_spec["content_hash"] != assignment_spec_sha256:
        raise PermissionError("assignment finalize specification differs from reservation")
    shards = value["shards"]
    if not isinstance(shards, list) or len(shards) != len(source_records):
        raise ValueError("assignment finalize shard registry differs from split")
    envelopes = [_assignment_envelope(row) for row in shards]
    selected_rows = selection.get("selected_rows")
    selected_identities = selection.get("identity_digests")
    if not isinstance(selected_rows, list) or not isinstance(selected_identities, list):
        raise ValueError("assignment selection identity registry differs")
    expected_by_source: dict[str, dict[int, str]] = {
        record.path: {} for record in source_records
    }
    source_hashes = {record.path: record.sha256 for record in source_records}
    for row, ordered_digest in zip(selected_rows, selected_identities, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("assignment selection row differs")
        source = str(row.get("source_path", ""))
        entry = row.get("source_entry")
        if (
            source not in expected_by_source or isinstance(entry, bool)
            or not isinstance(entry, int) or entry < 0
            or row.get("source_file_sha256") != source_hashes[source]
        ):
            raise ValueError("assignment selection source identity differs from split")
        identity = canonical_sha256({
            "source_file_sha256": source_hashes[source], "source_entry": entry,
        })
        if row.get("identity_digest") != identity or ordered_digest != identity:
            raise ValueError("assignment selection identity is not source-derived")
        if entry in expected_by_source[source]:
            raise ValueError("assignment selection repeats a source entry")
        expected_by_source[source][entry] = identity

    registry_tasks = registry.get("tasks")
    if not isinstance(registry_tasks, list):
        raise ValueError("final task registry lacks assignment tasks")
    from .hcwdl_shared_final import issue_role_capability
    from .highcov_cache import load_assignment_shard
    metadata_paths = []
    derived_identities = []
    for position, (source, envelope) in enumerate(zip(
        source_partitions, envelopes, strict=True,
    )):
        matches = [
            row for row in registry_tasks
            if isinstance(row, Mapping) and row.get("kind") == "assignment_shard"
            and row.get("source_partition") == source
        ]
        if len(matches) != 1:
            raise ValueError("final task registry assignment source differs")
        task_id = str(matches[0]["task_id"])
        capability = issue_role_capability(
            claim=claim, task_registry=registry, task_id=task_id,
            execution_lock_sha256=None,
        )
        branch_access = load_json(envelope.directory / "branch_access.json")
        branch_hash = validate_content_hash(
            branch_access, expected_contract=str(branch_access.get("contract")),
            expected_schema_version=1,
        )
        expected_parents = {
            "split_manifest": split["content_hash"],
            "selection": selection["content_hash"],
            "matcher_resources": resources["content_hash"],
            "capability": capability["content_hash"],
            "branch_access": branch_hash,
        }
        if envelope.commit.get("parents") != dict(sorted(expected_parents.items())):
            raise ValueError("assignment shard parent lineage differs")
        registered_row = envelope.commit["payload"]["registered_output_row"]
        if registered_row != {
            "task_key": "final_assignment_shards", "array_index": position,
            "registered_output": (
                "final/assignment/shards/${source_partition}/committed/${envelope_id}"
            ),
        }:
            raise ValueError("assignment shard registered task/output differs")
        metadata_path = envelope.directory / "assignment.json"
        metadata, arrays = load_assignment_shard(
            metadata_path, expected_parents=expected_parents,
        )
        if metadata.get("source_path") != source:
            raise ValueError("assignment shard source order differs from split")
        entries = [int(raw) for raw in arrays["entries"]]
        if entries != sorted(expected_by_source[source]):
            raise ValueError("assignment shard entries differ from selected rows")
        derived_identities.extend(expected_by_source[source][entry] for entry in entries)
        metadata_paths.append(metadata_path)
    if derived_identities != selected_identities:
        raise ValueError("assignment shard identities/order differ from selection")
    outputs = _runtime_helpers()["outputs"](task, runtime_row)
    manifest_paths = [path for path in outputs.values() if path.name == "manifest.json"]
    audit_paths = [path for path in outputs.values() if path.name == "audit.json"]
    if len(manifest_paths) != 1 or len(audit_paths) != 1:
        raise ValueError("assignment finalize output registry differs")
    parents = {
        "selection": selection["content_hash"],
        "assignment_spec": assignment_spec_sha256,
        **{
            f"assignment_shard_{position:04d}": envelope.commit["content_hash"]
            for position, envelope in enumerate(envelopes)
        },
    }
    expected_rows = int(value["expected_mapped_jets"])
    from .highcov_cache import publish_assignment_manifest, validate_assignment_manifest
    manifest = publish_assignment_manifest(
        manifest_paths[0], role="final_test", shard_metadata_paths=metadata_paths,
        expected_mapped_jets=expected_rows, parents=parents,
    )
    validate_assignment_manifest(
        manifest_paths[0], expected_role="final_test", expected_mapped_jets=expected_rows,
        expected_parents=parents,
        require_sub10pct_dustbins=bool(value["require_sub10pct_dustbins"]),
    )
    from .hcwdl_representation_final import build_assignment_audit
    audit = build_assignment_audit(
        selection=selection, assignment_manifest=manifest,
        assignment_spec=assignment_spec,
        assigned_identity_digests=derived_identities,
        population_sha256=selection["population_sha256"],
    )
    _runtime_helpers()["publish"](audit_paths[0], audit)
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="assignment_finalize",
    )


def _prediction_rows(
    *, model: Any, rows: Iterable[Any], domain: str, device: str, batch_size: int,
):
    import torch
    from .inputs import NativeOfflineInputs

    model.to(device).float().eval()
    if model.training or any(
        parameter.dtype != torch.float32 for parameter in model.parameters()
    ):
        raise ValueError("final prediction model is not eval-mode FP32")
    identities: list[str] = []
    logits: list[np.ndarray] = []
    pending = []

    def flush() -> None:
        if not pending:
            return
        views = [row.model_inputs for row in pending]
        if domain == "native_offline":
            if not all(isinstance(view, NativeOfflineInputs) for view in views):
                raise TypeError("native-offline finalist received an ordinary particle view")
            charged = _particle_concat([view.charged for view in views])
            neutral = _particle_concat([view.neutral for view in views])
            args = (
                charged.features, charged.vectors, charged.mask,
                neutral.features, neutral.vectors, neutral.mask,
            )
        else:
            if any(isinstance(view, NativeOfflineInputs) for view in views):
                raise TypeError("ordinary finalist received a native-offline particle view")
            ordinary = _particle_concat(views)
            args = (ordinary.features, ordinary.vectors, ordinary.mask)
        tensors = []
        for value in args:
            tensor = torch.as_tensor(value, device=device)
            tensors.append(tensor.float() if tensor.dtype.is_floating_point else tensor)
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type, enabled=False,
        ):
            output = model(*tensors).float()
        if output.shape != (len(pending), 15) or not torch.isfinite(output).all():
            raise FloatingPointError("final prediction model emitted invalid logits")
        identities.extend(str(row.identity_digest) for row in pending)
        logits.append(np.ascontiguousarray(output.cpu().numpy(), dtype=np.float32))
        pending.clear()

    for row in rows:
        pending.append(row)
        if len(pending) == batch_size:
            flush()
    flush()
    if not identities:
        raise ValueError("final prediction source emitted no selected rows")
    identity_array = np.asarray([
        np.frombuffer(bytes.fromhex(value), dtype=np.uint8) for value in identities
    ], dtype=np.uint8)
    values = np.ascontiguousarray(np.concatenate(logits, axis=0), dtype=np.float32)
    order = np.argsort(np.asarray(identities), kind="stable")
    return np.ascontiguousarray(identity_array[order]), np.ascontiguousarray(values[order])


def _validate_prediction_worker_runtime(
    *, frozen_signature: Mapping[str, Any], runtime_row: Mapping[str, Any],
) -> str:
    from .hcwdl_representation_final import validate_prediction_runtime_signature
    from .hcwdl_representation_worker_runtime import build_row_runtime_signature

    validate_prediction_runtime_signature(frozen_signature)
    frozen_row_sha256 = require_sha256(
        frozen_signature.get("row_runtime_signature_sha256"),
        name="frozen prediction row runtime",
    )
    registered_row_sha256 = require_sha256(
        runtime_row.get("runtime_signature_sha256"),
        name="registered prediction row runtime",
    )
    if frozen_row_sha256 != registered_row_sha256:
        raise PermissionError(
            "prediction specification runtime differs from registered live row"
        )
    live = runtime_row.get("_live_worker_runtime")
    if not isinstance(live, Mapping):
        raise PermissionError("prediction worker lacks measured live runtime evidence")
    if build_row_runtime_signature(live)["content_hash"] != frozen_row_sha256:
        raise PermissionError("prediction live runtime differs from frozen specification")
    if frozen_signature["device"] != str(runtime_row.get("device", "")):
        raise PermissionError("prediction device differs from frozen specification")
    return frozen_row_sha256


def _load_bound_final_assignment_store(
    path: Path, *, execution_lock: Mapping[str, Any],
    prediction_spec: Mapping[str, Any],
):
    from .highcov_cache import DenseAssignmentStore, validate_assignment_manifest

    parents = execution_lock.get("assignment_manifest_parents")
    if not isinstance(parents, Mapping):
        raise ValueError("execution lock lacks assignment manifest parents")
    manifest = validate_assignment_manifest(
        path, expected_role="final_test",
        expected_mapped_jets=int(execution_lock["row_count"]),
        expected_parents=parents, require_sub10pct_dustbins=True,
    )
    expected_manifest_sha256 = require_sha256(
        execution_lock.get("assignment_manifest_sha256"),
        name="execution-lock assignment manifest",
    )
    if (
        manifest.get("content_hash") != expected_manifest_sha256
        or prediction_spec.get("assignment_manifest_sha256")
        != expected_manifest_sha256
        or manifest.get("parents", {}).get("assignment_spec")
        != execution_lock.get("assignment_spec_sha256")
        or prediction_spec.get("assignment_spec_sha256")
        != execution_lock.get("assignment_spec_sha256")
    ):
        raise PermissionError("D100 assignment manifest/hash/spec differs from execution lock")
    source_counts = {
        str(row["source_path"]): int(row["rows"])
        for row in manifest.get("shards", ())
    }
    if source_counts != execution_lock.get("source_counts"):
        raise ValueError("D100 assignment manifest source counts differ from execution lock")
    return DenseAssignmentStore(path)


def prediction_shard_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_PREDICTION_ASSEMBLY_CONTRACT,
        required=(
            "split_manifest", "selection", "prediction_spec", "finalist_lock",
            "execution_lock", "claim", "task_registry", "task_id", "data_root",
            "source_partition", "finalist_id", "model_source",
            "assignment_manifest", "step_size", "producer_runtime_signature",
            "envelope_root", "producer_task_id", "registered_output_row",
            "publication_owner",
        ),
    )
    split = _versioned_reference(value["split_manifest"], name="prediction split")
    selection = _versioned_reference(value["selection"], name="prediction selection")
    prediction_spec = _versioned_reference(
        value["prediction_spec"], name="prediction specification",
    )
    finalist_lock = _versioned_reference(value["finalist_lock"], name="finalist lock")
    execution_claim, task_registry, execution_lock, capability = _final_capability(
        claim_reference=value["claim"], registry_reference=value["task_registry"],
        task_id=str(value["task_id"]),
        execution_lock_reference=value["execution_lock"],
    )
    capability_path = _publish_capability_output(task, runtime_row, capability)
    assert execution_lock is not None
    from .hcwdl_representation_final import validate_prediction_spec
    validate_prediction_spec(
        prediction_spec, finalist_lock=finalist_lock,
        execution_lock=execution_lock, row_selection=selection,
    )
    runtime_signature = prediction_spec["runtime_signature"]
    _validate_prediction_worker_runtime(
        frozen_signature=runtime_signature, runtime_row=runtime_row,
    )
    finalists = [
        row for row in finalist_lock["finalists"]
        if row["finalist_id"] == str(value["finalist_id"])
    ]
    if len(finalists) != 1:
        raise ValueError("prediction finalist is not uniquely frozen")
    finalist = finalists[0]
    if capability["task"].get("finalist_id") != finalist["finalist_id"] or capability[
        "task"
    ].get("checkpoint_sha256") != finalist["checkpoint_sha256"]:
        raise PermissionError("prediction capability finalist/checkpoint differs")
    model, _, checkpoint_sha256 = _load_model_source(
        value["model_source"], name=finalist["finalist_id"],
        device=str(runtime_row["device"]),
        expected_finalist=finalist,
    )
    if checkpoint_sha256 != finalist["checkpoint_sha256"]:
        raise ValueError("prediction model checkpoint differs from finalist lock")
    domain = str(finalist["domain"])
    assignment_store = None
    if domain == "shell_exact_d100":
        assignment_path = _reference(
            value["assignment_manifest"], name="final assignment manifest", json_value=False,
        )
        assignment_store = _load_bound_final_assignment_store(
            assignment_path, execution_lock=execution_lock,
            prediction_spec=prediction_spec,
        )
        stream_name = "shell_exact"
    elif domain == "native_offline":
        if value["assignment_manifest"] is not None:
            raise ValueError("native-offline prediction unexpectedly binds assignments")
        stream_name = "native_offline"
    elif domain == "hlt":
        if value["assignment_manifest"] is not None:
            raise ValueError("HLT prediction unexpectedly binds assignments")
        stream_name = "hlt"
    else:
        raise ValueError("prediction finalist domain differs")
    if (
        runtime_signature["autocast"] is not False
        or runtime_signature["tf32"] is not False
        or runtime_signature["deterministic_algorithms"] is not True
    ):
        raise ValueError("prediction worker runtime differs from prediction specification")
    _configure_target_backend()
    collectors: list[dict[str, Any]] = []
    common = dict(
        split_manifest=split, data_root=value["data_root"],
        population_sha256=execution_lock["population_sha256"],
        task_id=str(value["task_id"]), capability=capability,
        execution_claim=execution_claim, task_registry=task_registry,
        execution_lock_sha256=execution_lock["content_hash"], selection=selection,
        source_partition=str(value["source_partition"]),
        step_size=int(value["step_size"]), branch_access_collector=collectors,
    )
    from .hcwdl_final_stream import (
        iterate_final_hlt_inputs, iterate_final_native_offline_inputs,
        iterate_final_shell_exact_inputs,
    )
    if stream_name == "hlt":
        stream = iterate_final_hlt_inputs(**common)
    elif stream_name == "native_offline":
        stream = iterate_final_native_offline_inputs(**common)
    else:
        stream = iterate_final_shell_exact_inputs(
            **common, assignment_store=assignment_store,
        )
    identities, logits = _prediction_rows(
        model=model, rows=stream, domain=stream_name,
        device=str(runtime_row["device"]), batch_size=int(runtime_signature["batch_size"]),
    )
    if len(collectors) != 1:
        raise ValueError("prediction branch-access evidence differs")
    producer_signature = value["producer_runtime_signature"]
    if (
        not isinstance(producer_signature, Mapping)
        or dict(producer_signature) != dict(runtime_signature)
    ):
        raise ValueError(
            "prediction producer runtime differs from frozen prediction specification"
        )
    from .hcwdl_representation_final import publish_prediction_shard
    envelope = publish_prediction_shard(
        value["envelope_root"], finalist=finalist,
        source_partition=str(value["source_partition"]),
        identity_digests=identities, logits=logits,
        prediction_spec_sha256=prediction_spec["content_hash"],
        execution_lock_sha256=execution_lock["content_hash"],
        producer_runtime_signature=producer_signature,
        branch_access=collectors[0], producer_task_id=str(value["producer_task_id"]),
        registered_output_row=value["registered_output_row"],
        campaign_or_recovery_owner=value["publication_owner"],
    )
    envelope_outputs = [
        path for path in _runtime_helpers()["outputs"](task, runtime_row).values()
        if path != capability_path
    ]
    if len(envelope_outputs) != 1:
        raise ValueError("prediction-shard envelope output registry differs")
    envelope_logical = next(
        logical for logical in getattr(task, "registered_outputs", ())
        if "committed/${envelope_id}" in logical
    )
    if not _runtime_helpers()["published_matches"](
        task, runtime_row, envelope_logical, envelope.directory,
    ):
        raise ValueError("prediction envelope differs from registered output")
    del model
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="prediction_shard",
    )


def _load_prediction_envelope(value: Mapping[str, Any]):
    legacy = {
        "root", "envelope_id", "prediction_spec_sha256", "execution_lock_sha256",
        "checkpoint_sha256", "branch_access_sha256",
    }
    exact = {"committed_directory"}
    if set(value) not in (legacy, exact):
        raise ValueError("prediction-envelope reference fields differ")
    from .hcwdl_representation_final import load_prediction_shard
    if set(value) == exact:
        directory, root, envelope_id = _committed_coordinates(
            value["committed_directory"], name="prediction committed envelope",
        )
        commit = load_json(directory / "commit.json")
        parents = _exact_mapping(commit["parents"], name="prediction committed parents")
        if set(parents) != {
            "prediction_spec", "execution_lock", "checkpoint", "branch_access",
        }:
            raise ValueError("prediction committed parent registry differs")
    else:
        root = _registered_input_path(value["root"], name="prediction envelope root")
        envelope_id = str(value["envelope_id"])
        parents = {
            "prediction_spec": str(value["prediction_spec_sha256"]),
            "execution_lock": str(value["execution_lock_sha256"]),
            "checkpoint": str(value["checkpoint_sha256"]),
            "branch_access": str(value["branch_access_sha256"]),
        }
    return load_prediction_shard(
        root, envelope_id,
        prediction_spec_sha256=parents["prediction_spec"],
        execution_lock_sha256=parents["execution_lock"],
        checkpoint_sha256=parents["checkpoint"],
        branch_access_sha256=parents["branch_access"],
    )


def prediction_finalize_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_PREDICTION_ASSEMBLY_CONTRACT,
        required=(
            "selection", "prediction_spec", "execution_lock", "finalist",
            "shards", "expected_source_partitions",
        ),
    )
    selection = _versioned_reference(value["selection"], name="prediction selection")
    prediction_spec = _versioned_reference(
        value["prediction_spec"], name="prediction specification",
    )
    execution_lock = _versioned_reference(value["execution_lock"], name="execution lock")
    shards = value["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("prediction finalize shard registry is empty")
    loaded = [_load_prediction_envelope(row) for row in shards]
    from .hcwdl_representation_final import build_prediction_manifest
    manifest = build_prediction_manifest(
        finalist=value["finalist"], shard_records=[item[0] for item in loaded],
        shard_arrays=[item[1] for item in loaded],
        selected_identity_digests=selection["identity_digests"],
        prediction_spec_sha256=prediction_spec["content_hash"],
        execution_lock_sha256=execution_lock["content_hash"],
        expected_source_partitions=value["expected_source_partitions"],
    )
    _runtime_helpers()["publish"](
        next(iter(_runtime_helpers()["outputs"](task, runtime_row).values())), manifest,
    )
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="prediction_finalize",
    )


def _joined_prediction_arrays(
    manifest: Mapping[str, Any], shards: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
):
    by_partition = {
        str(record[0]["payload"]["source_partition"]): record for record in shards
    }
    if len(by_partition) != len(shards):
        raise ValueError("joined prediction shards repeat a source partition")
    order = [str(row["source_partition"]) for row in manifest["shards"]]
    if set(order) != set(by_partition):
        raise ValueError("joined prediction shard inventory differs from manifest")
    return {
        "identity_digests": np.ascontiguousarray(np.concatenate([
            by_partition[name][1]["identity_digests"] for name in order
        ], axis=0)),
        "logits": np.ascontiguousarray(np.concatenate([
            by_partition[name][1]["logits"] for name in order
        ], axis=0), dtype=np.float32),
    }


def _publish_paired_bootstraps(
    *, metric_join: Mapping[str, Any], escrow_sidecar: Mapping[str, Any],
    escrow_arrays: Mapping[str, np.ndarray], prediction_arrays: Mapping[str, Mapping[str, np.ndarray]],
    prediction_manifests: Mapping[str, Mapping[str, Any]], comparison_registry: object,
    bootstrap_root: str | Path, bootstrap_output_rows: object,
    publication_owner: Mapping[str, Any], producer_task_id_prefix: str,
):
    finalist_ids = set(prediction_arrays)
    if set(prediction_manifests) != finalist_ids:
        raise ValueError("paired-bootstrap prediction registry differs")
    comparisons = comparison_registry
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("paired comparison registry is empty")
    comparison_by_id = {}
    for raw in comparisons:
        if not isinstance(raw, Mapping) or set(raw) != {
            "comparison_id", "left_id", "right_id", "sign",
        }:
            raise ValueError("paired comparison registry row differs")
        row = dict(raw)
        comparison_id = str(row["comparison_id"])
        if (
            not comparison_id or comparison_id in comparison_by_id
            or row["left_id"] not in finalist_ids or row["right_id"] not in finalist_ids
            or row["left_id"] == row["right_id"] or row["sign"] != "left_minus_right"
        ):
            raise ValueError("paired comparison registry semantics differ")
        comparison_by_id[comparison_id] = row
    output_rows = _exact_mapping(
        bootstrap_output_rows, name="paired-bootstrap registered output rows",
    )
    if set(output_rows) != set(comparison_by_id):
        raise ValueError("paired-bootstrap output-row registry differs")
    label_ids = [bytes(row).hex() for row in escrow_arrays["identity_digests"]]
    label_by_identity = {
        identity: int(label)
        for identity, label in zip(label_ids, escrow_arrays["labels"], strict=True)
    }
    if len(label_by_identity) != len(label_ids):
        raise ValueError("paired-bootstrap label escrow repeats an identity")
    from .hcwdl_paired_bootstrap import (
        BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, DEFAULT_METRICS,
        paired_classification_bootstrap, publish_paired_bootstrap_envelope,
    )
    records = []
    root = Path(bootstrap_root)
    for comparison_id in sorted(comparison_by_id):
        row = comparison_by_id[comparison_id]
        left = prediction_arrays[row["left_id"]]
        right = prediction_arrays[row["right_id"]]
        left_ids = [bytes(item).hex() for item in left["identity_digests"]]
        right_ids = [bytes(item).hex() for item in right["identity_digests"]]
        if left_ids != right_ids or set(left_ids) != set(label_by_identity):
            raise ValueError("paired-bootstrap prediction/label identity alignment differs")
        labels = np.asarray([label_by_identity[item] for item in left_ids], dtype=np.int64)
        report, arrays = paired_classification_bootstrap(
            left_logits=np.asarray(left["logits"], dtype=np.float32),
            right_logits=np.asarray(right["logits"], dtype=np.float32),
            labels=labels, identity_digests=left["identity_digests"],
            left_id=str(row["left_id"]), right_id=str(row["right_id"]),
            comparison_id=comparison_id,
            parent_hashes={
                "metric_join": metric_join["content_hash"],
                "label_escrow": escrow_sidecar["content_hash"],
                "left_prediction_manifest": prediction_manifests[row["left_id"]]["content_hash"],
                "right_prediction_manifest": prediction_manifests[row["right_id"]]["content_hash"],
            },
            metrics=DEFAULT_METRICS, replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        envelope = publish_paired_bootstrap_envelope(
            root / comparison_id, bootstrap_report=report, arrays=arrays,
            producer_task_id=f"{producer_task_id_prefix}:{comparison_id}",
            registered_output_row=output_rows[comparison_id],
            campaign_or_recovery_owner=publication_owner,
        )
        records.append({
            "comparison_id": comparison_id,
            "sidecar": envelope.sidecar,
            "commit": envelope.commit,
            "directory": str(envelope.directory),
        })
    return records


def metric_join_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_JOIN_ASSEMBLY_CONTRACT,
        required=(
            "selection", "label_escrow", "prediction_spec", "execution_lock",
            "finalist_lock", "data_attestation", "claim", "task_registry",
            "task_id", "prediction_manifests", "prediction_shards",
            "evaluation_output_paths", "comparison_registry", "bootstrap_root",
            "bootstrap_output_rows", "publication_owner", "producer_task_id_prefix",
        ),
    )
    selection = _versioned_reference(value["selection"], name="metric selection")
    prediction_spec = _versioned_reference(
        value["prediction_spec"], name="prediction specification",
    )
    execution_lock = _versioned_reference(value["execution_lock"], name="execution lock")
    finalist_lock = _versioned_reference(value["finalist_lock"], name="finalist lock")
    data_attestation = _versioned_reference(
        value["data_attestation"], name="final data attestation",
    )
    execution_claim, task_registry, _, capability = _final_capability(
        claim_reference=value["claim"], registry_reference=value["task_registry"],
        task_id=str(value["task_id"]), execution_lock_reference=value["execution_lock"],
    )
    capability_path = _publish_capability_output(task, runtime_row, capability)
    escrow_ref = value["label_escrow"]
    legacy_escrow = {"root", "envelope_id", "capability_sha256"}
    exact_escrow = {"committed_directory"}
    if not isinstance(escrow_ref, Mapping) or set(escrow_ref) not in (
        legacy_escrow, exact_escrow,
    ):
        raise ValueError("label-escrow reference fields differ")
    if set(escrow_ref) == exact_escrow:
        escrow_directory, escrow_root, escrow_id = _committed_coordinates(
            escrow_ref["committed_directory"], name="label-escrow committed envelope",
        )
        escrow_commit = load_json(escrow_directory / "commit.json")
        escrow_parents = _exact_mapping(
            escrow_commit["parents"], name="label-escrow committed parents",
        )
        if (
            escrow_parents.get("selection") != selection["content_hash"]
            or escrow_parents.get("population") != selection["population_sha256"]
            or set(escrow_parents) != {"selection", "population", "capability"}
        ):
            raise PermissionError("label-escrow committed lineage differs")
        escrow_capability = escrow_parents["capability"]
    else:
        escrow_root = _registered_input_path(
            escrow_ref["root"], name="label-escrow envelope root",
        )
        escrow_id = str(escrow_ref["envelope_id"])
        escrow_capability = str(escrow_ref["capability_sha256"])
    from .hcwdl_final_stream import load_label_escrow
    escrow_sidecar, escrow_arrays = load_label_escrow(
        escrow_root, escrow_id,
        selection_sha256=selection["content_hash"],
        population_sha256=selection["population_sha256"],
        capability_sha256=escrow_capability,
    )
    manifest_refs = _exact_mapping(
        value["prediction_manifests"], name="prediction manifests",
    )
    shard_refs = _exact_mapping(value["prediction_shards"], name="prediction shards")
    finalist_ids = {row["finalist_id"] for row in finalist_lock["finalists"]}
    if set(manifest_refs) != finalist_ids or set(shard_refs) != finalist_ids:
        raise ValueError("metric-join finalist prediction registry differs")
    manifests = {
        finalist_id: _versioned_reference(reference, name=f"{finalist_id} prediction manifest")
        for finalist_id, reference in manifest_refs.items()
    }
    predictions = {}
    for finalist_id in sorted(finalist_ids):
        refs = shard_refs[finalist_id]
        if not isinstance(refs, list) or not refs:
            raise ValueError("metric-join prediction shard registry is empty")
        loaded = [_load_prediction_envelope(reference) for reference in refs]
        predictions[finalist_id] = _joined_prediction_arrays(
            manifests[finalist_id], loaded,
        )
    from .hcwdl_representation_final import locked_metric_join
    join, evaluations = locked_metric_join(
        label_escrow_sidecar=escrow_sidecar, label_arrays=escrow_arrays,
        finalists=finalist_lock["finalists"], prediction_arrays=predictions,
        prediction_manifests=manifests, execution_lock=execution_lock,
        finalist_lock=finalist_lock, prediction_spec=prediction_spec,
        data_attestation=data_attestation, capability=capability,
        task_id=str(value["task_id"]), execution_claim=execution_claim,
        task_registry=task_registry,
    )
    bootstrap_records = _publish_paired_bootstraps(
        metric_join=join, escrow_sidecar=escrow_sidecar,
        escrow_arrays=escrow_arrays, prediction_arrays=predictions,
        prediction_manifests=manifests,
        comparison_registry=value["comparison_registry"],
        bootstrap_root=value["bootstrap_root"],
        bootstrap_output_rows=value["bootstrap_output_rows"],
        publication_owner=value["publication_owner"],
        producer_task_id_prefix=str(value["producer_task_id_prefix"]),
    )
    outputs = _runtime_helpers()["outputs"](task, runtime_row)
    join_paths = [
        path for logical, path in outputs.items()
        if "metric_join" in logical and "capabilit" not in logical.lower()
    ]
    evaluation_roots = [
        path for logical, path in outputs.items() if "evaluations" in logical
    ]
    bootstrap_roots = [
        path for logical, path in outputs.items()
        if logical.rstrip("/").endswith("paired_bootstrap")
    ]
    if (
        len(join_paths) != 1 or len(evaluation_roots) != 1 or len(bootstrap_roots) != 1
        or capability_path in join_paths + evaluation_roots + bootstrap_roots
        or Path(str(value["bootstrap_root"])).resolve() != bootstrap_roots[0].resolve()
        or len(bootstrap_records) != len(value["comparison_registry"])
    ):
        raise ValueError("metric-join output registry differs")
    _runtime_helpers()["publish"](join_paths[0], join)
    output_paths = _exact_mapping(
        value["evaluation_output_paths"], name="evaluation output paths",
    )
    if set(output_paths) != finalist_ids:
        raise ValueError("evaluation output-path registry differs")
    root = evaluation_roots[0]
    root.mkdir(parents=True, exist_ok=True)
    for finalist_id, artifact in evaluations.items():
        path = Path(str(output_paths[finalist_id]))
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise PermissionError("evaluation output escapes its registered directory") from error
        _runtime_helpers()["publish"](path, artifact)
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="metric_join",
    )


def validation_only_aggregate_adapter(spec, task, index, runtime_row):
    del index
    value = _assembly(
        task, runtime_row, contract=VALIDATION_ONLY_ASSEMBLY_CONTRACT,
        required=(
            "screen_aggregate", "confirmation_aggregate", "final_disposition",
        ),
    )
    screen = _versioned_reference(value["screen_aggregate"], name="screen aggregate")
    confirmation = _versioned_reference(
        value["confirmation_aggregate"], name="confirmation aggregate",
    )
    disposition = _versioned_reference(
        value["final_disposition"], name="final disposition",
    )
    from .hcwdl_representation_reporting import build_validation_only_aggregate
    aggregate = build_validation_only_aggregate(
        screen_aggregate=screen, confirmation_aggregate=confirmation,
        campaign_spec_sha256=str(spec["content_hash"]),
        final_disposition_sha256=disposition["content_hash"],
    )
    _runtime_helpers()["publish"](
        next(iter(_runtime_helpers()["outputs"](task, runtime_row).values())), aggregate,
    )
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="validation_only_aggregate",
    )


def _load_exact_paired_bootstrap_root(
    *, bootstrap_root: object, comparison_registry: object,
    metric_join: Mapping[str, Any],
) -> list[dict[str, Mapping[str, Any]]]:
    """Load the closed set of committed bootstrap children under one root.

    Envelope identities are content-derived by the upstream metric-join task,
    so they cannot be placed in a pre-campaign runtime row.  The registered
    producer root and frozen comparison registry nevertheless make discovery
    closed: there must be one comparison directory per declared row, exactly
    one committed child in each directory, and no undeclared children.
    """

    comparisons = comparison_registry
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("paired comparison registry is empty")
    manifest_hashes = _exact_mapping(
        metric_join.get("prediction_manifests"),
        name="metric-join prediction manifests",
    )
    by_id: dict[str, dict[str, str]] = {}
    for raw in comparisons:
        if not isinstance(raw, Mapping) or set(raw) != {
            "comparison_id", "left_id", "right_id", "sign",
        }:
            raise ValueError("paired comparison registry row differs")
        comparison_id = str(raw["comparison_id"])
        left_id = str(raw["left_id"])
        right_id = str(raw["right_id"])
        if (
            not comparison_id
            or comparison_id in {".", ".."}
            or "/" in comparison_id
            or "\\" in comparison_id
            or comparison_id in by_id
            or left_id == right_id
            or raw["sign"] != "left_minus_right"
            or left_id not in manifest_hashes
            or right_id not in manifest_hashes
        ):
            raise ValueError("paired comparison registry identity/sign differs")
        by_id[comparison_id] = {
            "left_id": left_id, "right_id": right_id,
        }

    root = Path(_registered_input_path(
        bootstrap_root, name="paired-bootstrap aggregate root",
    ))
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("paired-bootstrap aggregate root is absent or unsafe")
    children = list(root.iterdir())
    if (
        {path.name for path in children} != set(by_id)
        or any(path.is_symlink() or not path.is_dir() for path in children)
    ):
        raise ValueError("paired-bootstrap comparison directory inventory differs")

    from .hcwdl_paired_bootstrap import PAIRED_BOOTSTRAP_CONTRACT
    from .hcwdl_representation_artifacts import validate_binary_envelope

    records: list[dict[str, Mapping[str, Any]]] = []
    for comparison_id in sorted(by_id):
        comparison_root = root / comparison_id
        root_entries = list(comparison_root.iterdir())
        if (
            not root_entries
            or {path.name for path in root_entries} - {"committed", "staging"}
            or any(path.is_symlink() for path in root_entries)
        ):
            raise ValueError("paired-bootstrap comparison root inventory differs")
        committed = comparison_root / "committed"
        if committed.is_symlink() or not committed.is_dir():
            raise FileNotFoundError("paired-bootstrap committed directory is absent")
        committed_children = list(committed.iterdir())
        if (
            len(committed_children) != 1
            or committed_children[0].is_symlink()
            or not committed_children[0].is_dir()
        ):
            raise ValueError("paired-bootstrap committed envelope inventory differs")
        staging = comparison_root / "staging"
        if staging.exists() and (
            staging.is_symlink()
            or not staging.is_dir()
            or any(path.is_file() or path.is_symlink() for path in staging.rglob("*"))
        ):
            raise ValueError("paired-bootstrap staging inventory differs")

        envelope_id = require_sha256(
            committed_children[0].name,
            name=f"paired-bootstrap envelope ID for {comparison_id}",
        )
        row = by_id[comparison_id]
        expected_parents = {
            "metric_join": metric_join["content_hash"],
            "label_escrow": metric_join["label_escrow_sha256"],
            "left_prediction_manifest": manifest_hashes[row["left_id"]],
            "right_prediction_manifest": manifest_hashes[row["right_id"]],
        }
        envelope = validate_binary_envelope(
            comparison_root, envelope_id,
            expected_contract=PAIRED_BOOTSTRAP_CONTRACT,
            expected_parents=expected_parents,
        )
        identity = envelope.commit["payload"]
        registered_row = identity["registered_output_row"]
        if (
            not isinstance(registered_row, Mapping)
            or set(registered_row) != {"comparison_id", "task_id"}
            or registered_row["comparison_id"] != comparison_id
            or not isinstance(registered_row["task_id"], str)
            or not registered_row["task_id"]
            or identity["producer_task_id"]
            != f"{registered_row['task_id']}:{comparison_id}"
        ):
            raise ValueError("paired-bootstrap registered output identity differs")
        records.append({"sidecar": envelope.sidecar, "commit": envelope.commit})
    return records


def execution_lock_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_EXECUTION_ASSEMBLY_CONTRACT,
        required=(
            "finalist_lock", "data_attestation", "claim", "task_registry",
            "row_selection", "prediction_runtime_signature", "source_partitions",
        ),
    )
    finalist_lock = _versioned_reference(value["finalist_lock"], name="finalist lock")
    data_attestation = _versioned_reference(
        value["data_attestation"], name="final data attestation",
    )
    claim = _versioned_reference(value["claim"], name="final execution claim")
    registry = _versioned_reference(value["task_registry"], name="final task registry")
    selection = _versioned_reference(value["row_selection"], name="final row selection")
    from .hcwdl_representation_final import build_execution_lock, build_prediction_spec
    lock = build_execution_lock(
        finalist_lock=finalist_lock, data_attestation=data_attestation,
        claim=claim, task_registry=registry,
    )
    runtime_signature = value["prediction_runtime_signature"]
    if not isinstance(runtime_signature, Mapping):
        raise ValueError("prediction runtime signature is not an immutable object")
    prediction_spec = build_prediction_spec(
        finalist_lock=finalist_lock, execution_lock=lock,
        row_selection=selection, runtime_signature=runtime_signature,
        source_partitions=value["source_partitions"],
    )
    outputs = _runtime_helpers()["outputs"](task, runtime_row)
    lock_paths = [path for logical, path in outputs.items() if "07_execution" in logical]
    prediction_paths = [
        path for logical, path in outputs.items() if "prediction_spec" in logical
    ]
    if len(lock_paths) != 1 or len(prediction_paths) != 1:
        raise ValueError("execution-lock composite output registry differs")
    _runtime_helpers()["publish"](lock_paths[0], lock)
    _runtime_helpers()["publish"](prediction_paths[0], prediction_spec)
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="execution_lock",
    )


def final_aggregate_adapter(spec, task, index, runtime_row):
    del spec, index
    value = _assembly(
        task, runtime_row, contract=FINAL_AGGREGATE_ASSEMBLY_CONTRACT,
        required=(
            "metric_join", "evaluations", "finalist_lock", "execution_lock",
            "confirmation_aggregate", "comparison_registry", "bootstrap_root",
        ),
    )
    metric_join = _versioned_reference(value["metric_join"], name="metric join")
    finalist_lock = _versioned_reference(value["finalist_lock"], name="finalist lock")
    execution_lock = _versioned_reference(value["execution_lock"], name="execution lock")
    confirmation = _versioned_reference(
        value["confirmation_aggregate"], name="confirmation aggregate",
    )
    finalist_ids = {row["finalist_id"] for row in finalist_lock["finalists"]}
    evaluation_refs = _exact_mapping(value["evaluations"], name="final evaluations")
    if set(evaluation_refs) != finalist_ids:
        raise ValueError("final aggregate finalist registry differs")
    evaluations = {
        finalist_id: _versioned_reference(reference, name=f"{finalist_id} evaluation")
        for finalist_id, reference in evaluation_refs.items()
    }
    comparisons = value["comparison_registry"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("paired comparison registry is empty")
    envelope_records = _load_exact_paired_bootstrap_root(
        bootstrap_root=value["bootstrap_root"],
        comparison_registry=comparisons,
        metric_join=metric_join,
    )
    from .hcwdl_representation_final import build_final_aggregate
    aggregate = build_final_aggregate(
        metric_join=metric_join, evaluations=evaluations,
        finalist_lock=finalist_lock, execution_lock=execution_lock,
        paired_bootstrap_envelopes=envelope_records,
        paired_comparison_registry=comparisons,
        confirmation_aggregate_sha256=confirmation["content_hash"],
    )
    outputs = _runtime_helpers()["outputs"](task, runtime_row)
    aggregate_paths = [
        path for logical, path in outputs.items() if "final_aggregate.json" in logical
    ]
    if len(aggregate_paths) != 1 or len(outputs) != 1:
        raise ValueError("final aggregate output registry differs")
    _runtime_helpers()["publish"](aggregate_paths[0], aggregate)
    return _runtime_helpers()["validate_outputs"](
        task, runtime_row, operation="final_aggregate",
    )
