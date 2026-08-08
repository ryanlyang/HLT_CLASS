"""Process-local PMARD input-view caches with exact sampler replay."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import os
from typing import Final

import numpy as np

from hlt_classification.data.cache_contracts import with_content_hash
from .dataset import (
    TRAIN_INTERLEAVE_FILES, TRAIN_SHUFFLE_BUFFER_ROWS, _take_batch,
)
from .inputs import NativeOfflineInputs, ParticleInputs
from .splits import SourceFileRecord
from .streaming import partition_files


EPHEMERAL_VIEW_CACHE_CONTRACT: Final = "hlt_classification_pmard_ephemeral_view_cache_v1"
EPHEMERAL_VIEW_CACHE_VERSION: Final = 1
VIEW_CACHE_MEMORY_HEADROOM: Final = 0.75


def _slurm_memory_bytes(environ: Mapping[str, str]) -> int | None:
    """Return the Slurm job memory request when the controller exposes it."""

    raw = environ.get("SLURM_MEM_PER_NODE")
    if not raw:
        return None
    value = raw.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if value[-1:] in multipliers:
        try:
            return int(float(value[:-1]) * multipliers[value[-1]])
        except ValueError as error:
            raise ValueError("SLURM_MEM_PER_NODE is not a valid memory quantity") from error
    try:
        # Slurm documents an unsuffixed value in MiB.
        return int(value) * 1024**2
    except ValueError as error:
        raise ValueError("SLURM_MEM_PER_NODE is not a valid memory quantity") from error


def view_cache_budget_bytes(
    max_gib: float, *, environ: Mapping[str, str] | None = None,
) -> int:
    """Resolve the explicit cache cap beneath a conservative Slurm headroom."""

    if not np.isfinite(max_gib) or max_gib <= 0:
        raise ValueError("view-cache maximum GiB must be positive and finite")
    configured = int(float(max_gib) * 1024**3)
    slurm = _slurm_memory_bytes(os.environ if environ is None else environ)
    return configured if slurm is None else min(
        configured, int(slurm * VIEW_CACHE_MEMORY_HEADROOM),
    )


def _particle_row_bytes(view: ParticleInputs) -> int:
    rows = len(view.raw_lengths)
    if rows <= 0:
        raise ValueError("view-cache seed batch is empty")
    return sum(array.nbytes for array in (
        view.features, view.vectors, view.mask, view.raw_lengths,
    )) // rows


def _view_row_bytes(view: ParticleInputs | NativeOfflineInputs) -> int:
    if isinstance(view, ParticleInputs):
        return _particle_row_bytes(view)
    if isinstance(view, NativeOfflineInputs):
        return _particle_row_bytes(view.charged) + _particle_row_bytes(view.neutral)
    raise TypeError("view cache received an unsupported view type")


def _validate_view(view: object) -> None:
    particles = (
        (view,) if isinstance(view, ParticleInputs)
        else (view.charged, view.neutral) if isinstance(view, NativeOfflineInputs)
        else ()
    )
    if not particles:
        raise ValueError("view cache requires canonical particle inputs")
    for particle in particles:
        if (
            particle.features.dtype != np.float32
            or particle.vectors.dtype != np.float32
            or particle.mask.dtype != np.bool_
            or particle.raw_lengths.dtype != np.int32
        ):
            raise ValueError("view cache requires canonical FP32/bool/int32 particle inputs")


def _allocate_particle(view: ParticleInputs, rows: int) -> ParticleInputs:
    return ParticleInputs(
        np.empty((rows, *view.features.shape[1:]), view.features.dtype),
        np.empty((rows, *view.vectors.shape[1:]), view.vectors.dtype),
        np.empty((rows, *view.mask.shape[1:]), view.mask.dtype),
        np.empty(rows, view.raw_lengths.dtype),
    )


def _allocate_view(view: ParticleInputs | NativeOfflineInputs, rows: int):
    if isinstance(view, ParticleInputs):
        return _allocate_particle(view, rows)
    return NativeOfflineInputs(
        _allocate_particle(view.charged, rows), _allocate_particle(view.neutral, rows),
    )


def _append_particle(target: ParticleInputs, source: ParticleInputs, start: int, stop: int) -> None:
    for name in ("features", "vectors", "mask", "raw_lengths"):
        source_array = getattr(source, name); target_array = getattr(target, name)
        if source_array.shape[1:] != target_array.shape[1:] or source_array.dtype != target_array.dtype:
            raise ValueError("view-cache particle shape or dtype changed within a role")
        target_array[start:stop] = source_array


def _append_view(target, source, start: int, stop: int) -> None:
    if isinstance(target, ParticleInputs) and isinstance(source, ParticleInputs):
        _append_particle(target, source, start, stop); return
    if isinstance(target, NativeOfflineInputs) and isinstance(source, NativeOfflineInputs):
        _append_particle(target.charged, source.charged, start, stop)
        _append_particle(target.neutral, source.neutral, start, stop); return
    raise ValueError("view-cache particle type changed within a role")


def _view_array_bytes(view: ParticleInputs | NativeOfflineInputs) -> int:
    if isinstance(view, ParticleInputs):
        return sum(getattr(view, name).nbytes for name in ("features", "vectors", "mask", "raw_lengths"))
    return _view_array_bytes(view.charged) + _view_array_bytes(view.neutral)


def _identity_digest(values: Sequence[str], *, ordered: bool) -> str:
    digest = hashlib.sha256()
    items = values if ordered else sorted(values)
    for value in items:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little")); digest.update(encoded)
    return digest.hexdigest()


class EphemeralPmardViewCache:
    """RAM-only particle views indexed by identity and replayed by epoch schedule."""

    def __init__(
        self, *, batch: Mapping[str, object], identities: Sequence[str],
        records: Sequence[SourceFileRecord], role: str, step_size: int,
        expected_source_rows: Mapping[str, int], header: Mapping[str, object],
    ) -> None:
        self._batch = dict(batch)
        self.identities = tuple(map(str, identities))
        self.records = tuple(records); self.role = role; self.step_size = int(step_size)
        self.header = dict(header)
        if self.step_size <= 0 or role not in {"train", "validation", "final_test"}:
            raise ValueError("view cache has an invalid role or chunk size")
        if len(self.identities) != len(set(self.identities)):
            raise ValueError("view cache contains duplicate jet identities")
        self._source_chunks: dict[str, tuple[np.ndarray, ...]] = {}
        rows_by_source: dict[str, list[tuple[int, int]]] = {}
        for cache_index, key in enumerate(self.identities):
            try:
                source, raw_entry = key.rsplit("::tree::", 1)
                entry = int(raw_entry)
            except (ValueError, TypeError) as error:
                raise ValueError("view cache contains a malformed identity") from error
            rows_by_source.setdefault(source, []).append((entry, cache_index))
        record_paths = {record.path for record in self.records}
        if set(rows_by_source) - record_paths:
            raise ValueError("view cache contains a source outside its split role")
        for record in self.records:
            pairs = sorted(rows_by_source.get(record.path, ()))
            if len(pairs) != int(expected_source_rows.get(record.path, -1)):
                raise ValueError(f"view-cache source coverage differs for {record.path!r}")
            entries = np.asarray([entry for entry, _ in pairs], np.int64)
            indexes = np.asarray([index for _, index in pairs], np.int64)
            if len(entries) and (entries[0] < 0 or entries[-1] >= record.raw_entries):
                raise ValueError("view cache identity lies outside its source entry range")
            chunks = []
            for start in range(0, record.raw_entries, self.step_size):
                lo = int(np.searchsorted(entries, start, side="left"))
                hi = int(np.searchsorted(entries, start + self.step_size, side="left"))
                chunks.append(indexes[lo:hi])
            self._source_chunks[record.path] = tuple(chunks)

    @classmethod
    def build(
        cls, batches: Iterable[Mapping[str, object]], *, expected_rows: int,
        records: Sequence[SourceFileRecord], role: str,
        expected_source_rows: Mapping[str, int], view_keys: Sequence[str],
        lineage: Mapping[str, object], max_gib: float = 320.0,
        step_size: int = 4096, environ: Mapping[str, str] | None = None,
    ) -> "EphemeralPmardViewCache":
        if expected_rows <= 0:
            raise ValueError("view cache requires a positive expected row count")
        keys = tuple(dict.fromkeys(map(str, view_keys)))
        if not keys or any(key not in {"hlt", "privileged", "toff"} for key in keys):
            raise ValueError("view cache accepts only canonical particle views")
        iterator = iter(batches)
        try:
            first = next(iterator)
        except StopIteration as error:
            raise ValueError("view-cache source stream is empty") from error
        for key in keys:
            _validate_view(first.get(key))
        first_rows = len(first["labels"])
        if first_rows <= 0:
            raise ValueError("view-cache source emitted an empty batch")
        row_bytes = sum(_view_row_bytes(first[key]) for key in keys)
        label_dtype = np.asarray(first["labels"]).dtype
        estimated_array_bytes = expected_rows * (row_bytes + label_dtype.itemsize)
        budget = view_cache_budget_bytes(max_gib, environ=environ)
        if estimated_array_bytes > budget:
            raise MemoryError(
                "PMARD view cache would require at least "
                f"{estimated_array_bytes / 1024**3:.2f} GiB of arrays, above its "
                f"{budget / 1024**3:.2f} GiB safe budget"
            )
        labels = np.empty(expected_rows, dtype=label_dtype)
        identities: list[str | None] = [None] * expected_rows
        views = {key: _allocate_view(first[key], expected_rows) for key in keys}
        cursor = 0

        def append(batch: Mapping[str, object]) -> None:
            nonlocal cursor
            count = len(batch["labels"]); stop = cursor + count
            if count <= 0 or stop > expected_rows or len(batch["identity_keys"]) != count:
                raise ValueError("view-cache source emitted an invalid row block")
            labels[cursor:stop] = np.asarray(batch["labels"])
            identities[cursor:stop] = map(str, batch["identity_keys"])
            for key in keys:
                source = batch.get(key); target = views[key]
                _append_view(target, source, cursor, stop)
            cursor = stop

        append(first)
        for batch in iterator:
            append(batch)
        if cursor != expected_rows or any(value is None for value in identities):
            raise ValueError(
                f"view-cache row count differs: expected {expected_rows}, observed {cursor}"
            )
        identity_values = tuple(str(value) for value in identities)
        array_bytes = labels.nbytes + sum(_view_array_bytes(view) for view in views.values())
        header = with_content_hash({
            "contract": EPHEMERAL_VIEW_CACHE_CONTRACT,
            "schema_version": EPHEMERAL_VIEW_CACHE_VERSION,
            "storage_mode": "process_local_ram_float32_particle_views_v1",
            "role": role, "rows": expected_rows, "view_keys": list(keys),
            "array_bytes": array_bytes, "safe_budget_bytes": budget,
            "identity_order_sha256": _identity_digest(identity_values, ordered=True),
            "identity_set_sha256": _identity_digest(identity_values, ordered=False),
            "step_size": int(step_size), "lineage": dict(sorted(lineage.items())),
            "durable_artifact_published": False,
        })
        batch = {
            "labels": labels,
            "identity_keys": np.asarray(identity_values, dtype=object),
            **views,
        }
        return cls(
            batch=batch, identities=identity_values, records=records, role=role,
            step_size=step_size, expected_source_rows=expected_source_rows,
            header=header,
        )

    def _scheduled_chunks(
        self, *, rng: np.random.Generator, rank: int, world_size: int,
        worker_id: int, num_workers: int, interleave_source_files: int,
    ) -> Iterable[np.ndarray]:
        ordered = list(self.records)
        if self.role == "train":
            rng.shuffle(ordered)
        assigned = partition_files(
            ordered, rank=rank, world_size=world_size,
            worker_id=worker_id, num_workers=num_workers,
        )
        for start in range(0, len(assigned), interleave_source_files):
            active = [
                iter(self._source_chunks[record.path])
                for record in assigned[start:start + interleave_source_files]
            ]
            while active:
                remaining = []
                for iterator in active:
                    try:
                        yield next(iterator); remaining.append(iterator)
                    except StopIteration:
                        pass
                active = remaining

    def iterate_batches(
        self, *, epoch: int, sampler_seed: int, batch_size: int,
        shuffle_buffer_rows: int = TRAIN_SHUFFLE_BUFFER_ROWS,
        interleave_source_files: int = TRAIN_INTERLEAVE_FILES,
        rank: int = 0, world_size: int = 1, worker_id: int = 0,
        num_workers: int = 1,
    ) -> Iterable[dict[str, object]]:
        if batch_size <= 0 or shuffle_buffer_rows < batch_size:
            raise ValueError("view-cache batch or shuffle-buffer size is invalid")
        if interleave_source_files <= 0:
            raise ValueError("view-cache source interleave must be positive")
        rng = np.random.default_rng(np.random.SeedSequence([sampler_seed, epoch]))
        pending = np.empty(0, np.int64)
        observed = 0
        for indexes in self._scheduled_chunks(
            rng=rng, rank=rank, world_size=world_size, worker_id=worker_id,
            num_workers=num_workers, interleave_source_files=interleave_source_files,
        ):
            if not len(indexes):
                continue
            if self.role == "train":
                indexes = indexes[rng.permutation(len(indexes))]
            observed += len(indexes)
            pending = np.concatenate((pending, indexes))
            drain_at = shuffle_buffer_rows if self.role == "train" else batch_size
            if len(pending) >= drain_at:
                if self.role == "train":
                    pending = pending[rng.permutation(len(pending))]
                while len(pending) >= batch_size and (
                    self.role != "train" or len(pending) - batch_size >= batch_size
                ):
                    yield _take_batch(self._batch, pending[:batch_size])
                    pending = pending[batch_size:]
        if len(pending):
            if self.role == "train":
                pending = pending[rng.permutation(len(pending))]
            while len(pending) > batch_size:
                yield _take_batch(self._batch, pending[:batch_size])
                pending = pending[batch_size:]
            if len(pending):
                yield _take_batch(self._batch, pending)
        expected = len(self.identities) if world_size == num_workers == 1 else observed
        if observed != expected:
            raise ValueError("view-cache replay did not cover its expected row population")


def expected_cache_source_rows(
    records: Sequence[SourceFileRecord], *, row_selection=None,
) -> dict[str, int]:
    result = {}
    for record in records:
        selected = -1 if row_selection is None else row_selection.source_rows(record.path)
        result[record.path] = record.mapped_entries if selected < 0 else selected
    return result


def should_cache_student_views(
    *, requested: bool, needs_privileged_training_views: bool,
    alpha: float, arm: str,
) -> bool:
    """Keep alpha-zero aliases and native-offline KD on their cheaper paths."""

    return bool(
        requested and needs_privileged_training_views
        and float(alpha) != 0.0 and arm != "K6"
    )


__all__ = [
    "EPHEMERAL_VIEW_CACHE_CONTRACT", "EPHEMERAL_VIEW_CACHE_VERSION",
    "EphemeralPmardViewCache", "expected_cache_source_rows",
    "should_cache_student_views", "view_cache_budget_bytes",
]
