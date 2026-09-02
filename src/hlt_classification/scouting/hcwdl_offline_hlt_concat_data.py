"""Canonical RAM-only offline-then-HLT view for concatenation oracles."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from pathlib import Path
from typing import Final

import numpy as np

from .dataset import _concat_batches, _slice_batch
from .hcwdl_homotopy import prepare_hlt_endpoints, prepare_offline_endpoints
from .hcwdl_homotopy_stream import _selected_record_rows
from .hcwdl_representation_data import (
    CHARGED_FAMILY, DIRECT_CHARGED_REASON, DIRECT_NEUTRAL_REASON,
    HCWDLTaggedParticleInputs, NEUTRAL_FAMILY, PADDED_FAMILY, PADDED_REASON,
    derive_hcwdl_token_metadata,
)
from .inputs import build_hlt_inputs
from .labels import baseline_mask, multiclass_labels
from .repair import full_endpoint_required_branches, transform_endpoint_features
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, hlt_required_branches
from .selective_assignment import RowSelection
from .splits import role_records
from .streaming import iterate_projected_chunks


TAGGED_CONCAT_VIEW_CONTRACT: Final = "HCWDL_OFFLINE_HLT_TAGGED_CONCAT_VIEW/v1"
OFFLINE_CONTENT: Final = np.int8(0)
HLT_CONTENT: Final = np.int8(1)
PADDED_CONTENT: Final = np.int8(-1)
CONCAT_CAPACITY: Final = 400


def tagged_concat_required_branches() -> set[str]:
    return (
        set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
        | set(hlt_required_branches()) | set(full_endpoint_required_branches())
        | {"n_cpfcands", "n_lts", "n_npfcands"}
    )


def build_tagged_concat_inputs(
    arrays: Mapping[str, object], *, capacity: int = CONCAT_CAPACITY,
) -> HCWDLTaggedParticleInputs:
    """Build exactly ``O followed by H`` without matching or deduplication."""

    if capacity != CONCAT_CAPACITY:
        raise ValueError("tagged concatenation capacity differs")
    offline = prepare_offline_endpoints(arrays)
    hlt = prepare_hlt_endpoints(arrays)
    if offline.rows != hlt.rows:
        raise ValueError("concatenation endpoint row counts differ")
    if np.any(hlt.raw_lengths > 200):
        raise ValueError("concatenation HLT endpoint exceeds authenticated capacity")
    hlt_inputs = build_hlt_inputs(arrays, max_length=200)
    hlt_metadata = derive_hcwdl_token_metadata(arrays, max_length=200)
    rows = offline.rows
    features = np.zeros((rows, 21, capacity), np.float32)
    vectors = np.zeros((rows, 4, capacity), np.float32)
    mask = np.zeros((rows, 1, capacity), np.bool_)
    lengths = np.zeros(rows, np.int32)
    visible_indices = np.full((rows, capacity), -1, np.int64)
    family = np.full((rows, capacity), PADDED_FAMILY, np.int8)
    reason = np.full((rows, capacity), PADDED_REASON, np.int8)
    sources = np.full((rows, capacity), PADDED_CONTENT, np.int8)

    for row in range(rows):
        offline_count = len(offline.raw_features[row])
        hlt_count = int(hlt.raw_lengths[row])
        total = offline_count + hlt_count
        if total > capacity:
            raise ValueError("offline+HLT concatenation requires hidden truncation")
        projected = transform_endpoint_features(
            offline.raw_features[row], offline.validity[row],
        )
        if projected.shape != (offline_count, 21):
            raise ValueError("offline projected endpoint shape differs")
        if offline_count:
            features[row, :, :offline_count] = projected.T
            vectors[row, :, :offline_count] = np.asarray(
                offline.p4[row], np.float32,
            ).T
            charged = int(offline.charged_counts[row])
            family[row, :charged] = CHARGED_FAMILY
            family[row, charged:offline_count] = NEUTRAL_FAMILY
            reason[row, :charged] = DIRECT_CHARGED_REASON
            reason[row, charged:offline_count] = DIRECT_NEUTRAL_REASON
            sources[row, :offline_count] = OFFLINE_CONTENT
        if hlt_count:
            start = offline_count
            stop = total
            features[row, :, start:stop] = hlt_inputs.features[row, :, :hlt_count]
            vectors[row, :, start:stop] = hlt_inputs.vectors[row, :, :hlt_count]
            family[row, start:stop] = hlt_metadata.family_codes[row, :hlt_count]
            reason[row, start:stop] = hlt_metadata.family_reason_codes[row, :hlt_count]
            sources[row, start:stop] = HLT_CONTENT
        mask[row, 0, :total] = True
        lengths[row] = total
        visible_indices[row, :total] = np.arange(total, dtype=np.int64)

    return HCWDLTaggedParticleInputs(
        features=np.ascontiguousarray(features),
        vectors=np.ascontiguousarray(vectors), mask=np.ascontiguousarray(mask),
        raw_lengths=lengths, visible_indices=visible_indices,
        family_codes=family, family_reason_codes=reason,
        content_source_codes=sources,
    )


def _slice(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}


def iterate_tagged_concat_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    row_selection: RowSelection, batch_size: int, step_size: int = 4096,
    completed_locks: Sequence[str] = (), source_index: int | None = None,
) -> Iterator[dict[str, object]]:
    """Stream one exact role in canonical source/entry order."""

    if role not in {"train", "validation"}:
        raise PermissionError("concatenation stream cannot access this split role")
    if batch_size <= 0:
        raise ValueError("concatenation stream batch size must be positive")
    records = role_records(split_manifest, role)
    if source_index is not None:
        if source_index < 0 or source_index >= len(records):
            raise IndexError("concatenation source index is out of range")
        records = (records[source_index],)
    branches = tagged_concat_required_branches()
    pending = None
    observed = 0
    for record in records:
        for chunk in iterate_projected_chunks(
            (Path(data_root) / record.path,), branches, data_root=data_root,
            role=role, completed_locks=completed_locks, step_size=step_size,
        ):
            labels = multiclass_labels(chunk.arrays)
            indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
            absolute = chunk.entry_start + indexes
            indexes = indexes[row_selection.mask(chunk.source_path, absolute)]
            if not len(indexes):
                continue
            entries = chunk.entry_start + indexes
            block = {
                "labels": labels[indexes],
                "identity_keys": np.asarray([
                    f"{chunk.source_path}::tree::{int(entry)}" for entry in entries
                ]),
                "privileged": build_tagged_concat_inputs(
                    _slice(chunk.arrays, indexes),
                ),
            }
            observed += len(indexes)
            pending = block if pending is None else _concat_batches((pending, block))
            while len(pending["labels"]) >= batch_size:
                yield _slice_batch(pending, 0, batch_size)
                pending = _slice_batch(pending, batch_size, len(pending["labels"]))
    if pending is not None and len(pending["labels"]):
        yield pending
    expected = (
        row_selection.rows if source_index is None
        else _selected_record_rows(row_selection, records[0])
    )
    if observed != expected:
        raise ValueError(
            f"concatenation stream coverage differs: expected {expected}, observed {observed}"
        )


_SOURCE_PROCESS_CONTEXT: dict[str, object] | None = None


def _initialize_source_process(context: Mapping[str, object]) -> None:
    global _SOURCE_PROCESS_CONTEXT
    if _SOURCE_PROCESS_CONTEXT is not None:
        raise RuntimeError("concatenation source process initialized twice")
    _SOURCE_PROCESS_CONTEXT = dict(context)


def _materialize_source_process(
    source: tuple[int, str],
) -> tuple[str, tuple[Mapping[str, object], ...]]:
    if _SOURCE_PROCESS_CONTEXT is None:
        raise RuntimeError("concatenation source process lacks context")
    source_index, source_path = source
    context = _SOURCE_PROCESS_CONTEXT
    stream = iterate_tagged_concat_batches(
        context["split_manifest"], data_root=str(context["data_root"]),
        role=str(context["role"]), row_selection=context["row_selection"],
        batch_size=int(context["batch_size"]),
        step_size=int(context["step_size"]), source_index=source_index,
    )
    try:
        batches = tuple(stream)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return source_path, batches


def parallel_tagged_concat_source_streams(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    row_selection: RowSelection, batch_size: int, source_workers: int,
    step_size: int = 4096,
) -> Iterable[tuple[str, Mapping[str, object]]]:
    """Build independent source streams with bounded spawned processes."""

    records = role_records(split_manifest, role)
    selected = [
        (index, record.path) for index, record in enumerate(records)
        if _selected_record_rows(row_selection, record) > 0
    ]
    if source_workers <= 0 or source_workers > max(1, len(selected)):
        raise ValueError("concatenation source worker count differs")
    if source_workers == 1:
        for index, path in selected:
            for batch in iterate_tagged_concat_batches(
                split_manifest, data_root=data_root, role=role,
                row_selection=row_selection, batch_size=batch_size,
                step_size=step_size, source_index=index,
            ):
                yield path, batch
        return
    context = {
        "split_manifest": split_manifest, "data_root": str(data_root),
        "role": role, "row_selection": row_selection,
        "batch_size": batch_size, "step_size": step_size,
    }
    iterator = iter(selected)
    executor = ProcessPoolExecutor(
        max_workers=source_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_source_process, initargs=(context,),
    )
    pending = {}

    def submit_next() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(_materialize_source_process, item)] = item
        return True

    for _ in range(min(source_workers, len(selected))):
        submit_next()
    try:
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda value: pending[value][0]):
                _, expected_path = pending.pop(future)
                path, batches = future.result()
                if path != expected_path:
                    raise ValueError("concatenation source process identity differs")
                for batch in batches:
                    yield path, batch
                submit_next()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


__all__ = [
    "CONCAT_CAPACITY", "HLT_CONTENT", "OFFLINE_CONTENT", "PADDED_CONTENT",
    "TAGGED_CONCAT_VIEW_CONTRACT", "build_tagged_concat_inputs",
    "iterate_tagged_concat_batches", "parallel_tagged_concat_source_streams",
    "tagged_concat_required_branches",
]
