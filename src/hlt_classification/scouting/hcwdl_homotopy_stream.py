"""Distinct authenticated ROOT stream for variable-support HCWDL U/J views."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .dataset import _concat_batches, _slice_batch
from .hcwdl_homotopy import (
    HomotopyCoordinate, build_homotopy_inputs, build_unified_balanced_inputs,
    build_unified_balanced_pairing_inputs,
)
from .hcwdl_unified_balanced_cache import BalancedCouplingStore
from .hcwdl_upper_cache import ResidualCouplingStore
from .highcov_cache import DenseAssignmentStore
from .labels import baseline_mask, multiclass_labels
from .repair import full_endpoint_required_branches
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, hlt_required_branches
from .selective_assignment import RowSelection
from .splits import role_records
from .streaming import iterate_projected_chunks


def _selected_record_rows(row_selection: RowSelection, record: object) -> int:
    """Resolve one source's selected population, including all-mapped ``-1``."""

    selected = int(row_selection.source_rows(record.path))
    mapped = int(record.mapped_entries)
    if mapped < 0:
        raise ValueError("HCWDL-UB mapped source rows are negative")
    if selected == -1:
        selected = mapped
    elif selected < -1:
        raise ValueError("HCWDL-UB row-selection sentinel differs")
    if selected > mapped:
        raise ValueError("HCWDL-UB selected source rows exceed mapped rows")
    return selected


def _slice(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}


def _build_block(arguments: tuple[object, ...]) -> dict[str, object]:
    (
        arrays, labels, identities, assignment, confidence, coupling_rows,
        coordinate, repair_seed, output_key,
    ) = arguments
    view = build_homotopy_inputs(
        arrays, assignments=assignment, confidence=confidence,
        coupling_rows=coupling_rows, coordinate=coordinate,
        identity_keys=identities, discrete_seed=repair_seed,
    )
    return {
        "labels": labels, "identity_keys": identities,
        str(output_key): view,
    }


def _ordered_blocks(
    arguments: Iterator[tuple[object, ...]], *, workers: int,
) -> Iterator[dict[str, object]]:
    """Bound in-flight work and emit completed chunks in canonical input order."""

    if workers <= 0:
        raise ValueError("HCWDL-UJ view-build worker count must be positive")
    if workers == 1:
        for item in arguments:
            yield _build_block(item)
        return
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="hcwuj-view",
    ) as executor:
        in_flight: deque[Future[dict[str, object]]] = deque()
        for item in arguments:
            in_flight.append(executor.submit(_build_block, item))
            if len(in_flight) >= workers:
                yield in_flight.popleft().result()
        while in_flight:
            yield in_flight.popleft().result()


def _build_balanced_block(arguments: tuple[object, ...]) -> dict[str, object]:
    (
        arrays, labels, identities, assignment, confidence, coupling_rows,
        coordinate, repair_seed, output_key, include_training_metadata,
        provenance_kind,
    ) = arguments
    common = dict(
        arrays=arrays, assignments=assignment, coupling_rows=coupling_rows,
        coordinate=coordinate, identity_keys=identities,
        discrete_seed=repair_seed,
        include_training_metadata=bool(include_training_metadata),
    )
    if provenance_kind == "pairing_validity":
        view = build_unified_balanced_pairing_inputs(
            **common, pairing_validity=confidence,
        )
    elif provenance_kind == "correspondence_confidence":
        view = build_unified_balanced_inputs(**common, confidence=confidence)
    else:
        raise ValueError("HCWDL-UB assignment-store provenance differs")
    return {"labels": labels, "identity_keys": identities, str(output_key): view}


def _ordered_balanced_blocks(
    arguments: Iterator[tuple[object, ...]], *, workers: int,
) -> Iterator[dict[str, object]]:
    if workers <= 0:
        raise ValueError("HCWDL-UB view-build worker count must be positive")
    if workers == 1:
        for item in arguments:
            yield _build_balanced_block(item)
        return
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="hcwub-view",
    ) as executor:
        in_flight: deque[Future[dict[str, object]]] = deque()
        for item in arguments:
            in_flight.append(executor.submit(_build_balanced_block, item))
            if len(in_flight) >= workers:
                yield in_flight.popleft().result()
        while in_flight:
            yield in_flight.popleft().result()


def iterate_homotopy_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    assignment_store: DenseAssignmentStore, coupling_store: ResidualCouplingStore,
    row_selection: RowSelection, coordinate: HomotopyCoordinate,
    repair_seed: int, batch_size: int, step_size: int = 4096,
    completed_locks: Sequence[str] = (), output_key: str = "privileged",
    workers: int = 1,
) -> Iterator[dict[str, object]]:
    """Stream one V(s,f) role exactly once in canonical source/entry order."""

    if role not in {"train", "validation"}:
        raise PermissionError("HCWDL-UJ homotopy stream is validation-only")
    if output_key not in {"hlt", "privileged"}:
        raise ValueError("HCWDL-UJ homotopy stream output key differs")
    if batch_size <= 0 or workers <= 0:
        raise ValueError("HCWDL-UJ stream batch size must be positive")
    records = role_records(split_manifest, role)
    branches = (
        set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
        | set(hlt_required_branches()) | set(full_endpoint_required_branches())
        | {"n_cpfcands", "n_lts", "n_npfcands"}
    )

    def tasks():
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
                arrays = _slice(chunk.arrays, indexes)
                identities = np.asarray([
                    f"{chunk.source_path}::tree::{int(entry)}" for entry in entries
                ])
                assignment, confidence = assignment_store.join(chunk.source_path, entries)
                coupling_rows = [
                    coupling_store.get(chunk.source_path, int(entry)).edits for entry in entries
                ]
                yield (
                    arrays, labels[indexes], identities, assignment, confidence,
                    coupling_rows, coordinate, repair_seed, output_key,
                )

    pending: dict[str, object] | None = None
    observed = 0
    for block in _ordered_blocks(tasks(), workers=workers):
        observed += len(block["labels"])
        pending = block if pending is None else _concat_batches((pending, block))
        while len(pending["labels"]) >= batch_size:
            yield _slice_batch(pending, 0, batch_size)
            pending = _slice_batch(pending, batch_size, len(pending["labels"]))
    if pending is not None and len(pending["labels"]):
        yield pending
    if observed != row_selection.rows:
        raise ValueError(
            f"HCWDL-UJ stream coverage differs: expected {row_selection.rows}, observed {observed}"
        )


def iterate_unified_balanced_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    assignment_store: DenseAssignmentStore,
    coupling_store: BalancedCouplingStore, row_selection: RowSelection,
    coordinate: HomotopyCoordinate, repair_seed: int, batch_size: int,
    step_size: int = 4096, completed_locks: Sequence[str] = (),
    output_key: str = "privileged", workers: int = 1,
    include_training_metadata: bool = False,
    source_index: int | None = None,
) -> Iterator[dict[str, object]]:
    """Stream HCWDL-UB V_UB(s,f) once in canonical source/entry order."""

    if role not in {"train", "validation"}:
        raise PermissionError("HCWDL-UB homotopy stream is validation-only")
    if output_key not in {"hlt", "privileged"} or batch_size <= 0 or workers <= 0:
        raise ValueError("HCWDL-UB stream output/batch/worker settings differ")
    records = role_records(split_manifest, role)
    if source_index is not None:
        if source_index < 0 or source_index >= len(records):
            raise IndexError("HCWDL-UB stream source index is out of range")
        records = (records[source_index],)
    branches = (
        set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
        | set(hlt_required_branches()) | set(full_endpoint_required_branches())
        | {"n_cpfcands", "n_lts", "n_npfcands"}
    )

    def tasks():
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
                arrays = _slice(chunk.arrays, indexes)
                identities = np.asarray([
                    f"{chunk.source_path}::tree::{int(entry)}" for entry in entries
                ])
                assignment, confidence = assignment_store.join(chunk.source_path, entries)
                coupling_rows = [
                    coupling_store.get(chunk.source_path, int(entry)).edits
                    for entry in entries
                ]
                yield (
                    arrays, labels[indexes], identities, assignment, confidence,
                    coupling_rows, coordinate, repair_seed, output_key,
                    include_training_metadata,
                    getattr(
                        assignment_store, "provenance_kind",
                        "correspondence_confidence",
                    ),
                )

    pending = None; observed = 0
    for block in _ordered_balanced_blocks(tasks(), workers=workers):
        observed += len(block["labels"])
        pending = block if pending is None else _concat_batches((pending, block))
        while len(pending["labels"]) >= batch_size:
            yield _slice_batch(pending, 0, batch_size)
            pending = _slice_batch(pending, batch_size, len(pending["labels"]))
    if pending is not None and len(pending["labels"]):
        yield pending
    expected_rows = (
        row_selection.rows if source_index is None
        else _selected_record_rows(row_selection, records[0])
    )
    if observed != expected_rows:
        raise ValueError(
            f"HCWDL-UB stream coverage differs: expected {expected_rows}, observed {observed}"
        )


__all__ = ["iterate_homotopy_batches", "iterate_unified_balanced_batches"]
