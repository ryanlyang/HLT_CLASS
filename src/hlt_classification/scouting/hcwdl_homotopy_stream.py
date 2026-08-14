"""Distinct authenticated ROOT stream for variable-support HCWDL U/J views."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .dataset import _concat_batches, _slice_batch
from .hcwdl_homotopy import HomotopyCoordinate, build_homotopy_inputs
from .hcwdl_upper_cache import ResidualCouplingStore
from .highcov_cache import DenseAssignmentStore
from .labels import baseline_mask, multiclass_labels
from .repair import full_endpoint_required_branches
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, hlt_required_branches
from .selective_assignment import RowSelection
from .splits import role_records
from .streaming import iterate_projected_chunks


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


__all__ = ["iterate_homotopy_batches"]
