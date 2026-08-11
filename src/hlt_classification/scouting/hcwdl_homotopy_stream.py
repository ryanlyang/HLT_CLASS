"""Distinct authenticated ROOT stream for variable-support HCWDL U/J views."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
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


def iterate_homotopy_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    assignment_store: DenseAssignmentStore, coupling_store: ResidualCouplingStore,
    row_selection: RowSelection, coordinate: HomotopyCoordinate,
    repair_seed: int, batch_size: int, step_size: int = 4096,
    completed_locks: Sequence[str] = (), output_key: str = "privileged",
    source_index: int | None = None,
) -> Iterator[dict[str, object]]:
    """Stream one V(s,f) role exactly once in canonical source/entry order."""

    if role not in {"train", "validation"}:
        raise PermissionError("HCWDL-UJ homotopy stream is validation-only")
    if output_key not in {"hlt", "privileged"}:
        raise ValueError("HCWDL-UJ homotopy stream output key differs")
    if batch_size <= 0:
        raise ValueError("HCWDL-UJ stream batch size must be positive")
    all_records = role_records(split_manifest, role)
    if source_index is None:
        records = all_records
        expected_rows = row_selection.rows
    else:
        if isinstance(source_index, bool) or not 0 <= int(source_index) < len(all_records):
            raise ValueError("HCWDL-UJ homotopy source index differs")
        records = (all_records[int(source_index)],)
        expected_rows = row_selection.source_rows(records[0].path)
        if expected_rows <= 0:
            raise ValueError("HCWDL-UJ selected source partition is empty")
    branches = (
        set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
        | set(hlt_required_branches()) | set(full_endpoint_required_branches())
        | {"n_cpfcands", "n_lts", "n_npfcands"}
    )
    pending: dict[str, object] | None = None
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
            arrays = _slice(chunk.arrays, indexes)
            identities = np.asarray([
                f"{chunk.source_path}::tree::{int(entry)}" for entry in entries
            ])
            assignment, confidence = assignment_store.join(chunk.source_path, entries)
            coupling_rows = [
                coupling_store.get(chunk.source_path, int(entry)).edits for entry in entries
            ]
            view = build_homotopy_inputs(
                arrays, assignments=assignment, confidence=confidence,
                coupling_rows=coupling_rows, coordinate=coordinate,
                identity_keys=identities, discrete_seed=repair_seed,
                include_training_metadata=True,
            )
            block = {
                "labels": labels[indexes], "identity_keys": identities,
                output_key: view,
            }
            observed += len(indexes)
            pending = block if pending is None else _concat_batches((pending, block))
            while len(pending["labels"]) >= batch_size:
                yield _slice_batch(pending, 0, batch_size)
                pending = _slice_batch(pending, batch_size, len(pending["labels"]))
    if pending is not None and len(pending["labels"]):
        yield pending
    if observed != expected_rows:
        raise ValueError(
            f"HCWDL-UJ stream coverage differs: expected {expected_rows}, observed {observed}"
        )


__all__ = ["iterate_homotopy_batches"]
