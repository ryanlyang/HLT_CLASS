"""Natural-population, bounded-memory Scouting model batch streams."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import numpy as np

from .inputs import build_hlt_inputs, build_native_offline_inputs
from .labels import baseline_mask, multiclass_labels
from .schema import (
    BASELINE_BRANCHES, LABEL_BRANCHES, OBSERVER_BRANCHES, hlt_required_branches,
    native_offline_required_branches,
)
from .splits import role_records
from .streaming import iterate_projected_chunks, partition_files


def _slice(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}  # Awkward and NumPy agree


def iterate_model_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    input_mode: str = "hlt", completed_locks: Sequence[str] = (), step_size: int = 4096,
    rank: int = 0, world_size: int = 1, worker_id: int = 0, num_workers: int = 1,
    epoch: int = 0, sampler_seed: int = 1337, shuffle_within_chunk: bool = True,
    batch_size: int = 512,
    include_observers: bool = False,
    max_rows: int | None = None,
) -> Iterator[dict[str, object]]:
    if input_mode not in {"hlt", "toff", "paired"}:
        raise ValueError("input_mode must be hlt, toff, or paired")
    if batch_size <= 0:
        raise ValueError("model batch size must be positive")
    records = role_records(split_manifest, role)
    ordered = list(records)
    rng = np.random.default_rng(np.random.SeedSequence([sampler_seed, epoch]))
    if role == "train":
        rng.shuffle(ordered)
    assigned = partition_files(
        ordered, rank=rank, world_size=world_size, worker_id=worker_id, num_workers=num_workers,
    )
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
    if include_observers:
        branches |= set(OBSERVER_BRANCHES)
    if input_mode in {"hlt", "paired"}:
        branches |= set(hlt_required_branches())
    if input_mode in {"toff", "paired"}:
        branches |= set(native_offline_required_branches())
    files = [Path(data_root) / item.path for item in assigned]
    emitted = 0
    class_targets = None
    class_emitted = np.zeros(15, np.int64)
    if max_rows is not None:
        class_targets = np.full(15, max_rows // 15, np.int64)
        class_targets[:max_rows % 15] += 1
    for chunk in iterate_projected_chunks(
        files, branches, data_root=data_root, role=role,
        completed_locks=completed_locks, step_size=step_size,
    ):
        arrays = chunk.arrays
        labels = multiclass_labels(arrays)
        keep = baseline_mask(arrays) & (labels >= 0)
        indexes = np.flatnonzero(keep)
        if not len(indexes):
            continue
        if role == "train" and shuffle_within_chunk:
            indexes = indexes[rng.permutation(len(indexes))]
        if class_targets is not None:
            provisional = class_emitted.copy(); retained = []
            for index in indexes:
                label = int(labels[index])
                if provisional[label] < class_targets[label]:
                    retained.append(int(index)); provisional[label] += 1
            indexes = np.asarray(retained, np.int64)
            if not len(indexes): continue
        selected = _slice(arrays, indexes)
        chunk_batch: dict[str, object] = {
            "labels": labels[indexes],
            "identity_keys": np.asarray([
                f"{chunk.source_path}::tree::{chunk.entry_start + int(index)}" for index in indexes
            ]),
        }
        if input_mode in {"hlt", "paired"}:
            chunk_batch["hlt"] = build_hlt_inputs(selected)
        if include_observers:
            chunk_batch["observers"] = {
                name: np.asarray(selected[name]) for name in OBSERVER_BRANCHES if name in selected
            }
        if input_mode in {"toff", "paired"}:
            chunk_batch["toff"] = build_native_offline_inputs(selected)
        for start in range(0, len(indexes), batch_size):
            stop = min(start + batch_size, len(indexes))
            batch: dict[str, object] = {
                "labels": chunk_batch["labels"][start:stop],
                "identity_keys": chunk_batch["identity_keys"][start:stop],
            }
            if "hlt" in chunk_batch:
                view = chunk_batch["hlt"]
                batch["hlt"] = type(view)(
                    view.features[start:stop], view.vectors[start:stop],
                    view.mask[start:stop], view.raw_lengths[start:stop],
                )
            if "toff" in chunk_batch:
                native = chunk_batch["toff"]
                batch["toff"] = type(native)(
                    charged=type(native.charged)(
                        native.charged.features[start:stop], native.charged.vectors[start:stop],
                        native.charged.mask[start:stop], native.charged.raw_lengths[start:stop],
                    ),
                    neutral=type(native.neutral)(
                        native.neutral.features[start:stop], native.neutral.vectors[start:stop],
                        native.neutral.mask[start:stop], native.neutral.raw_lengths[start:stop],
                    ),
                )
            if "observers" in chunk_batch:
                batch["observers"] = {
                    name: value[start:stop] for name, value in chunk_batch["observers"].items()
                }
                batch["observers"]["hlt_truncated"] = (
                    batch["hlt"].raw_lengths > batch["hlt"].features.shape[2]
                ).astype(np.int8)
            yield batch
            emitted += stop - start
            if class_targets is not None:
                class_emitted += np.bincount(batch["labels"], minlength=15)
            if max_rows is not None and emitted >= max_rows: return


__all__ = ["iterate_model_batches"]
