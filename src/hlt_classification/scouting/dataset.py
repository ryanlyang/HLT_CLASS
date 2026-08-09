"""Natural-population, bounded-memory Scouting model batch streams."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
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
from .selective_assignment import RowSelection


TRAIN_SHUFFLE_BUFFER_ROWS = 32768
TRAIN_INTERLEAVE_FILES = 8


def alias_hlt_as_privileged(
    batches: Iterable[Mapping[str, object]],
) -> Iterator[dict[str, object]]:
    """Expose the exact HLT object under the privileged key without copying arrays."""

    for batch in batches:
        if "hlt" not in batch:
            raise KeyError("alpha-zero privileged alias requires an HLT view")
        result = dict(batch)
        result["privileged"] = result["hlt"]
        yield result


def _slice(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}  # Awkward and NumPy agree


def _slice_particle_view(view, start: int, stop: int):
    values = [
        view.features[start:stop], view.vectors[start:stop],
        view.mask[start:stop], view.raw_lengths[start:stop],
    ]
    for name in ("visible_indices", "family_codes", "family_reason_codes"):
        if hasattr(view, name):
            values.append(getattr(view, name)[start:stop])
    return type(view)(*values)


def _take_particle_view(view, indexes: np.ndarray):
    values = [
        view.features[indexes], view.vectors[indexes],
        view.mask[indexes], view.raw_lengths[indexes],
    ]
    for name in ("visible_indices", "family_codes", "family_reason_codes"):
        if hasattr(view, name):
            values.append(getattr(view, name)[indexes])
    return type(view)(*values)


def _concat_particle_views(views):
    values = [
        np.concatenate([view.features for view in views]),
        np.concatenate([view.vectors for view in views]),
        np.concatenate([view.mask for view in views]),
        np.concatenate([view.raw_lengths for view in views]),
    ]
    for name in ("visible_indices", "family_codes", "family_reason_codes"):
        if hasattr(views[0], name):
            if not all(hasattr(view, name) for view in views):
                raise ValueError("particle-view metadata topology differs")
            values.append(np.concatenate([getattr(view, name) for view in views]))
    return type(views[0])(*values)


def _slice_batch(batch: Mapping[str, object], start: int, stop: int) -> dict[str, object]:
    result: dict[str, object] = {
        "labels": batch["labels"][start:stop],
        "identity_keys": batch["identity_keys"][start:stop],
    }
    for key in ("hlt", "privileged"):
        if key in batch:
            result[key] = _slice_particle_view(batch[key], start, stop)
    if "toff" in batch:
        native = batch["toff"]
        result["toff"] = type(native)(
            charged=_slice_particle_view(native.charged, start, stop),
            neutral=_slice_particle_view(native.neutral, start, stop),
        )
    if "observers" in batch:
        result["observers"] = {
            name: value[start:stop] for name, value in batch["observers"].items()
        }
    return result


def _take_batch(batch: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "labels": batch["labels"][indexes],
        "identity_keys": batch["identity_keys"][indexes],
    }
    for key in ("hlt", "privileged"):
        if key in batch:
            result[key] = _take_particle_view(batch[key], indexes)
    if "toff" in batch:
        native = batch["toff"]
        result["toff"] = type(native)(
            charged=_take_particle_view(native.charged, indexes),
            neutral=_take_particle_view(native.neutral, indexes),
        )
    if "observers" in batch:
        result["observers"] = {
            name: value[indexes] for name, value in batch["observers"].items()
        }
    return result


def _concat_batches(parts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "labels": np.concatenate([part["labels"] for part in parts]),
        "identity_keys": np.concatenate([part["identity_keys"] for part in parts]),
    }
    for key in ("hlt", "privileged"):
        if key in parts[0]:
            result[key] = _concat_particle_views([part[key] for part in parts])
    if "toff" in parts[0]:
        natives = [part["toff"] for part in parts]
        result["toff"] = type(natives[0])(
            charged=_concat_particle_views([view.charged for view in natives]),
            neutral=_concat_particle_views([view.neutral for view in natives]),
        )
    if "observers" in parts[0]:
        result["observers"] = {
            name: np.concatenate([part["observers"][name] for part in parts])
            for name in parts[0]["observers"]
        }
    return result


def iterate_model_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    input_mode: str = "hlt", completed_locks: Sequence[str] = (), step_size: int = 4096,
    rank: int = 0, world_size: int = 1, worker_id: int = 0, num_workers: int = 1,
    epoch: int = 0, sampler_seed: int = 1337, shuffle_within_chunk: bool = True,
    batch_size: int = 512,
    include_observers: bool = False,
    max_rows: int | None = None,
    shuffle_buffer_rows: int = TRAIN_SHUFFLE_BUFFER_ROWS,
    interleave_source_files: int = TRAIN_INTERLEAVE_FILES,
    row_selection: RowSelection | None = None,
    include_hcwdl_metadata: bool = False,
    canonical_order: bool = False,
) -> Iterator[dict[str, object]]:
    if input_mode not in {"hlt", "toff", "paired"}:
        raise ValueError("input_mode must be hlt, toff, or paired")
    if batch_size <= 0:
        raise ValueError("model batch size must be positive")
    if shuffle_buffer_rows < batch_size:
        raise ValueError("shuffle_buffer_rows must be at least batch_size")
    if canonical_order and (
        shuffle_within_chunk
        or interleave_source_files != 1
        or shuffle_buffer_rows != batch_size
    ):
        raise ValueError(
            "canonical model batches require no chunk shuffle, one-file interleave, "
            "and a one-batch drain buffer"
        )
    records = role_records(split_manifest, role)
    ordered = list(records)
    rng = np.random.default_rng(np.random.SeedSequence([sampler_seed, epoch]))
    if role == "train" and not canonical_order:
        rng.shuffle(ordered)
    assigned = partition_files(
        ordered, rank=rank, world_size=world_size, worker_id=worker_id, num_workers=num_workers,
    )
    if canonical_order and len(assigned) != 1:
        raise ValueError("canonical model batching requires a one-source manifest")
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
    if include_observers:
        branches |= set(OBSERVER_BRANCHES)
    if input_mode in {"hlt", "paired"}:
        branches |= set(hlt_required_branches())
    if input_mode in {"toff", "paired"}:
        branches |= set(native_offline_required_branches())
    files = [Path(data_root) / item.path for item in assigned]
    selected_rows = 0
    class_targets = None
    class_selected = np.zeros(15, np.int64)
    if max_rows is not None:
        class_targets = np.full(15, max_rows // 15, np.int64)
        class_targets[:max_rows % 15] += 1
    pending: dict[str, object] | None = None
    prior_canonical_entry: int | None = None
    for chunk in iterate_projected_chunks(
        files, branches, data_root=data_root, role=role,
        completed_locks=completed_locks, step_size=step_size,
        interleave_files=interleave_source_files if role == "train" else 1,
    ):
        arrays = chunk.arrays
        labels = multiclass_labels(arrays)
        keep = baseline_mask(arrays) & (labels >= 0)
        indexes = np.flatnonzero(keep)
        if not len(indexes):
            continue
        if row_selection is not None:
            absolute = chunk.entry_start + indexes
            indexes = indexes[row_selection.mask(chunk.source_path, absolute)]
            if not len(indexes):
                continue
        if role == "train" and shuffle_within_chunk:
            indexes = indexes[rng.permutation(len(indexes))]
        if class_targets is not None:
            provisional = class_selected.copy(); retained = []
            for index in indexes:
                label = int(labels[index])
                if provisional[label] < class_targets[label]:
                    retained.append(int(index)); provisional[label] += 1
            indexes = np.asarray(retained, np.int64)
            if not len(indexes): continue
        selected = _slice(arrays, indexes)
        if canonical_order:
            absolute_entries = chunk.entry_start + indexes
            if (
                np.any(absolute_entries[1:] <= absolute_entries[:-1])
                or (
                    prior_canonical_entry is not None
                    and int(absolute_entries[0]) <= prior_canonical_entry
                )
            ):
                raise ValueError("canonical model source entries reorder or overlap")
            prior_canonical_entry = int(absolute_entries[-1])
        chunk_batch: dict[str, object] = {
            "labels": labels[indexes],
            "identity_keys": np.asarray([
                f"{chunk.source_path}::tree::{chunk.entry_start + int(index)}" for index in indexes
            ]),
        }
        if input_mode in {"hlt", "paired"}:
            if include_hcwdl_metadata:
                from .hcwdl_representation_data import build_hcwdl_hlt_inputs

                chunk_batch["hlt"] = build_hcwdl_hlt_inputs(selected)
            else:
                chunk_batch["hlt"] = build_hlt_inputs(selected)
        if include_observers:
            chunk_batch["observers"] = {
                name: np.asarray(selected[name]) for name in OBSERVER_BRANCHES if name in selected
            }
        if input_mode in {"toff", "paired"}:
            chunk_batch["toff"] = build_native_offline_inputs(selected)
        if include_observers:
            chunk_batch["observers"]["hlt_truncated"] = (
                chunk_batch["hlt"].raw_lengths > chunk_batch["hlt"].features.shape[2]
            ).astype(np.int8)
        selected_rows += len(indexes)
        if class_targets is not None:
            class_selected += np.bincount(labels[indexes], minlength=15)
        pending = chunk_batch if pending is None else _concat_batches((pending, chunk_batch))
        drain_at = shuffle_buffer_rows if role == "train" else batch_size
        if len(pending["labels"]) >= drain_at:
            if role == "train" and not canonical_order:
                pending = _take_batch(pending, rng.permutation(len(pending["labels"])))
            while len(pending["labels"]) >= batch_size and (
                role != "train" or len(pending["labels"]) - batch_size >= batch_size
            ):
                yield _slice_batch(pending, 0, batch_size)
                pending = _slice_batch(pending, batch_size, len(pending["labels"]))
        if max_rows is not None and selected_rows >= max_rows:
            break
    if pending is not None and len(pending["labels"]):
        if role == "train" and not canonical_order:
            pending = _take_batch(pending, rng.permutation(len(pending["labels"])))
        while len(pending["labels"]) > batch_size:
            yield _slice_batch(pending, 0, batch_size)
            pending = _slice_batch(pending, batch_size, len(pending["labels"]))
        if len(pending["labels"]):
            yield pending


__all__ = [
    "TRAIN_INTERLEAVE_FILES", "TRAIN_SHUFFLE_BUFFER_ROWS", "alias_hlt_as_privileged",
    "iterate_model_batches",
]
