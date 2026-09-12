"""Ragged RAM-only views with bounded, ordered process preparation.

One file per block prevents a second full-population concatenate/copy. At most
one outstanding future per worker is admitted. No memmap or disk cache exists.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .campaign import coordinate
from .foundation import load_assignments, validate_foundation_spec
from .inputs import build_inputs
from .reader import DatasetReader
from .views import build_view


def _limit_worker_threads():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    # Also constrain numerical libraries already loaded during spawn imports.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


@dataclass
class RamBlock:
    file_index: int
    offsets: np.ndarray
    features: np.ndarray
    vectors: np.ndarray
    identities: np.ndarray
    labels: np.ndarray

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.offsets, self.features, self.vectors, self.identities, self.labels))


def _prepare_file(arguments) -> RamBlock:
    spec, data_root, foundation_root, task, coord = arguments
    u, f = coordinate(coord)
    needs_assignment = coord not in {"U000", "D000"}
    assigned = load_assignments(spec, root=Path(foundation_root), file_index=task["file_index"])[1] if needs_assignment else None
    reader = DatasetReader(Path(data_root), spec["inventory"], spec["splits"], role=task["role"],
                           include_offline=coord != "D000", file_paths=(task["path"],))
    offsets, features, vectors, ids, labels = [0], [], [], [], []
    for row, jet in enumerate(reader):
        identity = np.frombuffer(bytes.fromhex(jet.identity), np.uint8)
        mapping = None
        if assigned is not None:
            if not np.array_equal(identity, assigned["identities"][row]):
                raise ValueError("Assignment/view row identity join differs")
            start, end = assigned["offsets"][row:row+2]
            mapping = assigned["mapping"][start:end]
        view = build_view(jet, u=u, f=f, mapping=mapping)
        value = build_inputs(view, capacity=spec["inputs"]["capacity"])
        length = len(view)
        features.append(value.features[:, :length].T.copy())
        vectors.append(value.vectors[:, :length].T.copy())
        offsets.append(offsets[-1] + length)
        ids.append(identity)
        labels.append(jet.label)
    block = RamBlock(
        task["file_index"], np.asarray(offsets, np.int64),
        np.concatenate(features) if features else np.empty((0, 17), np.float32),
        np.concatenate(vectors) if vectors else np.empty((0, 4), np.float32),
        np.asarray(ids, np.uint8).reshape(-1, 32), np.asarray(labels, np.int64),
    )
    if len(block.labels) != task["rows"]:
        raise ValueError("RAM block row coverage differs")
    return block


class RamCache:
    def __init__(self, blocks: list[RamBlock], *, role: str, foundation_sha256: str, coordinate_name: str):
        if role not in {"train", "validation"}:
            raise PermissionError("Final-test RAM cache is sealed")
        self.blocks, self.role = blocks, role
        self.foundation_sha256, self.coordinate_name = foundation_sha256, coordinate_name
        self.ends = np.cumsum([len(b.labels) for b in blocks])
        self.labels = np.concatenate([b.labels for b in blocks]) if blocks else np.empty(0, np.int64)
        self.identities = np.concatenate([b.identities for b in blocks]) if blocks else np.empty((0, 32), np.uint8)
        self.nbytes = sum(b.nbytes for b in blocks) + self.labels.nbytes + self.identities.nbytes

    def __len__(self):
        return len(self.labels)

    def batch(self, indices: np.ndarray) -> dict:
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices) or np.any(indices < 0) or np.any(indices >= len(self)):
            raise ValueError("Invalid RAM cache row indices")
        selected = []
        for index in indices:
            block_index = int(np.searchsorted(self.ends, index, side="right"))
            row = int(index - (self.ends[block_index - 1] if block_index else 0))
            block = self.blocks[block_index]
            start, end = block.offsets[row:row + 2]
            selected.append((block, int(start), int(end)))
        length = max(16, max(e - s for _, s, e in selected))
        features = np.zeros((len(indices), 17, length), np.float32)
        vectors = np.zeros((len(indices), 4, length), np.float32)
        mask = np.zeros((len(indices), 1, length), np.bool_)
        for row, (block, start, end) in enumerate(selected):
            features[row, :, :end-start] = block.features[start:end].T
            vectors[row, :, :end-start] = block.vectors[start:end].T
            mask[row, :, :end-start] = True
        return dict(features=features, vectors=vectors, mask=mask,
                    labels=self.labels[indices], identities=self.identities[indices])


def preparation_bound(spec: dict, role: str, workers: int) -> int:
    """Conservative resident + bounded worker/IPC-copy envelope, in bytes."""
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test preprocessing is sealed")
    if type(workers) is not int or not 1 <= workers <= 72:
        raise ValueError("Invalid preparation worker count")
    upper = [t["rows"] * (spec["inputs"]["capacity"] * 84 + 1024)
             for t in spec["assignment_tasks"] if t["role"] == role]
    return sum(upper) + 4 * min(workers, len(upper)) * max(upper, default=0)


def cache_budgets(spec: dict, memory_mb: int, workers: int) -> dict[str, int]:
    """Share 75% of RAM in proportion to each selected role's upper bound.

    In particular, 1M validation must not inherit the old 22% validation slice
    when train contains only 500k. A quarter stays reserved for runtime use.
    """
    if type(memory_mb) is not int or memory_mb <= 0:
        raise ValueError("Positive integer memory allocation required")
    bounds = {r: preparation_bound(spec, r, workers) for r in ("train", "validation")}
    total = sum(bounds.values())
    available = memory_mb * 1024**2 * 3 // 4
    if not all(bounds.values()) or total > available:
        raise MemoryError(f"Selected-profile cache bounds {bounds} exceed 75% RAM budget {available}; reduce workers or request more RAM")
    train = available * bounds["train"] // total
    return dict(train=train, validation=available - train)


def prepare_cache(spec: dict, *, data_root: Path, foundation_root: Path, role: str,
                  coordinate_name: str, workers: int = 1, max_ram_bytes: int) -> RamCache:
    parent = validate_foundation_spec(spec)
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test preprocessing is sealed")
    if type(workers) is not int or not 1 <= workers <= 72 or max_ram_bytes <= 0:
        raise ValueError("Invalid preparation workers/RAM ceiling")
    tasks = [r for r in spec["assignment_tasks"] if r["role"] == role]
    # Conservative pre-allocation check including ragged metadata and bounded
    # worker/IPC/concatenation copies; no wait for an OS OOM after allocation.
    peak_upper = preparation_bound(spec, role, workers)
    if peak_upper > max_ram_bytes:
        raise MemoryError(f"Conservative RAM preparation bound {peak_upper} exceeds budget {max_ram_bytes}; reduce workers or increase RAM")
    args = [(spec, str(data_root), str(foundation_root), t, coordinate_name) for t in tasks]
    blocks, resident = [], 0
    started = time.monotonic()
    def accept(block):
        nonlocal resident
        # Reserve room for identities/labels concatenation, worker output, batches.
        resident += block.nbytes + len(block.labels) * 40
        if resident > max_ram_bytes:
            raise MemoryError("RAM cache exceeds explicit budget; disk spilling is forbidden")
        blocks.append(block)
        print(f"JC2 phase=cache role={role} coordinate={coordinate_name} files={len(blocks)}/{len(tasks)} resident_GiB={resident/2**30:.2f} seconds={time.monotonic()-started:.1f}", flush=True)
    if workers == 1:
        for arg in args:
            accept(_prepare_file(arg))
    else:
        # Bounded ordered results. Unlike executor.map this cannot enqueue the
        # full raw snapshot or accumulate all completed later files in memory.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_limit_worker_threads) as pool:
            iterator = iter(args)
            pending = [pool.submit(_prepare_file, arg) for arg in (next(iterator, None) for _ in range(workers)) if arg is not None]
            while pending:
                accept(pending.pop(0).result())
                arg = next(iterator, None)
                if arg is not None:
                    pending.append(pool.submit(_prepare_file, arg))
    cache = RamCache(blocks, role=role, foundation_sha256=parent, coordinate_name=coordinate_name)
    if len(cache) != spec["splits"]["role_counts"][role]:
        raise ValueError("Prepared population differs from frozen split")
    return cache
