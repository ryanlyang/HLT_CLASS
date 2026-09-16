"""Native offline + native HLT, packed in RAM without matching or duplication."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .cache import RamBlock, RamCache, _limit_worker_threads
from .inputs import build_inputs
from .reader import DatasetReader
from .salience_foundation import validate_foundation_spec


def jet_inputs(jet, capacity):
    if jet.offline is None:
        raise ValueError("Native concatenation requires both reconstructions")
    features, vectors = [], []
    for code, particles in enumerate((jet.offline, jet.hlt)):
        value = build_inputs(particles, capacity=capacity)
        count = len(particles)
        features.append(np.column_stack((
            value.features[:, :count].T, np.full(count, code, np.float32),
        )))
        vectors.append(value.vectors[:, :count].T)
    return np.concatenate(features), np.concatenate(vectors)


def _prepare_file(arguments):
    spec, data_root, task = arguments
    reader = DatasetReader(
        Path(data_root), spec["inventory"], spec["splits"], role=task["role"],
        include_offline=True, file_paths=(task["path"],),
    )
    offsets, features, vectors, identities, labels = [0], [], [], [], []
    for jet in reader:
        x, v = jet_inputs(jet, spec["inputs"]["capacity"])
        features.append(x)
        vectors.append(v)
        offsets.append(offsets[-1] + len(x))
        identities.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
        labels.append(jet.label)
    block = RamBlock(
        task["file_index"], np.asarray(offsets, np.int64),
        np.concatenate(features), np.concatenate(vectors),
        np.asarray(identities, np.uint8), np.asarray(labels, np.int64),
    )
    if len(labels) != task["rows"]:
        raise ValueError("Native concatenation row coverage differs")
    return block


class ConcatCache(RamCache):
    def __init__(self, blocks, *, role, foundation_sha256, capacity):
        super().__init__(blocks, role=role, foundation_sha256=foundation_sha256,
                         coordinate_name="NATIVE_OFFLINE_HLT_CONCAT")
        self.capacity = capacity
        self.lengths = np.concatenate([np.diff(block.offsets) for block in blocks])
        self.nbytes += self.lengths.nbytes
        if not len(self) or np.any(self.lengths > capacity) or np.any(self.lengths < 2):
            raise ValueError("Invalid native concatenation lengths")

    def batch(self, indices):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                or np.any(indices < 0) or np.any(indices >= len(self))):
            raise ValueError("Invalid concatenation batch indices")
        length = max(16, int(self.lengths[indices].max()))
        x = np.zeros((len(indices), 18, length), np.float32)
        x[:, 17] = -1
        v = np.zeros((len(indices), 4, length), np.float32)
        mask = np.zeros((len(indices), 1, length), bool)
        for out, index in enumerate(indices):
            b = int(np.searchsorted(self.ends, index, side="right"))
            row = int(index - (self.ends[b-1] if b else 0))
            block = self.blocks[b]
            start, stop = block.offsets[row:row+2]
            count = int(stop - start)
            x[out, :, :count] = block.features[start:stop].T
            v[out, :, :count] = block.vectors[start:stop].T
            mask[out, :, :count] = True
        return dict(features=x, vectors=v, mask=mask,
                    labels=self.labels[indices], identities=self.identities[indices])


def memory_bounds(spec, workers):
    if type(workers) is not int or not 1 <= workers <= 36:
        raise ValueError("Invalid concatenation preparation workers")
    bounds = {}
    for role in ("train", "validation"):
        sizes = [t["rows"] * (2 * spec["inputs"]["capacity"] * 88 + 1024)
                 for t in spec["assignment_tasks"] if t["role"] == role]
        # Both complete role caches plus bounded ordered worker/IPC copies.
        bounds[role] = sum(sizes) + 4 * min(workers, len(sizes)) * max(sizes, default=0)
    return bounds


def prepare_cache(spec, *, data_root, role, workers, max_ram_bytes):
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test concatenation is sealed")
    parent = validate_foundation_spec(spec)
    if memory_bounds(spec, workers)[role] > max_ram_bytes:
        raise MemoryError("Concatenation preparation exceeds RAM bound")
    tasks = [t for t in spec["assignment_tasks"] if t["role"] == role]
    arguments = [(spec, str(data_root), t) for t in tasks]
    blocks, resident = [], 0
    started = time.monotonic()

    def accept(block):
        nonlocal resident
        resident += block.nbytes + len(block.labels) * 48
        if resident > max_ram_bytes:
            raise MemoryError("Concatenation cache cannot spill to disk")
        blocks.append(block)
        print(f"JC2-CONCAT cache role={role} files={len(blocks)}/{len(tasks)} "
              f"GiB={resident/2**30:.2f} seconds={time.monotonic()-started:.1f}", flush=True)

    if workers == 1:
        for argument in arguments:
            accept(_prepare_file(argument))
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_limit_worker_threads) as pool:
            iterator = iter(arguments)
            pending = [pool.submit(_prepare_file, a) for a in
                       (next(iterator, None) for _ in range(workers)) if a is not None]
            while pending:
                accept(pending.pop(0).result())
                argument = next(iterator, None)
                if argument is not None:
                    pending.append(pool.submit(_prepare_file, argument))
    cache = ConcatCache(blocks, role=role, foundation_sha256=parent,
                        capacity=2 * spec["inputs"]["capacity"])
    if len(cache) != spec["splits"]["role_counts"][role]:
        raise ValueError("Concatenation population differs from frozen split")
    return cache
