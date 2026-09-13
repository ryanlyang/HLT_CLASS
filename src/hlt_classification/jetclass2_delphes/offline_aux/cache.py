"""Native HLT ragged RAM caches, with explicit auxiliary-study row capabilities."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import numpy as np

from ..cache import RamBlock, RamCache, _limit_worker_threads
from ..inputs import build_inputs
from .roles import read_rows, authorize_role


class AuxiliaryCache:
    # Reuse only the pure array batching operation, not the old reader or its
    # train/validation/foundation capability.
    batch = RamCache.batch
    __len__ = RamCache.__len__

    def __init__(self, blocks, *, role, split_sha256):
        self.blocks, self.role, self.split_sha256 = blocks, role, split_sha256
        self.ends = np.cumsum([len(b.labels) for b in blocks])
        self.labels = np.concatenate([b.labels for b in blocks])
        self.identities = np.concatenate([b.identities for b in blocks])
        self.nbytes = sum(b.nbytes for b in blocks) + self.labels.nbytes + self.identities.nbytes


def ordered_process(function, arguments, workers):
    if not 1 <= workers <= 36:
        raise ValueError("Invalid bounded SPORC worker count")
    if workers == 1:
        for arg in arguments:
            yield function(arg)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_limit_worker_threads) as pool:
        iterator = iter(arguments)
        pending = []
        for _ in range(workers):
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(function, item))
        while pending:
            yield pending.pop(0).result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(function, item))


def _file(args):
    data_root, inventory, split, metadata, role, lock, fi = args
    offsets, features, vectors, ids, labels = [0], [], [], [], []
    for _, jet in read_rows(data_root, inventory, split, metadata, role=role, include_offline=False,
                            reporting_lock=lock, file_index=fi):
        transformed = build_inputs(jet.hlt, capacity=240)
        n = len(jet.hlt)
        features.append(transformed.features[:, :n].T.copy())
        vectors.append(transformed.vectors[:, :n].T.copy())
        ids.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
        labels.append(jet.label)
        offsets.append(offsets[-1] + n)
    return RamBlock(fi, np.array(offsets, np.int64), np.concatenate(features), np.concatenate(vectors),
                    np.array(ids, np.uint8), np.array(labels, np.int64))


def prepare(data_root, inventory, split, metadata, *, role, workers, budget_bytes, reporting_lock=None):
    authorize_role(role, split, reporting_lock)
    counts = np.unique(metadata["file_index"], return_counts=True)[1]
    bound = int(len(metadata["labels"]) * (240*84+1024) + 4*min(workers, len(counts))*max(counts)*(240*84+1024))
    if bound > budget_bytes:
        raise MemoryError(f"Conservative RAM cache/worker bound {bound} exceeds {budget_bytes}")
    started = time.monotonic()
    args = [(data_root, inventory, split, metadata, role, reporting_lock, int(fi))
            for fi in np.unique(metadata["file_index"])]
    blocks = []
    for block in ordered_process(_file, args, workers):
        blocks.append(block)
        print(f"JC2AUX cache role={role} files={len(blocks)}/{len(args)} seconds={time.monotonic()-started:.1f}", flush=True)
    cache = AuxiliaryCache(blocks, role=role, split_sha256=split["content_hash"])
    if (not np.array_equal(cache.identities, metadata["identities"])
            or not np.array_equal(cache.labels, metadata["labels"])):
        raise ValueError("Native HLT cache ordered row join differs")
    return cache
