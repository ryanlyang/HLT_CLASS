"""Byte-identical ragged targets with zero filesystem access during training."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import numpy as np

from hlt_classification.cms_proxy_ladder.contracts import safe
from hlt_classification.luka_fullstat_attention.banks import Bank


class ResidentBank(Bank):
    def __init__(self, *args, max_bytes, **kwargs):
        # Retain EVERY original source, payload, probability and slot check.
        super().__init__(*args, **kwargs)
        elements = sum(int(n)**2 for n in self.counts)
        required = elements*4 + (len(self.counts)+1)*8
        if required > max_bytes:
            raise MemoryError(f'Resident targets need {required/2**30:.2f} GiB; budget {max_bytes/2**30:.2f} GiB')
        self.offsets = np.concatenate(([0], np.cumsum(self.counts**2))).astype(np.int64)
        self.values = np.empty(elements, np.float32)
        self.loaded_bytes = required
        for i, shard in enumerate(self.manifest['shards']):
            ref = shard['payload']
            # Hash the exact bytes being copied, not a prior read of this path.
            blob = safe(self.root, ref['path']).read_bytes()
            if len(blob) != ref['bytes'] or sha256(blob).hexdigest() != ref['sha256']:
                raise ValueError('Target shard changed during resident load')
            data = np.load(BytesIO(blob), allow_pickle=False)
            lo, hi = self.offsets[shard['start']], self.offsets[shard['stop']]
            self.values[lo:hi] = data
            del data, blob
            if i % 100 == 0:
                print(f'FAST attention resident load: {i+1}/{len(self.manifest["shards"])} shards', flush=True)
        for a in (self.values, self.offsets, self.probabilities, self.identities, self.counts):
            a.flags.writeable = False

    def attention_batch(self, indices, width):
        ix = np.asarray(indices)
        if (ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix)
                or np.any(ix < 0) or np.any(ix >= len(self.counts))
                or not isinstance(width, (int, np.integer)) or not 1 <= width <= 512
                or width < self.counts[ix].max()):
            raise ValueError('Invalid attention selection')
        out = np.zeros((len(ix), width, width), np.float32)
        for dest, index in enumerate(ix):
            n = self.counts[index]
            out[dest, :n, :n] = self.values[self.offsets[index]:self.offsets[index+1]].reshape(n, n)
        return out


def batches(cache, bank, order, *, depth=2):
    """One CPU producer; bounded futures, original exact order, no RNG/CUDA work.

    Preparation overlaps the previous GPU update. Exceptions propagate before
    their batch is consumed. Buffers are owned by each batch, never recycled.
    """
    if depth not in (0, 2):
        raise ValueError('Unregistered prefetch depth')
    def prepare(start):
        ix = order[start:start+128]
        raw = cache.batch(ix)
        target = bank.attention_batch(ix, raw['mask'].shape[-1])
        return ix, raw, target
    starts = iter(range(0, len(order), 128))
    if depth == 0:
        for start in starts:
            yield prepare(start)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='attention-input') as pool:
        pending = deque()
        for _ in range(depth):
            start = next(starts, None)
            if start is not None:
                pending.append(pool.submit(prepare, start))
        while pending:
            value = pending.popleft().result()
            start = next(starts, None)
            if start is not None:
                pending.append(pool.submit(prepare, start))
            yield value
