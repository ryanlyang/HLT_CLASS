"""Ragged FP32 TRAIN targets, immutable shards and bounded read-only mmap LRU."""
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref, safe
from hlt_classification.data.cache_contracts import (atomic_publish_bytes, deterministic_npz_bytes,
    load_npz_arrays, write_immutable_json, load_json)
from hlt_classification.jetclass2_delphes.runner import _forward
from .contracts import artifact, validate, SURFACE, TEACHER

TARGET_SHARD_BYTES = 64*2**20
MAX_MAPPED_BYTES = 4*2**30


def counts(cache):
    return np.concatenate([np.diff(b.offsets) for b in cache.blocks]).astype(np.int64)


def alignment(cache):
    n = counts(cache)
    if (len(n) != len(cache) or np.any((n < 1) | (n > 512))
            or len(np.unique(cache.identities, axis=0)) != len(cache)):
        raise ValueError('Invalid persistent-slot support/identities')
    digest = sha256()
    for array in (cache.identities, cache.labels.astype('<i8'), n.astype('<i8')):
        digest.update(array.tobytes())
    return dict(prepared=cache.foundation_sha256, role=cache.role,
                ordered_slots_sha256=digest.hexdigest(), rows=len(cache))


def check_matrices(values, offsets, lengths):
    if (values.dtype != np.float32 or offsets.dtype != np.int64 or lengths.dtype != np.int64
            or values.ndim != 1 or offsets.shape != (len(lengths)+1,)
            or offsets[0] != 0 or offsets[-1] != len(values)
            or not np.array_equal(np.diff(offsets), lengths**2)):
        raise ValueError('Invalid ragged attention schema')
    for i, n in enumerate(lengths):
        a = values[offsets[i]:offsets[i+1]].reshape(n, n)
        if (not np.isfinite(a).all() or np.any(a < 0)
                or not np.allclose(a.sum(-1), 1., atol=2e-6, rtol=0)):
            raise ValueError('Invalid attention target distribution')


def export(root, model, cache, *, parents, teacher, attention, device):
    if cache.role != 'train' or cache.coordinate_name != 'D033' or teacher != TEACHER or attention is not True:
        raise PermissionError('Only the frozen D033 TRAIN attention bank may be exported')
    binding, lengths = alignment(cache), counts(cache)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    probabilities = np.empty((len(cache), 11), np.float32)
    shards = []
    pending = []
    shard_start, pending_bytes, pending_rows = 0, 0, 0
    def flush(stop):
        nonlocal pending, shard_start, pending_bytes, pending_rows
        flat = np.concatenate(pending)
        offsets = np.concatenate(([0], np.cumsum(lengths[shard_start:stop]**2))).astype(np.int64)
        check_matrices(flat, offsets, lengths[shard_start:stop])
        buf = BytesIO()
        np.save(buf, flat, allow_pickle=False)
        name = f'attention-{shard_start:07d}.npy'
        atomic_publish_bytes(root/name, buf.getvalue())
        shards.append(dict(start=shard_start, stop=stop, offsets=offsets.tolist(),
                           payload=file_ref(root/name, root=root)))
        pending, shard_start, pending_bytes, pending_rows = [], stop, 0, 0
    model.to(device).eval()
    model.capture_attention = attention
    # Pack batches to ~64 MiB (at most one extra 128 MiB batch), <=4096 rows.
    # This avoids thousands of tiny random-open operations per shuffled pass.
    with torch.inference_mode():
        for start in range(0, len(cache), 128):
            stop = min(start+128, len(cache))
            ix = np.arange(start, stop)
            logits = _forward(model, cache.batch(ix), device, bf16=str(device).startswith('cuda'))
            probabilities[ix] = (logits.float()/2.).softmax(-1).cpu().numpy()
            if attention:
                a = model.take_attention().cpu().numpy()
                ns = lengths[ix]
                offsets = np.concatenate(([0], np.cumsum(ns**2))).astype(np.int64)
                flat = np.concatenate([row[:n, :n].reshape(-1) for row, n in zip(a, ns)])
                check_matrices(flat, offsets, ns)
                pending.append(flat)
                pending_bytes += flat.nbytes
                pending_rows += len(ix)
                if pending_bytes >= TARGET_SHARD_BYTES or pending_rows >= 4096:
                    flush(stop)
            if start % (128*100) == 0:
                print(f'LUKA attention export {teacher}: {stop}/{len(cache)}', flush=True)
    model.capture_attention = False
    if pending:
        flush(len(cache))
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(-1), 1., atol=2e-6, rtol=0)):
        raise ValueError('Invalid class probabilities')
    atomic_publish_bytes(root/'population.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, counts=lengths, probabilities=probabilities)))
    manifest = artifact('BANK', parents=parents, teacher=teacher, coordinate=cache.coordinate_name,
                        alignment=binding, attention=attention, surface=dict(SURFACE) if attention else None,
                        temperature=2., population=file_ref(root/'population.npz', root=root), shards=shards)
    write_immutable_json(root/'manifest.json', manifest)
    return manifest


class Bank:
    def __init__(self, root, cache, *, parents, teacher, require_attention):
        if cache.role != 'train':
            raise PermissionError('No validation/test target bank')
        self.root = Path(root)
        m = self.manifest = load_json(self.root/'manifest.json')
        validate(m, 'BANK', parents)
        # Teacher self-readback and the single registered HLT student only.
        if (cache.coordinate_name not in ('D033', 'D000') or teacher != TEACHER
                or m['teacher'] != teacher or m['coordinate'] != 'D033'
                or m['alignment'] != alignment(cache) or m['temperature'] != 2.
                or m['attention'] is not True
                or m['surface'] != (SURFACE if m['attention'] else None)):
            raise ValueError('Teacher/slot/surface binding differs')
        validate_file_ref(m['population'], root=self.root)
        pop = load_npz_arrays(safe(self.root, m['population']['path']))
        if (set(pop) != {'identities', 'labels', 'counts', 'probabilities'}
                or pop['counts'].dtype != np.int64 or pop['labels'].dtype != np.int64
                or pop['identities'].dtype != np.uint8
                or not np.array_equal(pop['counts'], counts(cache))
                or not np.array_equal(pop['identities'], cache.identities)
                or not np.array_equal(pop['labels'], cache.labels)):
            raise ValueError('Teacher population/order differs')
        self.probabilities, self.counts = pop['probabilities'], pop['counts']
        self.identities = pop['identities']
        q = self.probabilities
        if (q.shape != (len(cache), 11) or q.dtype != np.float32 or not np.isfinite(q).all()
                or np.any(q < 0) or not np.allclose(q.sum(-1), 1, atol=2e-6, rtol=0)):
            raise ValueError('Invalid teacher probabilities')
        stop, seen = 0, set()
        for row in m['shards']:
            a, b = row['start'], row['stop']
            if a != stop or not a < b <= min(a+4096, len(cache)) or row['payload']['path'] in seen:
                raise ValueError('Shard coverage/order differs')
            seen.add(row['payload']['path'])
            validate_file_ref(row['payload'], root=self.root)
            data = np.load(safe(self.root, row['payload']['path']), mmap_mode='r', allow_pickle=False)
            check_matrices(data, np.asarray(row['offsets'], np.int64), self.counts[a:b])
            del data
            stop = b
        if stop != (len(cache) if m['attention'] else 0):
            raise ValueError('Incomplete attention bank')
        self.ends = np.array([s['stop'] for s in m['shards']])
        self._lru = OrderedDict()

    def attention_batch(self, indices, width):
        if not self.manifest['attention']:
            raise ValueError('Bank has no attention targets')
        ix = np.asarray(indices)
        if (ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix)
                or np.any(ix < 0) or np.any(ix >= len(self.counts))
                or width < self.counts[ix].max()):
            raise ValueError('Invalid attention selection')
        out = np.zeros((len(ix), width, width), np.float32)
        for dest, index in enumerate(ix):
            which = int(np.searchsorted(self.ends, index, side='right'))
            row = self.manifest['shards'][which]
            if which not in self._lru:
                self._lru[which] = np.load(safe(self.root, row['payload']['path']), mmap_mode='r', allow_pickle=False)
                while (len(self._lru) > 64 or sum(a.nbytes for a in self._lru.values()) > MAX_MAPPED_BYTES):
                    self._lru.popitem(last=False)
            self._lru.move_to_end(which)
            data = self._lru[which]
            local, n = int(index-row['start']), self.counts[index]
            lo, hi = row['offsets'][local:local+2]
            out[dest, :n, :n] = data[lo:hi].reshape(n, n)
        return out
