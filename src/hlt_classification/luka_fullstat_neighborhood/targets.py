"""Bounded physical TRAIN extraction; original cache bytes must replay exactly."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import multiprocessing
import time
import numpy as np
from hlt_classification.luka_fullstat import cache, data
from hlt_classification.luka_fullstat.contracts import BLOCK_ROWS, identity
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_fullsim.stage2 import build_inputs
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.jetclass2_delphes.cache import _limit_worker_threads
from hlt_classification.luka_neighborhood.targets import descriptors, Moments

_WORK = None


def _init(path, record, membership, config, cache_root):
    global _WORK
    _limit_worker_threads()
    _WORK = path, record, membership, config, cache_root


def _work(chunk):
    path, record, member, config, root = _WORK
    if chunk['role'] != 'train':
        raise PermissionError('No validation/test neighborhood targets')
    reference = cache.read_block(root, chunk, 'D000', record, member)
    rows, values = [], []
    start = chunk['block_index']*BLOCK_ROWS
    # Inherited role-safe reader never requests sealed/unused particle entries.
    # Ordinary validation rows may be decoded, but never enter targets/statistics.
    for role, entry, raw_hlt, raw_off in cache.read_selected_chunk(
            path, record, member, start, min(start+BLOCK_ROWS, record['entries'])):
        if role != 'train':
            continue
        hlt, off = physical(raw_hlt, config, 'hlt'), physical(raw_off, config, 'offline')
        rows.append((SimpleNamespace(identity=identity(record, entry), label=int(member['labels'][entry])),
                     build_inputs(hlt, capacity=512)))
        values.append(descriptors(hlt, off, smooth=True, chunk=32))
    rebuilt = _block(chunk['file_index'], rows)
    if any(not np.array_equal(getattr(reference, k), getattr(rebuilt, k))
           for k in ('identities', 'labels', 'offsets', 'features', 'vectors')):
        raise ValueError('Physical HLT endpoint/order/identity replay differs')
    return dict(identities=reference.identities, labels=reference.labels,
                offsets=reference.offsets, values=np.concatenate(values))


def chunks(spec, meta):
    manifest, foundation, memberships = meta
    e = spec['anchor']['evidence']
    if manifest['content_hash'] != e['prepared']:
        raise ValueError('Prepared source binding differs')
    workers = spec['preparation_workers']
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError('Unbounded preparation worker request')
    seen, started = 0, time.monotonic()
    for i, (record, member) in enumerate(zip(foundation['files'], memberships, strict=True)):
        path = data.authenticate_source(e['raw_container'], record)
        selected = [c for c in manifest['chunks'] if c['file_index'] == i and c['role'] == 'train']
        args = str(path), record, member, manifest['conventions'], e['cache_root']
        if workers == 1:
            _init(*args)
            for chunk in selected:
                yield _work(chunk)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=_init, initargs=args) as pool:
                pending, iterator = deque(), iter(selected)
                for _ in range(workers):
                    chunk = next(iterator, None)
                    if chunk is not None:
                        pending.append(pool.submit(_work, chunk))
                while pending:
                    yield pending.popleft().result()
                    chunk = next(iterator, None)
                    if chunk is not None:
                        pending.append(pool.submit(_work, chunk))
        seen += sum(c['views']['D000']['rows'] for c in selected)
        print(f'FULLSTAT_NEIGHBOR targets rows={seen}/{spec["counts"]["train"]} '
              f'seconds={time.monotonic()-started:.1f}', flush=True)
        data.authenticate_source(e['raw_container'], record)
