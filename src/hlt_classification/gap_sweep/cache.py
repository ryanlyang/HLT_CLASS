"""Ephemeral baseline inputs; no matching, particle export, labels in degradation, or test access."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path
import numpy as np

from hlt_classification.cms_proxy_ladder.cache import (
    _limit_worker_threads, role_sources, preparation_bound, cache_budgets)
from hlt_classification.cms_proxy_ladder.cache_full import ordered_results
from hlt_classification.cms_proxy_ladder.data import iter_paired
from hlt_classification.cms_proxy_ladder.release import load_bank
from hlt_classification.correlated_topology.views import build_inputs
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from .kernel import generate, CANDIDATES


def _prepare(arguments):
    foundation, role, candidate, sources = arguments
    offsets, features, vectors, identities, labels = [0], [], [], [], []
    counts, stats = Counter(), Counter()
    for row in iter_paired(foundation['release'], release_root=Path(foundation['release_root']),
                           role=role, source_file_index=tuple(sources)):
        proxy, mechanisms = generate(row.proxy, row.identity, candidate)
        encoded = build_inputs(proxy, capacity=foundation['inputs']['capacity'])
        features.append(encoded.features); vectors.append(encoded.vectors)
        offsets.append(offsets[-1]+len(proxy))
        identities.append(np.frombuffer(bytes.fromhex(row.identity), np.uint8))
        labels.append(row.label)
        if role == 'train':
            counts.update(mechanisms)
            for side, p in (('OFFLINE', row.offline), ('BASE', row.proxy), (candidate, proxy)):
                stats[side+'/particles'] += len(p)
                stats[side+'/multiplicity_squared'] += len(p)**2
                stats[side+'/charged_particles'] += int((p.charge != 0).sum())
                for i, field in enumerate(('d0', 'dz', 'd0err', 'dzerr')):
                    stats[side+'/valid_'+field] += int(p.valid[:, i].sum())
    block = RamBlock(int(sources[0]), np.asarray(offsets, np.int64),
        np.concatenate(features) if features else np.empty((0, 17), np.float32),
        np.concatenate(vectors) if vectors else np.empty((0, 4), np.float32),
        np.asarray(identities, np.uint8).reshape(-1, 32), np.asarray(labels, np.int64))
    return block, dict(counts), dict(stats)


def prepare(parent, candidate, role, *, workers, input_identity, max_ram_bytes):
    if role not in ('train', 'validation'):
        raise PermissionError('Gap screen cannot open final test')
    if candidate not in CANDIDATES:
        raise ValueError('Unknown candidate')
    f = parent['foundation']
    if preparation_bound(f, role, workers) > max_ram_bytes:
        raise MemoryError('Conservative candidate preparation bound exceeds budget')
    sources = role_sources(f, role)
    chunks = [tuple(map(int, v)) for v in np.array_split(sources, min(workers, len(sources))) if len(v)]
    arguments = [(f, role, candidate, chunk) for chunk in chunks]
    blocks, counts, stats, resident = [], Counter(), Counter(), 0

    def accept(value):
        nonlocal resident
        block, mechanisms, statistics = value
        resident += block.nbytes+40*len(block.labels)
        if resident > max_ram_bytes:
            raise MemoryError('No disk spilling permitted')
        blocks.append(block); counts.update(mechanisms); stats.update(statistics)
        print(f'GAP cache={candidate}/{role} blocks={len(blocks)}/{len(chunks)} GiB={resident/2**30:.2f}', flush=True)

    if workers == 1:
        for argument in arguments:
            accept(_prepare(argument))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_limit_worker_threads) as pool:
            for value in ordered_results(pool, _prepare, arguments, workers):
                accept(value)
    cache = RamCache(blocks, role=role, foundation_sha256=input_identity, coordinate_name='D000')
    bank = load_bank(f['release'], root=Path(f['release_root']))
    mask = bank['role'] == (0 if role == 'train' else 1)
    if (len(cache) != f['role_counts'][role] or not np.array_equal(cache.identities, bank['identity'][mask])
            or not np.isin(cache.labels, np.arange(11)).all()):
        raise ValueError('Candidate population/order or authenticated source labels invalid')
    digest = hashlib.sha256()
    for block in blocks:
        for a in (block.offsets, block.features, block.vectors, block.identities, block.labels):
            digest.update(a.tobytes())
    return cache, dict(mechanisms=dict(counts), statistics=dict(stats), endpoint_sha256=digest.hexdigest())


def _replay_one(arguments):
    p, identity, candidate, capacity = arguments
    out, _ = generate(p, identity, candidate)
    encoded = build_inputs(out, capacity=capacity)
    digest = hashlib.sha256()
    for a in (out.p4, out.charge, out.category, out.tracking, out.valid, encoded.features, encoded.vectors):
        digest.update(a.tobytes())
    digest.update(repr(out.keys).encode())
    return digest.hexdigest()


def replay(parent, workers):
    f = parent['foundation']
    rows = []
    for row in iter_paired(f['release'], release_root=Path(f['release_root']), role='train'):
        rows.append(row)
        if len(rows) == 32:
            break
    if not rows:
        raise ValueError('No replay rows')
    args = [(r.proxy, r.identity, c, f['inputs']['capacity']) for c in CANDIDATES for r in rows]
    serial = [_replay_one(a) for a in args]
    with ProcessPoolExecutor(max_workers=min(2, workers), mp_context=multiprocessing.get_context('spawn'),
                             initializer=_limit_worker_threads) as pool:
        parallel = list(pool.map(_replay_one, args))
    if serial != parallel:
        raise ValueError('Candidate serial/process replay differs')
    return dict(jets_per_candidate=len(rows), candidate_order=list(CANDIDATES), hashes=serial)
