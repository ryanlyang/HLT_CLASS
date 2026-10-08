"""Ordinary-role RAM views; D000 delegates to the unchanged screen cache."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path
import numpy as np

from hlt_classification.gap_sweep import cache as screen_cache
from hlt_classification.cms_proxy_ladder.cache import _limit_worker_threads, role_sources, preparation_bound, cache_budgets
from hlt_classification.cms_proxy_ladder.cache_full import ordered_results
from hlt_classification.cms_proxy_ladder.data import iter_paired, load_assignments
from hlt_classification.cms_proxy_ladder.release import load_bank
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from .views import build_view, build_inputs, COORDINATES


def identities(parent, role):
    if role not in ('train', 'validation'):
        raise PermissionError('S3 final test is sealed')
    f = parent['foundation']
    bank = load_bank(f['release'], root=Path(f['release_root']))
    return bank['identity'][bank['role'] == (0 if role == 'train' else 1)]


def _prepare(arguments):
    parent, role, coordinate, sources = arguments
    f = parent['foundation']
    ids, starts, mapping = load_assignments(f, root=Path(parent['foundation_root']), role=role)
    offsets, features, vectors, keys, labels = [0], [], [], [], []
    for row in iter_paired(f['release'], release_root=Path(f['release_root']), role=role,
                           source_file_index=tuple(sources)):
        if bytes(ids[row.ordinal]).hex() != row.identity:
            raise ValueError('S3 assignment identity differs')
        lo, hi = starts[row.ordinal:row.ordinal+2]
        view = build_view(proxy=row.proxy, offline=row.offline, identity=row.identity,
                          coordinate=coordinate, mapping=mapping[int(lo):int(hi)])
        encoded = build_inputs(view, capacity=f['inputs']['capacity'])
        offsets.append(offsets[-1]+len(view)); features.append(encoded.features); vectors.append(encoded.vectors)
        keys.append(np.frombuffer(bytes.fromhex(row.identity), np.uint8)); labels.append(row.label)
    return RamBlock(int(sources[0]), np.asarray(offsets, np.int64),
        np.concatenate(features) if features else np.empty((0, 17), np.float32),
        np.concatenate(vectors) if vectors else np.empty((0, 4), np.float32),
        np.asarray(keys, np.uint8).reshape(-1, 32), np.asarray(labels, np.int64))


def prepare(parent, coordinate, role, *, workers, input_identity, max_ram_bytes):
    if role not in ('train', 'validation'):
        raise PermissionError('S3 final test is sealed')
    if coordinate not in COORDINATES:
        raise ValueError('Unknown S3 coordinate')
    if coordinate == 'D000':
        return screen_cache.prepare(parent, 'S3', role, workers=workers,
            input_identity=input_identity, max_ram_bytes=max_ram_bytes)
    f = parent['foundation']
    if preparation_bound(f, role, workers) > max_ram_bytes:
        raise MemoryError('S3 preparation bound exceeds RAM')
    sources = role_sources(f, role)
    chunks = [tuple(map(int, v)) for v in np.array_split(sources, min(workers, len(sources))) if len(v)]
    arguments = [(parent, role, coordinate, chunk) for chunk in chunks]
    blocks, resident = [], 0

    def accept(block):
        nonlocal resident
        resident += block.nbytes+40*len(block.labels)
        if resident > max_ram_bytes:
            raise MemoryError('S3 RAM spilling forbidden')
        blocks.append(block)
        print(f'S3 cache={coordinate}/{role} blocks={len(blocks)}/{len(chunks)} GiB={resident/2**30:.2f}', flush=True)

    if workers == 1:
        for args in arguments:
            accept(_prepare(args))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_limit_worker_threads) as pool:
            for block in ordered_results(pool, _prepare, arguments, workers):
                accept(block)
    value = RamCache(blocks, role=role, foundation_sha256=input_identity, coordinate_name=coordinate)
    if (len(value) != f['role_counts'][role] or not np.array_equal(value.identities, identities(parent, role))
            or not np.isin(value.labels, np.arange(11)).all()):
        raise ValueError('S3 population/order/labels differ')
    digest = hashlib.sha256()
    for block in blocks:
        for a in (block.offsets, block.features, block.vectors, block.identities, block.labels):
            digest.update(a.tobytes())
    return value, dict(endpoint_sha256=digest.hexdigest())


def _replay_one(args):
    proxy, offline, identity, mapping, coordinate, capacity = args
    out = build_view(proxy=proxy, offline=offline, identity=identity, mapping=mapping, coordinate=coordinate)
    enc = build_inputs(out, capacity=capacity)
    digest = hashlib.sha256()
    for a in (out.p4, out.charge, out.category, out.tracking, out.valid, enc.features, enc.vectors):
        digest.update(a.tobytes())
    digest.update(repr(out.keys).encode())
    return digest.hexdigest()


def replay(parent, workers):
    f = parent['foundation']
    ids, offsets, mapping = load_assignments(f, root=Path(parent['foundation_root']), role='train')
    args = []
    for row in iter_paired(f['release'], release_root=Path(f['release_root']), role='train'):
        if bytes(ids[row.ordinal]).hex() != row.identity:
            raise ValueError('Replay assignment identity differs')
        lo, hi = offsets[row.ordinal:row.ordinal+2]
        args.extend((row.proxy, row.offline, row.identity, mapping[int(lo):int(hi)], name,
                     f['inputs']['capacity']) for name in COORDINATES)
        if len(args) == 32*len(COORDINATES):
            break
    if not args:
        raise ValueError('No replay rows')
    serial = [_replay_one(a) for a in args]
    with ProcessPoolExecutor(max_workers=min(2, workers), mp_context=multiprocessing.get_context('spawn'),
                             initializer=_limit_worker_threads) as pool:
        parallel = list(pool.map(_replay_one, args))
    if serial != parallel:
        raise ValueError('S3 process replay differs')
    return dict(jets=len(args)//len(COORDINATES), coordinates=list(COORDINATES), hashes=serial, exact=True)
