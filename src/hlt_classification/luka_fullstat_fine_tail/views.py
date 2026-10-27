"""Exact-ninths physical views from immutable original assignments, built once in RAM."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import hashlib
import multiprocessing
import time
import numpy as np
from hlt_classification.cms_proxy_ladder import views as original
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.data.cache_contracts import load_npz_arrays
from hlt_classification.jetclass2_delphes.cache import RamCache, _limit_worker_threads
from hlt_classification.luka_fullstat import cache, data
from hlt_classification.luka_fullstat.contracts import BLOCK_ROWS, ROLES, identity
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.literature_context.transform import build_inputs
from .contracts import FRACTIONS, STAGES, view_recipe, artifact, validate

_WORK = None


def fraction_view(*, identity, proxy, offline, alpha, mapping):
    """Original D kernel, explicit rational alpha; same switch hash and groups.

    New registration lives here, leaving every old decimal-coordinate API intact.
    No ParT-feature interpolation and no rerun of the matching algorithm.
    """
    if not isinstance(alpha, Fraction) or not 0 <= alpha <= 1:
        raise ValueError('Exact bounded rational offline fraction required')
    if alpha == 0:
        return proxy
    if len(identity) != 64 or offline is None:
        raise ValueError('Authenticated identity and physical offline endpoint required')
    mapping = np.asarray(mapping)
    original.validate_pairing(mapping, nh=len(proxy), no=len(offline))
    p4, charge, category, tracking, valid = [np.array(getattr(proxy, k), copy=True)
        for k in ('p4', 'charge', 'category', 'tracking', 'valid')]
    beta = 1-alpha
    for slot in np.flatnonzero(mapping >= 0):
        src = int(mapping[slot])
        p4[slot] = float(beta)*proxy.p4[slot]+float(alpha)*offline.p4[src]
        take = original._choose_offline(identity, int(slot), 'identity', alpha)
        endpoint, index = (offline, src) if take else (proxy, int(slot))
        charge[slot], category[slot] = endpoint.charge[index], endpoint.category[index]
        if (proxy.charge[slot] != 0) != (offline.charge[src] != 0):
            tracking[slot], valid[slot] = endpoint.tracking[index], endpoint.valid[index]
            continue
        for value_index, error_index, group in ((0, 2, 'd0'), (1, 3, 'dz')):
            positions = [value_index, error_index]
            pv, ov = proxy.valid[slot, positions], offline.valid[src, positions]
            if np.array_equal(pv, ov):
                valid[slot, positions] = pv
                for position, index in enumerate(positions):
                    tracking[slot, index] = (float(beta)*proxy.tracking[slot, index]
                        + float(alpha)*offline.tracking[src, index]) if pv[position] else 0.
            else:
                take = original._choose_offline(identity, int(slot), group, alpha)
                endpoint, index = (offline, src) if take else (proxy, int(slot))
                tracking[slot, positions] = endpoint.tracking[index, positions]
                valid[slot, positions] = endpoint.valid[index, positions]
    return original._particles(p4, charge, category, tracking, valid,
                               tuple('proxy:'+key for key in proxy.keys))


def build_view(*, coordinate, **kwargs):
    if coordinate not in FRACTIONS:
        raise ValueError('Only registered exact-ninths fine-tail views')
    return fraction_view(alpha=FRACTIONS[coordinate], **kwargs)


def digest(block):
    h = hashlib.sha256()
    for name in ('identities', 'labels', 'offsets', 'features', 'vectors'):
        a = np.ascontiguousarray(getattr(block, name))
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def _init(path, record, membership, config, cache_root, chunks, coordinate):
    global _WORK
    _limit_worker_threads()
    _WORK = (path, record, membership, config, cache_root, chunks, coordinate)


def assignments(chunk, record, member, root):
    a = load_npz_arrays(validate_file_ref(chunk['assignment'], root=Path(root)))
    start = chunk['block_index']*BLOCK_ROWS
    entries = np.flatnonzero(member['roles'][start:start+BLOCK_ROWS] == ROLES.index(chunk['role']))+start
    lengths = member['hlt_jet_nparticles'][entries]
    ids = np.asarray([np.frombuffer(bytes.fromhex(identity(record, int(e))), np.uint8) for e in entries])
    if (set(a) != {'entries', 'offsets', 'mapping', 'identities', 'labels'}
            or a['entries'].dtype != np.int64 or not np.array_equal(a['entries'], entries)
            or a['offsets'].dtype != np.int64 or not np.array_equal(a['offsets'], np.r_[0,np.cumsum(lengths)])
            or a['mapping'].dtype != np.int32 or a['mapping'].shape != (int(lengths.sum()),)
            or a['identities'].dtype != np.uint8 or not np.array_equal(a['identities'], ids)
            or a['labels'].dtype != np.int64 or not np.array_equal(a['labels'], member['labels'][entries])):
        raise ValueError('Saved assignment TRAIN/validation membership differs')
    return a


def _work(block_index):
    path, record, member, config, root, chunks, coordinate = _WORK
    selected = [c for c in chunks if c['block_index'] == block_index]
    aa = {c['role']: assignments(c, record, member, root) for c in selected}
    indices = {r: {int(e):j for j,e in enumerate(a['entries'])} for r,a in aa.items()}
    new, endpoint = {r:[] for r in aa}, {r:[] for r in aa}
    start, stop = block_index*BLOCK_ROWS, min((block_index+1)*BLOCK_ROWS, record['entries'])
    for role, entry, raw_hlt, raw_off in cache.read_selected_chunk(path, record, member, start, stop):
        a, j = aa[role], indices[role][entry]
        row_id = identity(record, entry)
        hlt, off = physical(raw_hlt, config, 'hlt'), physical(raw_off, config, 'offline')
        lo, hi = a['offsets'][j:j+2]
        view = build_view(identity=row_id, proxy=hlt, offline=off,
                          coordinate=coordinate, mapping=a['mapping'][lo:hi])
        row = SimpleNamespace(identity=row_id, label=int(a['labels'][j]))
        new[role].append((row, build_inputs(view, capacity=512)))
        endpoint[role].append((row, build_inputs(hlt, capacity=512)))
    result = []
    for chunk in selected:
        role = chunk['role']; i = chunk['file_index']
        reference = cache.read_block(root, chunk, 'D000', record, member)
        rebuilt = _block(i, endpoint[role]); value = _block(i, new[role])
        if digest(rebuilt) != digest(reference):
            raise ValueError('Original physical D000 endpoint replay differs')
        if (not np.array_equal(value.offsets, reference.offsets)
                or not np.array_equal(value.identities, reference.identities)
                or not np.array_equal(value.labels, reference.labels) or value.nbytes != reference.nbytes):
            raise ValueError('Exact HLT support/order/population differs')
        result.append((role, value, dict(file_index=i, block_index=block_index, role=role,
            assignment=chunk['assignment'], original_D000=chunk['views']['D000'],
            arrays_sha256=digest(value), endpoint_replay=True)))
    return result


def bounded(pool, indices, workers):
    """At most workers in-flight/results, unlike unbounded Executor.map."""
    pending = deque(); iterator = iter(indices)
    for _ in range(workers):
        index = next(iterator, None)
        if index is not None:
            pending.append(pool.submit(_work, index))
    while pending:
        yield pending.popleft().result()
        index = next(iterator, None)
        if index is not None:
            pending.append(pool.submit(_work, index))


def prepare(spec, stage, authenticated):
    coordinate = STAGES[stage]
    if coordinate not in ('D022', 'D011'):
        raise ValueError('D000 must load its unchanged original cache')
    manifest, foundation, memberships = authenticated
    evidence = spec['anchor']['evidence']
    if manifest['content_hash'] != evidence['prepared']:
        raise ValueError('Prepared parent changed')
    workers = spec['preparation_workers']
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError('Unbounded preparation concurrency')
    blocks = dict(train=[], validation=[]); records = []; started = time.monotonic()
    for i, (record, member) in enumerate(zip(foundation['files'], memberships, strict=True)):
        path = data.authenticate_source(evidence['raw_container'], record)
        chunks = [c for c in manifest['chunks'] if c['file_index'] == i]
        indices = sorted({c['block_index'] for c in chunks})
        args = (str(path), record, member, manifest['conventions'], evidence['cache_root'], chunks, coordinate)
        def accept(products):
            for role, block, info in products:
                blocks[role].append(block); records.append(info)
            print(f'FINE_TAIL views coordinate={coordinate} file={i} chunks={len(records)} '
                  f'elapsed_seconds={time.monotonic()-started:.1f}', flush=True)
        if workers == 1:
            _init(*args)
            for index in indices:
                accept(_work(index))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=_init, initargs=args) as pool:
                for products in bounded(pool, indices, workers):
                    accept(products)
        data.authenticate_source(evidence['raw_container'], record)
    caches = {r: RamCache(bb, role=r, coordinate_name=coordinate,
                          foundation_sha256=manifest['content_hash']) for r,bb in blocks.items()}
    for role, value in caches.items():
        summary = manifest['summaries'][role]['D000']
        if (len(value) != spec['counts'][role] or value.nbytes != summary['resident_bytes']
                or 2*value.nbytes+256*2**20 > .35*spec['memory_mb']*1024**2):
            raise MemoryError('New view population/RAM budget differs')
    report = artifact('VIEW_BUILD', parents=dict(spec=spec['content_hash'], prepared=manifest['content_hash'],
        view=view_recipe(coordinate)['content_hash']), coordinate=coordinate, chunks=records,
        construction='raw_saved_assignments_once_in_process_RAM', seconds=time.monotonic()-started,
        workers=workers, summaries={r:manifest['summaries'][r]['D000'] for r in caches},
        no_test_particle_requests=True, endpoint_replay=True)
    validate_build(spec, stage, report, manifest)
    return caches['train'], caches['validation'], report


def validate_build(spec, stage, report, manifest):
    co = STAGES[stage]
    validate(report, 'VIEW_BUILD', dict(spec=spec['content_hash'], prepared=manifest['content_hash'],
        view=view_recipe(co)['content_hash']))
    if (co not in ('D022','D011') or report['coordinate'] != co
            or report['construction'] != 'raw_saved_assignments_once_in_process_RAM'
            or report['workers'] != spec['preparation_workers'] or report['endpoint_replay'] is not True
            or report['no_test_particle_requests'] is not True
            or not np.isfinite(report['seconds']) or report['seconds'] < 0
            or report['summaries'] != {r:manifest['summaries'][r]['D000'] for r in ('train','validation')}
            or len(report['chunks']) != len(manifest['chunks'])):
        raise ValueError('New view preparation receipt differs')
    for r, c in zip(report['chunks'], manifest['chunks'], strict=True):
        expected = dict(file_index=c['file_index'], block_index=c['block_index'], role=c['role'],
                        assignment=c['assignment'], original_D000=c['views']['D000'], endpoint_replay=True)
        digest_value = r.get('arrays_sha256','')
        if (any(r.get(k) != v for k,v in expected.items()) or len(digest_value) != 64
                or any(x not in '0123456789abcdef' for x in digest_value)):
            raise ValueError('View chunk provenance/order differs')
