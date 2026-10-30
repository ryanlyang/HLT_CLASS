"""FullStat input adapters; preserve raw-particle union and sealed role boundaries."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import multiprocessing
import numpy as np
from hlt_classification.jetclass2_delphes.cache import RamCache, _limit_worker_threads
from hlt_classification.luka_fullstat import cache, data
from hlt_classification.luka_fullstat.contracts import BLOCK_ROWS, ROLES, identity
from hlt_classification.luka_fullstat_attention.worker import metadata
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.luka_u100_multiview.views import union, PairedCache, lengths

_WORK = None


def _init(path, record, membership, conventions):
    global _WORK
    _limit_worker_threads()
    _WORK = (path, record, membership, conventions)


def _union_chunk(block):
    path, record, membership, conventions = _WORK
    rows = {'train': [], 'validation': []}
    start, stop = block*BLOCK_ROWS, min((block+1)*BLOCK_ROWS, record['entries'])
    for role, entry, rh, ro in cache.read_selected_chunk(path, record, membership, start, stop):
        if role not in rows: raise PermissionError('Sealed union row')
        h, o = physical(rh, conventions, 'hlt'), physical(ro, conventions, 'offline')
        particles = union(h, o)
        value = build_inputs(particles, capacity=max(16, len(particles)))
        if len(value.features) != len(h)+len(o): raise ValueError('Union particle loss')
        rows[role].append((SimpleNamespace(identity=identity(record, entry),
            label=int(membership['labels'][entry])), value))
    return {r: _block(0, values) for r, values in rows.items() if values}


def plain(spec, role, coordinate, meta):
    if role not in ('train','validation') or coordinate not in ('D000','OFFLINE'):
        raise PermissionError('Only ordinary native HLT/OFFLINE roles')
    return cache.load(spec['anchor']['evidence']['cache_root'], meta, role=role,
        coordinate=coordinate, max_ram_bytes=int(spec['memory_mb']*1024**2*.75))


def concat(spec, meta, *, workers=4):
    if workers not in (1,4): raise ValueError('Registered bounded union workers only')
    manifest, foundation, memberships = meta
    budget = int(spec['memory_mb']*1024**2*.75)
    residents = sum(manifest['summaries'][r][c]['resident_bytes']
                    for r in ('train','validation') for c in ('D000','OFFLINE'))
    if 2*residents+8*2**30 > budget:
        raise MemoryError('Union construction bound exceeds RAM; no token crop/fallback')
    blocks = {'train': [], 'validation': []}
    used = 0
    for i, (record, member) in enumerate(zip(foundation['files'], memberships)):
        path = data.authenticate_source(spec['anchor']['evidence']['raw_container'], record)
        indices = [b for b in range((record['entries']+BLOCK_ROWS-1)//BLOCK_ROWS)
                   if np.isin(member['roles'][b*BLOCK_ROWS:(b+1)*BLOCK_ROWS], (0,1)).any()]
        def accept(b, values):
            nonlocal used
            for role in ('train','validation'):
                lo = b*BLOCK_ROWS
                entries = np.flatnonzero(member['roles'][lo:lo+BLOCK_ROWS] == ROLES.index(role))+lo
                if not len(entries):
                    if role in values: raise ValueError('Unexpected union rows')
                    continue
                v = values[role]
                expected = np.asarray([np.frombuffer(bytes.fromhex(identity(record,int(e))),np.uint8) for e in entries])
                if (not np.array_equal(v.identities, expected) or
                    not np.array_equal(v.labels, member['labels'][entries]) or
                    not np.array_equal(np.diff(v.offsets), member['hlt_jet_nparticles'][entries]+member['jet_nparticles'][entries])):
                    raise ValueError('Union identity/label/full cardinality differs')
                v.file_index = i
                blocks[role].append(v); used += v.nbytes+len(v.labels)*40
                if used+8*2**30 > budget: raise MemoryError('Union RAM bound exceeded')
        args = (str(path), record, member, manifest['conventions'])
        if workers == 1:
            _init(*args)
            for b in indices: accept(b, _union_chunk(b))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init, initargs=args) as pool:
                remaining = iter(indices)
                pending = [(b,pool.submit(_union_chunk,b)) for b in [next(remaining,None) for _ in range(workers)] if b is not None]
                while pending:
                    b,future = pending.pop(0); accept(b,future.result())
                    b = next(remaining,None)
                    if b is not None: pending.append((b,pool.submit(_union_chunk,b)))
        data.authenticate_source(spec['anchor']['evidence']['raw_container'], record)
        print(f'FULLSTAT MULTIVIEW union files={i+1}/{len(memberships)} resident_bytes={used}',flush=True)
    return tuple(RamCache(blocks[r],role=r,coordinate_name='CONCAT',foundation_sha256=manifest['content_hash'])
                 for r in ('train','validation'))


def describe(value):
    parts = (value.primary,value.context) if isinstance(value,PairedCache) else (value,)
    h = hashlib.sha256()
    for part in parts:
        for block in part.blocks:
            for key in ('offsets','features','vectors','identities','labels'):
                a = getattr(block,key)
                h.update(str((key,a.dtype.str,a.shape)).encode()); h.update(a.tobytes())
    support = lengths(value)
    return dict(role=value.role,coordinate=value.coordinate_name,rows=len(value),resident_bytes=value.nbytes,
                maximum=int(support.max()),input_sha256=h.hexdigest())


def load(spec, task):
    meta = metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['u100']['summaries']: raise ValueError('Prepared summaries changed')
    if task == 'teacher_CONCAT': tr,val = concat(spec,meta)
    elif task in ('teacher_FUSION','train_multi3'):
        tr,val = (plain(spec,r,'D000',meta) for r in ('train','validation'))
        if task == 'teacher_FUSION':
            tr = PairedCache(tr,plain(spec,'train','OFFLINE',meta))
            val = PairedCache(val,plain(spec,'validation','OFFLINE',meta))
    else: raise PermissionError('Unregistered task')
    if any(len(v)!=spec['counts'][v.role] for v in (tr,val)):
        raise ValueError('FullStat input population differs')
    if 2*(tr.nbytes+val.nbytes)+12*2**30 > .85*spec['memory_mb']*1024**2:
        raise MemoryError('Input/target construction exceeds host envelope')
    return tr,val,meta
