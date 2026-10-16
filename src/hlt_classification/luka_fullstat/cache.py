"""Bounded merged-file preparation; durable ragged views, never test particles."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import multiprocessing
import time

import numpy as np
import uproot

from hlt_classification.cms_proxy_ladder import views
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays, write_immutable_json,
)
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache, _limit_worker_threads
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.luka_fullsim.particles import FIELDS, _matrix, _ranges, physical
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.luka_fullsim.stage2 import conventions as convention_artifact, contracts as view_contracts
from .contracts import (BLOCK_ROWS, COORDINATES, ORDINARY, ROLES, artifact, identity, validate)
from . import data

_WORK = None


def conventions(parent):
    return convention_artifact(audit_sha256=parent, offline_unit='mm', hlt_unit='mm',
        zero_error='error_only_unavailable', authority='operator_provisional',
        evidence='Operator carries trial mm and error-only zero-error convention to FullStat; producer unconfirmed.')


def storage_bound(foundation):
    # Each view has at most hlt+offline particles. Include all seven, maps,
    # block publication temporaries and a reserve; actual quota is separate.
    particles = sum(sum(r['role_particles'][role].values()) for r in foundation['files'] for role in ORDINARY)
    rows = sum(foundation['counts'][role] for role in ORDINARY)
    return int((particles*(7*84+4)+rows*7*96)*1.10)+10*2**30


def _init(path, record, membership, config, output, file_index):
    global _WORK
    _limit_worker_threads()
    _WORK = (Path(path), record, membership, config, Path(output), file_index)


def read_selected_chunk(path, record, membership, start, stop):
    """Request only spans without final-test/unused rows; ignored ineligible rows
    may be decoded inside a span, but never become prepared examples. ROOT may
    internally decompress a basket beyond the requested entry slice.
    """
    role_codes = membership['roles']
    authorized = np.flatnonzero(np.isin(role_codes[start:stop], (0, 1, 255)))+start
    if not len(authorized):
        return
    branches = ['hlt_jet_nparticles', 'jet_nparticles']
    branches += [prefix+field for prefix in ('hlt_part_', 'part_') for field in FIELDS]
    with uproot.open(path, array_cache=None) as handle:
        tree = handle[record['tree_key']]
        for lo, hi in _ranges(authorized.tolist(), BLOCK_ROWS):
            if not np.isin(role_codes[lo:hi], (0, 1)).any():
                continue
            arrays = tree.arrays(branches, entry_start=lo, entry_stop=hi, library='ak', how=dict)
            for offset, entry in enumerate(range(lo, hi)):
                code = int(role_codes[entry])
                if code == 255:
                    continue
                if code not in (0, 1):
                    raise PermissionError('Sealed particle row requested')
                hlt = _matrix(arrays, 'hlt_part_', offset, int(arrays['hlt_jet_nparticles'][offset]))
                off = _matrix(arrays, 'part_', offset, int(arrays['jet_nparticles'][offset]))
                if (len(hlt) != membership['hlt_jet_nparticles'][entry]
                        or len(off) != membership['jet_nparticles'][entry]):
                    raise ValueError('Scalar/particle cardinality differs')
                yield ROLES[code], entry, hlt, off


def _work(block_index):
    path, record, membership, config, output, file_index = _WORK
    start, stop = block_index*BLOCK_ROWS, min((block_index+1)*BLOCK_ROWS, record['entries'])
    rows = {r: {co: [] for co in COORDINATES} for r in ORDINARY}
    assignments = {r: dict(entries=[], offsets=[0], mapping=[], identities=[], labels=[]) for r in ORDINARY}
    for role, entry, raw_hlt, raw_off in read_selected_chunk(path, record, membership, start, stop):
        hlt, off = physical(raw_hlt, config, 'hlt'), physical(raw_off, config, 'offline')
        row_id, label = identity(record, entry), int(membership['labels'][entry])
        mapping = views.match_particles(hlt, off)
        a = assignments[role]
        a['entries'].append(entry)
        a['offsets'].append(a['offsets'][-1]+len(mapping))
        a['mapping'].append(mapping)
        a['identities'].append(np.frombuffer(bytes.fromhex(row_id), np.uint8))
        a['labels'].append(label)
        light = SimpleNamespace(identity=row_id, label=label)
        for co in COORDINATES:
            view = views.build_view(identity=row_id, proxy=hlt, offline=off, coordinate=co, mapping=mapping)
            value = build_inputs(view, capacity=512)
            rows[role][co].append((light, value))
    result = []
    for role in ORDINARY:
        a = assignments[role]
        if not a['entries']:
            continue
        prefix = f'chunks/f{file_index:02d}/b{block_index:06d}/{role}'
        payload = dict(entries=np.asarray(a['entries'], np.int64), offsets=np.asarray(a['offsets'], np.int64),
            mapping=np.concatenate(a['mapping']).astype(np.int32), identities=np.asarray(a['identities'], np.uint8),
            labels=np.asarray(a['labels'], np.int64))
        target = output/prefix/'assignment.npz'
        atomic_publish_bytes(target, deterministic_npz_bytes(payload))
        chunks = {}
        for co in COORDINATES:
            block = _block(file_index, rows[role][co])
            target = output/prefix/(co+'.npz')
            atomic_publish_bytes(target, deterministic_npz_bytes({k:v for k,v in vars(block).items() if k != 'file_index'}))
            chunks[co] = dict(file_ref(target, root=output), rows=len(block.labels),
                             particles=len(block.features), maximum=int(np.diff(block.offsets).max()),
                             resident_bytes=block.nbytes+len(block.labels)*40)
        result.append(dict(file_index=file_index, block_index=block_index, role=role,
            assignment=file_ref(output/prefix/'assignment.npz', root=output), views=chunks))
    return result


def prepare_file(container, foundation_root, output, file_index, *, workers=4):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Use 1..8 bounded chunk workers')
    foundation, memberships = data.load(foundation_root)
    if type(file_index) is not int or not 0 <= file_index < len(foundation['files']):
        raise ValueError('Unknown source file')
    output = Path(output)
    (output/'files'/f'{file_index:02d}').mkdir(parents=True, exist_ok=False)
    record, membership = foundation['files'][file_index], memberships[file_index]
    path = data.authenticate_source(container, record)
    # A per-file reservation; the entire gate also reserves the global bound.
    config = conventions(foundation['content_hash'])
    args = (str(path), record, membership, config, str(output), file_index)
    indices = [i for i in range((record['entries']+BLOCK_ROWS-1)//BLOCK_ROWS)
               if np.isin(membership['roles'][i*BLOCK_ROWS:(i+1)*BLOCK_ROWS], (0, 1)).any()]
    products, started = [], time.monotonic()
    def accept(value):
        products.extend(value)
        print(f'FullStat prepare file={file_index} chunks={len(products)} seconds={time.monotonic()-started:.1f}', flush=True)
    if workers == 1:
        _init(*args)
        for index in indices:
            accept(_work(index))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init, initargs=args) as pool:
            iterator = iter(indices)
            pending = [pool.submit(_work, i) for i in (next(iterator, None) for _ in range(workers)) if i is not None]
            while pending:
                accept(pending.pop(0).result())
                index = next(iterator, None)
                if index is not None:
                    pending.append(pool.submit(_work, index))
    data.authenticate_source(container, record)
    products.sort(key=lambda r: (r['block_index'], r['role']))
    result = artifact('FILE_PREPARED', parents=dict(foundation=foundation['content_hash']),
        file_index=file_index, conventions=config, view_contracts=view_contracts(), chunks=products,
        seconds=time.monotonic()-started, final_test_accessed=False)
    verify_file(result, foundation, membership, output)
    write_immutable_json(output/'files'/f'{file_index:02d}'/'receipt.json', result)
    return result


def read_block(output, chunk, co, record, membership):
    if chunk['role'] not in ORDINARY or co not in COORDINATES:
        raise PermissionError('No sealed role or unregistered coordinate')
    a = load_npz_arrays(validate_file_ref(chunk['views'][co], root=Path(output)))
    start = chunk['block_index']*BLOCK_ROWS
    entries = np.flatnonzero(membership['roles'][start:start+BLOCK_ROWS] == ROLES.index(chunk['role']))+start
    n = len(entries)
    ids = np.asarray([np.frombuffer(bytes.fromhex(identity(record, int(e))), np.uint8) for e in entries]).reshape(-1,32)
    expected = chunk['views'][co]
    if (set(a) != {'offsets', 'features', 'vectors', 'labels', 'identities'}
            or n != expected['rows'] or n == 0
            or a['offsets'].dtype != np.int64 or a['offsets'].shape != (n+1,)
            or a['labels'].dtype != np.int64 or a['identities'].dtype != np.uint8
            or not np.array_equal(a['labels'], membership['labels'][entries])
            or not np.array_equal(a['identities'], ids)
            or a['features'].dtype != np.float32 or a['features'].shape != (expected['particles'], 17)
            or a['vectors'].dtype != np.float32 or a['vectors'].shape != (expected['particles'], 4)
            or a['offsets'][0] != 0 or a['offsets'][-1] != expected['particles']
            or np.any(np.diff(a['offsets']) <= 0) or np.any(np.diff(a['offsets']) > 512)
            or int(np.diff(a['offsets']).max()) != expected['maximum']
            or not np.isfinite(a['features']).all() or not np.isfinite(a['vectors']).all()):
        raise ValueError('Prepared block shape/identity/label/cardinality differs')
    value = RamBlock(chunk['file_index'], **a)
    if value.nbytes+n*40 != expected['resident_bytes']:
        raise ValueError('Prepared resident-byte accounting differs')
    return value


def file_metadata(result, foundation, membership):
    validate(result, 'FILE_PREPARED', dict(foundation=foundation['content_hash']))
    i = result['file_index']
    record = foundation['files'][i]
    if (result['conventions'] != conventions(foundation['content_hash'])
            or result['view_contracts'] != view_contracts() or result['final_test_accessed'] is not False):
        raise ValueError('Prepared physical policy differs')
    expected = [(b, r) for b in range((record['entries']+BLOCK_ROWS-1)//BLOCK_ROWS) for r in sorted(ORDINARY)
                if np.any(membership['roles'][b*BLOCK_ROWS:(b+1)*BLOCK_ROWS] == ROLES.index(r))]
    if [(x['block_index'], x['role']) for x in result['chunks']] != expected:
        raise ValueError('Prepared chunk coverage/order differs')
    for chunk in result['chunks']:
        prefix = f'chunks/f{i:02d}/b{chunk["block_index"]:06d}/{chunk["role"]}'
        if chunk['file_index'] != i or set(chunk['views']) != set(COORDINATES):
            raise ValueError('Prepared view/file coverage differs')
        if chunk['assignment']['path'] != prefix+'/assignment.npz':
            raise ValueError('Assignment path differs')
        start = chunk['block_index']*BLOCK_ROWS
        n = int(np.count_nonzero(membership['roles'][start:start+BLOCK_ROWS] == ROLES.index(chunk['role'])))
        for co, ref in chunk['views'].items():
            if (ref['path'] != prefix+'/'+co+'.npz' or ref['rows'] != n
                    or not n <= ref['particles'] <= 512*n or not 1 <= ref['maximum'] <= 512
                    or ref['resident_bytes'] != ref['particles']*84+n*88+8):
                raise ValueError('Prepared block metadata differs')


def verify_file(result, foundation, membership, output):
    file_metadata(result, foundation, membership)
    record = foundation['files'][result['file_index']]
    for chunk in result['chunks']:
        assigned = load_npz_arrays(validate_file_ref(chunk['assignment'], root=Path(output)))
        start = chunk['block_index']*BLOCK_ROWS
        entries = np.flatnonzero(membership['roles'][start:start+BLOCK_ROWS] == ROLES.index(chunk['role']))+start
        lengths = membership['hlt_jet_nparticles'][entries]
        if (set(assigned) != {'entries', 'offsets', 'mapping', 'identities', 'labels'}
                or assigned['entries'].dtype != np.int64 or not np.array_equal(assigned['entries'], entries)
                or assigned['offsets'].dtype != np.int64
                or not np.array_equal(assigned['offsets'], np.r_[0, np.cumsum(lengths)])
                or assigned['mapping'].dtype != np.int32 or assigned['mapping'].shape != (int(lengths.sum()),)):
            raise ValueError('Assignment membership/shape differs')
        for j, e in enumerate(entries):
            lo, hi = assigned['offsets'][j:j+2]
            mapping = assigned['mapping'][lo:hi]
            active = mapping[mapping >= 0]
            if (np.any(mapping < -1) or np.any(active >= membership['jet_nparticles'][e])
                    or len(np.unique(active)) != len(active)):
                raise ValueError('Noninjective/out-of-range assignment')
        for co in COORDINATES:
            value = read_block(output, chunk, co, record, membership)
            if not np.array_equal(value.identities, assigned['identities']) or not np.array_equal(value.labels, assigned['labels']):
                raise ValueError('Assignment/input identity join differs')


def summarize(chunks, foundation):
    summaries = {r: {co: dict(rows=0, particles=0, maximum=0, resident_bytes=0) for co in COORDINATES} for r in ORDINARY}
    for chunk in chunks:
        for co, ref in chunk['views'].items():
            meter = summaries[chunk['role']][co]
            for key in ('rows', 'particles', 'resident_bytes'):
                meter[key] += ref[key]
            meter['maximum'] = max(meter['maximum'], ref['maximum'])
    if any(summaries[r][co]['rows'] != foundation['counts'][r] for r in ORDINARY for co in COORDINATES):
        raise ValueError('Full preparation row coverage differs')
    return summaries


def assemble(foundation_root, output, *, gate_hash):
    foundation, memberships = data.load(foundation_root, expected_gate=gate_hash)
    chunks, receipts = [], []
    for index, member in enumerate(memberships):
        path = Path(output)/'files'/f'{index:02d}'/'receipt.json'
        result = load_json(path)
        if result['file_index'] != index:
            raise ValueError('Wrong file preparation receipt')
        verify_file(result, foundation, member, output)
        chunks.extend(result['chunks'])
        receipts.append(file_ref(path, root=Path(output)))
    summaries = summarize(chunks, foundation)
    manifest = artifact('CACHE', parents=dict(gate=gate_hash, foundation=foundation['content_hash']),
        summaries=summaries, chunks=chunks, file_receipts=receipts, view_contracts=view_contracts(),
        conventions=conventions(foundation['content_hash']), final_test_accessed=False)
    write_immutable_json(Path(output)/'cache_manifest.json', manifest)
    return manifest


def metadata(foundation_root, output, *, gate_hash):
    foundation, memberships = data.load(foundation_root, expected_gate=gate_hash)
    manifest = load_json(Path(output)/'cache_manifest.json')
    validate(manifest, 'CACHE', dict(gate=gate_hash, foundation=foundation['content_hash']))
    if (manifest['conventions'] != conventions(foundation['content_hash'])
            or manifest['view_contracts'] != view_contracts() or manifest['final_test_accessed'] is not False):
        raise ValueError('Cache policies differ')
    combined = []
    if len(manifest['file_receipts']) != len(foundation['files']):
        raise ValueError('File receipt coverage differs')
    for index, ref in enumerate(manifest['file_receipts']):
        if ref['path'] != f'files/{index:02d}/receipt.json':
            raise ValueError('File receipt path differs')
        row = load_json(validate_file_ref(ref, root=Path(output)))
        if row['file_index'] != index:
            raise ValueError('File receipt index differs')
        file_metadata(row, foundation, memberships[index])
        combined.extend(row['chunks'])
    if combined != manifest['chunks'] or manifest['summaries'] != summarize(combined, foundation):
        raise ValueError('Cache/chunk receipt binding differs')
    return manifest, foundation, memberships


def load(output, authenticated, *, role, coordinate, max_ram_bytes, miniature=False):
    if role not in ORDINARY:
        raise PermissionError('Final-test/unused cache access is sealed')
    if coordinate not in COORDINATES or type(max_ram_bytes) is not int or max_ram_bytes <= 0:
        raise ValueError('Invalid view/RAM limit')
    manifest, foundation, memberships = authenticated
    summary = manifest['summaries'][role][coordinate]
    if not miniature and 2*summary['resident_bytes']+256*2**20 > max_ram_bytes:
        raise MemoryError('Full role cache plus read temporaries exceeds explicit budget')
    chunks = [c for c in manifest['chunks'] if c['role'] == role]
    if miniature:
        # Same row set for every coordinate: first chunk per source/class plus
        # all seven longest witnesses. This permits exact technical teacher joins.
        chosen = {}
        for chunk in chunks:
            i, b = chunk['file_index'], chunk['block_index']
            chosen.setdefault(('file', i), (i,b))
            a = memberships[i]
            lo, hi = b*BLOCK_ROWS, (b+1)*BLOCK_ROWS
            labels = a['labels'][lo:hi][a['roles'][lo:hi] == ROLES.index(role)]
            for label in np.unique(labels):
                chosen.setdefault(('class', int(label)), (i,b))
        for co in COORDINATES:
            longest = max(chunks, key=lambda c: c['views'][co]['maximum'])
            chosen[('longest', co)] = (longest['file_index'], longest['block_index'])
        keys = set(chosen.values())
        chunks = [c for c in chunks if (c['file_index'], c['block_index']) in keys]
    blocks = []
    for chunk in chunks:
        i = chunk['file_index']
        blocks.append(read_block(output, chunk, coordinate, foundation['files'][i], memberships[i]))
    result = RamCache(blocks, role=role, coordinate_name=coordinate, foundation_sha256=manifest['content_hash'])
    if result.nbytes > max_ram_bytes:
        raise MemoryError('Resident cache exceeds budget')
    if not miniature and (len(result) != summary['rows'] or result.nbytes != summary['resident_bytes']):
        raise ValueError('Loaded population/resident bytes differ')
    return result
