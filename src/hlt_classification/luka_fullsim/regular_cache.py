"""Deterministic file-parallel preparation and authenticated persistent inputs.

Only ordinary selected rows are decoded. Workers publish bounded file blocks;
large particle arrays never cross process queues. Legacy physics is reused.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import multiprocessing
import shutil
import time

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    sha256_file, write_immutable_json,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache, _limit_worker_threads
from . import stage2 as s
from .contracts import artifact, row_identity, source_path
from .foundation import load_foundation
from .particles import FIELDS, RawJet, _ranges, _matrix, physical
from .particle_audit import scan, Moments
from .preflight_cache import _block
from .splits import validate_splits as validate_legacy_splits

_WORK = None


def validate_splits(splits, inventory):
    if splits.get('contract') == 'LUKA_FULLSIM_ALLJETS_SPLITS/v1':
        from .alljets_data import validate_splits as validate_alljets
        return validate_alljets(splits, inventory)
    return validate_legacy_splits(splits, inventory)


def _init(container, inventory, splits, config, output):
    global _WORK
    _limit_worker_threads()
    validate_splits(splits, inventory)
    s.validate_conventions(config)
    _WORK = (Path(container), inventory, splits, config, Path(output))


def selected_file_rows(container, inventory, member):
    """Internal to a split-authenticated worker; identical selected-range reader."""
    index = member['file_index']
    record = inventory['files'][index]
    path, _ = source_path(Path(container)/'raw', record['path'])
    if sha256_file(path) != record['sha256']:
        raise ValueError('Particle source checksum differs')
    entries = {e: label for label, es in enumerate(member['entries_by_class']) for e in es}
    branches = ['hlt_jet_nparticles', 'jet_nparticles']
    branches += [prefix+f for prefix in ('hlt_part_', 'part_') for f in FIELDS]
    with uproot.open(path) as handle:
        for start, stop in _ranges(sorted(entries), 512):
            arrays = handle[record['tree_key']].arrays(branches, entry_start=start,
                                                     entry_stop=stop, library='ak', how=dict)
            for offset, entry in enumerate(range(start, stop)):
                yield RawJet(row_identity(record, entry), entries[entry], index, entry,
                    _matrix(arrays, 'hlt_part_', offset, int(arrays['hlt_jet_nparticles'][offset])),
                    _matrix(arrays, 'part_', offset, int(arrays['jet_nparticles'][offset])))
    if sha256_file(path) != record['sha256']:
        raise ValueError('Particle source changed while preparing')


def _work(task):
    container, inventory, splits, config, output = _WORK
    role, index = task
    if role not in ('train', 'validation'):
        raise PermissionError('No final-test preparation')
    members = [m for m in splits['memberships'][role] if m['file_index'] == index]
    if len(members) != 1:
        raise PermissionError('File is not an ordinary selected member')
    rows = {c: [] for c in s.COORDINATES}
    offsets, maps, ids, labels = [0], [], [], []

    def tapped(*args, **kwargs):
        if kwargs['role'] != role:
            return
        for jet in selected_file_rows(container, inventory, members[0]):
            hlt, offline = physical(jet.hlt, config, 'hlt'), physical(jet.offline, config, 'offline')
            mapping = s.views.match_particles(hlt, offline)
            maps.append(mapping)
            offsets.append(offsets[-1]+len(mapping))
            ids.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
            labels.append(jet.label)
            light = RawJet(jet.identity, jet.label, index, jet.entry, None, None)
            for coord in s.COORDINATES:
                view = s.views.build_view(identity=jet.identity, proxy=hlt, offline=offline,
                                          coordinate=coord, mapping=mapping)
                rows[coord].append((light, s.build_inputs(view, capacity=512)))
            yield jet

    audit = scan(container, inventory, splits, reader_factory=tapped)
    if len(labels) != sum(map(len, members[0]['entries_by_class'])):
        raise ValueError('Prepared file coverage differs')
    assigned = dict(offsets=np.asarray(offsets, np.int64), mapping=np.concatenate(maps).astype(np.int32),
                    identities=np.asarray(ids, np.uint8), labels=np.asarray(labels, np.int64))
    path = output/'assignments'/f'{role}_{index:04d}.npz'
    atomic_publish_bytes(path, deterministic_npz_bytes(assigned))
    assignment = dict(file_ref(path, root=output), file_index=index, rows=len(labels))
    refs = {}
    for coord in s.COORDINATES:
        block = _block(index, rows[coord])
        path = output/'inputs'/role/coord/f'{index:04d}.npz'
        atomic_publish_bytes(path, deterministic_npz_bytes({k: v for k, v in vars(block).items() if k != 'file_index'}))
        refs[coord] = dict(file_ref(path, root=output), file_index=index, rows=len(labels))
    return dict(role=role, index=index, assignment=assignment, blocks=refs,
                statistics=audit['statistics'][role], issues=audit['issues'][role])


def _merge(meter, row):
    """Parallel moment merge; only diagnostic roundoff differs from serial audit."""
    n = row['finite']
    meter.nonfinite += row['nonfinite']
    meter.zero += row['zero']
    meter.negative += row['negative']
    if row['abs_bin_edges'] != meter.edges.tolist():
        raise ValueError('Audit histogram edges differ')
    meter.histogram += np.asarray(row['abs_bin_counts'], np.int64)
    if not n:
        return
    delta, total = row['mean']-meter.mean, meter.n+n
    meter.m2 += row['sd']**2*n + delta**2*meter.n*n/total
    meter.mean += delta*n/total
    meter.n = total
    meter.minimum = row['minimum'] if meter.minimum is None else min(meter.minimum, row['minimum'])
    meter.maximum = row['maximum'] if meter.maximum is None else max(meter.maximum, row['maximum'])


def read_block(root, ref, member, inventory, meter=None):
    path = validate_file_ref(ref, root=Path(root))
    # validate_file_ref returns a path, not merely an existence check.
    a = load_npz_arrays(path)
    n = ref['rows']
    ordered = sorted((e, label) for label, es in enumerate(member['entries_by_class']) for e in es)
    identities = np.asarray([np.frombuffer(bytes.fromhex(row_identity(inventory['files'][member['file_index']], e)), np.uint8)
                             for e, _ in ordered])
    if (ref['file_index'] != member['file_index'] or n != len(ordered)
            or set(a) != {'offsets', 'features', 'vectors', 'identities', 'labels'}
            or a['offsets'].dtype != np.int64 or a['offsets'].shape != (n+1,)
            or a['features'].dtype != np.float32 or a['features'].ndim != 2 or a['features'].shape[1] != 17
            or a['vectors'].dtype != np.float32 or a['vectors'].shape != (len(a['features']), 4)
            or a['identities'].dtype != np.uint8 or not np.array_equal(a['identities'], identities)
            or a['labels'].dtype != np.int64 or not np.array_equal(a['labels'], [v for _, v in ordered])
            or a['offsets'][0] != 0 or a['offsets'][-1] != len(a['features'])
            or np.any(np.diff(a['offsets']) <= 0) or np.any(np.diff(a['offsets']) > 512)
            or not np.isfinite(a['features']).all() or not np.isfinite(a['vectors']).all()):
        raise ValueError('Durable input shape/identity/label/capacity differs')
    if meter is not None:
        for i, identity in enumerate(identities):
            lo, hi = a['offsets'][i:i+2]
            meter.add(SimpleNamespace(identity=bytes(identity).hex(), label=int(a['labels'][i])),
                      SimpleNamespace(features=a['features'][lo:hi], vectors=a['vectors'][lo:hi]))
    return RamBlock(member['file_index'], **a)


def build_rows(container, inventory, splits, config, output, *, workers):
    """Low-level fixture/production core; publication order independent of workers."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Use 1..8 bounded file workers')
    validate_splits(splits, inventory)
    args = (str(container), inventory, splits, config, str(output))
    tasks = [(role, m['file_index']) for role in ('train', 'validation') for m in splits['memberships'][role]]
    assignments = {r: [] for r in ('train', 'validation')}
    blocks = {r: {c: [] for c in s.COORDINATES} for r in assignments}
    meters = {r: {c: s.ViewMeter() for c in s.COORDINATES} for r in assignments}
    moments, issues = {r: {} for r in assignments}, {r: {} for r in assignments}
    done = 0
    def accept(row):
        nonlocal done
        role, index = row['role'], row['index']
        member = next(m for m in splits['memberships'][role] if m['file_index'] == index)
        assignments[role].append(row['assignment'])
        for coord, ref in row['blocks'].items():
            read_block(output, ref, member, inventory, meters[role][coord])
            blocks[role][coord].append(ref)
        for key, stats in row['statistics'].items():
            _merge(moments[role].setdefault(key, Moments()), stats)
        for side, values in row['issues'].items():
            for key, value in values.items():
                dest = issues[role].setdefault(side, {})
                dest[key] = dest.get(key, 0)+value
        done += 1
        print(f'LUKA durable files={done}/{len(tasks)} role={role} rows={meters[role]["D000"].rows}', flush=True)
    if workers == 1:
        _init(*args)
        for task in tasks:
            accept(_work(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init, initargs=args) as pool:
            iterator = iter(tasks)
            pending = [pool.submit(_work, task) for task in (next(iterator, None) for _ in range(workers)) if task is not None]
            while pending:
                accept(pending.pop(0).result())
                task = next(iterator, None)
                if task is not None:
                    pending.append(pool.submit(_work, task))
    summaries = {r: {c: m.report() for c, m in group.items()} for r, group in meters.items()}
    return dict(assignments=assignments, blocks=blocks, summaries=summaries,
                statistics={r: {k: v.report() for k, v in group.items()} for r, group in moments.items()}, issues=issues)


def prepare(gate):
    from . import regular_campaign as c
    c = c.for_spec(gate)
    c.validate_gate(gate)
    root = Path(gate['root'])
    foundation, inventory, splits = c.load_foundation(root/'foundation')
    c.check_population(inventory, splits)
    output = root/'prepared'
    output.mkdir(exist_ok=False)
    # Worst case for all 7 full-support views plus temporary file blocks.
    bound = sum(gate['counts'].values())*(512*84+128)*7 + 5*2**30
    if gate.get('storage_policy') == 'authenticated_scalar_ragged_upper_bound_with_reserve':
        from .alljets_data import storage_bound
        bound = storage_bound(foundation['input_container'], inventory, splits)
    print(f'LUKA preparation required_free_GiB={bound/2**30:.2f} (including reserve)', flush=True)
    if shutil.disk_usage(root).free < bound:
        raise OSError('Insufficient free storage for conservative nontruncated cache bound')
    started = time.monotonic()
    # Temporary parent only: numerical policies are frozen; final config binds
    # the completed integrated audit rather than pretending it existed earlier.
    temporary = c.conventions(gate['content_hash'])
    products = build_rows(foundation['input_container'], inventory, splits, temporary, output, workers=8)
    audit = c.data_artifact('PARTICLE_AUDIT', parents=dict(foundation=foundation['content_hash'], source=gate['source_snapshot']['content_hash']),
        source_snapshot=gate['source_snapshot'], counts=gate['counts'], statistics=products.pop('statistics'),
        issues=products.pop('issues'), units='stored_unconfirmed_no_conversion',
        moments='particle_weighted_except_jet_counts_SD_is_width', zero_error_meaning='unresolved_not_inferred',
        scientific_admission=False, final_test_accessed=False, aggregation='fixed_order_source_file_parallel_moments')
    config = c.conventions(audit['content_hash'])
    refs = products.pop('blocks')
    p = c.data_artifact('PREPARED', parents=dict(foundation=foundation['content_hash'], audit=audit['content_hash'],
        conventions=config['content_hash'], source=gate['source_snapshot']['content_hash']),
        source_snapshot=gate['source_snapshot'], foundation_root=str(root/'foundation'), audit=audit,
        conventions=config, **s.contracts(), **products, counts=gate['counts'],
        seconds=time.monotonic()-started, final_test_accessed=False, science_authorized=False)
    write_immutable_json(output/'prepared.json', p)
    manifest = c.make_artifact('CACHE', parents=dict(prepared=p['content_hash'], gate=gate['content_hash']),
        blocks=refs, serial_reference_passed=False)
    # Replay one actual TRAIN source file through the original reader/view path.
    # This is input parity, never a scientific miniature or validation selection.
    from .particles import ParticleReader
    if splits.get('contract') == 'LUKA_FULLSIM_ALLJETS_SPLITS/v1':
        from .alljets_data import ParticleReader
    from .preflight_cache import _values
    first = splits['memberships']['train'][0]
    assignment = s.load_assignment(output, p['assignments']['train'][0])
    originals = {co: read_block(output, refs['train'][co][0], first, inventory) for co in s.COORDINATES}
    checked = 0
    for jet in ParticleReader(foundation['input_container'], inventory, splits, role='train', include_offline=True):
        if jet.file_index != first['file_index']:
            break
        for co, value in _values(jet, assignment, checked, p, s.COORDINATES).items():
            block = originals[co]
            lo, hi = block.offsets[checked:checked+2]
            if not np.array_equal(value.features, block.features[lo:hi]) or not np.array_equal(value.vectors, block.vectors[lo:hi]):
                raise ValueError('Serial legacy input replay differs')
        checked += 1
        if checked == len(assignment['labels']):
            break
    if checked != len(assignment['labels']):
        raise ValueError('Serial parity coverage differs')
    manifest = c.make_artifact('CACHE', parents=manifest['parents'], blocks=refs,
                               serial_reference_passed=True, serial_reference_rows=checked)
    c.validate_gate(gate)
    if c.load_prepared(output)[0] != p:
        raise ValueError('Prepared metadata changed before cache commit')
    from .inventory import verify_source_bytes
    verify_source_bytes(foundation['input_container'], inventory)
    write_immutable_json(output/'cache_manifest.json', manifest)  # completion last
    return manifest


def metadata(root):
    from . import regular_campaign as c
    root = Path(root)
    manifest = load_json(root/'cache_manifest.json')
    c = c.for_spec(manifest)
    p, f, inventory, splits = c.load_prepared(root)
    c.validate(manifest, 'CACHE')
    if (manifest['parents']['prepared'] != p['content_hash'] or not manifest['serial_reference_passed']
            or set(manifest['blocks']) != {'train', 'validation'}):
        raise ValueError('Durable cache lineage/parity differs')
    for role in ('train', 'validation'):
        if set(manifest['blocks'][role]) != set(s.COORDINATES):
            raise ValueError('Durable coordinate coverage differs')
        expected = [m['file_index'] for m in splits['memberships'][role]]
        for coord, refs in manifest['blocks'][role].items():
            if [r['file_index'] for r in refs] != expected:
                raise ValueError('Durable file coverage/order differs')
            for ref in refs:
                if ref['path'] != f'inputs/{role}/{coord}/{ref["file_index"]:04d}.npz':
                    raise ValueError('Durable block path differs')
    return manifest, p, inventory, splits


def load(root, *, role, coordinate, max_ram_bytes, authenticated=None):
    if role not in ('train', 'validation'):
        raise PermissionError('No final-test/unused cache access')
    if coordinate not in s.COORDINATES or type(max_ram_bytes) is not int or max_ram_bytes <= 0:
        raise ValueError('Invalid cache coordinate/budget')
    manifest, p, inventory, splits = metadata(root) if authenticated is None else authenticated
    if 2*p['summaries'][role][coordinate]['resident_bytes']+64*2**20 > max_ram_bytes:
        raise MemoryError('Durable cache exceeds explicit RAM budget')
    meter, blocks = s.ViewMeter(), []
    for ref, member in zip(manifest['blocks'][role][coordinate], splits['memberships'][role]):
        blocks.append(read_block(root, ref, member, inventory, meter))
    if meter.report() != p['summaries'][role][coordinate]:
        raise ValueError('Durable input fingerprint replay differs')
    result = RamCache(blocks, role=role, coordinate_name=coordinate, foundation_sha256=p['content_hash'])
    if result.nbytes > max_ram_bytes:
        raise MemoryError('Resident cache exceeds budget')
    return result


def cache(spec, prepared, node, role):
    if node['context_coordinate'] is not None:
        raise PermissionError('Regular campaign forbids fusion inputs')
    value = load(spec['admission']['prepared_root'], role=role, coordinate=node['coordinate'],
                 max_ram_bytes=int(spec['memory_mb']*1024**2*.15))
    if value.foundation_sha256 != prepared['content_hash']:
        raise ValueError('Scientific cache preparation differs')
    return value
