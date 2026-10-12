"""Explicit consumed-lock capability for final-test particles and RAM views."""
from pathlib import Path
import itertools
import os

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.context_fusion.inputs import PairedCache
from . import final_test_campaign as c, stage2 as s, science_worker
from .contracts import row_identity, source_path
from .particles import FIELDS, RawJet, ParticleReader, physical, _matrix, _ranges
from .preflight_cache import _block, _values

_KEY = object()


class Access:
    def __init__(self, key, execution, p, f, inventory, splits, claim):
        if key is not _KEY:
            raise PermissionError('Test access requires consumed execution locks')
        self.execution, self.p, self.f = execution, p, f
        self.inventory, self.splits, self.claim = inventory, splits, claim
        self.pid = os.getpid()

    def require(self):
        execution, _ = c.authenticate(Path(self.execution['root'])/'execution_lock.json', full=False)
        claim = load_json(Path(execution['claim_dir'])/'claim.json')
        c.validate(claim, 'ACCESS_CLAIM', dict(execution=execution['content_hash'],
            finalists=execution['parents']['finalists']))
        if (execution != self.execution or claim != self.claim or self.pid != os.getpid()
                or str(os.environ.get('SLURM_JOB_ID')) != claim['job_id']):
            raise PermissionError('Final-test process/job/claim bindings differ')


def consume(path):
    execution, _, p, f, inv, splits = c.authenticate(path)
    parent = load_json(execution['parent']['spec']['path'])
    job = science_worker.execution(parent, load_json(execution['parent']['preflight']['path']))
    witness_path = Path(execution['root'])/'evaluate/train_smoke.json'
    witness = load_json(witness_path)
    c.validate_smoke(witness, execution)
    claim = c.artifact('ACCESS_CLAIM', parents=dict(execution=execution['content_hash'],
        finalists=execution['parents']['finalists']), job_id=str(job),
        train_smoke=file_ref(witness_path),
        final_test_accessed=True, meaning='authorization_consumed_before_any_test_particle_read')
    location = Path(execution['claim_dir'])
    try:
        location.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise PermissionError('Final-test already claimed: preserve evidence; no automatic retry') from exc
    write_immutable_json(location/'claim.json', claim)
    return Access(_KEY, execution, p, f, inv, splits, claim)


def require(access):
    if not isinstance(access, Access):
        raise PermissionError('A consumed final-test access capability is required')
    access.require()


def population_arrays(inventory, splits):
    """Frozen scalar membership only, without opening particle branches."""
    ids, labels, files = [], [], []
    for index, (record, group) in enumerate(zip(inventory['files'], splits['groups'])):
        if group['role'] != 'final_test':
            continue
        for entry, label in sorted((e, label) for label, es in enumerate(record['entries_by_class']) for e in es):
            ids.append(np.frombuffer(bytes.fromhex(row_identity(record, entry)), np.uint8))
            labels.append(label)
            files.append(index)
    return dict(identities=np.asarray(ids, np.uint8), labels=np.asarray(labels, np.int64),
                files=np.asarray(files, np.int64))


def test_jets(access):
    require(access)
    inventory, splits = access.inventory, access.splits
    total, class_counts = 0, np.zeros(11, np.int64)
    for index, (record, group) in enumerate(zip(inventory['files'], splits['groups'])):
        if group['role'] != 'final_test':
            continue
        require(access)
        if group['path'] != record['path'] or group['sha256'] != record['sha256']:
            raise ValueError('Final-test file binding differs')
        path, _ = source_path(Path(access.f['input_container'])/'raw', record['path'])
        if sha256_file(path) != record['sha256']:
            raise ValueError('Final-test source checksum differs')
        entries = {entry: label for label, es in enumerate(record['entries_by_class']) for entry in es}
        branches = ['jet_nparticles', 'hlt_jet_nparticles'] + [
            prefix+field for prefix in ('hlt_part_', 'part_') for field in FIELDS]
        if entries:
            with uproot.open(path) as handle:
                tree = handle[record['tree_key']]
                for start, stop in _ranges(sorted(entries), 512):
                    arrays = tree.arrays(branches, entry_start=start, entry_stop=stop, library='ak', how=dict)
                    for row, entry in enumerate(range(start, stop)):
                        label = entries[entry]
                        yield RawJet(row_identity(record, entry), label, index, entry,
                            _matrix(arrays, 'hlt_part_', row, int(arrays['hlt_jet_nparticles'][row])),
                            _matrix(arrays, 'part_', row, int(arrays['jet_nparticles'][row])))
                        total += 1
                        class_counts[label] += 1
        if sha256_file(path) != record['sha256']:
            raise ValueError('Final-test source changed during read')
    if total != splits['selected_counts']['final_test'] or class_counts.tolist() != splits['reservoir_class_counts']['final_test']:
        raise ValueError('Final-test coverage/class counts differ')


def values(jet, p):
    hlt, off = physical(jet.hlt, p['conventions'], 'hlt'), physical(jet.offline, p['conventions'], 'offline')
    mapping = s.views.match_particles(hlt, off)
    return mapping, {coord: s.build_inputs(s.views.build_view(identity=jet.identity,
        proxy=hlt, offline=off, coordinate=coord, mapping=mapping), capacity=p['inputs']['capacity'])
        for coord in s.COORDINATES}


class TestCache:
    """Reuse only the ordinary cache's pure tensor batching, not its role gate."""
    def __init__(self, blocks, coordinate, access):
        require(access)
        self.access, self.blocks, self.role = access, blocks, 'final_test'
        self.coordinate_name, self.foundation_sha256 = coordinate, access.p['content_hash']
        self.ends = np.cumsum([len(b.labels) for b in blocks])
        self.labels = np.concatenate([b.labels for b in blocks])
        self.identities = np.concatenate([b.identities for b in blocks])
        self.nbytes = sum(b.nbytes for b in blocks)+self.labels.nbytes+self.identities.nbytes

    __len__ = RamCache.__len__

    def batch(self, indices):
        require(self.access)
        return RamCache.batch(self, indices)


class TestPair:
    def __init__(self, primary, context, access):
        require(access)
        if (primary.role != 'final_test' or context.role != 'final_test'
                or primary.access is not access or context.access is not access
                or not np.array_equal(primary.identities, context.identities)
                or not np.array_equal(primary.labels, context.labels)):
            raise ValueError('Final-test paired view identity/role differs')
        self.primary, self.context, self.access = primary, context, access
        self.role, self.coordinate_name = primary.role, primary.coordinate_name
        self.foundation_sha256 = primary.foundation_sha256
        self.labels, self.identities = primary.labels, primary.identities
        self.nbytes = primary.nbytes+(0 if primary is context else context.nbytes)

    __len__ = PairedCache.__len__

    def batch(self, indices):
        require(self.access)
        return PairedCache.batch(self, indices)


def for_node(bundle, node, access=None):
    primary = bundle[node['coordinate']]
    other = node['context_coordinate']
    if other is None:
        if primary.role == 'final_test':
            require(access)
        return primary
    return (TestPair(primary, bundle[other], access) if primary.role == 'final_test'
            else PairedCache(primary, bundle[other]))


def build(access, max_ram_bytes):
    require(access)
    if type(max_ram_bytes) is not int or max_ram_bytes <= 0:
        raise ValueError('Explicit final-test RAM budget required')
    blocks, meters = {c: [] for c in s.COORDINATES}, {c: s.ViewMeter() for c in s.COORDINATES}
    resident, files, labels, ids, seen_files = 0, [], [], [], set()
    for index, group in itertools.groupby(test_jets(access), key=lambda jet: jet.file_index):
        rows = {c: [] for c in s.COORDINATES}
        for jet in group:
            # Reserve the largest possible next row before building any feature arrays.
            upper = len(s.COORDINATES)*(access.p['inputs']['capacity']*84+88)
            if 2*(resident+upper)+64*2**20 > max_ram_bytes:
                raise MemoryError('Final-test caches exceed reserved RAM; no spill/truncation')
            _, constructed = values(jet, access.p)
            for coord, value in constructed.items():
                meters[coord].add(jet, value)
                rows[coord].append((RawJet(jet.identity, jet.label, index, jet.entry, None, None), value))
                resident += value.features.nbytes+value.vectors.nbytes+88
            ids.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
            labels.append(jet.label)
            files.append(index)
            seen_files.add(index)
        for coord in s.COORDINATES:
            if rows[coord]:
                blocks[coord].append(_block(index, rows[coord]))
        print(f'LUKA final-test views: rows={len(labels)} files={len(seen_files)} RAM_GiB={resident/2**30:.2f}', flush=True)
    bundle = {coord: TestCache(blocks[coord], coord, access) for coord in s.COORDINATES}
    if len(bundle['D000']) != access.execution['protocol']['counts']:
        raise ValueError('Final-test cache row count differs')
    if 2*sum(cache.nbytes for cache in bundle.values())+64*2**20 > max_ram_bytes:
        raise MemoryError('Final-test residency exceeds bound')
    arrays = dict(identities=np.asarray(ids, np.uint8), labels=np.asarray(labels, np.int64), files=np.asarray(files, np.int64))
    expected = population_arrays(access.inventory, access.splits)
    if any(not np.array_equal(arrays[k], expected[k]) for k in expected):
        raise ValueError('Frozen test identity/label/order differs')
    return bundle, arrays, {co: m.report() for co, m in meters.items()}


def train_smoke(parent, p, f, inventory, splits, limit=64):
    """Recompute small ordinary rows; compare every value to saved assignments."""
    refs = {r['file_index']: r for r in p['assignments']['train']}
    assigned, used, rows = {}, set(), {c: [] for c in s.COORDINATES}
    positions = {}
    reader = iter(ParticleReader(f['input_container'], inventory, splits, role='train', include_offline=True))
    try:
        for jet in itertools.islice(reader, limit):
            index = jet.file_index
            used.add(index)
            if index not in assigned:
                assigned[index] = s.load_assignment(parent['prepared_root'], refs[index])
                positions[index] = 0
            row, saved = positions[index], assigned[index]
            positions[index] += 1
            mapping, actual = values(jet, p)
            lo, hi = saved['offsets'][row:row+2]
            if not np.array_equal(mapping, saved['mapping'][lo:hi]):
                raise ValueError('TRAIN smoke reconstructed mapping differs')
            expected = _values(jet, saved, row, p, s.COORDINATES)
            for coord in s.COORDINATES:
                for key in ('features', 'vectors'):
                    if not np.array_equal(getattr(actual[coord], key), getattr(expected[coord], key)):
                        raise ValueError('TRAIN smoke view bytes differ')
                rows[coord].append((jet, actual[coord]))
    finally:
        reader.close()
    for index in used:
        record = inventory['files'][index]
        path, _ = source_path(Path(f['input_container'])/'raw', record['path'])
        if sha256_file(path) != record['sha256']:
            raise ValueError('TRAIN smoke input changed')
    if not rows['D000']:
        raise ValueError('Empty TRAIN smoke')
    return {co: RamCache([_block(0, items)], role='train', coordinate_name=co,
                        foundation_sha256=p['content_hash']) for co, items in rows.items()}
