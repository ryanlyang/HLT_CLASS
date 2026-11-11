"""Exact historical model inputs; no cross-platform re-encoding of anchors.

The small delta fills only the gaps in Brown's original HLT/U100 bundle and
native OFFLINE transfer. Missing source caches may be replayed on SPORC, but
every replay must reproduce the historical file bytes and population digest.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import shutil

import numpy as np

from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    write_immutable_json,
)
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_cache import _block, _values
from hlt_classification.luka_fullsim.particles import RawJet
from hlt_classification.luka_teacher_sweep_hydra.transfer import Delta as OfflineDelta
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from hlt_classification.luka_u100_pair_transfer import bundle as original_transfer
from hlt_classification.provenance import validate_source_snapshot_payload
from .contracts import COUNTS, artifact, validate
from .storage import regular, join
from .transfer import fresh


HISTORICAL = ('D000', 'U100', 'D033', 'D066', 'OFFLINE')
REQUIRED = {'train': ('D033', 'D066'), 'validation': ('D033', 'D066', 'U100')}
POLICY = 'authenticated_historical_tensors_plus_new_quarter_views_v2'


def references(cache):
    return [dict(path=r['path'], bytes=r['bytes'], sha256=r['sha256'])
            for role, coords in REQUIRED.items() for c in coords
            for r in cache['blocks'][role][c]]


def replay_file(root, container, meta, role, member, meters):
    """At most one source file resident; never substitute for a corrupt cache."""
    cache, prepared, inventory, _ = meta
    index = member['file_index']; blocks = {}; missing = {}
    for coord in REQUIRED[role]:
        matches = [r for r in cache['blocks'][role][coord] if r['file_index'] == index]
        if len(matches) != 1 or matches[0]['path'] != f'inputs/{role}/{coord}/{index:04d}.npz':
            raise ValueError('Historical native block coverage/path differs')
        ref = matches[0]; path = regular(root, ref['path'])
        try:
            path.stat()
        except FileNotFoundError:
            missing[coord] = ref
        else:
            blocks[coord] = regular_cache.read_block(root, ref, member, inventory, meters[coord])
    if missing:
        assignments = [r for r in prepared['assignments'][role] if r['file_index'] == index]
        if len(assignments) != 1:
            raise ValueError('Original native assignment coverage differs')
        assigned = regular_cache.s.load_assignment(root, assignments[0])
        rows = {c: [] for c in missing}
        for i, jet in enumerate(regular_cache.selected_file_rows(container, inventory, member)):
            for coord, value in _values(jet, assigned, i, prepared, tuple(missing)).items():
                meters[coord].add(jet, value)
                light = RawJet(jet.identity, jet.label, index, jet.entry, None, None)
                rows[coord].append((light, value))
        for coord, ref in missing.items():
            if len(rows[coord]) != ref['rows'] or len(rows[coord]) != len(assigned['labels']):
                raise ValueError('Historical native replay population differs')
            blocks[coord] = _block(index, rows[coord])
    return blocks


def export(prepared_root, origin_spec, output, *, source_snapshot):
    old = original_transfer.checked(load_json(origin_spec))
    proof = original_transfer.authenticate(Path(origin_spec).resolve(), old)
    root = Path(prepared_root).resolve(strict=True); meta = regular_cache.metadata(root)
    cache, prepared, _, splits = meta
    anchor = old['anchor']['evidence']
    if (proof['counts'] != COUNTS or proof['prepared'] != prepared['content_hash']
            or proof['cache'] != cache['content_hash'] or root != Path(anchor['prepared_root']).resolve()):
        raise ValueError('Historical native preparation differs')
    validate_source_snapshot_payload(source_snapshot)
    output = fresh(output, [root, anchor['raw_container'], old['root']])
    files = references(cache); total = sum(r['bytes'] for r in files)
    if shutil.disk_usage(output.parent).free < total + 2**30:
        raise OSError('Insufficient native export space including one GiB headroom')
    output.mkdir(); expected = {r['path']: r for r in files}; seen = set()
    for role, coords in REQUIRED.items():
        meters = {c: regular_cache.s.ViewMeter() for c in coords}
        members = splits['memberships'][role]
        for number, member in enumerate(members, 1):
            index = member['file_index']
            blocks = replay_file(root, anchor['raw_container'], meta, role, member, meters)
            for coord, block in blocks.items():
                name = f'inputs/{role}/{coord}/{index:04d}.npz'; ref = expected[name]
                payload = deterministic_npz_bytes({k: v for k, v in vars(block).items() if k != 'file_index'})
                if len(payload) != ref['bytes'] or hashlib.sha256(payload).hexdigest() != ref['sha256']:
                    raise ValueError('Native export differs from historical bytes: ' + name)
                path = output / name; path.parent.mkdir(parents=True, exist_ok=True)
                atomic_publish_bytes(path, payload); seen.add(name)
            if number == 1 or number % 20 == 0 or number == len(members):
                print(f'EMBEDDING_BRIDGE native export {role}: {number}/{len(members)} files; historical bytes verified', flush=True)
        for c, meter in meters.items():
            if meter.report() != prepared['summaries'][role][c]:
                raise ValueError('Native export historical fingerprint differs: ' + role + '/' + c)
    if seen != set(expected):
        raise ValueError('Native export incomplete')
    value = artifact('NATIVE_TRANSFER', parents=dict(prepared=proof['prepared'], cache=proof['cache'],
        source=source_snapshot['content_hash']), origin=proof, source_snapshot=source_snapshot,
        coordinates={r: list(cs) for r, cs in REQUIRED.items()}, files=files, payload_bytes=total,
        validation_usage='evaluation_only')
    write_immutable_json(output / 'manifest.json', value)
    return dict(directory=str(output), manifest_hash=value['content_hash'], payload_bytes=total)


class Delta:
    def __init__(self, root, digest, bundle):
        root = Path(root).absolute()
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError('Redirected native delta root')
        self.root = root.resolve(strict=True)
        self.manifest = load_json(regular(self.root, 'manifest.json')); m = self.manifest
        validate(m, 'NATIVE_TRANSFER'); source = validate_source_snapshot_payload(m['source_snapshot'])
        origin = bundle.manifest['origin']; expected = references(bundle.json('prepared/cache_manifest.json'))
        if (m['content_hash'] != digest or m['origin'] != origin or m['files'] != expected
                or m['parents'] != dict(prepared=origin['prepared'], cache=origin['cache'], source=source)
                or m['coordinates'] != {r: list(cs) for r, cs in REQUIRED.items()}
                or m['validation_usage'] != 'evaluation_only'
                or m['payload_bytes'] != sum(r['bytes'] for r in expected)):
            raise ValueError('Native delta differs from historical inventory/lineage')
        self.rows = {r['path']: r for r in expected}
        if len(self.rows) != len(expected):
            raise ValueError('Duplicate native delta member')

    def path(self, name):
        if name not in self.rows:
            raise PermissionError('Undeclared native delta member')
        path = regular(self.root, name); validate_file_ref(self.rows[name], root=self.root)
        return path

    def verify_all(self):
        for i, name in enumerate(self.rows, 1):
            self.path(name)
            if i % 200 == 0:
                print(f'EMBEDDING_BRIDGE verified {i}/{len(self.rows)} native delta files', flush=True)


class Sources:
    def __init__(self, spec):
        self.bundle = Bundle(spec['bundle_root'])
        self.delta = Delta(spec['native_root'], spec['native_hash'], self.bundle)
        validate_file_ref(spec['parent_spec']); parent = load_json(spec['parent_spec']['path'])
        self.offline = OfflineDelta(parent['offline_root'], parent['offline_hash'], spec['bundle_root'])
        self.cache = self.bundle.json('prepared/cache_manifest.json')
        self.refs = {(role, c, r['file_index']): r for role in COUNTS for c in HISTORICAL
                     for r in self.cache['blocks'][role][c]}

    def path(self, role, coord, index):
        if (role, coord, index) not in self.refs:
            raise PermissionError('Undeclared historical native input')
        ref = self.refs[role, coord, index]; name = ref['path']
        if name != f'inputs/{role}/{coord}/{index:04d}.npz':
            raise ValueError('Historical input ownership differs')
        if coord == 'OFFLINE':
            return self.offline.path(name)
        if coord in REQUIRED[role]:
            return self.delta.path(name)
        return self.bundle.path('prepared/' + name)

    def verify_all(self):
        for key in self.refs:
            self.path(*key)

    def block(self, role, coord, index, physical):
        a = load_npz_arrays(self.path(role, coord, index)); join(a, physical)
        offsets = physical['OFFLINE_offsets' if coord == 'OFFLINE' else 'HLT_offsets']
        if (set(a) != {'identities', 'labels', 'offsets', 'features', 'vectors'}
                or a['offsets'].dtype != np.int64 or not np.array_equal(a['offsets'], offsets)
                or a['features'].dtype != np.float32 or a['features'].shape != (offsets[-1], 17)
                or a['vectors'].dtype != np.float32 or a['vectors'].shape != (offsets[-1], 4)
                or not np.isfinite(a['features']).all() or not np.isfinite(a['vectors']).all()):
            raise ValueError('Historical native shape/support differs')
        return SimpleNamespace(**a)
