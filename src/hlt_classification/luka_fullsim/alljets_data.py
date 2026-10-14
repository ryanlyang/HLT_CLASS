"""Explicit retirement of the historical holdout into TRAIN; legacy roles intact."""
from pathlib import Path

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.provenance import validate_source_snapshot_payload
from . import splits as legacy, stage2 as s
from .contracts import row_identity, source_path
from .inventory import build_inventory, validate_inventory, verify_source_bytes
from .particles import ParticleReader as LegacyReader

PREFIX = 'LUKA_FULLSIM_ALLJETS_'
FLAGS = dict(final_test_accessed=False, historical_test_evaluated=True,
             historical_test_repurposed_for_training=True, independent_test_available=False,
             evaluation_scope='exploratory_development_validation')


def artifact(kind, *, parents=None, version=1, **fields):
    return with_content_hash(dict(contract=f'{PREFIX}{kind}/v{version}', schema_version=version,
        parents=dict(parents or {}), **{**FLAGS, **fields}))


def validate(value, kind, parents=None, *, version=1):
    digest = validate_content_hash(value, expected_contract=f'{PREFIX}{kind}/v{version}', expected_schema_version=version)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('All-jets experiment must disclose retired test and exploratory scope')
    if parents is not None and value['parents'] != parents:
        raise ValueError('All-jets parent lineage differs')
    return digest


def derive_splits(inventory, original):
    """Deterministic union of existing file reservoirs; no new random assignment."""
    validate_inventory(inventory)
    legacy.validate_splits(original, inventory)
    members = {role: [] for role in ('train', 'validation')}
    groups, counts = [], {r: [0]*11 for r in members}
    for index, (record, old) in enumerate(zip(inventory['files'], original['groups'])):
        role = 'validation' if old['role'] == 'validation' else 'train'
        groups.append(dict(path=record['path'], sha256=record['sha256'], role=role, historical_role=old['role']))
        if sum(record['selected_class_counts']):
            members[role].append(dict(file_index=index, entries_by_class=record['entries_by_class']))
        counts[role] = [x+y for x, y in zip(counts[role], record['selected_class_counts'])]
    return artifact('SPLITS', parents=dict(inventory=inventory['content_hash'], original_splits=original['content_hash']),
        original_splits=original, algorithm='original_validation_files_held_out_all_other_files_train_v1',
        groups=groups, memberships=members, class_counts=counts,
        selected_counts={**{r: sum(v) for r, v in counts.items()}, 'final_test': 0}, unused_eligible_rows=0,
        comparison_memberships=original['memberships']['validation'],
        comparison_rows=original['selected_counts']['validation'],
        proven_disjointness='file_paths_and_content_hashes', cross_file_event_independence='unverified')


def validate_splits(value, inventory):
    digest = validate(value, 'SPLITS')
    if value != derive_splits(inventory, value['original_splits']):
        raise ValueError('All-jets split replay differs; never move validation files or omit eligible rows')
    return digest


def row_arrays(inventory, members):
    ordered = [(row_identity(inventory['files'][m['file_index']], e), label)
               for m in members for e, label in sorted((e, c) for c, es in enumerate(m['entries_by_class']) for e in es)]
    return (np.asarray([np.frombuffer(bytes.fromhex(key), np.uint8) for key, _ in ordered]),
            np.asarray([label for _, label in ordered], np.int64))


def comparison_indices(inventory, splits, identities, labels):
    expected_ids, expected_labels = row_arrays(inventory, splits['memberships']['validation'])
    if not np.array_equal(identities, expected_ids) or not np.array_equal(labels, expected_labels):
        raise ValueError('Full validation prediction identities/order/labels differ')
    wanted, wanted_labels = row_arrays(inventory, splits['comparison_memberships'])
    lookup = {bytes(key): i for i, key in enumerate(identities)}
    indices = np.asarray([lookup[bytes(key)] for key in wanted], np.int64)
    if len(indices) != splits['comparison_rows'] or not np.array_equal(labels[indices], wanted_labels):
        raise ValueError('Original validation subset join differs')
    return indices


class ParticleReader(LegacyReader):
    """New explicit capability; inherit only the unchanged selected-range iterator."""
    def __init__(self, container, inventory, splits, *, role, include_offline=False, chunk=512):
        if role not in ('train', 'validation'):
            raise PermissionError('All-jets experiment has no test role')
        if type(include_offline) is not bool or type(chunk) is not int or not 1 <= chunk <= 2048:
            raise ValueError('Invalid reader options')
        validate_splits(splits, inventory)
        self.raw = Path(container).resolve(strict=True)/'raw'
        self.inventory, self.splits, self.role = inventory, splits, role
        self.include_offline, self.chunk = include_offline, chunk


def build(gate):
    from . import alljets_campaign as c
    c.validate_gate(gate)
    output = Path(gate['root'])/'foundation'
    output.mkdir(exist_ok=False)
    inventory = build_inventory(Path(gate['input_container']))
    original = legacy.build_splits(inventory)
    splits = derive_splits(inventory, original)
    c.check_population(inventory, splits)
    for name, value in (('inventory.json', inventory), ('split_manifest.json', splits)):
        write_immutable_json(output/name, value)
    result = artifact('FOUNDATION', parents=dict(inventory=inventory['content_hash'], splits=splits['content_hash'],
        source=gate['source_snapshot']['content_hash']), source_snapshot=gate['source_snapshot'],
        input_container=gate['input_container'], counts=splits['selected_counts'],
        source_particles_validated=False, pre_split_scalar_metadata_accessed=True,
        artifacts={name: dict(sha256=sha256_file(output/name), bytes=(output/name).stat().st_size)
                   for name in ('inventory.json', 'split_manifest.json')})
    verify_source_bytes(Path(gate['input_container']), inventory)
    c.validate_gate(gate)
    write_immutable_json(output/'foundation.json', result)
    return result


def load_foundation(root):
    root = Path(root)
    f = load_json(root/'foundation.json')
    validate(f, 'FOUNDATION')
    validate_source_snapshot_payload(f['source_snapshot'])
    if (f['source_snapshot']['worktree_clean'] is not True or f['source_particles_validated'] is not False
            or f['pre_split_scalar_metadata_accessed'] is not True
            or set(f['artifacts']) != {'inventory.json', 'split_manifest.json'}):
        raise ValueError('All-jets foundation readiness/source differs')
    for name, ref in f['artifacts'].items():
        if (root/name).stat().st_size != ref['bytes'] or sha256_file(root/name) != ref['sha256']:
            raise ValueError('All-jets foundation payload differs')
    inventory, splits = load_json(root/'inventory.json'), load_json(root/'split_manifest.json')
    parents = dict(inventory=validate_inventory(inventory), splits=validate_splits(splits, inventory),
                   source=f['source_snapshot']['content_hash'])
    if f['parents'] != parents or f['counts'] != splits['selected_counts']:
        raise ValueError('All-jets foundation parents/counts differ')
    return f, inventory, splits


def load_prepared(root):
    root = Path(root)
    p = load_json(root/'prepared.json')
    validate(p, 'PREPARED')
    f, inventory, splits = load_foundation(p['foundation_root'])
    audit, config = p['audit'], p['conventions']
    validate(audit, 'PARTICLE_AUDIT', dict(foundation=f['content_hash'], source=p['source_snapshot']['content_hash']))
    validate_source_snapshot_payload(p['source_snapshot'])
    s.validate_conventions(config)
    counts = {r: f['counts'][r] for r in ('train', 'validation')}
    if (audit['source_snapshot'] != p['source_snapshot'] or audit['counts'] != counts
            or audit['scientific_admission'] is not False or audit['units'] != 'stored_unconfirmed_no_conversion'
            or p['source_snapshot'] != f['source_snapshot'] or p['source_snapshot']['worktree_clean'] is not True
            or p['parents'] != dict(foundation=f['content_hash'], audit=audit['content_hash'],
                conventions=config['content_hash'], source=p['source_snapshot']['content_hash'])
            or config['parents'] != dict(audit=audit['content_hash']) or p['counts'] != counts
            or p['science_authorized'] is not False or any(p[k] != v for k, v in s.contracts().items())):
        raise ValueError('All-jets preparation lineage/policy differs')
    for role in ('train', 'validation'):
        members = splits['memberships'][role]
        if ([r['file_index'] for r in p['assignments'][role]] != [m['file_index'] for m in members]
                or set(p['summaries'][role]) != set(s.COORDINATES)):
            raise ValueError('All-jets assignment/coordinate coverage differs')
        for ref, member in zip(p['assignments'][role], members):
            arrays = s.load_assignment(root, ref)
            ids, labels = row_arrays(inventory, [member])
            if not np.array_equal(arrays['identities'], ids) or not np.array_equal(arrays['labels'], labels):
                raise ValueError('All-jets assignment row join differs')
        for summary in p['summaries'][role].values():
            if summary['rows'] != counts[role] or not 0 < summary['maximum'] <= 512:
                raise ValueError('All-jets input coverage/capacity differs')
    return p, f, inventory, splits


def storage_bound(container, inventory, splits):
    """Conservative ragged-output bound from authenticated scalar counts, no particles."""
    validate_splits(splits, inventory)
    sizes = []
    for role in ('train', 'validation'):
        for member in splits['memberships'][role]:
            record = inventory['files'][member['file_index']]
            path, _ = source_path(Path(container)/'raw', record['path'])
            if sha256_file(path) != record['sha256']:
                raise ValueError('Storage source checksum differs')
            entries = sorted(e for es in member['entries_by_class'] for e in es)
            with uproot.open(path) as handle:
                a = handle[record['tree_key']].arrays(['jet_nparticles', 'hlt_jet_nparticles'], library='np')
            no, nh = (np.asarray(a[key][entries], np.int64) for key in ('jet_nparticles', 'hlt_jet_nparticles'))
            if np.any(no <= 0) or np.any(nh <= 0) or np.any(no > 512) or np.any(nh > 512):
                raise ValueError('Native support exceeds unchanged nontruncating capacity')
            # OFFLINE=no; two U-tail views <=nh+no; four other views=nh.
            particles = int((no + 2*np.minimum(nh+no, 512) + 4*nh).sum())
            sizes.append(particles*84 + len(entries)*7*128 + int(nh.sum())*4 + len(entries)*64 + 65536)
            if sha256_file(path) != record['sha256']:
                raise ValueError('Storage source changed')
    # NPZ overhead, bounded eight-worker temporary publication and 5 GiB reserve.
    return int(1.2*sum(sizes)) + 2*sum(sorted(sizes, reverse=True)[:8]) + 5*2**30
