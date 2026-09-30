"""Metadata-only fixed population and offline-only, bounded ROOT ingestion."""
import hashlib
from pathlib import Path

import awkward as ak
import numpy as np

from .contracts import artifact, validate, canonical_sha256, digest_ids, safe
from hlt_classification.jetclass2_delphes.inventory import validate_inventory, latest_tree
from hlt_classification.jetclass2_delphes.split_registry import (
    validate_split_profile, pack_entries, unpack_entries,
)
from hlt_classification.jetclass2_delphes.contracts import row_identity
from hlt_classification.cms2jc2_response.bridge import JC2_FIELDS, from_jc2
from hlt_classification.cms2jc2_response.readers import Pair, authenticated_open
from hlt_classification.cms2jc2_response.generation_benchmark_data import BRANCHES

COUNTS = dict(train=1_000_000, validation=250_000, final_test=1_000_000)
PROFILE = 'TRAIN_1M'
PILOT_JETS = 10_000
SHARD_JETS = 25_000


def ids(inventory_hash, row, entries):
    return (row_identity(inventory_hash, row['path'], row['tree_key'], int(e)) for e in entries)


def build(inventory, profile, donor_profile):
    ih = validate_inventory(inventory)
    validate_split_profile(profile, inventory)
    validate_split_profile(donor_profile, inventory)
    if (profile['profile'] != PROFILE or profile['registry_sha256'] != donor_profile['registry_sha256']
            or profile['groups'] != donor_profile['groups']
            or any(profile['memberships'][r] != donor_profile['memberships'][r]
                   for r in ('validation', 'final_test'))):
        raise ValueError('Require same-registry TRAIN_1M and unchanged evaluation reservoirs')
    files = {r['path']: r for r in inventory['files']}
    groups = {r['path']: r['role'] for r in profile['groups']}
    donor_train = {r['path']: r for r in donor_profile['memberships']['train']['files']}
    rows = []
    val_candidates = []
    for role in COUNTS:
        for member in sorted(profile['memberships'][role]['files'], key=lambda r: r['path']):
            source = files[member['path']]
            if groups[source['path']] != role:
                raise PermissionError('Split escaped its outer file role')
            entries = unpack_entries(member['entry_mask'], source['entries'])
            if role == 'train' and source['path'] in donor_train:
                old = unpack_entries(donor_train[source['path']]['entry_mask'], source['entries'])
                if not np.isin(old, entries).all():
                    raise ValueError('Existing training membership not nested')
                donor_train.pop(source['path'])
            row = dict(path=source['path'], sha256=source['sha256'], source=source['source'],
                tree_key=source['tree_key'], raw_entries=source['entries'], role=role,
                entry_mask=member['entry_mask'], jets=len(entries))
            if role == 'validation':
                # Hash rankings are label-blind; only selected entry metadata is used.
                for e, identity in zip(entries, ids(ih, row, entries)):
                    key = hashlib.sha256(b'CMS2JC2_PROXY_VAL_250K/v1\0'+bytes.fromhex(identity)).digest()
                    val_candidates.append((key, identity, len(rows), int(e)))
            rows.append(row)
    if donor_train:
        raise ValueError('Training subset omitted donor files')
    if len(val_candidates) < COUNTS['validation']:
        raise ValueError('Insufficient validation capacity')
    chosen = {}
    for _, _, index, entry in sorted(val_candidates)[:COUNTS['validation']]:
        chosen.setdefault(index, []).append(entry)
    for index, row in enumerate(rows):
        if row['role'] == 'validation':
            entries = np.asarray(sorted(chosen.get(index, [])), dtype=np.int64)
            row.update(entry_mask=pack_entries(entries, row['raw_entries']), jets=len(entries))
        else:
            entries = unpack_entries(row['entry_mask'], row['raw_entries'])
        row['ordered_identities'] = digest_ids(ids(ih, row, entries))
    rows = [r for r in rows if r['jets']]
    if any(sum(r['jets'] for r in rows if r['role'] == role) != count for role, count in COUNTS.items()):
        raise ValueError('Production population counts differ')
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('File roles overlap')
    return artifact('POPULATION', parents={'inventory': ih, 'profile': profile['content_hash'],
        'donor_profile': donor_profile['content_hash']}, counts=COUNTS, files=rows,
        validation_method='sha256_domain_and_row_identity_lowest_250k_v1',
        inherited_native_hlt_eligibility=True, labels_read=False)


def shards(population):
    validate(population, 'POPULATION', test=False)
    ih = population['parents']['inventory']
    train = sorted([r for r in population['files'] if r['role'] == 'train'],
                   key=lambda r: canonical_sha256(['PILOT_FILES/v1', r['sha256']]))
    distinct, rest, sources = [], [], set()
    for row in train:
        (rest if row['source'] in sources else distinct).append(row)
        sources.add(row['source'])
    pilots = (distinct+rest)[:2]
    if len(pilots) != 2 or min(r['jets'] for r in pilots) < 64:
        raise ValueError('Need two TRAIN files with at least 64 jets for retained pilot')
    specs, consumed = [], {}

    def append(row, start, stop, pilot):
        entries = unpack_entries(row['entry_mask'], row['raw_entries'])[start:stop]
        specs.append(dict(shard_id=f's{len(specs):05d}', role=row['role'], path=row['path'],
            start=start, stop=stop, jets=stop-start, pilot=pilot,
            ordered_identities=digest_ids(ids(ih, row, entries))))

    for row in pilots:
        count = min(row['jets'], PILOT_JETS)
        append(row, 0, count, True)
        consumed[row['path']] = count
    for row in population['files']:
        for start in range(consumed.get(row['path'], 0), row['jets'], SHARD_JETS):
            append(row, start, min(start+SHARD_JETS, row['jets']), False)
    return specs


def entries_for(population, shard):
    row = next(r for r in population['files'] if r['path'] == shard['path'])
    entries = unpack_entries(row['entry_mask'], row['raw_entries'])[shard['start']:shard['stop']]
    if (shard['role'] != row['role'] or len(entries) != shard['jets']
            or digest_ids(ids(population['parents']['inventory'], row, entries)) != shard['ordered_identities']):
        raise ValueError('Shard population differs')
    return row, entries


def iterate(data_root, population, shard, review, *, test_lock=None):
    if shard['role'] == 'final_test':
        if test_lock is None:
            raise PermissionError('Sealed test materialization requires build lock')
        validate(test_lock, 'TEST_BUILD_LOCK', test=False)
        if test_lock['parents']['population'] != population['content_hash'] or test_lock['evaluate'] is not False:
            raise PermissionError('Wrong sealed test build lock')
    row, entries = entries_for(population, shard)
    ih = population['parents']['inventory']
    with authenticated_open(safe(data_root, row['path']), row['sha256']) as handle:
        key, tree = latest_tree(handle)
        if key != row['tree_key'] or tree.num_entries != row['raw_entries']:
            raise ValueError('ROOT tree identity changed')
        for start in range(int(entries[0]), int(entries[-1])+1, 512):
            selected = entries[(entries >= start) & (entries < start+512)]
            if not len(selected):
                continue
            arrays = tree.arrays(list(BRANCHES), entry_start=start,
                entry_stop=min(start+512, int(entries[-1])+1, row['raw_entries']), library='ak', how=dict)
            for entry in selected:
                index = int(entry)-start
                n = int(arrays['jet_nparticles'][index])
                columns = {f: ak.to_numpy(arrays['part_'+f][index]) for f in JC2_FIELDS}
                if any(len(v) != n for v in columns.values()):
                    raise ValueError('Jagged offline counts differ')
                identity = row_identity(ih, row['path'], key, int(entry))
                yield Pair(identity, row['sha256'], from_jc2(columns, review,
                    keys=tuple(f'part:{j}' for j in range(n))), None)
