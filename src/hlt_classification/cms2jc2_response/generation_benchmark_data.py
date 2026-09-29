"""Frozen, bounded dzfix TRAIN membership; never native HLT or labels."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import awkward as ak
import numpy as np

from .bridge import JC2_FIELDS, from_jc2
from .contracts import artifact, canonical_sha256, load_json, validate, safe_relative
from .dev_data import checked_file
from .readers import Pair, authenticated_open
from hlt_classification.jetclass2_delphes.inventory import validate_inventory, latest_tree
from hlt_classification.jetclass2_delphes.splits import validate_splits, is_subset_profile
from hlt_classification.jetclass2_delphes.split_registry import unpack_entries
from hlt_classification.jetclass2_delphes.contracts import row_identity

COUNT = 10_000
RELEASE = 'jetclass2_10M_20260918_dzfix'
BRANCHES = ('jet_nparticles', *('part_'+f for f in JC2_FIELDS))


def build(inventory, profile):
    ih = validate_inventory(inventory)
    validate_splits(profile, inventory)
    if not is_subset_profile(profile):
        raise PermissionError('An explicit registered training membership is required')
    groups = {g['path']: g for g in profile['groups']}
    files = {r['path']: r for r in inventory['files']}
    available = []
    for member in profile['memberships']['train']['files']:
        row = files[member['path']]
        if groups[row['path']]['role'] != 'train':
            raise PermissionError('Benchmark escaped TRAIN')
        entries = unpack_entries(member['entry_mask'], row['entries'])
        if len(entries) >= (COUNT+3)//4:
            available.append((row, entries))
    # One hashed file per source first, then additional hashed files. This is
    # a disclosed engineering sample, not a label-balanced physics evaluation.
    ordered = sorted(available, key=lambda r: canonical_sha256(['GEN_FILES/v1', r[0]['sha256']]))
    first, rest, sources = [], [], set()
    for row in ordered:
        if row[0]['source'] not in sources:
            first.append(row); sources.add(row[0]['source'])
        else:
            rest.append(row)
    chosen = (first+rest)[:8]
    if len(chosen) < 4:
        raise ValueError('Need at least four TRAIN files with sufficient eligible capacity')
    members, identities = [], []
    for i, (row, entries) in enumerate(chosen):
        count = COUNT//len(chosen) + int(i < COUNT % len(chosen))
        start = int(canonical_sha256(['GEN_WINDOW/v1', row['sha256']]), 16) % (len(entries)-count+1)
        selected = entries[start:start+count].tolist()
        ids = [row_identity(ih, row['path'], row['tree_key'], e) for e in selected]
        identities.extend(ids)
        members.append(dict(path=row['path'], sha256=row['sha256'], tree_key=row['tree_key'],
                            raw_entries=row['entries'], source=row['source'], entries=selected, jets=count))
    if len(set(identities)) != COUNT:
        raise ValueError('Benchmark membership duplicate or incomplete')
    return artifact('GEN_MEMBERSHIP', parents={'inventory': ih, 'profile': profile['content_hash']},
        role='train', jets=COUNT, files=members, ordered_identities=identity_hash(identities),
        gate_ordered_identities=identity_hash(identities[:64]),
        method='source_diverse_hash_files_eligible_window_v1', native_hlt_conditioning_inherited=True)


def identity_hash(values):
    return hashlib.sha256(b''.join(bytes.fromhex(s) for s in values)).hexdigest()


def metadata(study):
    return (load_json(checked_file(study['inventory'])), load_json(checked_file(study['profile'])))


def validate_membership(study):
    inventory, profile = metadata(study)
    if study['membership'] != build(inventory, profile):
        raise PermissionError('Frozen TRAIN membership differs')


def _iterate(study, *, limit=None):
    if limit not in (None, 64):
        raise PermissionError('Only the registered gate prefix or full training sample is allowed')
    inventory, _ = metadata(study)
    root = Path(study['data_root'])
    remaining = study['membership']['jets'] if limit is None else limit
    seen = set()
    for row in study['membership']['files']:
        if remaining == 0:
            break
        entries = np.asarray(row['entries'][:remaining], dtype=np.int64)
        path = safe_relative(root, row['path'])
        with authenticated_open(path, row['sha256']) as handle:
            key, tree = latest_tree(handle)
            if key != row['tree_key'] or tree.num_entries != row['raw_entries']:
                raise ValueError('JC2 tree identity changed')
            # Bounded windows, not a scan of the whole file. Expressions are an
            # explicit offline-only capability; labels/HLT are never requested.
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
                    if any(len(a) != n for a in columns.values()):
                        raise ValueError('Offline jagged counts differ')
                    identity = row_identity(inventory['content_hash'], row['path'], key, int(entry))
                    if identity in seen:
                        raise ValueError('Duplicate benchmark jet')
                    seen.add(identity); remaining -= 1
                    yield Pair(identity, row['sha256'], from_jc2(columns, study['review'],
                        keys=tuple(f'part:{j}' for j in range(n))), None)
    if remaining:
        raise ValueError('Incomplete benchmark population')


@contextmanager
def stream(spec, study, task_id, *, limit=None):
    from . import dev_campaign as dev
    if spec['study']['path'] != str(Path(study['root'])/'study_spec.json'):
        raise PermissionError('Wrong study for particle read')
    task = next((t for t in spec['tasks'] if t['task_id'] == task_id), None)
    if task is None or task['action'] not in ('jg_gate', 'jg_run'):
        raise PermissionError('No TRAIN read capability for this task')
    if (limit == 64) != (task['action'] == 'jg_gate'):
        raise PermissionError('Population does not match registered task')
    claim = load_json(dev.stage_dir(spec)/'claims'/task_id/'claim.json')
    validate(claim, 'DEV_CLAIM', parents={'stage': spec['content_hash']})
    if claim['task_id'] != task_id:
        raise PermissionError('Wrong task execution claim')
    validate_membership(study)
    iterator = _iterate(study, limit=limit)
    try:
        yield iterator
    finally:
        iterator.close()
