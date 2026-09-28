"""Separate, locked full outer-confirmation capability; no development bypass."""
from contextlib import closing
import hashlib
from pathlib import Path

import awkward as ak
import numpy as np

from .audit import cms_particle_branches, latest_tree, validate_inventory
from .bridge import from_cms
from .bounded_data import metadata
from .contracts import artifact, canonical_sha256, load_json, safe_relative, validate, validate_compatibility
from .readers import Pair, authenticated_open
from .splits import unpack_entries, validate_roles

MIN_JETS = 250_000
SHARD_JETS = 5_000


def build(study):
    inventory, roles = metadata(study)
    validate_inventory(inventory)
    validate_roles(roles, inventory)
    rows = sorted((f for f in roles['files'] if f['response_role'] == 'response_confirm'), key=lambda f: f['path'])
    fit = [f for f in roles['files'] if f['response_role'] == 'response_fit']
    if (not rows or any(f['original_role'] != 'train' for f in rows)
            or {f['source'] for f in rows} != {f['source'] for f in fit}
            or {f['sha256'] for f in rows} & {f['sha256'] for f in fit}):
        raise PermissionError('Confirmation source coverage/disjointness differs')
    shards, files = [], []
    for f in rows:
        entries = unpack_entries(f['entry_mask'], f['raw_entries'])
        if len(entries) != f['selected_entries']:
            raise ValueError('Confirmation file count differs')
        files.append(dict(path=f['path'], sha256=f['sha256'], source=f['source'], jets=len(entries)))
        for start in range(0, len(entries), SHARD_JETS):
            chosen = entries[start:start+SHARD_JETS]
            digest = hashlib.sha256()
            for e in chosen:
                digest.update(canonical_sha256(['CMS2JC2_ROW/v1', f['sha256'], f['tree_key'], int(e)]).encode())
            shards.append(dict(index=len(shards), path=f['path'], sha256=f['sha256'], start=start,
                stop=start+len(chosen), jets=len(chosen), ordered_identities=digest.hexdigest()))
    jets = sum(f['jets'] for f in files)
    if jets < MIN_JETS or jets != roles['counts']['response_confirm']:
        raise ValueError('Need all reserved confirmation rows and at least 250000 jets; no borrowing')
    return artifact('FROZEN_MEMBERSHIP', parents={'inventory': inventory['content_hash'], 'roles': roles['content_hash']},
        outer_role='response_confirm', jets=jets, files=files, shards=shards,
        source_counts={s: sum(f['jets'] for f in files if f['source'] == s) for s in sorted({f['source'] for f in files})},
        particle_accessed=False, labels_accessed=False)


def validate_membership(value, study):
    validate(value, 'FROZEN_MEMBERSHIP')
    if value != build(study):
        raise PermissionError('Confirmation membership differs from complete canonical population')


def access_value(spec):
    if spec['stage'] != 'frozen_confirm' or spec['untouched_asserted'] is not True:
        raise PermissionError('No confirmation capability')
    return artifact('FROZEN_ACCESS', parents={'stage': spec['content_hash'],
        'protocol': spec['protocol']['content_hash'], 'membership': spec['membership']['content_hash'],
        'reuse': spec['reuse']['content_hash']}, selected='JOINT', reference='B_DZ',
        no_reselection=True, production_qualified=False, transfer_authorized=False)


def iter_confirmation(spec, study, *, shard):
    from . import frozen_joint_campaign as campaign, dev_campaign as dev
    authenticated_study = campaign.validate_stage(spec, source=False)
    if study != authenticated_study:
        raise PermissionError('Reader study differs from authenticated confirmation source')
    if type(shard) is not int or not 0 <= shard < len(spec['membership']['shards']):
        raise PermissionError('Unregistered confirmation shard')
    expected = access_value(spec)
    if load_json(dev.stage_dir(spec)/'confirmation_access.json') != expected:
        raise PermissionError('Published confirmation access lock required')
    task = f'jf_eval_{shard:04d}'
    claim = load_json(dev.stage_dir(spec)/'claims'/task/'claim.json')
    validate(claim, 'DEV_CLAIM', parents={'stage': spec['content_hash']})
    if claim['task_id'] != task:
        raise PermissionError('Exact confirmation worker claim required')
    inventory, roles = metadata(study)
    validate_compatibility(study['review'], inventory_hash=inventory['content_hash'])
    row = spec['membership']['shards'][shard]
    f = next(f for f in roles['files'] if f['path'] == row['path'] and f['response_role'] == 'response_confirm')
    selected = unpack_entries(f['entry_mask'], f['raw_entries'])[row['start']:row['stop']]
    branches = cms_particle_branches('offline')+cms_particle_branches('hlt')
    digest, count = hashlib.sha256(), 0
    with authenticated_open(safe_relative(Path(study['imported']['cms_root']), f['path']), f['sha256']) as handle:
        key, tree = latest_tree(handle)
        if key != f['tree_key'] or tree.num_entries != f['raw_entries']:
            raise ValueError('Confirmation ROOT identity changed')
        for start in sorted(set((selected//256*256).tolist())):
            entries = selected[np.searchsorted(selected, start):np.searchsorted(selected, start+256)]
            arrays = tree.arrays(list(branches), entry_start=start, entry_stop=start+256, library='ak', how=dict)
            for e in entries:
                cols = {name: ak.to_numpy(arrays[name][int(e)-start]) for name in branches}
                identity = canonical_sha256(['CMS2JC2_ROW/v1', f['sha256'], key, int(e)])
                digest.update(identity.encode()); count += 1
                yield Pair(identity, f['sha256'], from_cms(cols, study['review'], side='offline'),
                           from_cms(cols, study['review'], side='hlt'))
    if count != row['jets'] or digest.hexdigest() != row['ordered_identities']:
        raise ValueError('Confirmation reader membership incomplete')


def stream(spec, study, *, shard):
    return closing(iter_confirmation(spec, study, shard=shard))
