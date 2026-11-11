"""Reuse completed C03 subclass probabilities/prototypes byte for byte; never refit."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json, atomic_publish_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullstat_subclass import targets as previous

FILES = previous.FILES


def check_reuse(reuse):
    old = load_json(validate_file_ref(reuse['parent_spec']))
    folder = Path(old['root'])/'subclasses'
    proof = reuse['subclasses']
    expected = [file_ref(folder/name) for name in FILES]
    if proof['files'] != expected:
        raise ValueError('Frozen subclass files changed')
    manifest = previous.check_manifest(old)
    if manifest['content_hash'] != proof['manifest_hash']:
        raise ValueError('Frozen subclass identity changed')
    return old,manifest


def check_manifest(spec):
    # The bank retains its original v1 identity: its scientific meaning did not
    # change. Only the new training recipe/receipts get the six-teacher contract.
    manifest = previous.check_manifest(spec)
    if manifest['content_hash'] != spec['reuse']['subclasses']['manifest_hash']:
        raise ValueError('Copied subclass identity differs')
    folder = Path(spec['root'])/'subclasses'
    for name,ref in zip(FILES,spec['reuse']['subclasses']['files']):
        if file_ref(folder/name)['sha256'] != ref['sha256']:
            raise ValueError('Copied subclass bytes differ')
    return manifest


def prepare(spec,train,offline):
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or not np.array_equal(train.identities,offline.identities)
            or not np.array_equal(train.labels,offline.labels)
            or offline.manifest['content_hash'] != spec['reuse']['banks']['OFFLINE']['manifest_hash']):
        raise PermissionError('Frozen subclass reuse requires original ordered TRAIN and OFFLINE bank')
    old,_ = check_reuse(spec['reuse'])
    # Validate exact prepared identity, row ordering, dtype and soft probabilities
    # before copying. A fresh run never imports the old student's weights.
    previous.Bank(old,train)
    dest = Path(spec['root'])/'subclasses'; dest.mkdir(exist_ok=False)
    for name,ref in zip(FILES,spec['reuse']['subclasses']['files']):
        atomic_publish_bytes(dest/name,Path(validate_file_ref(ref)).read_bytes())
    check_manifest(spec)
    return previous.Bank(spec,train)
