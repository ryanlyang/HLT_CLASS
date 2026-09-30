"""Separate production schemas, truthful test-access flags and atomic records."""
from pathlib import Path
import hashlib
import json

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash,
    with_content_hash, atomic_publish_bytes,
)

GIB = 2**30
PREFIX = 'CMS2JC2_PROXY_'


def artifact(schema_kind, *, parents=None, test=False, **fields):
    reserved = {'contract', 'schema_version', 'content_hash', 'parents',
                'final_test_accessed', 'final_test_materialized', 'final_test_evaluated'}
    if type(test) is not bool or reserved.intersection(fields):
        raise ValueError('Reserved production fields')
    return with_content_hash(dict(contract=f'{PREFIX}{schema_kind}/v1', schema_version=1,
        parents=dict(parents or {}), final_test_accessed=bool(test),
        final_test_materialized=bool(test), final_test_evaluated=False, **fields))


def validate(value, kind, *, parents=None, test=None):
    validate_content_hash(value, expected_contract=f'{PREFIX}{kind}/v1')
    if (value.get('contract') != f'{PREFIX}{kind}/v1' or value.get('schema_version') != 1
            or value.get('final_test_evaluated') is not False
            or type(value.get('final_test_accessed')) is not bool
            or value['final_test_materialized'] != value['final_test_accessed']
            or (test is not None and value['final_test_accessed'] != test)
            or (parents is not None and value['parents'] != parents)):
        raise ValueError('Production contract/scope/lineage differs: '+kind)
    return value['content_hash']


def safe(root, relative):
    root = Path(root).resolve()
    parts = str(relative).split('/')
    if any(p in ('', '.', '..') for p in parts) or '\\' in str(relative) or ':' in str(relative):
        raise ValueError('Unsafe relative path')
    path = root
    for part in parts:
        path = path/part
        if path.is_symlink():
            raise ValueError('Symlinks forbidden in production outputs')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escaped production root')
    return path


def write(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    atomic_publish_bytes(Path(path), data)
    return ref(path)


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def checked(record):
    path = Path(record['path'])
    if path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']:
        raise ValueError('Artifact bytes differ: '+str(path))
    return path


def digest_ids(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()
