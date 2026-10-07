"""Separate namespace and explicit materialization/evaluation boundaries."""
from hlt_classification.data.cache_contracts import validate_content_hash, with_content_hash
from hlt_classification.literature_context_production.contracts import (
    safe, write, ref, checked, load_json, sha256_file, canonical_sha256, GIB,
)


def artifact(schema_kind, *, parents=None, test=False, **fields):
    reserved = {'contract', 'schema_version', 'content_hash', 'parents',
                'final_test_accessed', 'final_test_materialized', 'final_test_evaluated'}
    if reserved.intersection(fields) or type(test) is not bool:
        raise ValueError('Reserved context-v2 fields')
    return with_content_hash(dict(contract=f'JC2_CONTEXT_V2_{schema_kind}/v1', schema_version=1,
        parents=dict(parents or {}), final_test_accessed=test,
        final_test_materialized=test, final_test_evaluated=False, **fields))


def validate(value, kind, *, parents=None, test=False):
    digest = validate_content_hash(value, expected_contract=f'JC2_CONTEXT_V2_{kind}/v1',
                                   expected_schema_version=1)
    if (value['final_test_accessed'] is not test or value['final_test_materialized'] is not test
            or value['final_test_evaluated'] is not False
            or parents is not None and value['parents'] != parents):
        raise ValueError('Context-v2 scope/parents differ')
    return digest


def require(ok, message):
    if not ok:
        raise ValueError(message)
