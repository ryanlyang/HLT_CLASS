"""Independent, immutable scientific and execution artifacts."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash, load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref, write_json, safe


def artifact(kind, *, parents=None, **fields):
    if set(fields) & {'contract', 'schema_version', 'parents', 'content_hash', 'final_test_accessed'}:
        raise ValueError('Reserved context-fusion fields')
    return with_content_hash(dict(contract=f'JC2_CONTEXT_FUSION_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), final_test_accessed=False, **fields))


def validate(value, kind, *, parents=None):
    digest = validate_content_hash(value, expected_contract=f'JC2_CONTEXT_FUSION_{kind}/v1', expected_schema_version=1)
    if value.get('final_test_accessed') is not False:
        raise PermissionError('Final test is sealed')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Context-fusion parent lineage differs')
    return digest
