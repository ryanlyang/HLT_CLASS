"""New execution identities; original scientific recipe and bank format unchanged."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_multiview.contracts import FLAGS

PARENT_COMMIT = 'a1eb1a746875a2697216d45a7c8ea4b6acdabc8a'
BASELINE = '4b9630fa00f137c4e48b749eb86714719b9d26cf'
JOBS = {'teacher_FUSION': '8220911', 'train_multi3': '8220912'}
TASKS = {'export_FUSION': [], 'train_multi3': ['export_FUSION']}
MINUTES = {'export_FUSION': 360, 'train_multi3': 1440}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_MULTIVIEW_EXPORT_RECOVERY_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_MULTIVIEW_EXPORT_AUTOSUBMIT_V1'
CLAIM = 'fusion_export_recovery_claim_v1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_MULTIVIEW_EXPORT_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_MULTIVIEW_EXPORT_{kind}/v1')
    if value.get('schema_version') != 1 or any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Sealed-role flags differ')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Recovery parents differ')
    return value
