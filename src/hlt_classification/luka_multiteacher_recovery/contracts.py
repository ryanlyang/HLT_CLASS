"""New execution identity: replacements are never historical checkpoints."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_multiteacher_trial.contracts import ARMS, TEACHERS, COUNTS

AUTHORIZATION = 'AUTHORIZE_LUKA_MULTITEACHER_RECOVERY_V1'
REPLACEMENTS = ('LOGIT_D066', 'LOGIT_D033')


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_MULTITEACHER_RECOVERY_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_MULTITEACHER_RECOVERY_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Recovery parents differ')
    return digest


def tasks():
    return [dict(task_id='train_LOGIT_D066', dependencies=[]),
            dict(task_id='train_LOGIT_D033', dependencies=['train_LOGIT_D066']),
            dict(task_id='export_targets', dependencies=['train_LOGIT_D033'])] + [
                dict(task_id=a, dependencies=['export_targets']) for a in ARMS]
