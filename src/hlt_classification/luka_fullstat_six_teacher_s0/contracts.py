"""Distinct experiment identity; only the subclass objective coefficient changes."""
from copy import deepcopy
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_six_teacher import contracts as previous

TASK = previous.TASK
TASKS = {TASK: []}
COUNTS = previous.COUNTS
FLAGS = previous.previous.FLAGS
TEACHERS, PRIMARY, LADDER = previous.TEACHERS, previous.PRIMARY, previous.LADDER
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SIX_TEACHER_S0_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SIX_TEACHER_S0_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_SIX_TEACHER_S0_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_SIX_TEACHER_S0_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Scientific scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return value


def recipe():
    old = previous.recipe()
    fields = deepcopy({k:v for k,v in old.items() if k not in
                       ('contract','schema_version','parents','content_hash',*FLAGS)})
    old_subclass = fields['subclass']
    subclass_fields = {k:v for k,v in old_subclass.items() if k not in
                       ('contract','schema_version','parents','content_hash',*FLAGS)}
    subclass_fields['coefficient'] = 0.
    fields['subclass'] = artifact('SUBCLASS_OBJECTIVE',
        parents=dict(previous=old_subclass['content_hash']), **subclass_fields)
    fields['historical_control'] = 'completed_six_teacher_subclass05'
    return artifact('RECIPE', parents=dict(previous=old['content_hash']), **fields)


def node(parent):
    old = parent['nodes'][TASK]
    if (parent['recipe'] != previous.recipe() or old['coordinate'] != 'D000'
            or old['context_coordinate'] is not None or old['embedding_targets'] != list(TEACHERS)
            or old['logit_targets'] != list(TEACHERS) or old['total_embedding_weight'] != 15.
            or old['total_logit_weight'] != 1. or old['matching_coefficient'] != .3):
        raise ValueError('Expected original six-teacher subclass0.5 HLT student')
    return dict(old, branch='FULLSTAT_SIX_TEACHER_S0')


coefficient = previous.coefficient
auxiliary_ramp = previous.auxiliary_ramp
loss_names = previous.loss_names


def subclass_coefficient(update, steps):
    previous.auxiliary_ramp(update, steps)  # Keep the original cursor validation.
    return 0.
