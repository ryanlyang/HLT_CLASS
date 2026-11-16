"""One new scientific identity; the original balanced recipe remains immutable."""
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.luka_fullstat_balanced_ladder import contracts as base

PREFIX = 'LUKA_FULLSTAT_BALANCED_MAIN_KD'
FLAGS = dict(base.FLAGS)
COUNTS = dict(base.COUNTS)
STAGES = ('DE000',)
TEACHERS = {'DE000': base.TEACHERS['DE000']}
REFERENCES = ('OFFLINE', 'M0HLT', 'CONCAT', 'UE100', 'DE050')
COORDINATES = {'DE000': 'D000'}
ELIGIBLE, DRAWS, EPOCHS, BATCH, STEPS = base.ELIGIBLE, base.DRAWS, base.EPOCHS, base.BATCH, base.STEPS
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_BALANCED_MAIN_KD_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_BALANCED_MAIN_KD_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    if set(fields).intersection(('contract', 'schema_version', 'parents', *FLAGS)):
        raise ValueError('Reserved artifact fields')
    return meta.with_content_hash(dict(contract=f'{PREFIX}_{kind}/v1', schema_version=1,
                                      parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    meta.validate_content_hash(value, expected_contract=f'{PREFIX}_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Scientific scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent hashes differ')
    return value


def recipe():
    old = base.recipe()
    fields = {k:v for k,v in old.items() if k not in ('content_hash','contract','schema_version','parents',*FLAGS)}
    fields.update(base_recipe_sha256=old['content_hash'], ce_weight=.25,
                  main_kd_weight=.75, main_kd_teacher='DE050',
                  main_head_supervision='true_CE_plus_frozen_DE050_T2_forward_KL',
                  teacher_order={'DE000':list(TEACHERS['DE000'])}, coordinates=COORDINATES)
    return artifact('RECIPE', **fields)


def node(stage):
    if stage != 'DE000':
        raise PermissionError('Only one fresh HLT-only DE000 is authorized')
    return dict(base.node(stage), main_logit_teacher='DE050')


def coefficient(stage, update, steps=STEPS):
    if stage != 'DE000':
        raise PermissionError('Only DE000 is registered')
    return base.coefficient(stage, update, steps)


def tasks():
    return {'train_DE000': []}
