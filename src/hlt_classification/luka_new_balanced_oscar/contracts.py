"""New-data Oscar identity; exact Hydra population with direct KD at every rung."""
from hlt_classification.luka_new_balanced_hydra import contracts as donor
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash

PREFIX = 'LUKA_NEW_BALANCED_OSCAR'
FLAGS = dict(donor.FLAGS)
STAGES, COORDINATES, TEACHERS = donor.STAGES, donor.COORDINATES, donor.TEACHERS
CLASS_NAMES, EDGES = donor.CLASS_NAMES, donor.EDGES
ELIGIBLE, DRAWS, EPOCHS, BATCH, STEPS = donor.ELIGIBLE, donor.DRAWS, donor.EPOCHS, donor.BATCH, donor.STEPS
COUNTS, ADDITIONS, OUTSIDE = dict(donor.COUNTS), dict(donor.ADDITIONS), dict(donor.OUTSIDE)
PARENT_HASH, POPULATION_HASH = donor.PARENT_HASH, donor.POPULATION_HASH
BLOCK = donor.BLOCK
population_policy, coefficient = donor.population_policy, donor.coefficient
MAIN_TEACHERS = dict(UE100='OFFLINE', DE050='UE100', DE000='DE050')
GPUS = {'l40s': ('l40s', 'NVIDIA L40S')}
SITE = dict(cluster='slurmctld', partition='gpu', account='default', qos='norm-gpu')
AUTH = 'AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1'
RESULT_GIB = 14
TRANSFER_HASH = '7034f2153e585068000185ce4c1e7fb3e61fe96e5bbfeb5e650608a150071ae4'
DATASET = '/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1'


def artifact(kind, *, parents=None, **fields):
    if set(fields).intersection(('contract', 'schema_version', 'parents', *FLAGS)):
        raise ValueError('Reserved artifact fields')
    return with_content_hash(dict(contract=f'{PREFIX}_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'{PREFIX}_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise ValueError('Scientific scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent bindings differ')
    return value


def main_policy(stage):
    if stage not in STAGES:
        raise PermissionError('Unregistered stage')
    teacher = MAIN_TEACHERS.get(stage)
    return dict(teacher=teacher, ce_weight=.25 if teacher else 1., kd_weight=.75 if teacher else 0.,
                temperature=2., reduction='T2_forward_KL_class_sum_batch_mean')


def node(stage):
    return dict(donor.node(stage), main_head=main_policy(stage))


def recipe():
    old = donor.recipe()
    fields = {k:v for k,v in old.items() if k not in ('content_hash','contract','schema_version','parents',*FLAGS)}
    fields.pop('ce_weight')
    fields.update(base_recipe_sha256=old['content_hash'],
                  main_head={s:main_policy(s) for s in STAGES},
                  main_head_supervision='CE_only_baselines_CE_plus_predecessor_KD_all_students')
    return artifact('RECIPE', **fields)


def tasks():
    # The predecessor is also an auxiliary teacher, but require it explicitly.
    result = donor.tasks()
    for stage, teacher in MAIN_TEACHERS.items():
        if 'train_'+teacher not in result['train_'+stage]:
            raise ValueError('Missing main KD dependency')
    return result
