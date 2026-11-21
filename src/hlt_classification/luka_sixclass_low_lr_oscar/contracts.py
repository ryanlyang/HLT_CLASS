"""Versioned three-fit experiment; original population is reused unchanged."""
import math
from hlt_classification.luka_sixclass_balanced_oscar import contracts as original
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash

PREFIX = 'LUKA_SIXCLASS_LOW_LR_OSCAR'
FLAGS = dict(original.FLAGS)
STAGES = ('OFFLINE', 'DE000', 'M0HLT')
EXTERNAL = ('CONCAT', 'UE100', 'DE050')
COORDINATES = {s: original.COORDINATES[s] for s in STAGES}
TEACHERS = {s: original.TEACHERS[s] for s in STAGES}
MAIN_TEACHERS = {}
CLASS_NAMES, EDGES = original.CLASS_NAMES, original.EDGES
ELIGIBLE, DRAWS, EPOCHS, BATCH, STEPS = original.ELIGIBLE, original.DRAWS, 20, 128, original.STEPS
COUNTS, OUTSIDE, UNUSED = dict(original.COUNTS), dict(original.OUTSIDE), original.UNUSED
PARENT_HASH, ROLE_POPULATION_HASH, POPULATION_HASH = original.PARENT_HASH, original.ROLE_POPULATION_HASH, original.POPULATION_HASH
SAMPLER_SEED, NEGATIVE_SEED, BLOCK = original.SAMPLER_SEED, original.NEGATIVE_SEED, original.BLOCK
GPUS, SITE, TRANSFER_HASH, DATASET = dict(original.GPUS), dict(original.SITE), original.TRANSFER_HASH, original.DATASET
AUTH = 'AUTHORIZE_SIXCLASS_LOW_LR_OSCAR_V1'
RESULT_GIB = 8
PARENT_SPEC_HASH = 'a68c689360012272a1703f67e6da160a617a2d7937450dd5d83ac0a473a5c7a4'
PARENT_COMMIT = '8a25f12653d0bcfefe170273d3f3e0748b4b508f'
PARENT_SPEC = '/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_balanced_oscar_8a25f126_r1/campaign_spec.json'
coefficient, population_policy = original.coefficient, original.population_policy


def artifact(kind, *, parents=None, **fields):
    if set(fields).intersection(('contract', 'schema_version', 'parents', *FLAGS)):
        raise ValueError('Reserved artifact fields')
    return with_content_hash(dict(contract=f'{PREFIX}_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'{PREFIX}_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise ValueError('Scientific scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent bindings differ')
    return value


def main_policy(stage):
    if stage not in STAGES:
        raise PermissionError('Unregistered mini-campaign stage')
    return dict(teacher=None, ce_weight=1., kd_weight=0., temperature=2.,
                reduction='T2_forward_KL_class_sum_batch_mean')


def node(stage):
    return dict(original.node(stage), main_head=main_policy(stage))


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= 20:
        raise ValueError('Twenty-epoch schedule position must be in (0,20]')
    if position <= 2:
        return 1e-4 * position / 2
    return 5e-6 + (1e-4-5e-6)*.5*(1+math.cos(math.pi*(position-2)/18))


def recipe():
    old = original.recipe()
    fields = {k:v for k,v in old.items() if k not in ('content_hash','contract','schema_version','parents',*FLAGS)}
    fields.update(base_recipe_sha256=old['content_hash'], peak_lr=1e-4, floor_lr=5e-6,
                  teacher_order={s:list(TEACHERS[s]) for s in STAGES}, coordinates=dict(COORDINATES),
                  main_head={s:main_policy(s) for s in STAGES}, main_head_supervision='CE_only_all_three_fits',
                  outside_validation='every_epoch_full_Hbb_diagnostic_primary_selection_unchanged',
                  full_validation_rows=566032, full_hbb_selection='max_binary_auc_then_earliest_epoch',
                  offline_target_source='new_primary_selected_OFFLINE_all_three_target_types',
                  external_teachers=list(EXTERNAL), exported_banks=['OFFLINE'])
    return artifact('RECIPE', **fields)


def tasks():
    return dict(train_OFFLINE=[], train_DE000=['train_OFFLINE'], train_M0HLT=['train_DE000'])
