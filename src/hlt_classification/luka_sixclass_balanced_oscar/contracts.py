"""Six-class 600--2600 GeV identity; protected historical split roles."""
from hlt_classification.luka_new_balanced_hydra import contracts as donor
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash

PREFIX = 'LUKA_SIXCLASS_BALANCED_OSCAR'
FLAGS = dict(donor.FLAGS)
STAGES, COORDINATES, TEACHERS = donor.STAGES, donor.COORDINATES, donor.TEACHERS
CLASS_NAMES = ('QCD', 'X_bb', 'X_cc', 'X_ss', 'X_qq', 'X_gg')
EDGES = tuple(range(600, 2601, 100))
ELIGIBLE = DRAWS = 1658937
EPOCHS, BATCH, STEPS = 20, 128, 12961
COUNTS = dict(train=1658937, validation=328589, final_test=822313)
UNUSED = 78034
OUTSIDE = dict(train=1253533, validation=237443, final_test=610553, unused=103443)
PARENT_HASH, ROLE_POPULATION_HASH = donor.PARENT_HASH, donor.POPULATION_HASH
POPULATION_HASH = '5092ce94a5e32fac10bdbd5b91b1becddc3fe9123affc41a93a0b312db2ea616'
SAMPLER_SEED, NEGATIVE_SEED = donor.donor.SAMPLER_SEED, donor.donor.NEGATIVE_SEED
BLOCK = donor.BLOCK
coefficient = donor.coefficient
MAIN_TEACHERS = dict(UE100='OFFLINE', DE050='UE100', DE000='DE050')
GPUS = {'l40s': ('l40s', 'NVIDIA L40S')}
SITE = dict(cluster='slurmctld', partition='gpu', account='default', qos='norm-gpu')
AUTH = 'AUTHORIZE_SIXCLASS_BALANCED_OSCAR_V1'
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
    return dict(donor.node(stage), main_head=main_policy(stage), class_names=list(CLASS_NAMES),
                native_outputs=6, auxiliary_logit_outputs=6)


def population_policy():
    return dict(role_population_sha256=ROLE_POPULATION_HASH, preserve_all_extended_roles=True,
                additions=dict(train=0, validation=0, final_test=0), counts=COUNTS,
                class_names=list(CLASS_NAMES), original_labels=list(range(6)),
                pt_edges_GeV=list(EDGES), boundary='[600,2600)',
                excluded_raw_label=206, final_test_sealed=True,
                outside_validation='all_extended_validation_six_classes_outside_range',
                unused_in_range=UNUSED)


def recipe():
    old = donor.recipe()
    fields = {k:v for k,v in old.items() if k not in ('content_hash','contract','schema_version','parents',*FLAGS)}
    fields.pop('ce_weight')
    fields.update(base_recipe_sha256=old['content_hash'],
                  eligible_train_rows=ELIGIBLE, draws_per_epoch=DRAWS, updates_per_epoch=STEPS,
                  last_batch_size=DRAWS % BATCH, class_names=list(CLASS_NAMES),
                  pt_edges_GeV=list(EDGES), population_policy=population_policy(),
                  logit_head=[128,6], native_outputs=6,
                  selection_population='six_class_in_range_328589_validation',
                  outside_validation='extended_six_class_outside_validation_selected_checkpoint_only',
                  final_test_primary_rows=COUNTS['final_test'], final_test_outside_rows=OUTSIDE['final_test'],
                  negative_rng='sixclass_identity_epoch_occurrence_component_stream_v1',
                  negative_classes=6, hard_other_classes=5, hard_capacity_per_class=820,
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
