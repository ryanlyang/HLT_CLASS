"""Fresh six-fit 200--2600 GeV campaign, CE-only classifiers, LR1e-4/B256."""
import math
from hlt_classification.luka_new_balanced_hydra import contracts as donor
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash

PREFIX = 'LUKA_SIXCLASS_WIDE_B256_OSCAR'
FLAGS = dict(donor.FLAGS)
STAGES, COORDINATES, TEACHERS = donor.STAGES, donor.COORDINATES, donor.TEACHERS
CLASS_NAMES = ('QCD', 'X_bb', 'X_cc', 'X_ss', 'X_qq', 'X_gg')
EDGES = tuple(range(200, 2601, 100))
ELIGIBLE = DRAWS = 2488148
EPOCHS, BATCH, STEPS = 20, 256, 9720
EXPORT_BATCH = 128
BINS = len(EDGES)-1
COUNTS = dict(train=2488148, validation=497954, final_test=1227412)
UNUSED = 78270
OUTSIDE = dict(train=424322, validation=68078, final_test=205454, unused=103207)
PARENT_HASH, ROLE_POPULATION_HASH = donor.PARENT_HASH, donor.POPULATION_HASH
POPULATION_HASH = '57eb7febd6ee6ae2c9729bdcc5f58a4f34dea232d7b1eaf12b7901ca405025d1'
SAMPLER_SEED, NEGATIVE_SEED = donor.donor.SAMPLER_SEED, donor.donor.NEGATIVE_SEED
BLOCK = donor.BLOCK
coefficient = donor.coefficient
MAIN_TEACHERS = {}
GPUS = {'l40s': ('l40s', 'NVIDIA L40S')}
SITE = dict(cluster='slurmctld', partition='gpu', account='default', qos='norm-gpu')
AUTH = 'AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1'
RESULT_GIB = 18
TRANSFER_HASH = '7034f2153e585068000185ce4c1e7fb3e61fe96e5bbfeb5e650608a150071ae4'
DATASET = '/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1'


def artifact(kind, *, parents=None, version=1, **fields):
    if set(fields).intersection(('contract', 'schema_version', 'parents', *FLAGS)):
        raise ValueError('Reserved artifact fields')
    if version!=1 and (kind!='SPEC' or version!=2):raise ValueError('Unsupported artifact version')
    return with_content_hash(dict(contract=f'{PREFIX}_{kind}/v{version}', schema_version=version,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    version=value.get('schema_version') if kind=='SPEC' else 1
    if version not in (1,2):raise ValueError('Unsupported spec version')
    validate_content_hash(value, expected_contract=f'{PREFIX}_{kind}/v{version}', expected_schema_version=version)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise ValueError('Scientific scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent bindings differ')
    return value


def main_policy(stage):
    if stage not in STAGES:
        raise PermissionError('Unregistered stage')
    teacher = None
    return dict(teacher=teacher, ce_weight=1., kd_weight=0.,
                temperature=2., reduction='T2_forward_KL_class_sum_batch_mean')


def node(stage):
    return dict(donor.node(stage), main_head=main_policy(stage), class_names=list(CLASS_NAMES),
                native_outputs=6, auxiliary_logit_outputs=6)


def population_policy():
    return dict(role_population_sha256=ROLE_POPULATION_HASH, preserve_all_extended_roles=True,
                additions=dict(train=0, validation=0, final_test=0), counts=COUNTS,
                class_names=list(CLASS_NAMES), original_labels=list(range(6)),
                pt_edges_GeV=list(EDGES), boundary='[200,2600)',
                excluded_raw_label=206, final_test_sealed=True,
                outside_validation='all_extended_validation_six_classes_outside_range',
                unused_in_range=UNUSED)


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= EPOCHS:
        raise ValueError('Twenty-epoch schedule position must be in (0,20]')
    if position <= 2:
        return 1e-4 * position / 2
    return 5e-6 + (1e-4-5e-6)*.5*(1+math.cos(math.pi*(position-2)/18))


def recipe():
    old = donor.recipe()
    fields = {k:v for k,v in old.items() if k not in ('content_hash','contract','schema_version','parents',*FLAGS)}
    fields.update(base_recipe_sha256=old['content_hash'],
                  eligible_train_rows=ELIGIBLE, draws_per_epoch=DRAWS, updates_per_epoch=STEPS,
                  batch_size=BATCH, last_batch_size=DRAWS % BATCH, peak_lr=1e-4, floor_lr=5e-6,
                  class_names=list(CLASS_NAMES), pt_edges_GeV=list(EDGES), population_policy=population_policy(),
                  logit_head=[128,6], native_outputs=6,
                  selection_population='six_class_natural_in_range_validation',
                  outside_validation='every_epoch_full_macro_AUC_and_Hbb_diagnostic_only',
                  full_validation_rows=COUNTS['validation']+OUTSIDE['validation'],
                  full_hbb_selection='max_binary_auc_then_earliest_epoch_diagnostic_only',
                  final_test_primary_rows=COUNTS['final_test'], final_test_outside_rows=OUTSIDE['final_test'],
                  negative_rng='sixclass_identity_epoch_occurrence_component_stream_v1',
                  negative_classes=6, hard_other_classes=5, hard_capacity_per_class=820,
                  main_head={s:main_policy(s) for s in STAGES}, main_head_supervision='CE_only_all_six_fits',
                  export_batch_size=EXPORT_BATCH, teacher_source='fresh_same_campaign_only',
                  exported_banks=list(STAGES[:-1]))
    return artifact('RECIPE', **fields)


def tasks():
    return donor.tasks()
