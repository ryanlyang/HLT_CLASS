"""Immutable twelve-fit selective-information experiment."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe

BASE = '6fe7e3499767f0a2216586d965efe39a05c24bf7'
AUTHORIZATION = 'AUTHORIZE_LUKA_SELECTIVE_TEACHER_HYDRA_V1'
TEACHERS = ('K', 'I', 'T', 'IT')
# Insertion order is the authorized submission order, not a scheduler guarantee.
STUDENTS = dict(S6=('K', 'I', 'T', 'IT', 'OFFLINE'), S5=('K', 'I', 'T', 'IT'),
                S4=('K', 'I', 'T'), S3=('I', 'T'), S2IT=('IT',), S2T=('T',),
                S2I=('I',), S2K=('K',))
FITS = (*TEACHERS, *STUDENTS)
GROUPS = dict(K=(0, 1, 2, 3, 4, 15, 16), I=(5, 6, 7, 8, 9, 10), T=(11, 12, 13, 14))


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_SELECTIVE_TEACHER_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_SELECTIVE_TEACHER_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Population disclosure differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return digest


def view_recipe():
    return artifact('VIEWS', groups={k: list(v) for k, v in GROUPS.items()},
        teachers=list(TEACHERS), support='D000_original_slots_order_and_unmatched_fallback',
        mapping='original_authenticated_algorithmic_HLT_to_OFFLINE_not_truth',
        K='all_p4_and_own_view_kinematic_features_from_U100',
        I='charge_and_categories_only_from_U100', T='four_tracking_channels_only_from_U100',
        IT='I_and_T_from_U100', remaining='bitwise_D000', vectors='U100_only_for_K',
        validity='original_source_group_zero_invalid_convention_no_added_bits',
        identity_tracking_conflicts='controlled_hybrid_no_recoupling_between_groups',
        frontend='original_luka_fullsim_literature_context_native17',
        no_offline_only_particles=True, no_assignment_features=True)


def weights(task):
    if task not in STUDENTS:
        raise ValueError('Unknown student')
    return dict(u100=5., **{k.lower(): 5./len(STUDENTS[task]) for k in STUDENTS[task]})


def term_names(task):
    if task in TEACHERS: return {'ce'}
    return {'ce', *weights(task)}


def recipe():
    return artifact('RECIPE', parents=dict(u100=u100_recipe()['content_hash'], views=view_recipe()['content_hash']),
        base=u100_recipe(), epochs=30, batch_size=128, teachers=list(TEACHERS),
        students={k: list(v) for k, v in STUDENTS.items()}, order=list(FITS),
        student_weights={k: weights(k) for k in STUDENTS}, teacher_objective='CE_only',
        teacher_seed_offset=200000, predictor_seeds=dict(k=2026110701, i=2026110702, t=2026110703,
            it=2026110704, offline=2026110601), target_width=128, minimum_scale=.001,
        normalization='TRAIN_FP64_population_std_floor1e-3', export='clean_native_preclassifier_FP32',
        embedding_loss='SmoothL1_beta1_mean_per_head', selection='native_full_validation_original_ties',
        reference_retrained=False, native_hlt_inference=True, student_logit_kd=False,
        train_targets_only=True, initial_native_unchanged=True)
