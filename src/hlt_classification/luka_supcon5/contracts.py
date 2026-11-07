"""Immutable scientific meaning; teacher strength is distinct from student weight."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe

AUTHORIZATION = 'AUTHORIZE_LUKA_SUPCON5_DEBUG_V1'
TEACHERS = ('teacher_supcon5',)
STUDENTS = {'student_supcon5': 'teacher_supcon5'}
FITS = (*TEACHERS, *STUDENTS)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_SUPCON5_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_SUPCON5_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory population disclosure differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return digest


def recipe():
    base = u100_recipe()
    return artifact('RECIPE', parents=dict(u100=base['content_hash']), base=base,
        epochs=30, batch_size=128, teacher_coefficients=dict(teacher_supcon5=5.),
        ce_weight=1., student_weights=dict(U100=5., OFFLINE=5.), temperature=.1,
        projection_seed=84221, offline_predictor_seed=2026110601,
        projection='Linear128_GELU_Linear128_then_L2', positives='same_class_exclude_self',
        reduction='mean_positive_log_probability_then_mean_eligible_anchor',
        no_positive='differentiable_zero', teacher_export='native_preclassifier_128_not_projection',
        normalization='TRAIN_FP64_mean_population_std_floor1e-3',
        selected='native_full_validation_AUC_with_original_ties', fresh_initialization=True,
        teacher_seed_offset=200000, student_logit_kd=False, student_supcon=False,
        native_HLT_inference=True, train_targets_only=True, reference_retrained=False)
