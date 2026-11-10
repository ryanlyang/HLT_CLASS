"""Immutable eight-fit screen; reused ordinary U100-w5 control is not retrained."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe

BASE = '37ed91205764667588105948f95a1784964ac8bc'
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_OBJECTIVE_HYDRA_V1'
FITS = ('R8', 'R2', 'R4', 'R1', 'R7', 'R3', 'R5', 'R6')
TERMS = ('coordinate', 'class_span', 'fisher', 'discriminant', 'whitened')


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_OBJECTIVE_HYDRA_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_OBJECTIVE_HYDRA_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Evaluation disclosure differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Artifact lineage differs')
    return digest


def cell(task):
    settings = {'R1': ('coordinate', .5, False), 'R2': ('class_span', .5, False),
        'R3': ('class_span', .75, False), 'R4': ('fisher', .5, False),
        'R5': ('discriminant', .5, False), 'R6': ('whitened', .5, False),
        'R7': (None, 0., True), 'R8': ('class_span', .5, True)}
    if task not in settings:
        raise ValueError('Unknown objective task')
    term, fraction, protect = settings[task]
    return dict(term=term, fraction=fraction, protect_ce=protect, seed=0, width=128)


def term_names(task):
    cell(task)
    return {'ce', 'u100'}


def recipe():
    base = u100_recipe()
    return artifact('RECIPE', parents=dict(u100=base['content_hash']), base=base,
        cells={t: cell(t) for t in FITS}, order=list(FITS), epochs=30, batch_size=128,
        auxiliary_coefficient=5., predictor='original_128_GELU_128',
        covariance_shrinkage=.1, inverse_gain_cap=3., eigen_floor=1e-6, rank_rtol=1e-8,
        importance_clip=[.25, 4.], importance_final_bounds=[1/16, 16.], fisher_temperature=2.,
        calibration=dict(rows=2048, seed=2026111001, mode='fresh_eval_prediction_gradient_RMS',
                         scale_bounds=[.1, 10.], zero_gradient_scale=1.),
        gradient_projection='asymmetric_shared_encoder_CE_protection_FP64_dot', projection_epsilon=1e-20,
        diagnostic_epochs=[1, 3, 10, 20, 30], selection='original_native_full_validation_ties',
        geometry_fit_role='train_only', validation_targets='evaluation_only', control='reused_C0',
        deployment='native_HLT_mod_only', teacher_fits=0, new_student_fits=8)
