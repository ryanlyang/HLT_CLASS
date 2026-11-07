"""Versioned teacher-sweep scientific registration."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe

BASE = '0ae4f28fbc25df4aef0ddc617a16555f98b02591'
AUTHORIZATION = 'AUTHORIZE_LUKA_TEACHER_SWEEP_HYDRA_V1'
TEACHERS = tuple(f'T{i:02}' for i in range(12))
STUDENTS = {f'S{i:02}': f'T{i:02}' for i in range(12)}
PAIR_ORDER = (1, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 2)
FITS = tuple(t for i in PAIR_ORDER for t in (f'T{i:02}', f'S{i:02}'))
NAMES = ('ce', 'supcon5', 'supcon20', 'native_supcon5', 'supcon_instance5',
         'vicreg', 'barlow5', 'supcon_variance', 'masked_set5', 'summaries5',
         'thinning_consistency5', 'noisy_bottleneck')


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_TEACHER_SWEEP_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_TEACHER_SWEEP_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Population disclosure differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return digest


def recipe():
    return artifact('RECIPE', parents=dict(u100=u100_recipe()['content_hash']),
        base=u100_recipe(), epochs=30, batch_size=128, teachers=dict(zip(TEACHERS, NAMES)),
        students=STUDENTS, order=list(FITS), temperature=.1, projection_seed=84221,
        instance_seed=84222, offline_predictor_seed=2026110601, augmentation_seed=2026110612,
        teacher_seed_offset=200000, student_weights=dict(u100=5., offline=5.),
        supcon_weights=dict(T01=5., T02=20., T03=5., T04=5., T07=5.),
        instance_weight=5., vicreg_weights=[25., 25., 1.], barlow_weight=5., barlow_offdiag=.0051,
        variance_weight=25., covariance_weight=1., thinning=.3, masking=.4,
        reconstruction_weight=5., summary_weight=5., consistency_weight=5., noise_scale=.5,
        target_width=128, normalization='TRAIN_FP64_population_std_floor1e-3',
        export='clean_native_preclassifier_not_projection', selection='native_full_validation_original_ties',
        reconstruction='TRAIN_standardized_native17_plus_p4_Hungarian_SmoothL1_mean_balanced_presence_count',
        decoder='pooled_h_plus_learned_query_MLP256_GELU_MLP256_GELU',
        summaries='physical36_v1', moments_role='train', variance_epsilon=1e-4,
        covariance='sample_offdiag_square_sum_div128', barlow='sum_diag_error_plus_0p0051_sum_offdiag',
        masking_semantics='token_removal_original_full_jet_normalization',
        initial_native_unchanged=True, train_targets_only=True, reference_retrained=False,
        native_hlt_inference=True, student_logit_kd=False)


def term_names(task):
    if task in STUDENTS:
        return {'ce', 'u100', 'offline'}
    if task not in TEACHERS:
        raise ValueError('Unknown teacher/student')
    extra = dict(T00=(), T01=('supcon',), T02=('supcon',), T03=('supcon',),
        T04=('supcon', 'instance'), T05=('invariance', 'variance', 'covariance'),
        T06=('barlow',), T07=('supcon', 'variance', 'covariance'),
        T08=('reconstruction',), T09=('summary',), T10=('consistency',), T11=())
    return {'ce', *extra[task]}
