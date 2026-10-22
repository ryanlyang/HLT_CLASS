"""Independent scientific identities; never reinterpret earlier banks or fits."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe

TEACHERS = {'OFFLINE': 'OFFLINE', 'COARSE_U100': 'U100',
            'COARSE_D066': 'D066', 'COARSE_D033': 'D033'}
WEIGHT = 1.25
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_FOUR_TEACHER_EMBEDDING_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_MULTITEACHER_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value,
        expected_contract=f'LUKA_FULLSTAT_MULTITEACHER_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Sealed FullStat roles differ')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return result


def target_recipe(teacher):
    if teacher not in TEACHERS:
        raise ValueError('Unregistered embedding teacher')
    return artifact('TARGET_RECIPE', teacher=teacher, coordinate=TEACHERS[teacher],
        role='train', width=128, tap='native_mod.fc_input', teacher_eval=True,
        teacher_fp32=True, export_batch=128, minimum_scale=1e-3,
        normalization='TRAIN_population_moments_FP64_targets_FP32')


def recipe():
    original_seed = auxiliary_recipe()['predictor_seed']
    # The U100 head retains the single-teacher predictor seed. Other heads use
    # independent CPU-only initialization streams without advancing native RNG.
    seeds = dict(COARSE_U100=original_seed, OFFLINE=original_seed+1,
                 COARSE_D066=original_seed+2, COARSE_D033=original_seed+3)
    return artifact('RECIPE', parents={t: target_recipe(t)['content_hash'] for t in TEACHERS},
        base=base_recipe(128), base_usage='optimizer_and_schedule_only_not_KD_loss',
        epochs=20, batch_size=128, ce_coefficient=1., auxiliary_total=5.,
        weights={t: WEIGHT for t in TEACHERS}, kd_coefficient=0.,
        teacher_order=list(TEACHERS), predictors={t: [128, 'GELU', 128] for t in TEACHERS},
        predictor_seeds=seeds, auxiliary_loss='per_teacher_mean_SmoothL1_beta1',
        inference='native_HLT_without_predictors', attention=False, relational=False,
        warm_start=False, teacher_embedding_average=False)


def node(base):
    if (base['coordinate'] != 'D000' or base['teacher'] is not None
            or base['context_coordinate'] is not None or not base['deployable']):
        raise ValueError('Expected native HLT-only from-scratch base')
    result = dict(base, node_id='MULTI4_EMBED_W5', branch='multi4',
                  auxiliary_weight=5., embedding_weights={t: WEIGHT for t in TEACHERS})
    return result


def tasks():
    return [dict(task_id='setup', dependencies=[]),
            dict(task_id='train_multi4', dependencies=['setup'])]
