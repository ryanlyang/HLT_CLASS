"""New semantic identities; original representations and executions remain immutable."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe
from hlt_classification.cms_proxy_ladder.campaign import paired_seed, coordinate as view_coordinate

STAGES = {'UE000': 'U000', 'UE100': 'U100', 'DE066': 'D066', 'DE033': 'D033', 'DE000': 'D000'}
TEACHERS = {'OFFLINE': 'OFFLINE', 'M0HLT': 'D000', **dict(list(STAGES.items())[:-1])}
EMBEDDING_TEACHERS = {
    'UE000': ['OFFLINE', 'M0HLT'],
    'UE100': ['OFFLINE', 'M0HLT'],
    'DE066': ['OFFLINE', 'UE100', 'M0HLT'],
    'DE033': ['OFFLINE', 'UE100', 'DE066', 'M0HLT'],
    'DE000': ['OFFLINE', 'UE100', 'DE066', 'DE033'],
}
EMBEDDING_EXPORTS = ('M0HLT', 'UE100', 'DE066', 'DE033')
KD_EXPORTS = ('OFFLINE', 'UE000', 'UE100', 'DE066', 'DE033')
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_HYBRID_EMBEDDING_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_HYBRID_EMBEDDING_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value,
        expected_contract=f'LUKA_FULLSTAT_HYBRID_EMBEDDING_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Sealed FullStat role flags differ')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return result


def teachers(stage):
    if stage not in STAGES:
        raise ValueError('Unregistered hybrid stage')
    return list(EMBEDDING_TEACHERS[stage])


def ancestors(stage):
    if stage not in STAGES:
        raise ValueError('Unregistered hybrid stage')
    return list(STAGES)[:list(STAGES).index(stage)]


def kd_teacher(stage):
    previous = ancestors(stage)
    return previous[-1] if previous else 'OFFLINE'


def kd_recipe(teacher):
    if teacher not in KD_EXPORTS:
        raise ValueError('Unregistered logit teacher')
    return artifact('KD_RECIPE', teacher=teacher, coordinate=TEACHERS[teacher],
        role='train', temperature=2., class_count=11, export_batch=128,
        native_logits=True, cuda_forward='BF16', softmax='FP32_logits_divided_by_T',
        auxiliary_predictions_are_not_targets=True)


def target_recipe(teacher):
    if teacher not in ('OFFLINE', *EMBEDDING_EXPORTS):
        raise ValueError('Unregistered hybrid teacher')
    return artifact('TARGET_RECIPE', teacher=teacher, coordinate=TEACHERS[teacher],
        role='train', width=128, tap='native_mod.fc_input', teacher_eval=True,
        teacher_fp32=True, export_batch=128, minimum_scale=1e-3,
        normalization='TRAIN_population_moments_FP64_targets_FP32')


def recipe(stage):
    order = teachers(stage)
    seed = auxiliary_recipe()['predictor_seed']
    seeds = dict(OFFLINE=seed+1, UE100=seed, DE066=seed+2, DE033=seed+3, M0HLT=seed+4)
    return artifact('RECIPE', parents={**{t: target_recipe(t)['content_hash'] for t in order},
        'kd': kd_recipe(kd_teacher(stage))['content_hash']},
        stage=stage, coordinate=STAGES[stage], base=base_recipe(128),
        base_usage='matched_optimizer_schedule_and_75KD_25CE', epochs=20, batch_size=128,
        ce_coefficient=.25, auxiliary_total=5., weights={t: 5./len(order) for t in order},
        kd_coefficient=.75, temperature=2., kd_teacher=kd_teacher(stage),
        kd_loss='T_squared_forward_KL_teacher_to_student_batchmean',
        teacher_order=order, predictors={t: [128, 'GELU', 128] for t in order},
        predictor_seeds={t: seeds[t] for t in order},
        auxiliary_loss='per_teacher_mean_SmoothL1_beta1',
        inference='native_HLT_without_predictors' if stage == 'DE000' else 'richer_input_oracle',
        attention=False, relational=False, warm_start=False, teacher_embedding_average=False)


def node(base, stage):
    if (base['coordinate'] != 'D000' or base['teacher'] is not None
            or base['context_coordinate'] is not None or not base['deployable']):
        raise ValueError('Expected original native M0HLT base')
    coordinate = STAGES[stage]
    u, f = view_coordinate(coordinate)
    return dict(base, node_id=stage, coordinate=coordinate, teacher=kd_teacher(stage), branch='hybrid_embedding',
        context_coordinate=None, deployable=stage == 'DE000', auxiliary_weight=5.,
        embedding_weights=recipe(stage)['weights'],
        u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
        initialization_seed=paired_seed(coordinate, 'initialization'),
        sampler_seed=paired_seed(coordinate, 'sampler'))


def tasks():
    return [dict(task_id='train_'+stage, dependencies=['train_'+t for t in ancestors(stage)])
            for stage in STAGES]
