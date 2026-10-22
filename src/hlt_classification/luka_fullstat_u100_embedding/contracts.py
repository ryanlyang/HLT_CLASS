"""Separate identities for U100-target weight5; old evidence is not relabelled."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe

WEIGHTS = {'u100_w5': 5.}
REFERENCES = ('M0HLT', 'OFFLINE', 'COARSE_U100')
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_U100_EMBEDDING_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_U100_EMBEDDING_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value,
        expected_contract=f'LUKA_FULLSTAT_U100_EMBEDDING_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise PermissionError('Sealed FullStat evaluation scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('U100 embedding parent hashes differ')
    return result


def target_recipe():
    return artifact('TARGET_RECIPE', teacher='COARSE_U100', coordinate='U100',
        role='train', width=128, tap='native_mod.fc_input', teacher_eval=True,
        teacher_fp32=True, export_batch=128, minimum_scale=1e-3,
        normalization='TRAIN_population_moments_FP64_targets_FP32')


def recipe(weight=5.):
    if type(weight) not in (int, float) or weight != 5.:
        raise ValueError('Only U100 embedding weight5 is registered')
    return artifact('RECIPE', parents=dict(target=target_recipe()['content_hash']),
        base=base_recipe(128), base_usage='optimizer_and_schedule_only_not_KD_loss',
        epochs=20, batch_size=128, ce_coefficient=1., auxiliary_coefficient=5.,
        kd_coefficient=0., teacher=None, embedding_teacher='COARSE_U100',
        width=128, predictor=[128, 'GELU', 128],
        predictor_seed=auxiliary_recipe()['predictor_seed'], auxiliary_loss='mean_SmoothL1_beta1',
        inference='native_HLT_without_predictor', attention=False, warm_start=False)


def node(base):
    if (base['coordinate'] != 'D000' or base['teacher'] is not None
            or base['context_coordinate'] is not None or not base['deployable']):
        raise ValueError('Expected original HLT-only CE initialization')
    return dict(base, node_id='U100_EMBED_W5', branch='u100_w5', auxiliary_weight=5.)


def tasks():
    return [dict(task_id='export_U100', dependencies=[]),
            dict(task_id='train_w5', dependencies=['export_U100'])]
