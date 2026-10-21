"""Independent sealed-FullStat identities; never trial/fixed30 or logit KD."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe

WEIGHTS = {'w0': 0., 'w2': 2., 'w10': 10., 'w50': 50.,
           'w100': 100., 'w500': 500., 'w1000': 1000.}
REFERENCES = ('M0HLT', 'OFFLINE')
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_EMBEDDING_SWEEP_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_EMBEDDING_SWEEP_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value,
        expected_contract=f'LUKA_FULLSTAT_EMBEDDING_SWEEP_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('FullStat sealed evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Embedding sweep parent hashes differ')
    return result


def target_recipe():
    return artifact('TARGET_RECIPE', teacher='OFFLINE', width=128,
        normalization='TRAIN_population_moments_FP64_targets_FP32', minimum_scale=1e-3,
        role='train', tap='native_mod.fc_input', teacher_eval=True, teacher_fp32=True)


def recipe(weight):
    if type(weight) not in (int, float) or weight not in WEIGHTS.values():
        raise ValueError('Unregistered embedding coefficient')
    return artifact('RECIPE', parents=dict(target=target_recipe()['content_hash']),
        base=base_recipe(128), base_usage='optimizer_and_schedule_only_not_KD_loss',
        epochs=20, batch_size=128, ce_coefficient=1.,
        auxiliary_coefficient=float(weight), kd_coefficient=0., teacher=None,
        embedding_teacher='OFFLINE', width=128, predictor=[128, 'GELU', 128],
        predictor_seed=auxiliary_recipe()['predictor_seed'], auxiliary_loss='mean_SmoothL1_beta1',
        inference='native_HLT_without_predictor', attention=False, warm_start=False)


def nodes(parent):
    base = next(n for n in parent['nodes'] if n['node_id'] == 'M0HLT')
    if (base['coordinate'] != 'D000' or base['teacher'] is not None
            or base['context_coordinate'] is not None or not base['deployable']):
        raise ValueError('Expected native CE-only M0HLT anchor')
    return {arm: dict(base, node_id='CE_EMBED_'+arm.upper(), branch=arm,
                      auxiliary_weight=weight) for arm, weight in WEIGHTS.items()}


def tasks():
    rows = [dict(task_id=arm, kind='train', dependencies=[]) for arm in WEIGHTS]
    return rows + [dict(task_id='aggregate', kind='aggregate', dependencies=list(WEIGHTS))]


def arm_gate(gate, arm):
    if arm not in WEIGHTS:
        raise ValueError('Unknown weight arm')
    fields = {k: v for k, v in gate.items()
              if k not in {'contract', 'schema_version', 'content_hash', 'parents', *FLAGS}}
    return artifact('ARM', parents=dict(gate=gate['content_hash']), **fields,
                    arm=arm, node=gate['nodes'][arm], recipe=recipe(WEIGHTS[arm]))
