"""Independent FullStat identities, not trial fixed30 artifacts."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS, REFERENCES
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe

AUTHORIZATION = 'AUTHORIZE FULLSTAT LAST EMBEDDING EXACT PLAN'
TEACHER = 'COARSE_D033'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_LAST_EMBEDDING_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_LAST_EMBEDDING_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('FullStat evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Embedding artifact parent hashes differ')
    return result


def recipe():
    return artifact('RECIPE', base=base_recipe(128), epochs=20, batch_size=128,
        teacher=TEACHER, embedding_teacher='OFFLINE', temperature=2.,
        ce_coefficient=.25, kd_coefficient=.75, auxiliary_coefficient=.3,
        width=128, predictor=[128, 'GELU', 128],
        predictor_seed=auxiliary_recipe()['predictor_seed'], minimum_scale=1e-3,
        normalization='TRAIN_population_moments_FP64_targets_FP32',
        auxiliary_loss='mean_SmoothL1_beta1', tap='native_mod.fc_input',
        inference='native_HLT_without_predictor', attention=False, warm_start=False)


def node(parent):
    old = next(n for n in parent['nodes'] if n['node_id'] == 'COARSE_D000')
    if (old['coordinate'] != 'D000' or old['context_coordinate'] is not None
            or old['teacher'] != TEACHER or not old['deployable']):
        raise ValueError('Expected existing regular D033 -> single-HLT D000')
    return dict(old, node_id='EMBED_D000', branch='EMBED', embedding_coefficient=.3)


def tasks():
    return [dict(task_id='train_EMBED_D000', kind='train', dependencies=[]),
            dict(task_id='aggregate', kind='aggregate', dependencies=['train_EMBED_D000'])]
