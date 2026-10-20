"""Independent FullStat identities and an exactly matched replacement contrast."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS, REFERENCES
from hlt_classification.luka_gating.contracts import GATE_CONFIG

AUTHORIZATION = 'AUTHORIZE FULLSTAT REPLACE GATE EXACT PLAN'
TEACHER = 'COARSE_D033'
NAMES = ('PRIV_D000', 'REPLACE_LOGIT', 'REPLACE_GATE')


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_REPLACE_GATE_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_REPLACE_GATE_{kind}/v1')
    if value.get('schema_version') != 1 or any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('FullStat sealed evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Replacement artifact parents differ')
    return result


def recipe():
    return artifact('RECIPE', base=base_recipe(128), epochs=20, batch_size=128,
        gate=dict(GATE_CONFIG), ce=.25, kd=.75, temperature=2., gate_weight=.1,
        auxiliary='centered_bounded_log_gate_MSE_per_jet_including_diagonal',
        initialization='fresh_seeded_all_weights_buffers_optimizer', freeze_epochs=0,
        teacher_forcing=False, retain_gate_at_inference=True,
        factor_storage='authenticated_FP32_ragged_shards_resident_contiguous',
        automatic_deletion=False)


def nodes(parent):
    old = next(n for n in parent['nodes'] if n['node_id'] == 'COARSE_D000')
    if (old['coordinate'] != 'D000' or old['context_coordinate'] is not None
            or old['teacher'] != TEACHER or not old['deployable']):
        raise ValueError('Expected regular D033 -> HLT D000 parent')
    return [dict(old, node_id=name, branch='PRIVILEGED' if name == NAMES[0] else 'REPLACEMENT',
        teacher=TEACHER if name == NAMES[0] else NAMES[0],
        context_coordinate='D033' if name == NAMES[0] else None,
        gate_mode='privileged' if name == NAMES[0] else 'self',
        deployable=name != NAMES[0], gate_weight=.1 if name == 'REPLACE_GATE' else 0.,
        warm_start=None, freeze_epochs=0) for name in NAMES]


def tasks():
    return [
        dict(task_id='train_PRIV_D000', kind='train', dependencies=[]),
        dict(task_id='export_PRIV_D000', kind='export', dependencies=['train_PRIV_D000']),
        dict(task_id='train_REPLACE_LOGIT', kind='train', dependencies=['export_PRIV_D000']),
        dict(task_id='train_REPLACE_GATE', kind='train', dependencies=['export_PRIV_D000']),
        dict(task_id='aggregate', kind='aggregate',
             dependencies=['train_PRIV_D000', 'train_REPLACE_LOGIT', 'train_REPLACE_GATE'])]
