"""Distinct cold initialization and bounded-storage scientific registration."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_gating.contracts import GATE_CONFIG, input_label
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

AUTHORIZATION = 'AUTHORIZE LUKA COLD GATE TIER3 EXACT PLAN'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_COLD_GATE_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_COLD_GATE_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise PermissionError('Cold gate evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Cold gate parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=base_recipe(), gate=GATE_CONFIG, epochs=30,
        initialization='fresh_seeded_all_weights_buffers_optimizer', freeze_epochs=0,
        gate_loss='valid_key_centered_log_gate_MSE_query_then_jet', gate_weight=.1,
        anchor='COARSE_U100', coordinates=['D066','D033','D000'],
        retention='selected_outputs_permanent_targets_owned_job_scratch_v1')


def nodes(parent_nodes):
    old = {n['node_id']:n for n in parent_nodes}
    rows = []
    for branch, mode in [('LOGIT','none'),('SELF','self'),('PRIV','privileged')]:
        teacher, richer = 'ANCHOR', 'U100'
        for co in recipe()['coordinates']:
            original = old['COARSE_'+co]
            if original['coordinate'] != co or original['context_coordinate'] is not None:
                raise ValueError('Wrong fixed30 coordinate/seeds')
            name = branch+'_'+co
            rows.append(dict(original, node_id=name, branch=branch, teacher=teacher,
                gate_mode=mode, context_coordinate=richer if mode=='privileged' else None,
                gate_weight=0., freeze_epochs=0, warm_start=None,
                deployable=co=='D000' and mode!='privileged'))
            teacher, richer = name, co
    for name, teacher, weight in [('REPLACE_LOGIT','PRIV_D000',0.),
            ('REPLACE_GATE','PRIV_D000',.1),('SELF_REDISTILL','SELF_D000',0.)]:
        rows.append(dict(old['COARSE_D000'], node_id=name, branch='REPLACEMENT',
            teacher=teacher, gate_mode='self', context_coordinate=None,
            gate_weight=weight, freeze_epochs=0, warm_start=None, deployable=True))
    return rows


def tasks(models):
    rows = [dict(task_id='train_'+n['node_id'], kind='train', node_id=n['node_id'],
        dependencies=[] if n['teacher']=='ANCHOR' else ['train_'+n['teacher']]) for n in models]
    return rows+[dict(task_id='aggregate', kind='aggregate', node_id=None,
                      dependencies=[r['task_id'] for r in rows])]
