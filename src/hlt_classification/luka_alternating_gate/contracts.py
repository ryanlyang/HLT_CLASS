"""Versioned stagewise replacement DAG; no changes to prior gate semantics."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_gating.contracts import GATE_CONFIG, input_label
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

AUTHORIZATION = 'AUTHORIZE LUKA ALTERNATING GATE DEBUG EXACT PLAN'
ARMS = ('CONTROL', 'LOGIT', 'GATE')
COORDINATES = ('D066', 'D033', 'D000')


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_ALTERNATING_GATE_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_ALTERNATING_GATE_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Alternating gate evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Alternating gate parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=base_recipe(), gate=GATE_CONFIG, epochs=30,
        initialization='fresh_seeded_all_weights_buffers_optimizer', freeze_epochs=0,
        gate_loss='valid_key_centered_log_gate_MSE_query_then_jet', gate_weight=.1,
        anchor='COARSE_U100', coordinates=list(COORDINATES), arms=list(ARMS),
        stages=['TEACH', 'SELF'], next_rung_teacher='previous_SELF_only',
        retention='selected_outputs_permanent_targets_owned_job_scratch_v1')


def nodes(parent_nodes):
    old = {n['node_id']: n for n in parent_nodes}
    rows = []
    for arm in ARMS:
        teacher, richer = 'ANCHOR', 'U100'
        for co in COORDINATES:
            original = old['COARSE_'+co]
            if original['coordinate'] != co or original['context_coordinate'] is not None:
                raise ValueError('Wrong fixed30 coordinate/seeds')
            for stage in ('TEACH', 'SELF'):
                privileged = stage == 'TEACH' and arm != 'CONTROL'
                name = f'{arm}_{stage}_{co}'
                rows.append(dict(original, node_id=name, branch=arm, stage=stage, teacher=teacher,
                    teacher_coordinate=richer if stage == 'TEACH' else co,
                    gate_mode='privileged' if privileged else 'self',
                    context_coordinate=richer if privileged else None,
                    gate_weight=.1 if arm == 'GATE' and stage == 'SELF' else 0.,
                    export_factors=arm == 'GATE' and stage == 'TEACH',
                    freeze_epochs=0, warm_start=None, deployable=co == 'D000' and not privileged))
                teacher = name
            richer = co
    return rows


def teacher_node(models, student):
    if student['teacher'] == 'ANCHOR':
        return dict(models[0], node_id='ANCHOR', coordinate='U100', gate_mode='none',
            u=[1,1], f=[0,1], context_coordinate=None, teacher=None, teacher_coordinate=None,
            branch='ANCHOR', stage='ANCHOR', deployable=False, gate_weight=0., export_factors=False)
    return next(n for n in models if n['node_id'] == student['teacher'])


def tasks(models):
    rows = [dict(task_id='train_'+n['node_id'], kind='train', node_id=n['node_id'],
        dependencies=[] if n['teacher']=='ANCHOR' else ['train_'+n['teacher']]) for n in models]
    return rows+[dict(task_id='aggregate', kind='aggregate', node_id=None,
                      dependencies=[r['task_id'] for r in rows])]
