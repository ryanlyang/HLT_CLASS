"""Registered warm ladders and HLT-only gate replacements, version 1."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe
from hlt_classification.luka_attention.contracts import FLAGS, COUNTS

GATE_CONFIG = dict(rank=4, hidden=32, blocks=[6, 7], log_bound=1.3862943611198906,
                   heads='shared', input='own17_context17', factor_channels=9)


def artifact(kind, *, parents=None, **fields):
    version = 2 if kind == 'PREFLIGHT' else 1
    return with_content_hash(dict(contract=f'LUKA_PRIVILEGED_GATE_{kind}/v{version}', schema_version=version,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    version = 2 if kind == 'PREFLIGHT' else 1
    digest = validate_content_hash(value, expected_contract=f'LUKA_PRIVILEGED_GATE_{kind}/v{version}',
                                   expected_schema_version=version)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Gate experiment evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Gate parent hashes differ')
    return digest


def execution_policy():
    # Numerical execution only; scientific RECIPE/v1 and checkpoint keys stay fixed.
    return artifact('ATTENTION_EXECUTION', backend='SDPBackend.MATH',
        scope='registered_gated_attention_call_only',
        condition='cuda_grad_enabled_trainable_bias_frozen_attention_parameters_and_inputs',
        otherwise='native_dispatch', restore_on_exception=True)


def recipe():
    return artifact('RECIPE', base=base_recipe(), gate=GATE_CONFIG, epochs=30,
        gate_loss='valid_key_centered_log_gate_MSE_query_then_jet', gate_weight=.1,
        freeze_epochs=3, warm_start='selected_parent_all_compatible_weights_fresh_optimizer',
        anchor='COARSE_U100', coordinates=['D066', 'D033', 'D000'])


def nodes(parent_nodes):
    original = {n['node_id']: n for n in parent_nodes}
    rows = []
    for branch, mode in [('WARM','none'), ('SELF','self'), ('PRIV','privileged')]:
        teacher = 'ANCHOR'
        richer = 'U100'
        for co in recipe()['coordinates']:
            old = original['COARSE_'+co]
            if old['coordinate'] != co or old['context_coordinate'] is not None:
                raise ValueError('Expected original regular coordinate/seeds')
            name = branch+'_'+co
            rows.append(dict(old, node_id=name, branch=branch, teacher=teacher,
                gate_mode=mode, context_coordinate=richer if mode=='privileged' else None,
                gate_weight=0., freeze_epochs=0, warm_start=teacher,
                deployable=co=='D000' and mode!='privileged'))
            teacher, richer = name, co
    base = original['COARSE_D000']
    for name, weight, freeze in [('REPLACE_LOGIT',0.,0), ('REPLACE_GATE',.1,0),
                                  ('REPLACE_STAGED',.1,3), ('SELF_CONT',0.,0)]:
        parent = 'SELF_D000' if name=='SELF_CONT' else 'PRIV_D000'
        rows.append(dict(base, node_id=name, branch='REPLACEMENT', teacher=parent,
            gate_mode='self', context_coordinate=None, gate_weight=weight,
            freeze_epochs=freeze, warm_start=parent))
    return rows


def anchor_node(models):
    return dict(models[0], node_id='ANCHOR', coordinate='U100', context_coordinate=None,
                u=[1,1],f=[0,1],deployable=False,branch='ANCHOR',
                gate_mode='none', teacher=None, warm_start=None)


def tasks(models):
    exported = {n['teacher'] for n in models}
    rows = [dict(task_id='export_ANCHOR', kind='export', node_id='ANCHOR', dependencies=[])]
    for n in models:
        name = n['node_id']
        deps = ['export_'+n['teacher']]
        if n['warm_start'] != 'ANCHOR':
            deps.append('train_'+n['warm_start'])
        rows.append(dict(task_id='train_'+name, kind='train', node_id=name, dependencies=deps))
        if name in exported:
            rows.append(dict(task_id='export_'+name, kind='export', node_id=name,
                             dependencies=['train_'+name]))
    rows.append(dict(task_id='aggregate', kind='aggregate', node_id=None,
                     dependencies=[r['task_id'] for r in rows]))
    return rows


def input_label(node):
    if node['gate_mode']=='privileged':
        return 'OFFLINE-DEPENDENT ORACLE (privileged gate)'
    if node['coordinate']!='D000':
        return 'OFFLINE-DEPENDENT ORACLE (mixed view)'
    return 'HLT ONLY, '+('native ParT' if node['gate_mode']=='none' else 'single encoder + predicted gate')
