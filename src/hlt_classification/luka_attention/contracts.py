"""Exact v1 science and content-addressed artifacts."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

FLAGS = dict(final_test_accessed=False, historical_test_evaluated=True,
             historical_test_repurposed_for_training=True, independent_test_available=False,
             evaluation_scope='exploratory_development_validation')
SURFACE = dict(block_index=7, head_reduction='mean_probabilities', dropout='before',
               pair_bias='native_own_view', precision='float32', epsilon=1e-8,
               diagonal='native', slots='persistent_hlt_all_real', temperature=1.)
COUNTS = dict(train=299234, validation=74895)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_D_ATTENTION_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_D_ATTENTION_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Attention experiment evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Attention parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=base_recipe(), surface=dict(SURFACE),
                    coefficient=.1, control_coefficient=0., epochs=30,
                    anchor='COARSE_U100', coordinates=['D066', 'D033', 'D000'])


def nodes(parent_nodes):
    by_name = {n['node_id']: n for n in parent_nodes}
    result = []
    for arm in ('LOGIT', 'ATTN'):
        teacher = 'ANCHOR'
        for co in recipe()['coordinates']:
            old = by_name['COARSE_'+co]
            if old['coordinate'] != co or old['context_coordinate'] is not None:
                raise ValueError('Expected regular parent coordinate/seeds')
            name = arm+'_'+co
            result.append(dict(old, node_id=name, teacher=teacher, branch=arm,
                               attention_coefficient=.1 if arm == 'ATTN' else 0.))
            teacher = name
    return result


def tasks(models):
    rows = [dict(task_id='export_ANCHOR', kind='export', node_id='ANCHOR', dependencies=[])]
    for n in models:
        name = n['node_id']
        rows.append(dict(task_id='train_'+name, kind='train', node_id=name,
                         dependencies=['export_'+n['teacher']]))
        if n['coordinate'] != 'D000':
            rows.append(dict(task_id='export_'+name, kind='export', node_id=name,
                             dependencies=['train_'+name]))
    rows.append(dict(task_id='aggregate', kind='aggregate', node_id=None,
                     dependencies=[r['task_id'] for r in rows]))
    return rows
