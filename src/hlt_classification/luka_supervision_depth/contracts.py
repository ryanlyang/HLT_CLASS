"""Exact twenty-fit scientific registration; old campaigns remain immutable."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_auxiliary_grid.contracts import recipe as grid_recipe, TEACHERS, COUNTS

AUTHORIZATION = 'AUTHORIZE_LUKA_SUPERVISION_DEPTH_20_STUDENTS_2_TEACHERS_DEBUG_V1'
TEACHER_TASKS = ('teacher_ce', 'teacher_supcon')
REFERENCES = ('M0HLT', 'OFFLINE', 'OFFLINE_EMBED_W5', 'U100_EMBED_W5', 'C50_DE000', 'K1_E40_PLAIN')
_ROWS = (
    ('baseline', (), 0., 'none', 0, 'original'),
    ('set8_p3', (8,), 3., 'set', 0, 'original'),
    ('set8_p10', (8,), 10., 'set', 0, 'original'),
    ('set48_p3', (4,8), 3., 'set', 0, 'original'),
    ('set48_p10', (4,8), 10., 'set', 0, 'original'),
    ('aligned8_p10', (8,), 10., 'aligned', 0, 'original'),
    ('set248_p3', (2,4,8), 3., 'set', 0, 'original'),
    ('set248_p10', (2,4,8), 10., 'set', 0, 'original'),
    ('baseline_seed1', (), 0., 'none', 1, 'original'),
    ('set48_p10_seed1', (4,8), 10., 'set', 1, 'original'),
    ('duplicate_offline', (), 0., 'none', 0, 'duplicate'),
    ('ce_diverse', (), 0., 'none', 0, 'ce'),
    ('supcon_diverse', (), 0., 'none', 0, 'supcon'),
    ('supcon_low', (), 0., 'none', 0, 'low'),
    ('supcon_replace', (), 0., 'none', 0, 'replace'),
    ('supcon_set8', (8,), 10., 'set', 0, 'supcon'),
    ('supcon_set48', (4,8), 10., 'set', 0, 'supcon'),
    ('set8_p1', (8,), 1., 'set', 0, 'original'),
    ('set8_p30', (8,), 30., 'set', 0, 'original'),
    ('set48_p30', (4,8), 30., 'set', 0, 'original'),
)
ARMS = {name:dict(depths=list(d), token_weight=p, matching=m, seed=s, diversity=v)
        for name,d,p,m,s,v in _ROWS}


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_SUPERVISION_DEPTH_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_SUPERVISION_DEPTH_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent lineage differs')
    return digest


def recipe(arm):
    r = ARMS[arm]; old = grid_recipe('k1_e40_plain'); delta = 100000*r['seed']
    weights = {t:10. for t in TEACHERS}; targets = {t:t for t in TEACHERS}
    variant = r['diversity']
    if variant == 'replace':
        # Change the target, not the original head's initialization/capacity.
        targets['OFFLINE'] = 'teacher_supcon'
    elif variant != 'original':
        weights['OFFLINE'] = 8. if variant == 'low' else 5.
        weights['OFFLINE_ALT'] = 10.-weights['OFFLINE']
        targets['OFFLINE_ALT'] = 'OFFLINE' if variant == 'duplicate' else 'teacher_ce' if variant == 'ce' else 'teacher_supcon'
    seeds = {t:s+delta for t,s in old['embedding_seeds'].items()}
    seeds['OFFLINE_ALT'] = seeds['OFFLINE']+20000
    return artifact('RECIPE', parents=dict(baseline=old['content_hash']), arm=arm, **r,
        base=old['base'], epochs=30, batch_size=128, ce_coefficient=.5, logit_total=1., embedding_total=40.,
        temperature=2., logit_weights={t:.25 for t in TEACHERS}, embedding_weights=weights,
        target_names=targets, embedding_seeds={t:seeds[t] for t in weights},
        logit_seeds={t:s+delta for t,s in old['logit_seeds'].items()},
        token_seeds={str(d):83100+d+delta for d in r['depths']}, token_teacher='c50_UE100',
        token_assignment='one_detached_Hungarian_mean_depth_cost' if r['matching']=='set' else r['matching'],
        token_reduction='mean_channels_real_tokens_depths_jets', normalization='TRAIN_FP64_population_floor1e-3',
        relation=False, inference='native_HLT_only_auxiliary_heads_removed')


def teacher_recipe(task):
    if task not in TEACHER_TASKS: raise ValueError('Unknown teacher')
    return artifact('TEACHER_RECIPE', task=task, epochs=30, batch_size=128, ce_coefficient=1.,
        supcon_coefficient=.1 if task=='teacher_supcon' else 0., temperature=.1,
        projection=[128,'GELU',128], projection_seed=84221, positive_rule='same_class_exclude_self',
        no_positive='exclude_anchor_or_differentiable_zero', augmentation=False,
        base=grid_recipe('k1_e40_plain')['base'], export='native_preclassifier_not_projection')


def nodes(parent):
    base = parent['nodes']['c50']['DE000']
    if base['coordinate']!='D000' or base['context_coordinate'] is not None or not base['deployable']:
        raise ValueError('Require native c50 DE000')
    result = {a:dict(base, node_id='DEPTH_'+a, arm=a, teacher=None,
        initialization_seed=base['initialization_seed']+100000*r['seed'],
        sampler_seed=base['sampler_seed']+100000*r['seed']) for a,r in ARMS.items()}
    offline = parent['anchor']['evidence']['models']['OFFLINE']
    # Original authenticated training report supplies the exact OFFLINE node.
    from hlt_classification.data.cache_contracts import load_json
    original = load_json(offline['report']['path'])['node']
    for task in TEACHER_TASKS:
        result[task] = dict(original, node_id=task, arm=task, coordinate='OFFLINE', teacher=None,
            context_coordinate=None, deployable=False, initialization_seed=original['initialization_seed']+200000,
            sampler_seed=original['sampler_seed']+200000)
    return result


def tasks():
    rows = [dict(task_id='targets', dependencies=[])]; previous = 'targets'
    for index, arm in enumerate(ARMS):
        if index == 10:
            for teacher in TEACHER_TASKS:
                rows.append(dict(task_id=teacher, dependencies=[previous])); previous=teacher
        rows.append(dict(task_id=arm, dependencies=[previous])); previous=arm
    rows.append(dict(task_id='report', dependencies=[previous]))
    rows.append(dict(task_id='cleanup', dependencies=['report']))
    return rows
