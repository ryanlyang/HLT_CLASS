"""Explicit trial scope, independent branches and fixed registered schedules."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe
from hlt_classification.luka_fullstat_hybrid_embedding.contracts import (
    STAGES, TEACHERS, teachers, ancestors, kd_teacher,
    recipe as donor_recipe, node as donor_node)

COUNTS = dict(train=299234, validation=74895)
ARMS = ('c25', 'c50', 'c75', 'anneal')
EMBED_EXPORTS = ('OFFLINE', 'M0HLT', 'UE100', 'DE066', 'DE033')
KD_EXPORTS = ('OFFLINE', 'UE000', 'UE100', 'DE066', 'DE033')
AUTHORIZATION = 'AUTHORIZE_LUKA_TRIAL_HYBRID_20_FITS_DEBUG_NO_PREFLIGHT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_TRIAL_HYBRID_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_TRIAL_HYBRID_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory trial scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent lineage differs')
    return digest


def coefficients(arm, epoch):
    if arm not in ARMS or type(epoch) is not int or not 1 <= epoch <= 30:
        raise ValueError('Unknown arm/epoch')
    kd = .75*max(0., (25-epoch)/24) if arm == 'anneal' else dict(c25=.75, c50=.5, c75=.25)[arm]
    return 1.-kd, kd


def recipe(arm, stage):
    order = teachers(stage)
    donor = donor_recipe(stage)
    return artifact('RECIPE', parents=dict(donor=donor['content_hash'], base=base_recipe()['content_hash']),
        arm=arm, stage=stage, coordinate=STAGES[stage], base=base_recipe(), epochs=30, batch_size=128,
        coefficients=[list(coefficients(arm,e)) for e in range(1,31)], temperature=2.,
        kd_teacher=kd_teacher(stage), teacher_order=order, weights={t:5./len(order) for t in order},
        auxiliary_total=5., predictor_seeds=donor['predictor_seeds'],
        target='selected_native_mod_fc_input_128_not_prediction', normalization='TRAIN_FP64_population_floor_1e-3',
        initialization='fresh_coordinate_paired', warm_start=False, fusion=False,
        inference='native_single_view_predictors_removed')


def nodes(parent):
    base = next(n for n in parent['nodes'] if n['node_id'] == 'M0HLT')
    return {arm:{stage:dict(donor_node(base,stage), arm=arm, branch='trial_hybrid_'+arm)
                 for stage in STAGES} for arm in ARMS}


def task_id(arm, stage):
    if arm not in ARMS or stage not in STAGES:
        raise ValueError('Unknown arm/stage')
    return arm+'_'+stage


def tasks():
    rows = [dict(task_id='targets', dependencies=[])]
    for arm in ARMS:
        for stage in STAGES:
            rows.append(dict(task_id=task_id(arm,stage),
                dependencies=['targets']+[task_id(arm,t) for t in ancestors(stage)]))
    return rows+[dict(task_id='report',dependencies=[r['task_id'] for r in rows])]


def split_task(task):
    for arm in ARMS:
        for stage in STAGES:
            if task == task_id(arm,stage): return arm,stage
    raise ValueError('Not a registered fit task')


def export_kinds(teacher):
    return (['embedding'] if teacher in EMBED_EXPORTS else []) + (['kd'] if teacher in KD_EXPORTS else [])
