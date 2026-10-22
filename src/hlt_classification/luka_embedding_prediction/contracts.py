"""Immediate-predecessor prediction, distinct from fixed-OFFLINE or logit KD."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as schedule

COUNTS = dict(train=299234, validation=74895)
COORDINATES = ('U100', 'D066', 'D033', 'D000')
VIEWS = ('U000', *COORDINATES)
PREVIOUS = dict(zip(COORDINATES, VIEWS[:-1]))
TASKS = {'setup': [], **{co: ['setup'] + ([] if i == 0 else [COORDINATES[i-1]]) for i, co in enumerate(COORDINATES)}}
MINUTES = {'setup': 480, **{co: 360 for co in COORDINATES}}
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_PREDICTION_DEBUG_V1'
WIDTH = 128
WEIGHT = 5.


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_PREDICTION_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_PREDICTION_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()): raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents: raise ValueError('Parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=schedule(), coordinates=list(COORDINATES), starting_teacher='U000',
        target='immediate_predecessor_selected_native_preclassifier_embedding', width=WIDTH,
        auxiliary_coefficient=WEIGHT, ce_coefficient=1., loss='mean_SmoothL1_beta1',
        predictor='Linear128_GELU_Linear128', normalization_fit_role='train', target_roles=['train'],
        minimum_scale=1e-3, target_precision='FP32', moment_precision='FP64',
        initialization='fresh_original_coordinate_seed', warm_start=False, logit_kd=False, attention_kd=False,
        inference='native_single_view_predictor_removed', final_deployable_coordinate='D000')


def teacher_id(coordinate):
    if coordinate not in VIEWS: raise ValueError('Unknown teacher coordinate')
    return 'U000' if coordinate == 'U000' else 'PRED_'+coordinate


def nodes(parent):
    by_name = {n['node_id']: n for n in parent['nodes']}
    result = []
    for co in COORDINATES:
        base = by_name['COARSE_'+co]
        if base['coordinate'] != co or base['context_coordinate'] is not None:
            raise ValueError('Expected original single-view coordinate')
        result.append(dict(base, node_id=teacher_id(co), branch='EMBEDDING_PREDICTION',
            teacher=None, teacher_coordinate=None, auxiliary_weight=WEIGHT,
            embedding_teacher=teacher_id(PREVIOUS[co]), embedding_teacher_coordinate=PREVIOUS[co]))
    return result


def stage_node(spec, co):
    if co not in COORDINATES: raise ValueError('Unknown stage')
    node = spec['nodes'][COORDINATES.index(co)]
    if (node['node_id'] != teacher_id(co) or node['coordinate'] != co or node['teacher'] is not None
            or node['teacher_coordinate'] is not None or node['context_coordinate'] is not None
            or node['branch'] != 'EMBEDDING_PREDICTION' or node['auxiliary_weight'] != WEIGHT
            or node['embedding_teacher'] != teacher_id(PREVIOUS[co])
            or node['embedding_teacher_coordinate'] != PREVIOUS[co]):
        raise ValueError('Immediate-predecessor CE+embedding registration differs')
    return node
