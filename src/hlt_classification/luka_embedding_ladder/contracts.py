"""Versioned five-rung cold experiment, independent of legacy artifacts."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as schedule
from hlt_classification.luka_embedding_aux.contracts import recipe as embedding_recipe
from hlt_classification.luka_fullsim.alljets_data import FLAGS

COUNTS = dict(train=299234, validation=74895)
COORDINATES = ('U050', 'U100', 'D066', 'D033', 'D000')
VIEWS = ('U000', *COORDINATES)
ARMS = ('LOGIT', 'EMBED')
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_LADDER_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_LADDER_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_LADDER_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Artifact parents differ')


def recipe():
    return artifact('RECIPE', schedule=schedule(), embedding=embedding_recipe(),
        coordinates=list(COORDINATES), ce=.25, kd=.75, temperature=2.,
        embedding_weight=.3, attention_kd=False, warm_start=False,
        starting_teacher='U000', embedding_teacher='OFFLINE',
        inference='native_single_view_predictor_removed')


def nodes(parent, arm):
    if arm not in ARMS:
        raise ValueError('Unregistered arm')
    by_name = {n['node_id']: n for n in parent['nodes']}
    result, teacher, coordinate = [], 'U000', 'U000'
    for co in COORDINATES:
        original = by_name['COARSE_'+co]
        if original['coordinate'] != co or original['context_coordinate'] is not None:
            raise ValueError('Expected original single-view coordinate')
        node = dict(original, node_id=arm+'_'+co, branch=arm, teacher=teacher,
                    teacher_coordinate=coordinate, embedding_coefficient=.3 if arm == 'EMBED' else 0.)
        result.append(node)
        teacher, coordinate = node['node_id'], co
    return result
