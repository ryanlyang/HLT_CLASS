"""Independent scientific and retention contracts; old campaigns stay immutable."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_attention.contracts import FLAGS, SURFACE, COUNTS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

# Immutable v1 routing. New GATE/v2 registrations use tier3 for BOTH arms/phases.
ARMS = {'w03': (.3, 'sporc_a100_debug'), 'w10': (1., 'sporc_a100')}
GATE_VERSION = 2
COORDINATES = ('D066', 'D033', 'D000')
AUTHORIZATION = 'AUTHORIZE LUKA ATTENTION WEIGHT EXACT PLAN AND OWNED RETIREMENT'


def artifact(kind, *, parents=None, version=None, **fields):
    version = (GATE_VERSION if kind == 'GATE' else 1) if version is None else version
    if type(version) is not int or version not in ((1, GATE_VERSION) if kind == 'GATE' else (1,)):
        raise ValueError('Attention weight schema version differs')
    return with_content_hash(dict(contract=f'LUKA_ATTENTION_WEIGHT_{kind}/v{version}',
        schema_version=version, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    version = value.get('schema_version')
    if type(version) is not int or version not in ((1, GATE_VERSION) if kind == 'GATE' else (1,)):
        raise ValueError('Attention weight schema version differs')
    digest = validate_content_hash(value, expected_contract=f'LUKA_ATTENTION_WEIGHT_{kind}/v{version}',
                                   expected_schema_version=version)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Historical test/development scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent lineage differs')
    return digest


def recipe(coefficient):
    if type(coefficient) not in (int, float) or coefficient not in (.3, 1.):
        raise ValueError('Only registered coefficients 0.3 and 1.0')
    return artifact('RECIPE', base=base_recipe(), surface=dict(SURFACE),
        coefficient=float(coefficient), epochs=30, coordinates=list(COORDINATES),
        initialization='fresh', target_storage='job_local_fp32_rolling',
        retirement='after_verified_last_consumer_v1', keep_validation_predictions=True)


def nodes(parent_nodes, arm):
    coefficient, _ = ARMS[arm]
    old = {n['node_id']: n for n in parent_nodes}
    out = []
    teacher = 'ANCHOR'
    for coordinate in COORDINATES:
        node = old['COARSE_'+coordinate]
        if node['coordinate'] != coordinate or node['context_coordinate'] is not None:
            raise ValueError('Expected regular fixed30 nodes')
        name = arm.upper()+'_'+coordinate
        out.append(dict(node, node_id=name, teacher=teacher, branch='ATTN',
                        attention_coefficient=coefficient))
        teacher = name
    return out
