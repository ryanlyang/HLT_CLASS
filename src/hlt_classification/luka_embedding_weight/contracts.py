"""Only the three explicitly requested objectives are registered."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_embedding_aux.contracts import recipe as original_recipe, node as original_node, COUNTS
from hlt_classification.luka_fullsim.alljets_data import FLAGS

WEIGHTS = {'w075': .75, 'w15': 1.5, 'w10': 10.}
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_WEIGHT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_WEIGHT_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_WEIGHT_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Weight experiment parent hashes differ')
    return result


def recipe(weight):
    if type(weight) not in (int, float) or weight not in WEIGHTS.values():
        raise ValueError('Registered embedding weights are 0.75, 1.5 and 10 only')
    original = original_recipe()
    fields = {k: v for k, v in original.items()
              if k not in {'contract', 'schema_version', 'content_hash', 'parents', *FLAGS}}
    fields['auxiliary_coefficient'] = float(weight)
    return artifact('RECIPE', parents=dict(original_target_recipe=original['content_hash']), **fields)


def nodes(parent):
    base = original_node(parent)
    return {arm: dict(base, node_id='HLT_EMBED_'+arm.upper(), branch=arm,
                     auxiliary_weight=weight) for arm, weight in WEIGHTS.items()}


def arm_gate(gate, arm):
    if arm not in WEIGHTS:
        raise ValueError('Unknown weight arm')
    fields = {k: v for k, v in gate.items()
              if k not in {'contract', 'schema_version', 'content_hash', 'parents', *FLAGS}}
    return artifact('ARM', parents=dict(gate=gate['content_hash']), **fields,
                    arm=arm, node=gate['nodes'][arm], recipe=recipe(WEIGHTS[arm]))
