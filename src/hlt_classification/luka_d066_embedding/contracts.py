"""A new target meaning, not a relabelled OFFLINE bank."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_embedding_aux.contracts import recipe as old_recipe, node as old_node, COUNTS
from hlt_classification.luka_fullsim.alljets_data import FLAGS

AUTHORIZATION = 'AUTHORIZE_LUKA_D066_EMBEDDING_V1'
WIDTH = 128
WEIGHT = 5.


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_D066_EMBEDDING_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_D066_EMBEDDING_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('D066 experiment parent hashes differ')
    return digest


def recipe():
    donor = old_recipe()
    fields = {k: v for k, v in donor.items()
              if k not in {'contract', 'schema_version', 'content_hash', 'parents', *FLAGS}}
    fields.update(auxiliary_coefficient=WEIGHT, teacher='LOGIT_D066', target_coordinate='D066')
    return artifact('RECIPE', parents=dict(donor=donor['content_hash']), **fields)


def node(parent):
    return dict(old_node(parent), node_id='HLT_D066_EMBED_W5', branch='D066_EMBED', auxiliary_weight=WEIGHT)
