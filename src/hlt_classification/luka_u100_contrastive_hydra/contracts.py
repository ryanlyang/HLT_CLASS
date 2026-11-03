"""Execution identity; scientific definitions remain in the donor package."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_contrastive.contracts import ARMS, COUNTS, recipe
from hlt_classification.luka_u100_pair_hydra.contracts import SITE, MANIFEST_HASH

BASE = '6f7a6cbc587800049c7e5eea6c22cc9af9026d03'
PARENT_COMMIT = '431341db5eced8eaa3e52b37ba047f8cd4beed50'
AUTHORIZATION = 'AUTHORIZE_LUKA_CONTRASTIVE_HYDRA_INDEPENDENT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_U100_CONTRASTIVE_HYDRA_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_U100_CONTRASTIVE_HYDRA_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()) or (parents is not None and value['parents'] != parents):
        raise ValueError('Contrastive Hydra scope/parents differ')
    return result
