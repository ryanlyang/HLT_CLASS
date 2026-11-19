"""Hydra execution identity; scientific semantics remain the Oscar v1 recipe."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_sixclass_balanced_oscar import contracts as science
from hlt_classification.luka_sixclass_balanced_oscar.contracts import (
    FLAGS, COUNTS, EPOCHS, BATCH, STEPS, TEACHERS, COORDINATES, PARENT_HASH,
    POPULATION_HASH, ROLE_POPULATION_HASH, TRANSFER_HASH, CLASS_NAMES, OUTSIDE,
    UNUSED, recipe, node, main_policy, population_policy)

PREFIX = 'LUKA_SIXCLASS_TAIL_HYDRA'
IMPORTED = ('OFFLINE', 'M0HLT', 'CONCAT', 'UE100')
STAGES = ('DE050', 'DE000')
ALL_STAGES = IMPORTED + STAGES
JOBS = dict(zip(IMPORTED, ('9198207', '9198208', '9198209', '9198210')))
GPUS = {'l40': ('nvidia_l40', 'NVIDIA L40')}
SITE = dict(cluster='hydra', partition='gpus', account='default', qos='normal')
BASE = '/cs/home/rlyang/hlt_u100_pairs'
DATASET = BASE + '/datasets/luka_qcdrun2com_higgs2p_sb_v1'
AUTH = 'AUTHORIZE_SIXCLASS_TAIL_HYDRA_V1'
RESULT_GIB = 14  # Reserve new results separately from the already received bundle.


def tasks():
    return {'train_DE050': [], 'train_DE000': ['train_DE050']}


def artifact(kind, *, parents=None, **fields):
    if set(fields).intersection(('contract', 'schema_version', 'parents', *FLAGS)):
        raise ValueError('Reserved artifact fields')
    return with_content_hash(dict(contract=f'{PREFIX}_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'{PREFIX}_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise ValueError('Scientific scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent bindings differ')
    return value
