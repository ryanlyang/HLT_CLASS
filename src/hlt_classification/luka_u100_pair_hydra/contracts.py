"""New execution identity; scientific recipe stays in the original namespace."""
from hlt_classification.luka_fullsim.alljets_data import artifact as base_artifact, validate as base_validate
from hlt_classification.luka_u100_pair_correction.contracts import ARMS, COUNTS, recipe

ARCHIVE_SHA256 = '2fbb97036df805967c651b9a15a4a14f89ef3a9a75fab1f9e96f1819ecc986f5'
MANIFEST_HASH = 'b87e76450441c78892af092a1cb67d6abeb23963b4c983eaf756cd4c4535116c'
BASE = '65952ed3805f812c06d1ea54117c35b630cc684e'
AUTHORIZATION = 'AUTHORIZE_LUKA_U100_PAIR_HYDRA_V1'
SITE = dict(cluster='hydra', partition='gpus', account='default', qos='normal',
            gres='gpu:nvidia_l40:1', gpu_name='NVIDIA L40', architecture='x86_64')
PARITY = dict(max_abs=0.005, rms=0.0002, auc_abs=0.00005)


def artifact(kind, *, parents=None, **fields):
    return base_artifact('PAIR_HYDRA_'+kind, parents=parents, **fields)


def validate(value, kind, parents=None):
    return base_validate(value, 'PAIR_HYDRA_'+kind, parents)
