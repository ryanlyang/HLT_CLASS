"""Registered matched warm-start science; historical recipes remain unchanged."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_embedding_aux.contracts import COUNTS
from hlt_classification.luka_u100_embedding.contracts import node as original_node
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as fixed30, learning_rate as lr30

AUTHORIZATION = 'AUTHORIZE_LUKA_CE_WARMSTART_DEBUG_V1'
ARMS = {'ce': 'CE_CONTINUE', 'embed': 'CE_EMBED_U100_W5'}
TASKS = {'setup': [], 'ce': ['setup'], 'embed': ['setup']}
MINUTES = {'setup': 480, 'ce': 360, 'embed': 360}


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_CE_WARMSTART_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_CE_WARMSTART_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=fixed30(), teacher='COARSE_U100', target_coordinate='U100',
        student='D000', width=128, target_roles=['train'], normalization_fit_role='train',
        target_precision='FP32', moment_precision='FP64', minimum_scale=1e-3,
        predictor='Linear128_GELU_Linear128', loss='mean_SmoothL1_beta1',
        arms={'ce': {'ce': 1., 'embedding': 0.}, 'embed': {'ce': 1., 'embedding': 5.}},
        epochs=30, batch_size=128, lr_scale=0.1, peak_lr=3e-5, minimum_lr=1.5e-6,
        initialization='selected_M0HLT_all_native_parameters_and_buffers',
        predictor_initialization='fresh_deterministic_no_native_RNG_consumption',
        optimizer='fresh_AdamW_all_native_trainable', selection='native_AUC_CE_rejection_update',
        logit_kd=False, attention_kd=False, inference='native_HLT_only')


def node(parent):
    return dict(original_node(parent), node_id='HLT_CE_WARMSTART', branch='CE_WARMSTART')


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= 30:
        raise ValueError('Invalid continuation schedule position')
    return 0.1 * lr30(position)
