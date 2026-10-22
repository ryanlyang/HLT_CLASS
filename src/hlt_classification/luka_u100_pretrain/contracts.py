"""Distinct two-stage meanings; existing auxiliary recipes remain immutable."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_embedding_aux.contracts import COUNTS, node as original_node
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as fixed30, learning_rate as lr30

AUTHORIZATION = 'AUTHORIZE_LUKA_U100_PRETRAIN_DEBUG_V1'
TASKS = {'setup': [], 'pretrain': ['setup'], 'finetune': ['setup', 'pretrain']}
MINUTES = {'setup': 480, 'pretrain': 360, 'finetune': 360}


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_U100_PRETRAIN_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_U100_PRETRAIN_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=fixed30(), teacher='COARSE_U100', coordinate='U100',
        student='D000', width=128, target_precision='FP32', moment_precision='FP64',
        minimum_scale=1e-3, normalization_fit_role='train', target_roles=['train', 'validation'],
        predictor='Linear128_GELU_Linear128', loss='mean_SmoothL1_beta1',
        batch_size=128, pretrain_epochs=20, finetune_epochs=30,
        pretrain_ce=0., pretrain_aux=1., finetune_ce=1., finetune_aux=5.,
        pretrain_selection='minimum_validation_embedding_loss_then_earliest',
        finetune_selection='native_AUC_CE_rejection_update',
        transition='selected_full_backbone_and_predictor_reset_optimizer_scheduler',
        logit_kd=False, attention_kd=False, inference='native_HLT_only')


def node(parent):
    return dict(original_node(parent), node_id='HLT_U100_PRETRAIN_W5', branch='U100_PRETRAIN', auxiliary_weight=5.)


def learning_rate(epoch_position, stage):
    if stage == 'finetune':
        return lr30(epoch_position)
    if stage != 'pretrain' or not math.isfinite(epoch_position) or not 0 < epoch_position <= 20:
        raise ValueError('Invalid pretraining schedule position')
    if epoch_position <= 3:
        return 3e-4 * epoch_position / 3
    return 1.5e-5 + (3e-4-1.5e-5)*.5*(1+math.cos(math.pi*(epoch_position-3)/17))
