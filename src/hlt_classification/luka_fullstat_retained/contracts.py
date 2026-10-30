"""Distinct FullStat20 retained-graph science and artifact identities."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat_u100_embedding.contracts import recipe as original_recipe
from hlt_classification.luka_u100_multiview.contracts import HEAD_SEEDS

ARMS = {'train_retained': 'u100_use_prediction', 'train_control': 'hlt_capacity_control'}
TASKS = {'train_retained': [], 'train_control': ['train_retained']}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_RETAINED_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_RETAINED_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_RETAINED_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_RETAINED_{kind}/v1', expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()): raise PermissionError('Sealed-role flags differ')
    if parents is not None and value['parents']!=parents: raise ValueError('Artifact parents differ')
    return value


def recipe():
    old=original_recipe()
    return artifact('RECIPE', parents=dict(original=old['content_hash']), base=old['base'],
        epochs=20, batch_size=128, ce_weight=1., embedding_weight=5.,
        arms=ARMS, predictor_seed=HEAD_SEEDS['U100'],
        predictor='Linear128_GELU_Linear128_FP32', classifier='Linear256x11_FP32',
        classifier_input='concatenate_native_h_and_predicted_standardized_U100',
        initialization='fresh_native_classifier_half_copied_prediction_half_zero',
        teacher='COARSE_U100', targets='authenticated_original_TRAIN_bank',
        control='same_architecture_CE_only_no_teacher_targets',
        teacher_readout=False, logit_kd=False, relational=False, warm_start=False,
        inference='HLT_only_encoder_predictor_and_combined_classifier_retained',
        save_every_completed_epoch=True, automatic_resume=False)


def node(base, task):
    if (task not in ARMS or base['node_id']!='U100_EMBED_W5' or base['coordinate']!='D000'
            or base.get('teacher') is not None or base.get('context_coordinate') is not None
            or base['auxiliary_weight']!=5. or base['deployable'] is not True):
        raise ValueError('Expected original U100-w5 native HLT node')
    return dict(base, node_id='RETAINED_U100_W5' if task=='train_retained' else 'RETAINED_CE_CONTROL',
        branch=ARMS[task], auxiliary_weight=5. if task=='train_retained' else 0., retained_predictor=True)


def check_node(value, task):
    if (task not in ARMS or value['branch']!=ARMS[task] or value['coordinate']!='D000'
            or value['teacher'] is not None or value['context_coordinate'] is not None
            or value['deployable'] is not True or value['retained_predictor'] is not True
            or value['auxiliary_weight']!=(5. if task=='train_retained' else 0.)):
        raise ValueError('Retained model/input/objective differs')
