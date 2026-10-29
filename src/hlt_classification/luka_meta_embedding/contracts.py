"""Registered matched continuation, not an edit to historical experiments."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_embedding_aux.contracts import COUNTS, node as original_node
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as fixed30, learning_rate as lr30

AUTHORIZATION = 'AUTHORIZE_LUKA_META_EMBEDDING_DEBUG_V1'
ARMS = {'fixed': 'W5_FIXED_CONTINUE', 'meta': 'W5_STUDENT_BENEFIT'}
TASKS = {'setup': [], 'fixed': ['setup'], 'meta': ['setup']}
MINUTES = {'setup': 720, 'fixed': 720, 'meta': 720}
SPLIT_SALT = b'LUKA_META_EMBEDDING_INNER_QUERY_v1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_META_EMBEDDING_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_META_EMBEDDING_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=fixed30(), teacher='OFFLINE', student='D000', width=128,
        ce_weight=1., embedding_weight=5., logit_kd=False, target_role='train',
        epochs=30, batch_size=128, lr_scale=.1, meta_fraction=.05,
        split_salt=SPLIT_SALT.decode(), split='lowest_sha256_floor_N_over20',
        calibration_epochs=3, calibration_lr=3e-4, calibration_decay=0.,
        initialization='selected_OFFLINE_w5_native_plus_shared_calibrated_predictor',
        predictor='Linear128_GELU_Linear128', optimizer='fresh_AdamW',
        adapter='z_plus_0.1_tanh_Linear32_GELU_Linear128_zero_last',
        adapter_lr=1e-4, adapter_clip=1., teacher_preservation_kl_weight=.1, anchor_weight=1.,
        meta_steps=5, meta_every=100, support_batch=16, query_batch=64,
        meta_epochs=[4,25], freeze_epochs=[26,30],
        virtual_optimizer='functional_AdamW_cloned_real_moments_exact_bias_correction',
        virtual_precision='FP32_math_SDPA_no_TF32', virtual_bn='cloned_support_train_query_eval',
        real_precision='native_BF16_cuda_FP32_cpu', target_normalization='original_TRAIN_FP64_floor1e-3',
        inference='native_HLT_only', selection='full_validation_AUC_CE_rejection_earliest')


def node(parent):
    return dict(original_node(parent), node_id='HLT_META_EMBEDDING', branch='META_EMBEDDING', auxiliary_weight=5.)


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= 30:
        raise ValueError('Invalid continuation schedule position')
    return .1*lr30(position)


def meta_due(epoch, batch):
    return 4 <= epoch <= 25 and batch > 0 and batch % 100 == 0
