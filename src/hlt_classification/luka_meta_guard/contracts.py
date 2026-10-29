"""Versioned science: a bounded TRAIN-guard heuristic, not a no-harm guarantee."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_embedding_aux.contracts import COUNTS

AUTHORIZATION = 'AUTHORIZE_LUKA_META_GUARD_DEBUG_V3'
INITIALIZERS = {
    'completed_fixed': 'completed_parent_FIXED_selected_full_student',
    'shared_w5': 'completed_parent_setup_calibrated_OFFLINE_w5_full_student',
}
ARMS = {'fixed': 'GUARDED_FIXED', 'meta': 'GUARDED_META'}
TASKS = {'setup': [], 'fixed': ['setup'], 'meta': ['setup']}
MINUTES = {k: 720 for k in TASKS}
SALT = b'LUKA_META_GUARD_QUERY_GUARD_v1'
BLOCKS = 20
INITIAL_LR = 5e-6
TOLERANCE = 2e-4
PATIENCE = 2
MAX_ROLLBACKS = 3
# Only the separately evaluated FIXED query losses receive a numerical tolerance.
# Target identity and the arithmetic of the recorded signed delta remain exact.
FIXED_QUERY_CE_ATOL = 1e-6


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_META_GUARD_{kind}/v3', schema_version=3,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_META_GUARD_{kind}/v3', expected_schema_version=3)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return digest


def events(arm, block):
    if arm not in ARMS or not 1 <= block <= BLOCKS:
        raise ValueError('Unregistered arm/block')
    return 0 if arm == 'fixed' else (100 if block == 1 else 25 if block <= 16 else 0)


def recipe(initializer_source='completed_fixed'):
    if initializer_source not in INITIALIZERS:
        raise ValueError('Unknown initializer source; no automatic fallback')
    return artifact('RECIPE', initialization=INITIALIZERS[initializer_source],
        optimizer='fresh_AdamW_same_groups_as_parent', ce_weight=1., embedding_weight=5., logit_kd=False,
        blocks=BLOCKS, passes=10, batch_size=128, initial_lr=INITIAL_LR, warmup=False,
        sampling='two_consecutive_half_blocks_partition_one_shuffled_inner_pass',
        split='retain_original_95pct_inner_split_original_5pct_into_query_and_guard',
        split_salt=SALT.decode(), guard_metric='macro_ovr_auc', guard_tolerance=TOLERANCE,
        guard_patience=PATIENCE, max_rollbacks=MAX_ROLLBACKS, rollback_lr_factor=.5,
        rollback='accepted_full_student_buffers_and_optimizer_only_keep_adapter_and_its_optimizer',
        guard_accept='auc_exceeds_accepted_anchor_by_tolerance',
        rng='fresh_deterministic_block_seed_not_rewound_on_rollback',
        budget='always_all_20_blocks_even_after_rollback_cap',
        adapter='original_bounded_identity_plus_0.1_tanh_bottleneck32', adapter_lr=1e-4,
        adapter_headstart=100, adapter_between_blocks=25, adapter_freeze_blocks=[17,20],
        teacher='frozen_original_OFFLINE_including_classifier', normalization='unchanged_original_TRAIN_stats',
        meta_steps=5, support_batch=16, query_batch=64, virtual_lr='current_real_lr',
        teacher_preservation_kl_weight=.1, residual_anchor_weight=1.,
        real_precision='native_BF16_cuda_FP32_cpu', virtual_precision='FP32_math_SDPA_no_TF32',
        diagnostic=dict(fixed_query_ce_atol=FIXED_QUERY_CE_ATOL, fixed_query_ce_rtol=0.,
            fixed_target_residual='exact_zero', signed_delta='exact_adapted_minus_identity'),
        selection='initial_plus_post_guard_full_validation_AUC_CE_rejection_earliest',
        guard_not_independent_test=True, inference='native_HLT_only_no_auxiliary_heads')
