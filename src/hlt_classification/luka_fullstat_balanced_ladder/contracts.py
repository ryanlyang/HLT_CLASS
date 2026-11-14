"""Versioned scientific identity; historical FullStat recipes remain unchanged."""
import hashlib
import math

from hlt_classification.operations.fullstat_pt_metadata import (
    CLASS_NAMES, COUNTS, with_content_hash, validate_content_hash,
)

PREFIX = 'LUKA_FULLSTAT_BALANCED_LADDER'
FLAGS = dict(final_test_accessed=False, final_test_sealed=True,
             event_independence_verified=False, old_trial_overlap_verified=False,
             evaluation_scope='development_validation')
STAGES = ('OFFLINE', 'M0HLT', 'CONCAT', 'UE100', 'DE050', 'DE000')
COORDINATES = dict(OFFLINE='OFFLINE', M0HLT='D000', CONCAT='CONCAT',
                   UE100='U100', DE050='D050', DE000='D000')
TEACHERS = dict(OFFLINE=(), M0HLT=(), CONCAT=(),
    UE100=('OFFLINE', 'CONCAT', 'M0HLT'),
    DE050=('OFFLINE', 'CONCAT', 'UE100', 'M0HLT'),
    DE000=('OFFLINE', 'CONCAT', 'UE100', 'DE050'))
ELIGIBLE = 2425536
DRAWS = ELIGIBLE
EPOCHS = 20
BATCH = 128
STEPS = math.ceil(DRAWS / BATCH)
EDGES = tuple(range(200, 2001, 100))
SAMPLER_SEED = 2026111301
INITIALIZATION_SEED = 2026111302
MATCH_SEED = 2026111303
NEGATIVE_SEED = 2026111304
AUDIT_HASH = '8383cbacb0aeb7664a534ab5e8f3287b8e67c4963dbe69e7c90146786c25522c'
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_BALANCED_LADDER_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_BALANCED_LADDER_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    if set(fields).intersection(('contract', 'schema_version', 'parents', *FLAGS)):
        raise ValueError('Reserved artifact fields')
    return with_content_hash(dict(contract=f'{PREFIX}_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'{PREFIX}_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Scientific scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent hashes differ')
    return value


def seed(domain, teacher):
    return int.from_bytes(hashlib.sha256(
        f'{PREFIX}/v1/{domain}/{teacher}'.encode()).digest()[:4], 'big')


def recipe():
    return artifact('RECIPE', original_counts=dict(COUNTS), eligible_train_rows=ELIGIBLE,
        draws_per_epoch=DRAWS, epochs=EPOCHS, batch_size=BATCH, updates_per_epoch=STEPS,
        last_batch_size=DRAWS % BATCH, accumulation=1, early_stopping=False,
        class_names=list(CLASS_NAMES), pt_edges_GeV=list(EDGES), pt_boundary='[low,high)',
        pt_definition='saved_offline_reconstructed_jet_pt_GeV',
        sampler='PCG64_cell_then_row_iid_replacement_v1', sampler_seed=SAMPLER_SEED,
        same_draw_order_all_stages=True, loss_class_weights=None, loss_pt_weights=None,
        optimizer='AdamW', betas=[.9, .999], eps=1e-8, weight_decay=.01,
        peak_lr=3e-4, floor_lr=1.5e-5, warmup_epochs=2, decay_epoch=20,
        schedule='warmup2_cosine_to20_draw_epochs', precision='bf16_forward_fp32_loss',
        ce_weight=1., embedding_total_weight=10., auxiliary_kd_total_weight=1.,
        kd_temperature=2., kd_reduction='T2_forward_KL_class_sum_batch_mean',
        embedding_loss='SmoothL1_beta1_mean_rows_dimensions',
        normalization='unique_eligible_TRAIN_FP64_population_std_floor_0.001',
        embedding_export='native_FP32_batch128', probability_export='native_BF16_FP32_softmax_T2_batch128',
        teacher_order={k:list(v) for k,v in TEACHERS.items()},
        coordinates=COORDINATES, embedding_head=[128,128,128], logit_head=[128,11],
        matching_head=[128,128,128], matching_weight=.3, matching_ramp_epochs=3,
        matching_temperature=.1, matching_negatives=1024, matching_components=[256,256,512],
        hard_proposals=4096, hard_rank_interval=[257,1024], matching_stage='DE000',
        negative_seed=NEGATIVE_SEED, negative_rng='identity_epoch_occurrence_component_stream_v1',
        negative_population='unique_eligible_TRAIN_donor_mix_not_pt_flattened',
        selection=['maximum_macro_auc','minimum_cross_entropy','maximum_macro_log_R50','earliest_update'],
        selection_population='original_natural_500000_validation',
        final_test_operation=False)


def node(stage):
    if stage not in STAGES:
        raise PermissionError('Unregistered balanced stage')
    return dict(node_id=stage, coordinate=COORDINATES[stage],
        initialization_seed=INITIALIZATION_SEED, sampler_seed=SAMPLER_SEED,
        teachers=list(TEACHERS[stage]),
        embedding_seeds={t:seed('embedding',t) for t in TEACHERS[stage]},
        logit_seeds={t:seed('logit',t) for t in TEACHERS[stage]}, matching_seed=MATCH_SEED)


def coefficient(stage, update, steps=STEPS):
    if stage not in STAGES or type(update) is not int or update < 1 or steps <= 0:
        raise ValueError('Invalid contrastive schedule coordinate')
    return .3 * min(1., update/(3*steps)) if stage == 'DE000' else 0.


def tasks():
    return {f'train_{s}':[f'train_{t}' for t in TEACHERS[s]] for s in STAGES}
