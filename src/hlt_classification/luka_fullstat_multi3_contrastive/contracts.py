"""New scientific identity; historical recipes remain immutable."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat_multiview.contracts import recipe as multi3_recipe, TEACHERS
from hlt_classification.luka_u100_contrastive.contracts import sampler_recipe, ramp

TASK = 'train_multi3_mix'
TASKS = {TASK: []}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_MULTI3_CONTRASTIVE_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_MULTI3_CONTRASTIVE_AUTOSUBMIT_V1'
MATCH_SEED = 2026103101


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_MULTI3_CONTRASTIVE_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_MULTI3_CONTRASTIVE_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Scientific scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return value


def recipe():
    old, sampler = multi3_recipe(), sampler_recipe()
    return artifact('RECIPE', parents=dict(multi3=old['content_hash'], sampler=sampler['content_hash']),
        multi3=old, sampler=sampler, epochs=20, batch_size=128, ce_weight=1.,
        embedding_weights={t:[10,3] for t in TEACHERS}, matching_teacher='OFFLINE',
        matching_strategy='mix', matching_coefficient=.3, matching_ramp_epochs=3,
        matching_head_seed=MATCH_SEED, matching_head=[128,128,128], temperature=.1,
        key_normalization='FP64_TRAIN_population_std_floor_0.001_then_FP32_L2_eps1e-12',
        mining='exact_reference_chunk8_reductions_stable_GPU_lexsort_one_index_readback', candidate_lookahead_batches=1,
        candidate_cpu_workers=3,
        deployed_graph='native_HLT', historical_control='completed_multi3',
        scientific_fits=1, teacher_fits=0, preflight_jobs=0)


def node(parent):
    old = parent['nodes']['train_multi3']
    if (old['coordinate'] != 'D000' or old['context_coordinate'] is not None
            or old['embedding_targets'] != list(TEACHERS) or old['total_embedding_weight'] != 10.):
        raise ValueError('Expected completed native HLT Multi3 node')
    return dict(old, node_id=TASK, branch='FULLSTAT_MULTI3_CONTRASTIVE',
                matching_teacher='OFFLINE', matching_coefficient=.3)


def coefficient(update, steps):
    return ramp(.3, update, steps)
