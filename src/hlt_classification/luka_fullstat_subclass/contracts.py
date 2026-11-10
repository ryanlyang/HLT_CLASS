"""New scientific identity; historical recipes remain immutable."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat_multiview.contracts import recipe as multi3_recipe, TEACHERS
from hlt_classification.luka_u100_contrastive.contracts import sampler_recipe, ramp

TASK = 'train_subclass'
TASKS = {TASK: []}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SUBCLASS_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SUBCLASS_AUTOSUBMIT_V1'
MATCH_SEED = 2026103101


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_SUBCLASS_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_SUBCLASS_{kind}/v1',
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
        candidate_cpu_workers=3, subclass=subclass_recipe(),
        deployed_graph='native_HLT', historical_control='completed_multi3_contrastive',
        scientific_fits=1, teacher_fits=0, preflight_jobs=0)


def node(parent):
    old = parent['nodes']['train_multi3_mix']
    if (old['coordinate'] != 'D000' or old['context_coordinate'] is not None
            or old['embedding_targets'] != list(TEACHERS) or old['total_embedding_weight'] != 10.):
        raise ValueError('Expected completed native HLT Multi3 node')
    return dict(old, node_id=TASK, branch='FULLSTAT_SUBCLASS',
                matching_teacher='OFFLINE', matching_coefficient=.3)


def coefficient(update, steps):
    return ramp(.3, update, steps)


SUBCLASS_SEED = 2026110901
CLUSTER_SEED = 2026110902
COUNTS_PER_CLASS = (10,) + (5,)*10


def subclass_recipe():
    return artifact('SUBCLASS_RECIPE',teacher='OFFLINE',role='train',
        class_counts=list(COUNTS_PER_CLASS),head=[128,60],head_seed=SUBCLASS_SEED,
        coefficient=.5,ramp_epochs=3,loss='conditional_KL_teacher_to_student_batchmean',
        embedding_fraction=.75,margin_fraction=.25,margin='other_logit_minus_true_logit',
        embedding_transform='existing_FP64_TRAIN_population_standardization_floor_0.001',
        margins_transform='within_class_FP64_TRAIN_population_standardization_floor_0.001',
        logits='FP64_native_linear_readout_of_reconstructed_cached_FP32_embedding',
        dimension_scaling=True,algorithm='sklearn_MiniBatchKMeans',cluster_seed=CLUSTER_SEED,
        batch_size=4096,n_init=5,max_iter=40,max_no_improvement=20,reassignment_ratio=0.,
        threads=1,softmax='negative_squared_distance_over_class_temperature',
        temperature_goal=.8,temperature_bounds=[1e-8,1e8],temperature_bisections=48,
        degenerate_temperature='nearest_bound_record_achieved_no_quality_rejection',
        normalization_population='all_TRAIN',clustering_population='all_TRAIN',
        occupancy='unconstrained',target_dtype='float32',padded_width=10)
