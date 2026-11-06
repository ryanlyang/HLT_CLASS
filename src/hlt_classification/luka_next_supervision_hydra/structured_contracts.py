"""Group B's frozen target construction and nine selected scientific rows."""
from .contracts import ORDER, artifact, recipe as group_a_recipe

BASE='342e5d3affba78f9d6add4cbb95d6264fc4b516b'
ARMS={}
for name in ORDER:
    if name[0] not in 'QV': continue
    ARMS[name]=dict(family=name[0],source='OFFLINE',coefficient=.3,seed_index=0,
        law='structured',group_size=0,independent=False,dot=False,
        question_families=[0,1,2,3],count_only=False)
for name in ('Q2','V2'): ARMS[name]['seed_index']=1
ARMS['Q3']['coefficient']=.1
ARMS['Q5']['source']='HLT'; ARMS['V6']['source']='HLT'
ARMS['Q6']['question_families']=[0,1]
ARMS['V7']['count_only']=True
TASKS=tuple(n for n in ORDER if n in ARMS)
AUTHORIZATION='AUTHORIZE_NEXT_SUPERVISION_GROUP_B_V1'
QUESTION_COLUMNS=('family','category','charge','field','radius_low','radius_high',
    'fraction_low','fraction_high','displacement_threshold','minimum_count',
    'separation_low','separation_high','pair_charge')
SEMANTIC_WIDTH=31


def recipe():
    return artifact('STRUCTURED_RECIPE',group_a_science=group_a_recipe()['content_hash'],arms=ARMS,
        global_order=list(ORDER),implemented_group='B',epochs=30,batch_size=128,embedding_weight=5.,
        queries=64,question_proposals_per_family=2048,question_goal_per_family=1024,
        question_columns=list(QUESTION_COLUMNS),semantic_width=SEMANTIC_WIDTH,
        question_seed=2026110511,query_seed=2026110512,design_jets=32768,particles_per_jet=8,
        question_categories='global_TRAIN_measured_category_charge_support_per_valid_field',
        statistics_batch_size=32,statistics_outstanding_batches=8,
        measurement_tail='largest128_source_counts_largest128_valid_charged_plus_native_extremes_unique_partial_ge2',
        support='full_TRAIN_exact_min_max_sums_no_rare_prevalence_cut',
        constant_family='fixed_proposals_with_zero_applicability_reported_not_failed',
        catalog_shortfall='retain_all_supported_report_no_duplicates_to_fill_goal',
        query_sampling='identity_epoch_replicate_family_32_draws_then_prefix_source_independent',
        question_frame='paired_HLT_sum_p4_axis_source_scalar_pt_fractions',
        pair_chunk=512,question_chunk=64,codebook_size=128,
        dictionary='MiniBatchKMeans_kmeans++_ninit3_maxiter100_batch4096_max_no_improvement20_reassignment0',
        dictionary_seed=2026110513,assignment='FP64_direct_squared_distance_argmin_first_chunk1024',
        feature_weighting='equal_TRAIN_mean_squared_norm_for_kinematic_composition_tracking_blocks',
        dictionary_reservoir='lowest_identity_SHA256_jets_uniform_up_to8_particles_per_jet',
        dictionary_labels_used=False,frequency_heads=['count','scalar_pt'],
        frequency_loss='mean_heads_mean_jets_sum_codes_soft_cross_entropy_FP32',
        question_loss='mean_selected_families_mean_questions_BCE_logits_FP32',
        ramp_epochs=3,deployed_architecture='native_HLT_unchanged',target_role='TRAIN_only')
