"""Group C v1: whole-vector densities and full-set velocity denoising."""
from .contracts import artifact, ORDER, recipe as base_recipe

BASE='3b839fc3aef7e1e24e886b9dd512a6a8114b5473'
AUTHORIZATION='AUTHORIZE_NEXT_SUPERVISION_GROUP_C_V1'
ARMS={}
for name in ORDER:
    if name not in ('D1','D2','D3','D5','S1','S2','S5'): continue
    family=name[0]
    ARMS[name]=dict(family=family,source='U100' if family=='D' else 'OFFLINE',
        coefficient=.3 if family=='D' else .1,seed_index=int(name in ('D2','S2')),
        components=1 if name=='D5' else 4 if family=='D' else 0,
        law='probabilistic',group_size=0,independent=False,dot=False)
ARMS['D3']['coefficient']=.1
ARMS['S5']['source']='HLT'
TASKS=tuple(ARMS)
WIDTH=22
CONTINUOUS=(0,1,2,3,4,14,15,16,17)
GROUPS=((0,5),(5,14),(14,22))
FEATURES=('log_pt_over_HLT_pt','log_energy_over_HLT_pt','HLT_deta','HLT_sin_dphi','HLT_cos_dphi',
    'pid0','pid1','pid2','pid3','pid4','pid_unknown','charge_minus','charge_zero','charge_plus',
    'asinh_d0_mm','asinh_dz_mm','log_d0err_mm','log_dzerr_mm',
    'valid_d0','valid_dz','valid_d0err','valid_dzerr')


def recipe():
    return artifact('PROBABILISTIC_RECIPE',parents=dict(base=base_recipe()['content_hash']),
        arms=ARMS,global_order=list(ORDER),embedding_weight=5.,epochs=30,batch_size=128,
        ramp_epochs=3,auxiliary_seed=2026110507,target_seed=2026110510,
        density=dict(width=128,hidden=[256,256],components=[4,1],std_bounds=[.1,2.],initial_std=1.,
            likelihood='FP32_joint_coordinate_sum_then_component_logsumexp_then_divide128',
            initialization=dict(algorithm='sklearn_MiniBatchKMeans',seed=2026110509,n_init=3,
                batch_size=4096,max_iter=100,max_no_improvement=20,reassignment_ratio=0.,tol=0.),
            residual_weight_std=.001,negative_nll_valid=True),
        denoising=dict(features=list(FEATURES),continuous=list(CONTINUOUS),groups=[list(g) for g in GROUPS],
            frame='paired_HLT_sum_p4_axis_and_vector_pt',length_unit='mm',positive_floor=1e-12,
            normalization='source_specific_all_TRAIN_particle_masked_FP64_mean_std_floor1e-3',
            width=128,blocks=4,heads=4,feed_forward=256,dropout=0.,positional_ids=False,
            conditioning='pooled_HLT128_and_t_alpha_sigma_FiLM_each_block',noise_range=[.2,.95],
            target='cosine_schedule_velocity_all_numeric_channels',
            reduction='masked_per_jet_group_mean_then_equal3_groups_then_jets',
            count_weight=.1,count_normalization='log_source_TRAIN_max_count_plus1',
            full_collection=True,loss_chunk_jets=8,padding='batch_max_with_explicit_real_mask'),
        diagnostics='first_TRAIN_batch_same_noise_zero_global_and_within_class_coarse_HLT_kinematics_shuffle',
        deployment='unchanged_native_HLT_only_all_auxiliary_modules_removed',quality_gate=False)
