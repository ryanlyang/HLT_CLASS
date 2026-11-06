"""Additive direct measured-set reconstruction; the original forty rows are frozen."""
from .contracts import artifact as common_artifact, validate as common_validate, recipe as original_recipe
from .probabilistic_contracts import WIDTH, CONTINUOUS, FEATURES

BASE='505b07364041d4f51c42ac9e6a2f85353258606a'
SCOPE='direct_reconstruction_only'
AUTHORIZATION='AUTHORIZE_DIRECT_RECONSTRUCTION_AUTO_V2'
TASKS=('R1','R2','R3','R4','R11','R12','R5','R6','R7','R8','R9','R10')
CHUNK=8


def setting(task):
    rows={
        'R1':('global','OFFLINE',False,.1,0), 'R2':('tokens','OFFLINE',False,.1,0),
        'R3':('global','OFFLINE',False,.1,1), 'R4':('tokens','OFFLINE',False,.1,1),
        'R5':('global','OFFLINE',False,.03,0), 'R6':('tokens','OFFLINE',False,.03,0),
        'R7':('global','OFFLINE',False,.3,0), 'R8':('tokens','OFFLINE',False,.3,0),
        'R9':('global','OFFLINE',True,.1,0), 'R10':('tokens','OFFLINE',True,.1,0),
        'R11':('global','HLT',False,.1,0), 'R12':('tokens','HLT',False,.1,0)}
    if task not in rows: raise NotImplementedError(task)
    memory,source,consistency,weight,seed=rows[task]
    return dict(family='R',memory=memory,source=source,consistency=consistency,coefficient=weight,
        seed_index=seed,law='exact_set_reconstruction',dot=False,group_size=0,independent=False)


def artifact(kind,**fields):
    return common_artifact('DIRECT_RECONSTRUCTION_'+kind,**fields)


def validate(value,kind,parents=None):
    return common_validate(value,'DIRECT_RECONSTRUCTION_'+kind,parents)


def recipe():
    return artifact('RECIPE',parents=dict(native_recipe=original_recipe()['content_hash']),
        tasks=list(TASKS),settings={t:setting(t) for t in TASKS},embedding_weight=5,
        decoder=dict(width=256,blocks=4,heads=8,feedforward=1024,dropout=0.,norm_first=True,
            query_std=.02,output_std=.001,auxiliary_seed=2026110601,seed_offset=1000003,
            uniform_heads=['category','charge','count'],presence_prior=.1,
            initialization_order=['queries','jet','optional_tokens','blocks','film','norm','field_heads','count','output_initialization'],
            film='identity_initialized_per_block',memory='global_or_global_plus_block8',
            precision='FP32',chunk_size=CHUNK,count='global_only_0_through_K'),
        targets=dict(fields=list(FEATURES),width=WIDTH,continuous=list(CONTINUOUS),role='TRAIN',
            full_collection=True,frame='paired_HLT_sum_p4_axis_and_vector_pt',std_floor=.001,
            support='max_OFFLINE_and_HLT_TRAIN_count_minimum1',particle_matching='none'),
        objective=dict(assignment='scipy_linear_sum_assignment_exact_float64_detached_cost',
            assignment_cost='field/N+(softplus(-presence)-0.1*softplus(presence))/(N+0.1*(K-N))',
            fields='mean(kinematic,composition,track)',empty_weight=.1,count_weight=.1,
            consistency_weight=.1,coefficient_ramp_epochs=3,huber_beta=1.),
        inference='unchanged_native_HLT_no_decoder',selection='native_validation_AUC_then_CE_then_earliest',
        quality_gate=False)
