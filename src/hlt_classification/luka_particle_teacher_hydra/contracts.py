"""Twelve native HLT fits with removable, aligned oracle-token predictors."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

BASE='cc55765a5e6a80ed04b660d99dc182a4aaf4b017'
AUTHORIZATION='AUTHORIZE_PARTICLE_TEACHER_HYDRA_V1'
ARMS={name:dict(depths=list(depths),coefficient=weight,seed_index=seed,centered=centered)
    for name,depths,weight,seed,centered in (
        ('final_l1_s0',(8,),1.,0,False),('control_s0',(),0.,0,False),
        ('final_l0p3_s0',(8,),.3,0,False),('multi_l0p3_s0',(6,8),.3,0,False),
        ('final_l1_s1',(8,),1.,1,False),('control_s1',(),0.,1,False),
        ('middle_l0p3_s0',(6,),.3,0,False),('final_l3_s0',(8,),3.,0,False),
        ('centered_l1_s0',(8,),1.,0,True),('multi_l1_s0',(6,8),1.,0,False),
        ('middle_l1_s0',(6,),1.,0,False),('final_l0p1_s0',(8,),.1,0,False))}
TASKS=tuple(ARMS)


def artifact(kind,*,parents=None,**fields):
    return with_content_hash(dict(contract=f'LUKA_PARTICLE_TEACHER_HYDRA_{kind}/v1',
        schema_version=1,parents=dict(parents or {}),**FLAGS,**fields))


def validate(value,kind,parents=None):
    validate_content_hash(value,expected_contract=f'LUKA_PARTICLE_TEACHER_HYDRA_{kind}/v1',expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value['parents']!=parents):
        raise ValueError('Particle-teacher scope or parents differ')


def recipe():
    return artifact('RECIPE',base=base_recipe(),arms=ARMS,epochs=30,batch_size=128,
        embedding_weight=5.,student_depth=8,teacher_depths=[6,8],particle_head=[128,'GELU',128],
        particle_head_seed=2026110501,seed_offset=1000003,ramp_epochs=3.,technical_position=7.,
        normalization='TRAIN_real_particle_FP64_population_mean_std_floor1e-3',std_floor=.001,
        loss='SmoothL1_beta1_mean_channels_real_particles_depths_jets',centered_mixture=.5,
        target_forward='frozen_teacher_m16_oracle_with_privileged_injection',
        target_cache='allocation_local_FP32_fixed_batches128',selection='HLT_only_full_validation',
        deployment='native_HLT_all_auxiliary_heads_removed',logit_kd=False)


def coefficient(setting,position):
    import math
    if not math.isfinite(position) or position<0: raise ValueError('Invalid schedule position')
    return setting['coefficient']*min(position/3.,1.)


def seed(imported,setting,kind):
    return imported['node'][kind]+1000003*setting['seed_index']
