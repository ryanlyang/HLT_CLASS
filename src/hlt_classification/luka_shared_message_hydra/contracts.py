"""Immutable twelve-arm scientific and execution registration."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

BASE = '546077aa7fce428bf3f9f136ab2277ac5eed5f85'
AUTHORIZATION = 'AUTHORIZE_SHARED_MESSAGE_HYDRA_V1'
ARMS = {}
for name, mode, alpha, beta, seed, detach, fade in (
    ('joint_b0p1_s0','joint',.25,.1,0,False,False),
    ('control_s0','control',0.,0.,0,False,False),
    ('joint_b0p3_s0','joint',.25,.3,0,False,False),
    ('joint_b0p03_s0','joint',.25,.03,0,False,False),
    ('frozen_b0p1_s0','frozen',0.,.1,0,False,False),
    ('joint_b0p1_s1','joint',.25,.1,1,False,False),
    ('control_s1','control',0.,0.,1,False,False),
    ('joint_a0p1_s0','joint',.1,.1,0,False,False),
    ('joint_b1_s0','joint',.25,1.,0,False,False),
    ('joint_no_agreement_s0','joint',.25,0.,0,False,False),
    ('joint_student_only_s0','joint',.25,.1,0,True,False),
    ('joint_fade_s0','joint',.25,.1,0,False,True)):
    ARMS[name] = dict(mode=mode,alpha=alpha,beta=beta,seed_index=seed,detach_target=detach,fade=fade)
TASKS = tuple(ARMS)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_SHARED_MESSAGE_HYDRA_{kind}/v1',
        schema_version=1,parents=dict(parents or {}),**FLAGS,**fields))


def validate(value, kind, parents=None):
    validate_content_hash(value,expected_contract=f'LUKA_SHARED_MESSAGE_HYDRA_{kind}/v1',expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value['parents']!=parents):
        raise ValueError('Shared-message scope or parents differ')
    return value


def recipe():
    return artifact('RECIPE',base=base_recipe(),arms=ARMS,epochs=30,batch_size=128,
        block_index=3,width=16,hidden=128,heads=4,embedding_weight=5.,
        normalization='shared_nonaffine_layernorm_before_injection_and_agreement',
        agreement_warmup_epochs=3,agreement_full_epoch=6,fade_epochs=[10,20],
        seed_offset=1000003,head_seed=2026110403,technical_position=7.,
        selection='HLT_only_full_validation',logit_kd=False,
        deployment='native_HLT_plus_attention_predictor_and_shared_expansion')


def coefficients(setting, position):
    import math
    if not math.isfinite(position) or position < 0: raise ValueError('Invalid schedule position')
    fade = max(0.,min(1.,(20-position)/10)) if setting['fade'] else 1.
    return setting['alpha']*fade, setting['beta']*max(0.,min(1.,(position-3)/3))*fade


def seed(imported, setting, kind):
    return imported['node'][kind]+1000003*setting['seed_index']
