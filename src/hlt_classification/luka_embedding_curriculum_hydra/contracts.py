"""Twenty distinct rows; stage clocks and optimizer choices are scientific identity."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import recipe as original_recipe, COUNTS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate as original_lr

BASE = 'b5464e73c94b322a81a33032baea18f1615a5a66'
AUTHORIZATION = 'AUTHORIZE_U100_EMBEDDING_CURRICULUM_V1'
TASKS = tuple(f'E{i:02d}' for i in range(20))
ORDER = tuple(f'E{i:02d}' for i in (1,6,12,19,3,15,17,18,2,4,7,8,9,10,11,13,14,16,5))


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_CURRICULUM_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_CURRICULUM_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()) or (parents is not None and value['parents'] != parents):
        raise ValueError('Curriculum scope or parents differ')


def setting(task):
    if task not in TASKS: raise ValueError('Unknown curriculum')
    c=dict(pretrain=10, switch=10, pre_ce=0., ramp=5., shape='linear', ce_max=1.,
           reset_optimizer=True, restart_lr=True, selection='embedding_best', embedding_end=5., fade_epochs=0.)
    if task=='E00': c.update(pretrain=0,switch=0,pre_ce=1.,ramp=0.,reset_optimizer=False,restart_lr=False,selection='none')
    if task in ('E02','E03','E04','E05'):
        n={'E02':3,'E03':5,'E04':15,'E05':20}[task]; c.update(pretrain=n,switch=n)
    if task in ('E06','E07','E08'): c['ramp']={'E06':0.,'E07':2.,'E08':10.}[task]
    if task=='E09': c['shape']='cosine'
    if task=='E10': c['shape']='steps'
    if task=='E11': c['ce_max']=2.
    if task in ('E12','E13'): c['reset_optimizer']=False
    if task in ('E12','E14'): c['restart_lr']=False
    if task=='E15': c.update(embedding_end=1.,fade_epochs=15.)
    if task=='E16': c.update(embedding_end=0.,fade_epochs=10.)
    if task=='E17': c['pre_ce']=.05
    if task=='E18': c['selection']='last'
    if task=='E19': c.update(pretrain=0,switch=10,pre_ce=1.,ramp=0.,selection='last')
    return c


def weights(task, position):
    if not math.isfinite(position) or not 0 <= position <= 30: raise ValueError('Invalid epoch clock')
    c=setting(task)
    if position <= c['switch']: return c['pre_ce'],5.
    s=position-c['switch']
    if c['shape']=='steps': ce=.1 if s<=2 else .3 if s<=5 else 1.
    else:
        f=min(s/c['ramp'],1.) if c['ramp'] else 1.
        if c['shape']=='cosine': f=.5*(1-math.cos(math.pi*f))
        ce=c['pre_ce']+(c['ce_max']-c['pre_ce'])*f
    fade=min(max(s-c['ramp'],0.)/c['fade_epochs'],1.) if c['fade_epochs'] else 0.
    return ce,5.+(c['embedding_end']-5.)*fade


def learning_rate(task, position):
    c=setting(task)
    if not math.isfinite(position) or not 0 < position <= 30: raise ValueError('Invalid LR clock')
    if not c['restart_lr'] or position<=c['switch']: return original_lr(position)
    s=position-c['switch']; length=30-c['switch']
    if s<=2: return 3e-4*s/2
    return 1.5e-5+(3e-4-1.5e-5)*.5*(1+math.cos(math.pi*(s-2)/(length-2)))


def recipe():
    return artifact('RECIPE',baseline=original_recipe(),settings={t:setting(t) for t in TASKS},order=list(ORDER),
        epochs=30,batch_size=128,seed_index=0,predictor='original_AuxiliaryParT',
        loss='FP32_standardized_SmoothL1_beta1_mean',validation_targets='frozen_COARSE_U100_FP32_TRAIN_statistics',
        checkpoint_tie='earliest_minimum_embedding_validation_loss',final_selection='original_native_AUC_tiebreaks_post_pretraining',
        restart_warmup_epochs=2,clock='spent_epoch_clock_never_rewound',rng='restore_selected_model_CPU_CUDA_rng_at_transition',
        sampling='original_initialization_and_absolute_epoch_sampler_seeds',
        independent_fits=True,pretraining_shared=False,deployment='native_HLT_only',reference_retrained=False)
