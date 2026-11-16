"""Fourteen fixed configurations across three fresh seeds on four Pascal GPUs."""
from copy import deepcopy
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS

DONOR = '58d0c0edf5f41cff2398fb168fc16fc92e169294'
BENCHMARK = '1094dea4ccfb36d08e29f740a97d90fc770335ec'
CURRICULUM = 'a549fef16390f7ef5b5b426e8dae6f2951876d89'
IMPLEMENTATION_DONOR = '15c35c428fa0548b97700d639323e593ea3aabac'
AUTHORIZATION = 'AUTHORIZE_RECONSTRUCTION_STABILITY_42_FOUR_1080TI_V1'
QUALIFICATION_PROTOCOL = 'stability_global_gradients_accumulation_readouts_transport/v1'
LANES = ('pascal',)
SEED_INDICES = (7, 8, 9)
RECIPES = ('C128','R24','G30','C_L50','R_L50','C_L25','R_L25','PROJ','DEC2','C256','R256','C512','R512','SOFT')
CONTROLS = ('C128','C_L50','C_L25','C256','C512')
PASSES = (('G30','C_L25','R_L25','C256','R256','SOFT'),('PROJ','C_L50','R_L50','C512','R512','DEC2'))
FITS = tuple(f'{r}_s{s}' for s in SEED_INDICES for r in ('C128','R24')) + tuple(
    f'{r}_s{s}' for group in PASSES for s in SEED_INDICES for r in group)
QUALIFICATION_TASKS = tuple(f'{r}_s7' for r in RECIPES)
READOUTS = ('RAW','RAW_BN','EMA05','EMA10')
LATE_EPOCHS = tuple(range(21,31))
RECOVERY_THRESHOLD = 52.
TRAIN_ROWS = 299234
MICROBATCH = 128
SOFT_PROTOCOL = dict(iterations=500, marginal_tolerance=1e-4, gap_tolerance=1e-4,
    log_relaxation=1.8, relaxation_start=50,
    calibration_rows=512, initialization_seed=2026111501, scale_floor=1e-6,
    temperature_start=.1, temperature_end=.01, start=3., end=9.)


def cell(task):
    if task not in FITS: raise ValueError('Unknown stability task')
    name,index=task.rsplit('_s',1)
    return dict(recipe=name,seed_index=int(index),control=name in CONTROLS,lane='pascal',
        world_size=4,worker='stability',precision='fp32',delay=3.,ramp=3.,shape='linear',
        maximum=0. if name in CONTROLS else .3,token_layers=[8],decoder_warm=False,
        accumulation=2 if name.endswith('256') else 4 if name.endswith('512') else 1,
        encoder_lr_multiplier=.5 if name.endswith('L50') else .25 if name.endswith('L25') else 1.,
        gradient_rule=name if name in ('G30','PROJ') else 'ordinary',
        decoder_extra_steps=2 if name=='DEC2' else 0,soft_assignment=name=='SOFT')


def tasks(hardware):
    if hardware not in LANES: raise ValueError('Unknown hardware lane')
    return FITS


def lane(task):
    if task in (*FITS,'setup_pascal','release_pascal','report_pascal'): return 'pascal'
    raise ValueError('Unregistered task')


def control(task):
    c=cell(task);name=c['recipe']
    base=name if c['control'] else {'R_L50':'C_L50','R_L25':'C_L25','R256':'C256','R512':'C512'}.get(name,'C128')
    return f'{base}_s{c["seed_index"]}'


def original_task(task):return 'C0' if cell(task)['control'] else 'R24'


def scientific_cfg(task):
    from hlt_classification.luka_next_supervision_hydra.contracts import setting
    cfg=deepcopy(setting(original_task(task)))
    cfg['seed_index']=cell(task)['seed_index']
    return cfg


def seeds(spec,task):
    from hlt_classification.luka_next_supervision_hydra.contracts import seed
    cfg=scientific_cfg(task)
    init=seed(spec['imported'],cfg,'initialization_seed')
    return dict(native_initialization=init,dropout_base=init,
        sampler=seed(spec['imported'],cfg,'sampler_seed'))


def weights(task,position):
    if not math.isfinite(position) or not 0<=position<=30: raise ValueError('Invalid epoch position')
    c=cell(task); f=min(1.,max(0.,(position-c['delay'])/c['ramp']))
    if c['shape']=='cosine': f=.5*(1.-math.cos(math.pi*f))
    encoder=c['maximum']*f
    return encoder,c['maximum'] if c['decoder_warm'] else encoder


def artifact(kind,*,parents=None,**values):
    return with_content_hash(dict(contract=f'LUKA_RECONSTRUCTION_STABILITY_{kind}/v1',schema_version=1,
        parents=dict(parents or {}),**FLAGS,**values))


def validate(value,kind,parents=None):
    validate_content_hash(value,expected_contract=f'LUKA_RECONSTRUCTION_STABILITY_{kind}/v1',expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value['parents']!=parents):
        raise ValueError('Stability artifact scope or parents differ')


def encoder_multiplier(task,position):
    return 1.-(1.-cell(task)['encoder_lr_multiplier'])*min(1.,max(0.,position-3.))


def recipe():
    return artifact('RECIPE',cells={t:cell(t) for t in FITS},order=list(FITS),epochs=30,global_batch=128,
        scientific_configs={t:scientific_cfg(t) for t in FITS},ce_weight=1.,embedding_weight=5.,
        decoder='original_R8_chunk8_equal_fields',original_worker=DONOR,curriculum_donor=CURRICULUM,
        implementation_donor=IMPLEMENTATION_DONOR,selection='original_selection_key',
        deployment='native_HLT_only',dropout='reset_after_init_rank_offset_isolated_decoder_and_readouts',
        validation='full_development_validation',statistical_scope='three_fresh_paired_seeds',
        late_epochs=list(LATE_EPOCHS),recovery_thresholds=[50.,52.],readouts=list(READOUTS),
        precision='fp32_tf32_disabled',gradients='sample_weighted_SUM_before_transform',normalization='SyncBatchNorm_micro128',
        clock='group_end_consumed_examples_over_TRAIN',shared_parameters='mod_except_mod.fc',
        gradient_budget=dict(rho=.3,ema_half_life_epochs=.1,eps=1e-12,instantaneous_cap=True),
        projection=dict(protect='CE_plus_U100_w5',zero_squared_norm=1e-20),
        decoder_extra_interval=[3.,9.],ema_half_lives=[.5,1.],ema_start=3.,bn_train_rows=8192,
        soft_protocol=SOFT_PROTOCOL,optimizer_resets=False,automatic_lr_scaling=False)
