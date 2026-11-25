"""Full-population, three-seed RAW science and explicit five-epoch segments."""
import math
from hlt_classification.data.cache_contracts import with_content_hash,validate_content_hash
from hlt_classification.luka_sixclass_wide_b256_oscar import contracts as population_parent
from hlt_classification.luka_fullstat.recipe import recipe as base_optimizer

PREFIX='LUKA_D20_FULL_HYDRA'
FLAGS=dict(population_parent.FLAGS)
SEEDS=(20,21,22)
BASE_ARMS=('OFFLINE','M0HLT','U100','U100_W5','D20')
VARIANTS={'D20_E01':dict(maximum=.1),'D20_E06':dict(maximum=.6),
    'D20_D1':dict(delay=1.),'D20_D5':dict(delay=5.),
    'D20_R3':dict(ramp=3.),'D20_R9':dict(ramp=9.),
    'D20_NO_PREP':dict(decoder_warm=False),'D20_EMB25':dict(embedding_weight=2.5)}
ARMS=(*BASE_ARMS,*VARIANTS)
FITS=tuple(f'{arm}_s{seed}' for seed in SEEDS for arm in BASE_ARMS)+tuple(f'{arm}_s{SEEDS[0]}' for arm in VARIANTS)
READOUTS=('RAW','EMA05')
COUNTS=dict(train=2488148,validation=497954)
BATCH,EPOCHS,BN_ROWS=256,20,8192
EDGES=tuple(range(200,2601,100))
CLASS_NAMES=population_parent.CLASS_NAMES
POPULATION_HASH=population_parent.POPULATION_HASH
BASE='/cs/home/rlyang/hlt_u100_pairs'
DATASET=BASE+'/datasets/luka_qcdrun2com_higgs2p_sb_v1'
AUTH='AUTHORIZE_D20_FULL_23_FOUR_1080TI_V1'
SITE=dict(cluster='hydra',partition='gpus',account='default',qos='normal')
RESOURCES=dict(gpus=4,gpu_type='geforce_gtx_1080_ti',gpu_name='NVIDIA GeForce GTX 1080 Ti',
    cpus=12,memory_mb=96000,minutes=2880)
QUALIFICATION_MINUTES=2880
RESULT_BYTES,RESERVE_BYTES=24*2**30,2*2**30
SEGMENTS=4
GPU_TASKS=tuple(f'{fit}_p{part:02d}' for fit in FITS for part in range(1,SEGMENTS+1))


def artifact(kind,*,parents=None,**fields):
    if set(fields)&{'contract','schema_version','parents',*FLAGS}:raise ValueError('Reserved artifact fields')
    return with_content_hash(dict(contract=PREFIX+'_'+kind+'/v1',schema_version=1,
        parents=dict(parents or {}),**FLAGS,**fields))


def validate(value,kind,parents=None):
    validate_content_hash(value,expected_contract=PREFIX+'_'+kind+'/v1',expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value.get('parents')!=parents):
        raise ValueError('Scientific scope/parents differ')
    return value


def cell(task):
    if task not in FITS:raise ValueError('Unregistered scientific fit')
    arm,seed=task.rsplit('_s',1);reco=arm=='D20' or arm in VARIANTS
    result=dict(arm=arm,seed=int(seed),coordinate='OFFLINE' if arm=='OFFLINE' else 'U100' if arm=='U100' else 'D000',
        deployable=arm not in ('OFFLINE','U100'),
        embedding_weight=5. if reco or arm=='U100_W5' else 0.,reconstruction=reco,
        main_kd_weight=.75 if arm=='U100' else 0.,ce_weight=.25 if arm=='U100' else 1.,
        delay=3.,ramp=6.,shape='cosine',maximum=.3 if reco else 0.,
        decoder_maximum=.3 if reco else 0.,decoder_warm=reco,token_layers=[8],
        teacher=f'OFFLINE_s{seed}' if arm=='U100' else f'U100_s{seed}' if reco or arm=='U100_W5' else None)
    result.update(VARIANTS.get(arm,{}));return result


def seeds_for(arm):return SEEDS if arm in BASE_ARMS else (SEEDS[0],) if arm in VARIANTS else ()


def readouts(task):return READOUTS if cell(task)['arm'] in ('OFFLINE','U100') else ('RAW',)


def seeds(task):
    index=cell(task)['seed']
    return dict(native=2026112411+1000003*index,predictor=2026112413+1000003*index,
        dropout=2026112417+1000003*index,sampler=2026112421+1000003*index,
        decoder_dropout=2026112423+1000003*index,bn=2026112427)


def weights(task,position):
    if not math.isfinite(position) or not 0<=position<=EPOCHS:raise ValueError('Invalid draw clock')
    x=cell(task);f=min(1.,max(0.,(position-x['delay'])/x['ramp']))
    f=.5*(1-math.cos(math.pi*f));enc=x['maximum']*f
    return enc,x['decoder_maximum'] if x['decoder_warm'] else enc


def learning_rate(position):
    if not math.isfinite(position) or not 0<position<=EPOCHS:raise ValueError('Invalid draw clock')
    if position<=2:return 1e-4*position/2
    return 5e-6+(1e-4-5e-6)*.5*(1+math.cos(math.pi*(position-2)/18))


def decoder_config(task):
    x=cell(task)
    return dict(family='R',source='OFFLINE',memory='tokens',seed_index=x['seed'],token_layers=[8],field_weights=(1.,1.,1.))


def segment(task):
    if task not in GPU_TASKS:raise ValueError('Unregistered scientific segment')
    fit,part=task.rsplit('_p',1);part=int(part)
    return fit,part,(part-1)*5+1,part*5


def tasks():
    result=dict(prepare=[],qualify=['prepare'],release=['qualify'])
    for task in GPU_TASKS:
        fit,part,_,_=segment(task);teacher=cell(fit)['teacher']
        result[task]=[f'{fit}_p{part-1:02d}'] if part>1 else [teacher+'_p04'] if teacher else []
    result['report']=list(GPU_TASKS);return result


def recipe():
    base=base_optimizer(256)
    opt=artifact('OPTIMIZER',algorithm='AdamW',betas=base['betas'],eps=base['eps'],
        weight_decay=base['weight_decay'],peak_lr=1e-4,floor_lr=5e-6)
    return artifact('RECIPE',population_sha256=POPULATION_HASH,counts=COUNTS,class_names=list(CLASS_NAMES),
        edges=list(EDGES),sampling='uniform_144_cells_then_uniform_TRAIN_row_with_replacement',draws=COUNTS['train'],
        epochs=EPOCHS,batch=BATCH,last_batch=84,updates_per_epoch=9720,clock='actual_consumed_draws',
        peak_lr=1e-4,floor_lr=5e-6,warmup=2,decay_through=20,seeds=list(SEEDS),fits=list(FITS),
        nodes={t:cell(t) for t in FITS},optimizer=opt,precision='FP32_TF32_off',normalization='SyncBatchNorm_global256',
        decoder='original_full_support_four_block256_chunk8_exact_assignment',
        zero_weight_decoder='explicit_zero_gradients_preserve_AdamW_step_and_decay',
        target='all_eligible_TRAIN_mixed22_FP64_per_jet_moments_std_floor_1e-3_paired_HLT_frame',
        teacher_readout='EMA05',teacher_temperature=2.,primary='RAW',readouts={t:list(readouts(t)) for t in FITS},
        ema_start=3.,ema_half_life=.5,bn_rows=BN_ROWS,bn='fixed_balanced_TRAIN_draws_128_batch',
        bank_standardization='fixed_65536_row_two_pass_FP64',segments=4,epochs_per_segment=5,
        resume='completion_last_epoch_full_model_optimizer_teacher_EMA_selected_states_rank_RNG',
        selection='macro_OVR_AUC_then_CE_then_macro_log_R50_then_earliest',late_epochs=list(range(16,21)))
