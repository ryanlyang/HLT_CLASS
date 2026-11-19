"""Frozen smaller-population and eighteen-fit scientific contract."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_sixclass_balanced_oscar import contracts as parent
from hlt_classification.luka_fullstat.recipe import recipe as optimizer_recipe

PREFIX='LUKA_BALANCED_RECONSTRUCTION_HYDRA'
FLAGS=dict(parent.FLAGS)
SEEDS=(10,11)
ARMS=('OFFLINE','M0HLT','U100','U100_W5','R24','D20','R20','R22','R28')
FITS=tuple(f'{arm}_s{seed}' for seed in SEEDS for arm in ARMS)
READOUTS=('RAW','RAW_BN','EMA05','EMA10')
COUNTS=dict(train=600000,validation=150000)
BATCH=128
EPOCHS=20
BN_ROWS=8192
SUBSET_SEED=2026111901
POPULATION_HASH='21175a03adc3d847e62d84a447b04525fc813207685a37d0b0dcb630acafd1f1'
CLASS_NAMES=parent.CLASS_NAMES
EDGES=parent.EDGES
BASE='/cs/home/rlyang/hlt_u100_pairs'
DATASET=BASE+'/datasets/luka_qcdrun2com_higgs2p_sb_v1'
AUTH='AUTHORIZE_BALANCED_RECONSTRUCTION_18_FOUR_1080TI_V1'
SITE=dict(cluster='hydra',partition='gpus',account='default',qos='normal')
RESOURCES=dict(gpus=4,gpu_type='geforce_gtx_1080_ti',gpu_name='NVIDIA GeForce GTX 1080 Ti',
               cpus=12,memory_mb=64000,minutes=2880)
RESULT_BYTES=8*2**30  # Audited full inventory is4.50 GiB; retain2 GiB separate headroom.
RESERVE_BYTES=2*2**30
OTHER_ROOT=BASE+'/runs/sixclass_tail_504a8314_r1'
OTHER_RESERVATION=14*2**30


def artifact(kind,*,parents=None,**fields):
    if set(fields).intersection(('contract','schema_version','parents',*FLAGS)):
        raise ValueError('Reserved artifact fields')
    return with_content_hash(dict(contract=f'{PREFIX}_{kind}/v1',schema_version=1,
                                  parents=dict(parents or {}),**FLAGS,**fields))


def validate(value,kind,parents=None):
    validate_content_hash(value,expected_contract=f'{PREFIX}_{kind}/v1',expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value.get('parents')!=parents):
        raise ValueError('Scientific scope/parents differ')
    return value


def cell(task):
    if task not in FITS:raise ValueError('Unregistered scientific fit')
    arm,seed=task.rsplit('_s',1)
    reco=arm in ('R24','D20','R20','R22','R28')
    return dict(arm=arm,seed=int(seed),coordinate='OFFLINE' if arm=='OFFLINE' else 'U100' if arm=='U100' else 'D000',
        embedding_weight=5. if arm in ('U100_W5','R24','D20','R20','R22','R28') else 0.,
        main_kd_weight=.75 if arm=='U100' else 0.,ce_weight=.25 if arm=='U100' else 1.,
        reconstruction=reco,delay=3. if arm in ('R24','D20') else 0.,
        ramp=6. if arm=='D20' else 3.,shape='cosine' if arm=='D20' else 'linear',
        maximum={'R20':2.,'R22':10.}.get(arm,.3) if reco else 0.,
        decoder_warm=arm=='D20',token_layers=[4,8] if arm=='R28' else [8],
        teacher=f'OFFLINE_s{seed}' if arm=='U100' else f'U100_s{seed}' if arm not in ('M0HLT','OFFLINE') else None)


def seeds(task):
    index=cell(task)['seed']
    return dict(native=2026111911+1000003*index,predictor=2026111913+1000003*index,
                dropout=2026111917+1000003*index,sampler=2026111921+1000003*index,
                decoder_dropout=2026111923+1000003*index,bn=2026111927)


def weights(task,position):
    if not math.isfinite(position) or not 0<=position<=EPOCHS:raise ValueError('Invalid draw clock')
    x=cell(task);f=min(1.,max(0.,(position-x['delay'])/x['ramp']))
    if x['shape']=='cosine':f=.5*(1-math.cos(math.pi*f))
    enc=x['maximum']*f
    return enc,x['maximum'] if x['decoder_warm'] else enc


def decoder_config(task):
    x=cell(task)
    return dict(family='R',source='OFFLINE',memory='tokens',seed_index=x['seed'],
                token_layers=x['token_layers'],field_weights=(1.,1.,1.))


def tasks():
    result=dict(prepare=[],qualify=['prepare'],release=['qualify'])
    for task in FITS:
        teacher=cell(task)['teacher'];result[task]=[teacher] if teacher else []
    result['report']=list(FITS)
    return result


def recipe():
    # Flags already declare final-test sealing; never duplicate reserved fields.
    return artifact('RECIPE',population_parent=parent.POPULATION_HASH,counts=COUNTS,
        class_names=list(CLASS_NAMES),edges=list(EDGES),subset_seed=SUBSET_SEED,
        subset='proportional_largest_remainders_then_sha256_identity_rank_preserve_parent_order',
        sampling='uniform_120_cells_then_uniform_row_with_replacement',draws=COUNTS['train'],
        epochs=EPOCHS,batch=BATCH,clock='actual_draws_consumed_over_draws_per_epoch',
        seeds=list(SEEDS),arms=list(ARMS),fits=list(FITS),nodes={t:cell(t) for t in FITS},
        optimizer=optimizer_recipe(),precision='FP32_TF32_off',normalization='SyncBatchNorm_global128',
        decoder='original_full_support_four_block256_chunk8_exact_assignment',
        zero_weight_decoder='explicit_zero_gradients_preserve_original_AdamW_steps_and_decay',
        target='selected_TRAIN_mixed22_FP64_moments_std_floor_1e-3_paired_HLT_frame',
        target_support='max_selected_TRAIN_OFFLINE_and_HLT_count',
        teacher_readout='EMA05',teacher_temperature=2.,readouts=list(READOUTS),primary='EMA05',
        ema_start=3.,ema_half_lives=[.5,1.],bn_rows=BN_ROWS,bn='fixed_balanced_TRAIN_draws',
        selection='macro_OVR_AUC_then_CE_then_macro_log_R50_then_earliest',late_epochs=list(range(16,21)))
