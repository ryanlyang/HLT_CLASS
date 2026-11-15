"""Six fixed recipes across three fresh, matched seeds on four Pascal GPUs."""
from copy import deepcopy
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS

DONOR = '58d0c0edf5f41cff2398fb168fc16fc92e169294'
BENCHMARK = '1094dea4ccfb36d08e29f740a97d90fc770335ec'
CURRICULUM = 'a549fef16390f7ef5b5b426e8dae6f2951876d89'
IMPLEMENTATION_DONOR = '4a61a34c97888d3ad5f479e29a5b94f9b6afa22a'
AUTHORIZATION = 'AUTHORIZE_RECONSTRUCTION_REPEAT_18_FOUR_1080TI_V1'
QUALIFICATION_PROTOCOL = 'common_start_six_recipes/v1'
LANES = ('pascal',)
SEED_INDICES = (4, 5, 6)
RECIPES = ('C', 'R24', 'R20', 'R22', 'R28', 'D20')
FITS = tuple(f'{r}_s{s}' for s in SEED_INDICES for r in RECIPES)
QUALIFICATION_TASKS = tuple(f'{r}_s4' for r in RECIPES)
LATE_EPOCHS = tuple(range(21, 31))
RECOVERY_THRESHOLD = 52.


def cell(task):
    if task not in FITS: raise ValueError('Unknown reconstruction repeat task')
    name, index = task.split('_s')
    return dict(recipe=name, seed_index=int(index), control=name=='C', lane='pascal',
        world_size=4, worker='curriculum', precision='fp32',
        delay=3. if name in ('R24','D20') else 0.,
        ramp=6. if name=='D20' else 3., shape='cosine' if name=='D20' else 'linear',
        maximum={'C':0.,'R20':2.,'R22':10.}.get(name,.3),
        decoder_warm=name=='D20', token_layers=[4,8] if name=='R28' else [8])


def tasks(hardware):
    if hardware not in LANES: raise ValueError('Unknown hardware lane')
    return FITS


def lane(task):
    if task in (*FITS,'setup_pascal','release_pascal','report_pascal'): return 'pascal'
    raise ValueError('Unregistered task')


def control(task): return f'C_s{cell(task)["seed_index"]}'


def original_task(task):
    name=cell(task)['recipe']
    return {'C':'C0','D20':'R24'}.get(name,name)


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
    return with_content_hash(dict(contract=f'LUKA_RECONSTRUCTION_REPEAT_{kind}/v1',schema_version=1,
        parents=dict(parents or {}),**FLAGS,**values))


def validate(value,kind,parents=None):
    validate_content_hash(value,expected_contract=f'LUKA_RECONSTRUCTION_REPEAT_{kind}/v1',expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value['parents']!=parents):
        raise ValueError('Repeat artifact scope or parents differ')


def recipe():
    return artifact('RECIPE',cells={t:cell(t) for t in FITS},order=list(FITS),epochs=30,global_batch=128,
        scientific_configs={t:scientific_cfg(t) for t in FITS},ce_weight=1.,embedding_weight=5.,
        decoder='original_R8_chunk8_equal_fields',original_worker=DONOR,curriculum_donor=CURRICULUM,
        implementation_donor=IMPLEMENTATION_DONOR,selection='original_selection_key',
        deployment='native_HLT_only',dropout='reset_after_init_rank_offset',
        validation='full_development_validation',statistical_scope='three_fresh_paired_seeds',
        late_epochs=list(LATE_EPOCHS),recovery_threshold=RECOVERY_THRESHOLD,
        precision='fp32_tf32_disabled',gradients='sample_weighted_SUM',normalization='SyncBatchNorm')
