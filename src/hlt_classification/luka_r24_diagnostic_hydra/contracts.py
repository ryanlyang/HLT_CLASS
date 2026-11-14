"""Immutable fourteen-fit matrix; no changes to historical campaign semantics."""
from copy import deepcopy
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS

DONOR = '58d0c0edf5f41cff2398fb168fc16fc92e169294'
BENCHMARK = '1094dea4ccfb36d08e29f740a97d90fc770335ec'
CURRICULUM = 'a549fef16390f7ef5b5b426e8dae6f2951876d89'
AUTHORIZATION = 'AUTHORIZE_R24_DIAGNOSTIC_SELECTED_LANES_V2'
VERSIONS = {'SPEC':2,'PLAN':2,'QUALIFICATION':2,'PARITY_STEP':2}
QUALIFICATION_PROTOCOL = 'same_start_update_and_observed_trajectory/v2'
LANES = ('pascal', 'a6000')
PAIRS = ('P0', 'PD', 'PO', 'PI', 'OB', 'NB', 'NF')
FITS = tuple(p+'_'+s for p in PAIRS for s in ('R','C'))
OFFSET = 1000003


def cell(task):
    if task not in FITS: raise ValueError('Unknown R24 diagnostic task')
    pair, kind = task.split('_')
    return dict(pair=pair, control=kind=='C', seed_index=0,
        lane='pascal' if pair.startswith('P') else 'a6000',
        world_size=4 if pair.startswith('P') else 1,
        worker='original' if pair=='OB' else 'curriculum',
        precision='bf16_native' if pair in ('OB','NB') else 'fp32',
        native_initialization_offset=OFFSET if pair=='PI' else 0,
        dropout_offset=OFFSET if pair=='PD' else 0,
        sampler_offset=OFFSET if pair=='PO' else 0,
        delay=3., ramp=3., maximum=0. if kind=='C' else .3)


def tasks(lane):
    if lane not in LANES: raise ValueError('Unknown hardware lane')
    return tuple(t for t in FITS if cell(t)['lane']==lane)


def lane(task):
    if task in FITS: return cell(task)['lane']
    for prefix in ('setup_', 'release_', 'report_'):
        if task.startswith(prefix) and task[len(prefix):] in LANES: return task[len(prefix):]
    raise ValueError('Unregistered task')


def control(task): return cell(task)['pair']+'_C'
def original_task(task): return 'C0' if cell(task)['control'] else 'R24'


def scientific_cfg(task):
    from hlt_classification.luka_next_supervision_hydra.contracts import setting
    return deepcopy(setting(original_task(task)))


def seeds(spec,task):
    from hlt_classification.luka_next_supervision_hydra.contracts import seed
    c=cell(task); cfg=scientific_cfg(task)
    init=seed(spec['imported'],cfg,'initialization_seed')
    return dict(native_initialization=init+c['native_initialization_offset'],
        dropout_base=init+c['dropout_offset'],
        sampler=seed(spec['imported'],cfg,'sampler_seed')+c['sampler_offset'])


def weights(task,position):
    if not math.isfinite(position) or not 0<=position<=30: raise ValueError('Invalid epoch position')
    c=cell(task); w=c['maximum']*min(1.,max(0.,(position-c['delay'])/c['ramp']))
    return w,w


def artifact(kind,*,parents=None,**values):
    version=VERSIONS.get(kind,1)
    return with_content_hash(dict(contract=f'LUKA_R24_DIAGNOSTIC_{kind}/v{version}',schema_version=version,
        parents=dict(parents or {}),**FLAGS,**values))


def validate(value,kind,parents=None):
    version=VERSIONS.get(kind,1)
    validate_content_hash(value,expected_contract=f'LUKA_R24_DIAGNOSTIC_{kind}/v{version}',expected_schema_version=version)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value['parents']!=parents):
        raise ValueError('Diagnostic artifact scope or parents differ')


def recipe():
    return artifact('RECIPE',cells={t:cell(t) for t in FITS},order=list(FITS),epochs=30,global_batch=128,
        ce_weight=1.,embedding_weight=5.,decoder='original_R24_chunk8_equal_fields',
        original_worker=DONOR,curriculum_donor=CURRICULUM,selection='original_selection_key',
        deployment='native_HLT_only',dropout='reset_after_init_rank_offset_except_original',
        validation='full_development_validation',statistical_scope='paired_diagnostic_screen')
