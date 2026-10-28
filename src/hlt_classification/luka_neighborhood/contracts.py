"""Frozen fourteen-arm neighborhood experiment, additive v1 artifacts."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_supervision_depth.contracts import recipe as baseline_recipe, TEACHERS, COUNTS

AUTHORIZATION = 'AUTHORIZE_LUKA_NEIGHBORHOOD_14_DEBUG_NO_PREFLIGHT_V1'
BASE = '5d55d602549ff47c6301e74b712786650c3bf8f4'
REFERENCES = ('M0HLT', 'OFFLINE', 'OFFLINE_EMBED_W5', 'U100_EMBED_W5', 'C50_DE000', 'K1_E40_PLAIN', 'BASELINE_REPEAT')
ROWS = (('smooth_w3','smooth',3.), ('smooth_w10','smooth',10.),
        ('ann_w3','ann',3.), ('ann_w10','ann',10.),
        ('hlt_ann_w3','hlt_ann',3.), ('hlt_ann_w10','hlt_ann',10.),
        ('correction_w3','correction',3.), ('correction_w10','correction',10.),
        ('smooth_w1','smooth',1.), ('ann_w1','ann',1.), ('correction_w1','correction',1.),
        ('smooth_w30','smooth',30.), ('ann_w30','ann',30.), ('correction_w30','correction',30.))
ARMS = {name:dict(neighborhood=kind, neighborhood_weight=weight) for name,kind,weight in ROWS}
FAMILIES = ('ann','smooth','hlt_ann')
TEACHER_TASKS = ()


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_NEIGHBORHOOD_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest=validate_content_hash(value, expected_contract=f'LUKA_NEIGHBORHOOD_{kind}/v1', expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()): raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents']!=parents: raise ValueError('Parent lineage differs')
    return digest


def target_recipe():
    return artifact('TARGET_RECIPE', role='train', query='native_D000_order', context='native_OFFLINE_or_HLT',
        categories=['charged_hadron','neutral_hadron','photon','electron','muon','unknown'],
        smooth_sigmas=[.02,.05,.10], annular_edges=[0.,.05,.10,.20], width=36,
        geometry='physical_eta_wrapped_phi_common_frame', context_order='canonical_p4_category',
        count_transform='log1p(count)', pt_transform='log1p(100*sum_pt/sum_HLT_pt)',
        channels='count_then_pt_scale_major_category_minor', dtype='float32', reduction_dtype='float64',
        normalization='equal_jet_TRAIN_population', scale_floor=.1,
        correction='(offline_annular-hlt_annular)/offline_annular_scale', self_included=True,
        clipping=False, truncation=False, matching=False, label_conditioned=False)


def recipe(arm):
    old=baseline_recipe('baseline')
    # Keep inherited fields required by the unchanged objective/reductions.
    fields={k:v for k,v in old.items() if k not in ('contract','schema_version','parents','content_hash',*FLAGS)}
    fields.update(arm=arm, **ARMS[arm], neighborhood_seed=94271, neighborhood_head=[128,'GELU',36],
                  neighborhood_block=8, neighborhood_reduction='mean_channels_real_tokens_jets')
    return artifact('RECIPE', parents=dict(baseline=old['content_hash'], targets=target_recipe()['content_hash']), **fields)


def teacher_recipe(task):
    raise ValueError('No new teacher fits')


def tasks():
    rows=[dict(task_id='targets',dependencies=[],success_dependencies=[],slot_dependencies=[])]
    lanes=[[],[]]
    for index,arm in enumerate(ARMS):
        lane=lanes[index%2]
        rows.append(dict(task_id=arm,dependencies=['targets',*lane],success_dependencies=['targets'],slot_dependencies=list(lane)))
        lane.append(arm)
    rows.append(dict(task_id='report',dependencies=['targets',*ARMS],success_dependencies=[],slot_dependencies=['targets',*ARMS]))
    return rows
