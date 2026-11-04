"""Two message teachers and twelve matched student fits; scientific v1."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS
from hlt_classification.luka_u100_pair_hydra.contracts import SITE, MANIFEST_HASH
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

BASE = 'a40ae47ba87ee23020b482c17d467e2a42282e8f'
AUTHORIZATION = 'AUTHORIZE_CROSS_MESSAGE_HYDRA_V1'
TEACHERS = ('teacher_m32', 'teacher_m16')
ARMS = {}
for name, width, mode, weight, seed in (
    ('m32_both_l0p3_s0',32,'both',.3,0), ('m32_ce_s0',32,'ce',0.,0),
    ('m32_aux_l0p3_s0',32,'aux',.3,0), ('m32_both_l0p1_s0',32,'both',.1,0),
    ('m32_aux_l0p1_s0',32,'aux',.1,0), ('m32_both_l1_s0',32,'both',1.,0),
    ('m32_aux_l1_s0',32,'aux',1.,0), ('m16_both_l0p3_s0',16,'both',.3,0),
    ('m16_ce_s0',16,'ce',0.,0), ('m16_both_l1_s0',16,'both',1.,0),
    ('m32_both_l0p3_s1',32,'both',.3,1), ('m32_ce_s1',32,'ce',0.,1)):
    ARMS[name] = dict(width=width,mode=mode,embedding=5,correction_weight=weight,seed_index=seed)
TASKS = (*TEACHERS, *ARMS)


def artifact(kind, *, parents=None, version=1, **fields):
    return with_content_hash(dict(contract=f'LUKA_CROSS_MESSAGE_HYDRA_{kind}/v{version}',schema_version=version,
        parents=dict(parents or {}),**FLAGS,**fields))


def validate(value,kind,parents=None,*,version=1):
    validate_content_hash(value,expected_contract=f'LUKA_CROSS_MESSAGE_HYDRA_{kind}/v{version}',expected_schema_version=version)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value.get('parents')!=parents):
        raise ValueError('Cross-message scope or parents differ')
    return value


def recipe():
    return artifact('RECIPE',base=base_recipe(),epochs=30,batch_size=128,block_index=3,
        cross_width=128,cross_heads=4,message_widths=[32,16],predictor_hidden=128,
        correction_weights=[.1,.3,1.],correction_ramp_epochs=3,rms_floor=.05,statistics_rows=4096,
        teacher_ce=.25,teacher_kd=.75,temperature=2.,student_ce=1.,student_embedding=5.,
        predictor_seed=2026110401,expansion_seed=2026110402,seed_offset=1000003,arms=ARMS,
        initialization='fresh_native_and_message_parameters',teacher='frozen_COARSE_U100_block7',
        feature_cache='allocation_local_ragged_FP32_from_BF16_eval',teacher_forcing=False,
        geometry='own17_features_and_masked_pooled_HLT_context',target='compact_pre_expansion_message',
        selection='full_validation_AUC',independent_test=False)


def settings(task):
    if task in TEACHERS:
        return dict(width=int(task.removeprefix('teacher_m')),mode='teacher',embedding=0,correction_weight=0.,seed_index=0)
    return ARMS[task]


def teacher_task(task):
    return 'teacher_m'+str(settings(task)['width'])


def dependencies(task):
    if task=='report': return list(TASKS)
    if task in ARMS and ARMS[task]['mode']!='ce': return [teacher_task(task)]
    if task not in (*TASKS,'setup'): raise ValueError('Unknown task')
    return []


def architecture_control(task):
    cfg=settings(task)
    return f'm{cfg["width"]}_ce_s{cfg["seed_index"]}' if cfg['mode'] in ('both','ce') else 'U100_W5_HYDRA_REPLAY'


def seed(imported,setting,kind):
    return imported['node'][kind]+recipe()['seed_offset']*setting['seed_index']
