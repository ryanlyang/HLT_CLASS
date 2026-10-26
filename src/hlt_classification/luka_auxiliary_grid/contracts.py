"""Immutable twenty-arm grid; old k5/k10 recipes are never reinterpreted."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_trial_hybrid.contracts import COUNTS, recipe as parent_recipe

WAVES = ((5.,10.), (5.,20.), (2.5,10.), (2.5,20.), (1.,10.), (1.,20.),
         (5.,40.), (2.5,40.), (1.,40.), (10.,20.))


def arm_name(k, e, relation):
    return 'k'+format(k,'g').replace('.','p')+'_e'+format(e,'g')+('_rel' if relation else '_plain')


ARMS = {arm_name(k,e,rel): dict(kd=k, embedding=e, relation=rel, wave=i+1)
        for i,(k,e) in enumerate(WAVES) for rel in (True,False)}
WEIGHTS = {a:r['kd'] for a,r in ARMS.items()}
FIT_SUBMISSION_ORDER = tuple(ARMS)
PROBES = ('k5_e10_plain', 'k5_e40_rel', 'k10_e20_rel')
TEACHERS = ('OFFLINE','UE100','DE066','DE033')
REFERENCES = ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','U100_EMBED_W5','C50_DE000','K5_EMBED5')
AUTHORIZATION = 'AUTHORIZE_LUKA_AUXILIARY_GRID_TWENTY_DEBUG_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_AUXILIARY_GRID_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest=validate_content_hash(value,expected_contract=f'LUKA_AUXILIARY_GRID_{kind}/v1',expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()):
        raise PermissionError('Exploratory trial scope differs')
    if parents is not None and value['parents']!=parents:
        raise ValueError('Parent lineage differs')
    return digest


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('Unregistered grid configuration')
    r=ARMS[arm]; old=parent_recipe('c50','DE000')
    return artifact('RECIPE',parents=dict(parent=old['content_hash']),arm=arm,
        base=old['base'],epochs=30,batch_size=128,teacher_order=list(TEACHERS),
        ce_coefficient=.5,logit_total=r['kd'],embedding_total=r['embedding'],temperature=2.,
        logit_weights={t:r['kd']/4 for t in TEACHERS},
        embedding_weights={t:r['embedding']/4 for t in TEACHERS},
        embedding_seeds=old['predictor_seeds'],
        logit_seeds={t:old['predictor_seeds'][t]+10000 for t in TEACHERS},
        logit_head=[128,11],embedding_head=[128,'GELU',128],
        main_head_loss='CE_only',logit_loss='T_squared_batchmean_forward_KL',
        embedding_loss='mean_SmoothL1_beta1',normalization=old['normalization'],
        target=old['target'],initialization='from_scratch_c50_DE000_paired',
        inference='native_HLT_auxiliary_heads_removed',
        relation=dict(enabled=r['relation'],anchors=2048,per_class=186,extra_classes=2,
            seed=314159,temperature=.2,coefficient=1.,ramp_epochs=2,
            reduction='sum_anchors_mean_queries_mean_teachers',mask='same_jet',
            temperature_squared=False,precision='FP32'))


def nodes(parent):
    original=parent['nodes']['c50']['DE000']
    if original['coordinate']!='D000' or original['context_coordinate'] is not None or not original['deployable']:
        raise ValueError('Expected deployable c50 DE000')
    return {a:dict(original,node_id='AUXGRID_'+a,arm=a,branch='auxiliary_grid',
        teacher=None,teacher_order=list(TEACHERS),logit_weight=r['kd'],
        embedding_weight=r['embedding'],relation=r['relation']) for a,r in ARMS.items()}


def tasks():
    rows=[dict(task_id='targets',dependencies=[])]
    previous=[]
    for k,e in WAVES:
        current=[arm_name(k,e,rel) for rel in (True,False)]
        rows.extend(dict(task_id=a,dependencies=['targets',*previous]) for a in current)
        previous=current
    rows.append(dict(task_id='report',dependencies=['targets',*ARMS]))
    return rows
