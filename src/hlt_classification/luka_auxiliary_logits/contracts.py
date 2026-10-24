"""Four final-rung losses; never reinterpret the parent hybrid contract."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_trial_hybrid.contracts import COUNTS, recipe as parent_recipe

WEIGHTS = {'k0': 0., 'k025': .25, 'k05': .5, 'k1': 1.}
FIT_SUBMISSION_ORDER = ('k1', 'k05', 'k025', 'k0')
TEACHERS = ('OFFLINE', 'UE100', 'DE066', 'DE033')
REFERENCES = ('M0HLT', 'OFFLINE', 'OFFLINE_EMBED_W5', 'U100_EMBED_W5', 'C50_DE000')
AUTHORIZATION = 'AUTHORIZE_LUKA_AUXILIARY_LOGITS_FOUR_FITS_DEBUG_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_AUXILIARY_LOGITS_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_AUXILIARY_LOGITS_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory trial scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent lineage differs')
    return digest


def recipe(arm):
    if arm not in WEIGHTS:
        raise ValueError('Unregistered auxiliary-logit weight')
    old = parent_recipe('c50', 'DE000')
    return artifact('RECIPE', parents=dict(parent=old['content_hash']), arm=arm,
        base=old['base'], epochs=30, batch_size=128, teacher_order=list(TEACHERS),
        ce_coefficient=.5, logit_total=WEIGHTS[arm], embedding_total=5., temperature=2.,
        logit_weights={t: WEIGHTS[arm]/4 for t in TEACHERS},
        embedding_weights={t: 1.25 for t in TEACHERS},
        embedding_seeds=old['predictor_seeds'],
        logit_seeds={t: old['predictor_seeds'][t]+10000 for t in TEACHERS},
        logit_head=[128, 11], embedding_head=[128, 'GELU', 128],
        main_head_loss='CE_only', logit_loss='T_squared_batchmean_forward_KL',
        embedding_loss='mean_SmoothL1_beta1', normalization=old['normalization'],
        target=old['target'], initialization='from_scratch_c50_DE000_paired',
        inference='native_HLT_auxiliary_heads_removed', zero_control='no_logit_head_gradients')


def nodes(parent):
    original = parent['nodes']['c50']['DE000']
    if original['coordinate'] != 'D000' or original['context_coordinate'] is not None or not original['deployable']:
        raise ValueError('Expected deployable c50 DE000')
    return {a: dict(original, node_id='AUXLOGIT_'+a, arm=a, branch='auxiliary_logits',
                    teacher=None, teacher_order=list(TEACHERS), logit_weight=w)
            for a, w in WEIGHTS.items()}


def tasks():
    return ([dict(task_id='targets', dependencies=[])]
            + [dict(task_id=a, dependencies=['targets']) for a in FIT_SUBMISSION_ORDER]
            + [dict(task_id='report', dependencies=['targets', *FIT_SUBMISSION_ORDER])])
