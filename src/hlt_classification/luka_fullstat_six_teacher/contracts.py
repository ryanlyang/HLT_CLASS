"""Six frozen teachers; native CE, separate KD and embedding supervision."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_subclass import contracts as previous

TASK = 'train_six'
TASKS = {TASK: []}
COUNTS = previous.COUNTS
PRIMARY = ('OFFLINE', 'CONCAT', 'FUSION')
LADDER = ('UE100', 'DE066', 'DE033')
TEACHERS = PRIMARY + LADDER
EMBED_SEEDS = dict(OFFLINE=2026111001, CONCAT=2026102901, FUSION=2026102902,
                   UE100=2026111002, DE066=2026111003, DE033=2026111004)
KD_SEEDS = {t:2026111010+i for i,t in enumerate(TEACHERS)}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SIX_TEACHER_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SIX_TEACHER_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_SIX_TEACHER_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **previous.FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_SIX_TEACHER_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k, v in previous.FLAGS.items()):
        raise PermissionError('Scientific scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return value


def recipe():
    old = previous.recipe()
    fields = {k:v for k,v in old.items() if k not in
              ('contract','schema_version','parents','content_hash',*previous.FLAGS)}
    fields.update(matching_coefficient=.3, historical_control='completed_subclass_c03',
        subclass_targets='byte_identical_completed_subclass_c03',
        embedding_weights={t:[10,3] if t in PRIMARY else [5,3] for t in TEACHERS},
        teacher_order=list(TEACHERS), embedding_ramp_teachers=list(LADDER), embedding_ramp_epochs=3,
        embedding_head=[128,128,128], embedding_head_seeds=EMBED_SEEDS,
        logit_weights={t:[1,6] for t in TEACHERS}, logit_head=[128,11], logit_head_seeds=KD_SEEDS,
        logit_ramp_epochs=3, logit_temperature=2., logit_loss='T2_forward_KL_class_sum_batch_mean',
        logit_export='native_BF16_forward_FP32_softmax_T2_batch128_TRAIN_only',
        original_U100_supervision=False, classifier_supervision='CE_only')
    fields['multi3'] = dict(base=old['multi3']['base'])
    return artifact('RECIPE', parents=dict(previous=old['content_hash']), **fields)


def node(parent):
    old = parent['nodes'][previous.TASK]
    if (parent['recipe'] != previous.recipe() or old['coordinate'] != 'D000'
            or old['context_coordinate'] is not None or old['embedding_targets'] != list(previous.TEACHERS)
            or old['total_embedding_weight'] != 10. or old['matching_coefficient'] != .3):
        raise ValueError('Expected completed contrastive0.3/subclass0.5 HLT node')
    return dict(old, node_id=TASK, branch='FULLSTAT_SIX_TEACHER', matching_coefficient=.3,
                embedding_targets=list(TEACHERS), total_embedding_weight=15.,
                logit_targets=list(TEACHERS), total_logit_weight=1.)


def coefficient(update, steps):
    return previous.ramp(.3, update, steps)


def auxiliary_ramp(update, steps):
    return previous.ramp(1., update, steps)


def loss_names():
    raw = ['ce', *('embedding_'+t for t in TEACHERS), *('kd_'+t for t in TEACHERS), 'matching', 'subclass']
    weighted = ['weighted_'+n for n in raw if n != 'ce']
    return raw, weighted


def subclass_coefficient(update, steps):
    # Preserve the old floating-point arithmetic exactly, independently of the
    # new contrastive coefficient. No matching coefficient is an input here.
    return .5 * (previous.coefficient(update, steps) / .3)
