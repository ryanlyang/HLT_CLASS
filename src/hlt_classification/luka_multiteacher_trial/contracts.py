"""Fixed five-arm scientific and durable identities."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_embedding_aux.contracts import COUNTS, node as base_node
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

TEACHERS = dict(OFFLINE='OFFLINE', COARSE_U100='U100', LOGIT_D066='D066', LOGIT_D033='D033')
ARMS = dict(multi_w25=2.5, multi_w5=5., multi_w10=10., u100x4_w5=5., multi_w5_rel=5.)
SEEDS = dict(OFFLINE=20261020, COARSE_U100=20261019, LOGIT_D066=20261021, LOGIT_D033=20261022)
AUTHORIZATION = 'AUTHORIZE_LUKA_MULTITEACHER_TRIAL_DIRECT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_MULTITEACHER_TRIAL_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_MULTITEACHER_TRIAL_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory population flags differ')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return digest


def target_recipe(teacher):
    return artifact('TARGET_RECIPE', teacher=teacher, coordinate=TEACHERS[teacher], role='train',
        width=128, precision='FP32_eval', tap='native_mod.fc_input', normalization='TRAIN_population_FP64', minimum_scale=1e-3)


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('Unregistered arm')
    return artifact('RECIPE', base=base_recipe(), epochs=30, batch_size=128,
        ce_coefficient=1., total_weight=ARMS[arm], weights={t: ARMS[arm]/4 for t in TEACHERS},
        targets={t: 'COARSE_U100' if arm == 'u100x4_w5' else t for t in TEACHERS},
        predictor_seeds=SEEDS, predictor='Linear128_GELU_Linear128',
        pointwise='mean_SmoothL1_beta1', warm_start=False, logit_kd=False,
        inference='native_HLT_predictors_removed',
        relation=dict(enabled=arm == 'multi_w5_rel', anchors=2048, per_class=186,
            extra_classes=2, seed=314159, temperature=.2, coefficient=1., ramp_epochs=2,
            reduction='sum_anchors_mean_queries_mean_teachers', mask='same_jet',
            temperature_squared=False, precision='FP32'))


def node(parent, arm):
    return dict(base_node(parent), node_id=arm.upper(), branch=arm,
                auxiliary_weight=ARMS[arm])


def tasks():
    return [dict(task_id='export_targets', dependencies=[])] + [
        dict(task_id=arm, dependencies=['export_targets']) for arm in ARMS]
