"""One registered scientific comparison; no runtime hyperparameter sweep."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe
from hlt_classification.luka_fullsim.alljets_data import FLAGS

COUNTS = dict(train=299234, validation=74895)
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_AUX_V1'
WIDTH = 128


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_AUX_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_AUX_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Embedding auxiliary parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=base_recipe(), epochs=30, ce_coefficient=1.,
        auxiliary_coefficient=.3, loss='smooth_l1_mean_jets_and_dimensions', beta=1.,
        teacher='OFFLINE', target_role='train', width=WIDTH,
        tap='mod.fc_input_final_normalized_class_vector',
        predictor='Linear128x128_GELU_Linear128x128', predictor_seed=20261019,
        normalization='TRAIN_population_mean_std_float64', minimum_scale=1e-3,
        teacher_precision='float32_no_autocast', predictor_precision='float32',
        frozen_teacher=True, warm_start=False, logit_kd=False, attention_kd=False,
        inference='unchanged_native_HLT_ParT_predictor_removed',
        initialization='exact_M0HLT_seed_and_sampler_head_rng_isolated')


def node(parent):
    old = next(n for n in parent['nodes'] if n['node_id'] == 'M0HLT')
    if old['coordinate'] != 'D000' or old['context_coordinate'] is not None or old['teacher'] is not None:
        raise ValueError('Matched baseline must be native HLT CE only')
    return dict(old, node_id='HLT_EMBED_AUX', branch='EMBED_AUX')
