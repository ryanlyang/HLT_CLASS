"""Versioned two-arm scientific registration, distinct from jet embeddings."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, node as old_node, recipe as old_recipe

ARMS = ('aligned', 'set')
AUTHORIZATION = 'AUTHORIZE_LUKA_PARTICLE_PREDICTION_DEBUG_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_PARTICLE_PREDICTION_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_PARTICLE_PREDICTION_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Particle prediction parent hashes differ')
    return digest


def recipe():
    donor = old_recipe()
    return artifact('RECIPE', base=donor['base'], epochs=30, batch_size=128,
        teacher='COARSE_U100', target_coordinate='U100', target_role='train',
        tap='mod.blocks[7]_output_before_class_attention_native_layout_authenticated', width=128,
        predictor='Linear128x128_GELU_Linear128x128', predictor_seed=donor['predictor_seed'],
        teacher_precision='float32', predictor_precision='float32',
        export_backend='deterministic_algorithms_no_tf32_no_cudnn_benchmark_restore_after_export',
        regenerated_targets='exact_preflight_payload_and_index_hashes',
        normalization='TRAIN_all_real_tokens_population_mean_std_float64', minimum_scale=1e-3,
        ce_coefficient=1., auxiliary_coefficient=5., beta=1.,
        reduction='mean_coordinates_then_real_tokens_then_jets',
        set_matching='exact_Hungarian_mean_SmoothL1_detached_per_jet',
        frozen_teacher=True, warm_start=False, logit_kd=False, attention_kd=False,
        inference='native_HLT_predictor_removed')


def node(parent, arm):
    if arm not in ARMS:
        raise ValueError('Unknown particle prediction arm')
    return dict(old_node(parent), node_id='HLT_TOKEN_'+arm.upper()+'_W5', branch='TOKEN_'+arm.upper(), arm=arm)


def tasks():
    return [dict(task_id='preflight', dependencies=[])] + [
        dict(task_id=a, dependencies=['preflight']) for a in ARMS]
