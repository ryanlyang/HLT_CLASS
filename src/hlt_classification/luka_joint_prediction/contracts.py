"""Explicit user-selected CE + 4*jet + 3*token direct debug experiment."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_particle_prediction.contracts import ARMS, COUNTS, node as old_node, recipe as old_recipe

AUTHORIZATION = 'AUTHORIZE_LUKA_JOINT_PREDICTION_DEBUG_NO_PREFLIGHT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_JOINT_PREDICTION_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_JOINT_PREDICTION_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Joint prediction parent hashes differ')
    return digest


def recipe():
    donor = old_recipe()
    fields = {k:v for k,v in donor.items() if k not in
        {'contract','schema_version','content_hash','parents','auxiliary_coefficient','regenerated_targets',*FLAGS}}
    fields.update(jet_tap='mod.fc_input_final_normalized_native_class_vector',
        jet_normalization='TRAIN_jet_population_mean_std_float64', jet_reduction='mean_jets_and_coordinates',
        jet_coefficient=4., token_coefficient=3., predictors='two_independent_Linear128_GELU_Linear128_heads',
        regenerated_targets='token_payload_exact_original_preflight_hash_global_targets_same_across_new_arms',
        inference='native_HLT_both_predictors_removed')
    return artifact('RECIPE', parents=dict(token_donor=donor['content_hash']), **fields)


def node(parent, arm):
    n = old_node(parent, arm)
    n.pop('auxiliary_weight')
    return dict(n, node_id='HLT_JOINT_'+arm.upper()+'_J4_T3', branch='JOINT_'+arm.upper(),
        jet_weight=4., token_weight=3.)


def tasks():
    return [dict(task_id=a, dependencies=[]) for a in ARMS]
