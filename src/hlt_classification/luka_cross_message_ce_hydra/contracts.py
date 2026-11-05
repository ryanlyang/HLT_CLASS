"""Two seed-matched, CE-only fits; existing campaign contracts are unchanged."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_cross_message_hydra.contracts import recipe as donor_recipe

BASE = '199928da991f8a4194dcf42df009130cfcf84118'
AUTHORIZATION = 'AUTHORIZE_CROSS_MESSAGE_CE_HYDRA_V1'
TASKS = ('m32_ce_only_s0', 'm16_ce_only_s0')
# Execution timing changes require new specs/checks; scientific artifacts and
# the CE-only recipe retain their original meanings. Read old runs with their
# pinned source rather than interpreting a cold v1 check as warmed evidence.
VERSIONS = {'SPEC': 2, 'RESOURCE_CHECK': 2}


def resource_policy():
    return dict(warmup_fits=1, measured_fits=1, epochs_per_fit=1,
        timing='fresh_production_miniature_after_disposable_warmup',
        startup='count_once_in_elapsed_seconds', safety_multiplier=2,
        fixed_overhead_seconds=900, memory_fraction=.85,
        request='parent_matching_control_plus_10_minutes')


def settings(task):
    if task not in TASKS:
        raise ValueError('Unregistered CE-only control')
    return dict(width=32 if task == TASKS[0] else 16, mode='ce', embedding=0,
                correction_weight=0., seed_index=0)


def matched_task(task):
    return f'm{settings(task)["width"]}_ce_s0'


def artifact(kind, *, parents=None, **fields):
    version = VERSIONS.get(kind, 1)
    return with_content_hash(dict(contract=f'LUKA_CROSS_MESSAGE_CE_HYDRA_{kind}/v{version}',
        schema_version=version, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    version = VERSIONS.get(kind, 1)
    validate_content_hash(value, expected_contract=f'LUKA_CROSS_MESSAGE_CE_HYDRA_{kind}/v{version}',
                          expected_schema_version=version)
    if (any(value.get(k) != v for k, v in FLAGS.items())
            or (parents is not None and value.get('parents') != parents)):
        raise ValueError('CE-only scope or parents differ')
    return value


def recipe():
    donor = donor_recipe()
    return artifact('RECIPE', donor_commit=BASE, donor_recipe=donor['content_hash'],
        base=donor['base'], epochs=30, batch_size=128, tasks={t: settings(t) for t in TASKS},
        objective='ground_truth_CE_only', embedding_weight=0., message_weight=0., logit_kd_weight=0.,
        initialization='fresh_matched_seed0', selection='full_validation_AUC',
        predictor_inputs='HLT_block4_128_own_features17_masked_mean128',
        predictor_seed=donor['predictor_seed'], expansion_seed=donor['expansion_seed'],
        injection='zero_initialized_expansion_before_block5',
        training_views=['D000'], privileged_targets=False)
