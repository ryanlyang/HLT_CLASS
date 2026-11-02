"""Versioned scientific meaning; finite performance never controls admission."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import recipe as original_recipe, COUNTS

AUTHORIZATION = 'AUTHORIZE_LUKA_U100_PAIR_CORRECTION_DEBUG_V1'
REFERENCE_JOB = '21993056'
# Insertion order is scheduling priority, not a promise about performance.
ARMS = {
    'p4_pair_u100': dict(particle=True, direct=False, particle_weight=.5, pair_weight=.5),
    'pair_u100': dict(particle=False, direct=True, particle_weight=0., pair_weight=1.),
    'dual_u100': dict(particle=True, direct=True, particle_weight=.5, pair_weight=.5),
    'p4_u100': dict(particle=True, direct=False, particle_weight=1., pair_weight=0.),
}
PAIR_CHUNK = 4096


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_U100_PAIR_CORRECTION_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_U100_PAIR_CORRECTION_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory data scope changed')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return result


def recipe():
    base = original_recipe()
    return artifact('RECIPE', parents=dict(u100=base['content_hash']), u100=base, arms=ARMS,
        epochs=30, batch_size=128, capacity=512, ce_weight=1., embedding_weight=5., correction_weight=1.,
        predictor_input='particle_block1_output_128', corrected_blocks=list(range(2, 8)),
        prediction_frequency='once_per_forward', pair_features=['log_kt', 'log_z', 'log_delta', 'log_m2'],
        particle_fields=['log_pt', 'rapidity', 'phi', 'log1p_m2_over_pt2'],
        particle_seed=2026110201, direct_seed=2026110202, pair_chunk=PAIR_CHUNK,
        checkpoint_pair_mlp=True, smooth_l1_beta=1., reduction='coordinates_then_eligible_items_then_ALL_jets',
        target_role='train', supervision_mask='authenticated_matched_slots_only', inference_mask='all_real_HLT',
        ramp_epochs=3, evaluation_alpha=1., correction_cap=6., bias_bound='log(2)',
        pair_scale_floor=1e-3, particle_scale_floor=[1e-3, 1e-4, 1e-4, 1e-4],
        mass_latent_floor=1e-8, small_negative_m2_tolerance=1e-6,
        final_test_access=False, student_logit_kd=False, fresh_initialization=True,
        deployment='native_HLT_plus_correction_remove_only_jet_predictor', reference_retrained=False)


def ramp(update, steps):
    if update < 1 or steps < 1:
        raise ValueError('One-based positive update/steps required')
    return min(1., update/(3*steps))
