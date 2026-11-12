"""Versioned scientific meaning and lineage, distinct from all donor campaigns."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as original
from .registry import FITS, cell, dependencies, CONDITIONS, COORDINATES, ALIASES

BASE = '45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08'
# Freeze all donor runtime at the published repair base; only this campaign
# namespace may change afterwards. The original scientific recipe stays v1.
REPAIR_BASE = '58d0c0edf5f41cff2398fb168fc16fc92e169294'  # Reviewed R8 commit; consumed shared helpers unchanged.
VERSIONS = {'SPEC': 6, 'U100_REPLAY': 3, 'RESOURCES': 3, 'ACCEPTANCE': 3}
U100_EXTRACTION = 'original_consecutive_128_whole_batches_per_rank_fp32_v1'
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_BRIDGE_HYDRA_V1'
AUX_SEED = 20261110


def crop_execution_policy():
    return artifact('CROP_EXECUTION_POLICY', method='batched_analytic_frontend',
        authority='regions.crops', required_parity='byte_exact_inputs_masks_labels',
        reduction='original_per_crop_contiguous_float64_sum',
        physical_validation='full_sources_before_subsets', persistent_cache=False,
        global_padding='unchanged_global_batch128', scientific_recipe_unchanged=True)


def tp_execution_policy():
    return artifact('TP_EXECUTION_POLICY', method='native_encoder_and_stateless_blocks_checkpoint',
        use_reentrant=False, preserve_rng_state=True,
        recompute_buffers='disposable_clones_restore_original_references',
        scope='TP_full_and_each_crop_training_only', global_batch=128, precision='fp32',
        early_checks=['checkpoint_update_parity_all_six_conditions',
                      'largest_support_backward_all_six_conditions', 'partial_batches'],
        scientific_recipe_unchanged=True)


def u100_numerical_policy():
    """Explicit admission bounds; never inferred from the current execution."""
    return artifact('U100_NUMERICAL_POLICY',
        historical_limits=dict(standardized=dict(max_abs=.002, rms=.0001),
                               raw=dict(max_abs=.002, rms=.0001)),
        local_limits=dict(max_abs=1e-5, rms=2e-6),
        legacy_elementwise=dict(atol=1e-4, rtol=1e-4, diagnostic_only=True),
        witness_rows=4096, probe_selection='four_evenly_spaced_original_batches_plus_tail',
        probe_modes=['gpu_default', 'gpu_math', 'cpu_math'], cpu_threads=6,
        basis=dict(job='164226', rows=610,
            spec='ff91de37f837925b6d437be35c25ffb0ef89b649c7ee0e509d788b9f22c2cc34',
            historical_standardized_max=.0010596513748168945,
            historical_standardized_rms=6.245392502114526e-5,
            local_standardized_max=5.364418029785156e-6,
            local_standardized_rms=5.161255827756094e-7,
            full_witness_job='163911', full_witness_max=.0014574527740478516,
            full_witness_rms=5.6815192022950564e-5),
        interpretation='bounded_historical_export_difference_not_exact_setting_attribution',
        original_train_targets='byte_exact_unchanged')


def artifact(kind, *, parents=None, **fields):
    version = VERSIONS.get(kind, 1)
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_BRIDGE_HYDRA_{kind}/v{version}',
        schema_version=version, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    version = VERSIONS.get(kind, 1)
    digest = validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_BRIDGE_HYDRA_{kind}/v{version}',
                                   expected_schema_version=version)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Evaluation disclosures differ')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Artifact parent lineage differs')
    return digest


def ramp(position):
    if not math.isfinite(position) or position <= 0:
        raise ValueError('Positive fractional epoch required')
    return min(1., position/3.)


def recipe():
    from .regions import catalog
    return artifact('RECIPE', parents=dict(u100=original()['content_hash']), base=original(),
        cells={t: cell(t) for t in FITS}, order=list(FITS), aliases=ALIASES,
        dependencies={t: dependencies(t) for t in FITS}, epochs=30, global_batch=128,
        precision='fp32', world_size=2, normalization='SyncBatchNorm', width=128,
        ce_weight=1., embedding_weight=5., ramp_epochs=3, assistant_predecessor_max=.5,
        loss='SmoothL1_beta1_coordinate_query_scale_jet_mean', conditions=list(CONDITIONS),
        coordinates=list(COORDINATES), region_catalog=catalog(), auxiliary_seed=AUX_SEED,
        teacher_condition_seed=20261110, diagnostic_epochs=[1,3,10,20,30], diagnostic_panel=4096,
        response_scale_floor=.1, coordinate_scale_floor=.001, response_kit_anchor=1.,
        region_teacher_full_ce=1., region_teacher_aggregate_ce=1.,
        deployment='native_mod_only', scientific_fits=30, final_hlt_fits=22,
        assistant_selection='min_original_U100_validation_error_then_AUC_then_earlier',
        response_selection='mean_six_AUC_then_mean_CE_then_earlier',
        region_selection='mean_eight_AUC_then_mean_CE_then_earlier',
        student_selection='original_native_full_validation_ties')
