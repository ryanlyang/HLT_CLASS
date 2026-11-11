"""Versioned scientific meaning and lineage, distinct from all donor campaigns."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as original
from .registry import FITS, cell, dependencies, CONDITIONS, COORDINATES, ALIASES

BASE = '45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08'
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_BRIDGE_HYDRA_V1'
AUX_SEED = 20261110


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_BRIDGE_HYDRA_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_BRIDGE_HYDRA_{kind}/v1',
                                   expected_schema_version=1)
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
