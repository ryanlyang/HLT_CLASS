"""Ten independent coefficients over one frozen TRAIN subclass bank."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe
from hlt_classification.luka_fullstat_subclass.contracts import (
    subclass_recipe as kernel_recipe, COUNTS_PER_CLASS, SUBCLASS_SEED)

BASE = '5571240c8c42fce1103344dff94c88bf8edf13c5'
AUTHORIZATION = 'AUTHORIZE_LUKA_SUBCLASS_HYDRA_V2'
WEIGHTS = dict(S0p03=.03, S0p1=.1, S0p3=.3, S0p5=.5, S1=1., S2=2.,
               S5=5., S10=10., S40=40., S100=100.)
FITS = ('S0p5', 'S5', 'S40', 'S1', 'S10', 'S100', 'S0p3', 'S2', 'S0p1', 'S0p03')


def artifact(kind, *, parents=None, **fields):
    version = 1 if kind in ('RECIPE', 'TARGET_RECIPE') else 2
    return with_content_hash(dict(contract=f'LUKA_SUBCLASS_HYDRA_{kind}/v{version}', schema_version=version,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    version = 1 if kind in ('RECIPE', 'TARGET_RECIPE') else 2
    digest = validate_content_hash(value, expected_contract=f'LUKA_SUBCLASS_HYDRA_{kind}/v{version}', expected_schema_version=version)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Evaluation disclosure differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Artifact lineage differs')
    return digest


def cell(task):
    if task not in WEIGHTS:
        raise ValueError('Unknown subclass task')
    return dict(subclass_weight=WEIGHTS[task], seed=0, width=128)


def coefficient(task, position):
    if not math.isfinite(position) or position <= 0:
        raise ValueError('Positive one-based fractional epoch required')
    return cell(task)['subclass_weight']*min(1., position/3.)


def target_recipe():
    # Reuse only the pure array kernel. Its legacy .5 training metadata is not
    # the scientific coefficient here; WEIGHTS independently binds every fit.
    kernel = kernel_recipe()
    return artifact('TARGET_RECIPE', parents=dict(kernel=kernel['content_hash']),
        kernel=kernel, teacher='small_data_T00_CE_OFFLINE', fitting_role='train',
        population=COUNTS['train'], online_refit=False, validation_fitting=False)


def term_names(task):
    cell(task)
    return {'ce', 'embedding', 'subclass', 'weighted_embedding', 'weighted_subclass', 'total'}


def recipe():
    base = u100_recipe()
    return artifact('RECIPE', parents=dict(u100=base['content_hash'], targets=target_recipe()['content_hash']),
        base=base, targets=target_recipe(), cells={t: cell(t) for t in FITS}, order=list(FITS),
        epochs=30, batch_size=128, ce_weight=1., embedding_weight=5.,
        subclass_ramp_epochs=3, subclass_head=[128,60], subclass_seed=SUBCLASS_SEED,
        subclass_counts=list(COUNTS_PER_CLASS), subclass_loss='conditional_KL_target_student_batchmean',
        diagnostic_epochs=[1,3,10,20,30], gradient_scope='shared_mod_excluding_fc_weighted_losses',
        control='reused_C0', teacher_fits=0, new_student_fits=10,
        selection='original_native_full_validation_ties', deployment='native_HLT_mod_only')
