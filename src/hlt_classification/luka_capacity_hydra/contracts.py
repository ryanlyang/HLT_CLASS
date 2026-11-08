"""Versioned scientific matrix and fixed analysis registration."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe
from hlt_classification.jetclass2_delphes.model import model_config as native_config
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe

BASE = '3fb9e7f11f628ad9d0d8c4c0d3694cc918e6e8a6'
AUTHORIZATION = 'AUTHORIZE_LUKA_CAPACITY_HYDRA_V1'
CELLS = ('W_UI', 'W_CE', 'N_UI', 'N_CE', 'W_U', 'N_U')
FITS = tuple(f'{cell}_s{seed}' for seed in (0, 1) for cell in CELLS)
MEMBERS = ('S2IT', 'S4', 'S6')
ENSEMBLES = {'S2IT_S4': ('S2IT', 'S4'), 'S2IT_S6': ('S2IT', 'S6'),
             'S4_S6': ('S4', 'S6'), 'S2IT_S4_S6': MEMBERS}


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_CAPACITY_HYDRA_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_CAPACITY_HYDRA_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Population disclosure differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact lineage differs')
    return digest


def cell(task):
    if task not in FITS:
        raise ValueError('Unknown capacity task')
    architecture, objective, seed = task.split('_')
    return dict(width=128 if architecture == 'N' else 256, objective=objective, seed=int(seed[1:]))


def weights(task):
    objective = cell(task)['objective']
    return {} if objective == 'CE' else dict(u100=5.) if objective == 'U' else dict(u100=5., it=5.)


def term_names(task):
    return {'ce', *weights(task)}


def model_config(width):
    if width not in (128, 256):
        raise ValueError('Unregistered representation width')
    config = native_config()
    config['embed_dims'] = [width, 4*width, width]
    return config


def recipe():
    return artifact('RECIPE', parents=dict(u100=u100_recipe()['content_hash']),
        base=u100_recipe(), cells={t: cell(t) for t in FITS}, order=list(FITS),
        models={str(w): model_config(w) for w in (128, 256)}, epochs=30, batch_size=128,
        seed_offset=100003, predictor_seeds=dict(u100=auxiliary_recipe()['predictor_seed'], it=2026110704),
        predictor='student_width_128_128_GELU', targets_width=128,
        weights={t: weights(t) for t in FITS}, embedding_loss='TRAIN_standardized_mean_SmoothL1_beta1',
        selection='native_full_validation_original_ties', fresh_all_cells=True,
        deployment='width_specific_mod_only_HLT', train_targets_only=True,
        ensembles={k: list(v) for k, v in ENSEMBLES.items()}, ensemble_weights='equal_probability_average',
        bootstrap=dict(replicates=500, seed=2026110709, unit='class_stratified_jet_IID_sensitivity',
                       interval=[.025, .975], event_independence='unverified', selection_corrected=False))
