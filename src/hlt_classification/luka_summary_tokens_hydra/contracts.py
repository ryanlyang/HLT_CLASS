"""Five matched summary-token configurations; immutable training and deployment semantics."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe
from hlt_classification.jetclass2_delphes.model import model_config as native_config
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe

BASE = '54ff594df4fb7dddc56c0e99ffb13b9448a6a6c9'
AUTHORIZATION = 'AUTHORIZE_LUKA_SUMMARY_TOKENS_HYDRA_V1'
CELLS = ('E', 'B', 'D', 'C', 'A')
FITS = tuple(f'{cell}_s{seed}' for seed in (0, 1) for cell in CELLS)
EXTRA_PARAMETERS = 3072


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_SUMMARY_TOKENS_HYDRA_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_SUMMARY_TOKENS_HYDRA_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Population disclosure differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact lineage differs')
    return digest


def cell(task):
    if task not in FITS:
        raise ValueError('Unknown summary-token task')
    name, seed = task.split('_')
    return dict(configuration=name, width=128, summaries=1 if name in ('A', 'B') else 3,
                objective='CE' if name in ('A', 'C') else 'UI', seed=int(seed[1:]),
                prediction_sources={} if name in ('A', 'C') else
                dict(u100=1, it=2) if name == 'E' else dict(u100=0, it=0))


def weights(task):
    return {} if cell(task)['objective'] == 'CE' else dict(u100=5., it=5.)


def term_names(task):
    return {'ce', *weights(task)}


def model_config(width=128):
    if width != 128:
        raise ValueError('Only native width128 is registered')
    return native_config()


def recipe():
    return artifact('RECIPE', parents=dict(u100=u100_recipe()['content_hash']),
        base=u100_recipe(), cells={t: cell(t) for t in FITS}, order=list(FITS),
        model=model_config(), epochs=30, batch_size=128, seed_offset=100003,
        predictor_seeds=dict(u100=auxiliary_recipe()['predictor_seed'], it=2026110704),
        predictor='128_128_128_GELU', targets_width=128, extra_query_seed=2026110901,
        extra_query_initialization='isolated_rng_trunc_normal_std0p02',
        extra_classifier_columns='zero', general_classifier_columns='native_initialization',
        summary_order=['general', 'u100', 'it'], pooling='independent_queries_shared_original_class_blocks_and_norm',
        query_to_query_attention=False, classifier='concatenated_summaries_single_linear',
        additional_deployed_parameters=EXTRA_PARAMETERS,
        fp32_equivalence=dict(rtol=2e-5, atol=5e-6),
        weights={t: weights(t) for t in FITS}, embedding_loss='TRAIN_standardized_mean_SmoothL1_beta1',
        selection='native_full_validation_original_ties', fresh_all_cells=True,
        deployment='architecture_specific_mod_only_HLT', train_targets_only=True,
        comparisons=['E-D', 'C-A', 'D-B', 'E-B', 'B-A', 'E-C', '(E-C)-(B-A)'])
