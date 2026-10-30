"""Frozen first-screen identities; no runtime sweep or historical reinterpretation."""
from fractions import Fraction

from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import recipe as donor_recipe
from hlt_classification.luka_embedding_aux.contracts import COUNTS

AUTHORIZATION = 'AUTHORIZE_LUKA_U100_MULTIVIEW_DEBUG_V1'
HEAD_SEEDS = dict(U100=20261019, CONCAT=2026102901, FUSION=2026102902)
TEACHERS = ('U100', 'CONCAT', 'FUSION')
ARMS = {
    'u100_w5': dict(targets=['U100'], readout=None, retained=False),
    'u100_concat': dict(targets=['U100', 'CONCAT'], readout=None, retained=False),
    'u100_fusion': dict(targets=['U100', 'FUSION'], readout=None, retained=False),
    'u100_concat_fusion': dict(targets=list(TEACHERS), readout=None, retained=False),
    'u100_readout_ce': dict(targets=['U100'], readout='ce', retained=False),
    'u100_readout_kd': dict(targets=['U100'], readout='kd', retained=False),
    'u100_use_prediction': dict(targets=['U100'], readout=None, retained=True),
    'hlt_capacity_control': dict(targets=[], readout=None, retained=True),
}
# No student depends on another student's scientific performance or success.
TASKS = {
    'accept': [],
    'export_U100': ['accept'],
    'teacher_CONCAT': ['accept'],
    'teacher_FUSION': ['accept'],
    **{a: ['accept'] + ['export_U100' if t == 'U100' else 'teacher_'+t
                        for t in v['targets']] for a, v in ARMS.items()},
}
TASKS['report'] = list(TASKS)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_U100_MULTIVIEW_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_U100_MULTIVIEW_{kind}/v1',
                                  expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory role differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Multiview parent binding differs')
    return result


def weights(arm):
    targets = ARMS[arm]['targets']
    return {t: Fraction(5, len(targets)) for t in targets}


def node(imported, task):
    """Keep authenticated seeds, but report the actual new graph/objective identity."""
    if task.startswith('teacher_'):
        name = task.removeprefix('teacher_')
        if name not in ('CONCAT', 'FUSION'):
            raise ValueError('Only two fresh oracle teachers are registered')
        return dict(imported['oracle_node'], node_id=task, branch='MULTIVIEW_ORACLE', teacher=None,
                    coordinate='CONCAT' if name == 'CONCAT' else 'D000',
                    context_coordinate='OFFLINE' if name == 'FUSION' else None,
                    context_initialization_seed=(imported['oracle_node']['initialization_seed']+200000
                                                 if name == 'FUSION' else None), deployable=False)
    config = ARMS[task]
    base = {k: v for k, v in imported['node'].items() if k != 'auxiliary_weight'}
    return dict(base, node_id=task, branch='U100_MULTIVIEW', teacher=None, coordinate='D000',
                context_coordinate=None, embedding_targets=config['targets'],
                total_embedding_weight=5 if config['targets'] else 0, readout=config['readout'],
                retained_predictor=config['retained'], deployable=True)


def recipe():
    donor = donor_recipe()
    return artifact('RECIPE', parents=dict(donor=donor['content_hash']), base=donor['base'],
        counts=COUNTS, epochs=30, batch_size=128, ce_weight=1, heads=HEAD_SEEDS,
        arms={a: dict(v, weights={t: [w.numerator, w.denominator] for t, w in weights(a).items()})
              for a, v in ARMS.items()},
        target_width=128, normalization='TRAIN_fp64_population_std_floor_0.001',
        embedding_loss='SmoothL1_beta1_mean_batch_and_coordinates',
        readout=dict(warmup_passes=3, coefficient=1, temperature=2, de_standardize=True,
                     classifier='exact_frozen_U100_fc', native_KD=False),
        concat='physical_multiset_union_source_blind_sort_then_ALLJETS_transform_once',
        fusion=dict(primary='D000', context='OFFLINE', injections=[2, 4, 6, 8],
                    alpha=1, context_seed_offset=200000, zero_residual_projection=True),
        retained='Linear256x11_native_half_exact_predictor_half_zero',
        technical_policy='separate_real_debug_acceptance_then_review_measured_science_plan',
        automatic_retry=False, student_warm_start=False)
