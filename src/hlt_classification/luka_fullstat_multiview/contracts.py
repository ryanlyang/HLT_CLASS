"""Separate FullStat20 scientific identities; no retained-predictor graph."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat_u100_embedding.contracts import recipe as original_recipe
from hlt_classification.luka_u100_multiview.contracts import HEAD_SEEDS

TEACHERS = ('U100', 'CONCAT', 'FUSION')
TASKS = {'teacher_CONCAT': [], 'teacher_FUSION': ['teacher_CONCAT'],
         'train_multi3': ['teacher_CONCAT', 'teacher_FUSION']}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_MULTIVIEW_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_MULTIVIEW_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_MULTIVIEW_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_MULTIVIEW_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()): raise PermissionError('Sealed-role flags differ')
    if parents is not None and value['parents'] != parents: raise ValueError('Artifact parents differ')
    return value


def recipe():
    old = original_recipe()
    return artifact('RECIPE', parents=dict(original=old['content_hash']), base=old['base'],
        epochs=20, batch_size=128, ce_weight=1., total_embedding_weight=10.,
        weights={t: [10, 3] for t in TEACHERS}, head_seeds=HEAD_SEEDS,
        target_width=128, normalization='TRAIN_fp64_population_std_floor_0.001',
        embedding_loss='SmoothL1_beta1_mean_batch_and_coordinates',
        concat='physical_multiset_union_source_blind_sort_then_transform_once_full_support',
        fusion=dict(primary='D000', context='OFFLINE', injections=[2,4,6,8], alpha=1.,
                    context_seed_offset=200000, zero_residual_projection=True),
        teacher_loss='CE_only', inference='native_HLT_predictors_removed',
        teacher_readout=False, logit_kd=False, relational=False, retained=False,
        warm_start=False, save_every_completed_epoch=True, automatic_resume=False)


def nodes(reuse):
    base = reuse['u100']['base_node']
    if base['node_id'] != 'U100_EMBED_W5' or base['coordinate'] != 'D000' or base['auxiliary_weight'] != 5:
        raise ValueError('Expected original U100-w5 HLT node')
    # Original OFFLINE initialization/sampler, not a trained initialization.
    oracle = reuse['oracle_node']
    if oracle['node_id'] != 'OFFLINE' or oracle['coordinate'] != 'OFFLINE':
        raise ValueError('Expected original OFFLINE seed reference')
    result = {}
    for task in TASKS:
        teacher = task.startswith('teacher_')
        name = task.removeprefix('teacher_') if teacher else None
        value = {k: v for k,v in (oracle if teacher else base).items() if k != 'auxiliary_weight'}
        value.update(node_id=task, branch='FULLSTAT_MULTIVIEW', teacher=None,
            coordinate='CONCAT' if name == 'CONCAT' else 'D000',
            context_coordinate='OFFLINE' if name == 'FUSION' else None,
            deployable=not teacher, embedding_targets=[] if teacher else list(TEACHERS),
            total_embedding_weight=0. if teacher else 10., retained_predictor=False)
        result[task] = value
    return result
