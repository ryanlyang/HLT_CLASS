"""Four fixed native teachers and exactly one new HLT student."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_fullstat_fine_tail.contracts import recipe as tail_recipe
from hlt_classification.cms_proxy_ladder.campaign import paired_seed

STAGES = {'SMOOTH_W10': 'D000'}
EXTERNAL = ('OFFLINE', 'UE100', 'DE066', 'DE033')
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_NEIGHBORHOOD_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_NEIGHBORHOOD_AUTOSUBMIT_V1'
CONFIGURATION = 'CE0.5_K1_E40_smooth10_20epochs'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_NEIGHBORHOOD_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_NEIGHBORHOOD_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Sealed FullStat scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parent lineage differs')
    return result


def teachers(stage):
    if stage not in STAGES:
        raise ValueError('Unregistered neighborhood arm')
    return list(EXTERNAL)


def target_recipe():
    return artifact('TARGET_RECIPE', role='train', query='native_D000_order', context='native_OFFLINE',
        categories=['charged_hadron', 'neutral_hadron', 'photon', 'electron', 'muon', 'unknown'],
        smooth_sigmas=[.02, .05, .10], width=36, maximum_support=512,
        geometry='physical_eta_wrapped_phi_common_frame', context_order='canonical_p4_category',
        count_transform='log1p(count)', pt_transform='log1p(100*sum_pt/sum_HLT_pt)',
        channels='count_then_pt_scale_major_category_minor', dtype='float32', reduction_dtype='float64',
        normalization='equal_jet_TRAIN_population', scale_floor=.1, query_chunk=32,
        self_included=True, clipping=False, truncation=False, matching=False, label_conditioned=False)


def recipe(stage):
    order = teachers(stage)
    donor = tail_recipe('DE000')
    return artifact('RECIPE', parents=dict(targets=target_recipe()['content_hash']),
        configuration=CONFIGURATION, stage=stage, coordinate='D000', base=base_recipe(128),
        epochs=20, batch_size=128, ce_coefficient=.5, auxiliary_total=40.,
        weights={t:10. for t in order}, logit_total=1., logit_weights={t:.25 for t in order}, temperature=2.,
        teacher_order=order, predictors={t:[128, 'GELU', 128] for t in order},
        predictor_seeds={t:donor['predictor_seeds'][t] for t in order},
        logit_seeds={t:donor['logit_seeds'][t] for t in order},
        neighborhood_weight=10.,
        neighborhood_seed=94271, neighborhood_head=[128, 'GELU', 36], neighborhood_block=8,
        neighborhood_reduction='mean_channels_real_tokens_jets', main_head_supervision='CE_only',
        initialization='from_scratch_coordinate_paired', inference='native_HLT_without_auxiliary_heads',
        attention=False, relational=False, warm_start=False, teacher_embedding_average=False)


def node(base, stage):
    if (base['coordinate'] != 'D000' or base['teacher'] is not None
            or base['context_coordinate'] is not None or not base['deployable']):
        raise ValueError('Expected original native M0HLT base')
    return dict(base, node_id=stage, coordinate='D000', teacher=None, context_coordinate=None,
        branch='fullstat_neighborhood', deployable=True, auxiliary_weight=40.,
        initialization_seed=paired_seed('D000', 'initialization'), sampler_seed=paired_seed('D000', 'sampler'))


def tasks():
    return [dict(task_id='targets', dependencies=[]),
            dict(task_id='train_SMOOTH_W10', dependencies=['targets'])]
