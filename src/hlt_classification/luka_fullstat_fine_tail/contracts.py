"""Exact-ninths K1/E40 continuation; independent scientific identities."""
from fractions import Fraction
import hashlib
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe
from hlt_classification.cms_proxy_ladder.campaign import paired_seed
from hlt_classification.cms_proxy_ladder.views import view_contract

STAGES = {'DE022': 'D022', 'DE011': 'D011', 'DE000': 'D000'}
FRACTIONS = {'D022': Fraction(2, 9), 'D011': Fraction(1, 9), 'D000': Fraction(0)}
EXTERNAL = ('OFFLINE', 'UE100', 'DE066', 'DE033', 'M0HLT')
TEACHERS = dict(OFFLINE='OFFLINE', UE100='U100', DE066='D066', DE033='D033',
                M0HLT='D000', DE022='D022', DE011='D011')
EMBEDDING_EXPORTS = KD_EXPORTS = ('DE022', 'DE011')
CONFIGURATION = 'k1_e40_plain'
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_FINE_TAIL_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_FINE_TAIL_THREE_FITS_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_FINE_TAIL_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value,
        expected_contract=f'LUKA_FULLSTAT_FINE_TAIL_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Sealed FullStat role flags differ')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return result


def ancestors(stage):
    if stage not in STAGES:
        raise ValueError('Unregistered fine-tail stage')
    return list(STAGES)[:list(STAGES).index(stage)]


def teachers(stage):
    return list(EXTERNAL) + ancestors(stage)


def view_recipe(coordinate):
    if coordinate not in FRACTIONS:
        raise ValueError('Unregistered fine-tail coordinate')
    f = FRACTIONS[coordinate]
    return artifact('VIEW_RECIPE', parents=dict(kernel=view_contract()['content_hash']),
        coordinate=coordinate, retained_offline=[f.numerator, f.denominator],
        source='authenticated_original_raw_and_saved_assignments',
        semantics='original_D_physical_p4_atomic_identity_validity_tracking',
        row_order='original_prepared_chunks_and_entries', capacity=512,
        matching='reuse_exact_saved_mapping_no_recompute',
        switch_hash='original_view_contract_and_identity_domain',
        model_visible_assignment=False, endpoint_D000='original_cache_byte_identical')


def seed(coordinate, domain):
    if coordinate not in FRACTIONS or domain not in ('initialization', 'sampler'):
        raise ValueError('Unknown fine-tail RNG coordinate/domain')
    if coordinate == 'D000':
        return paired_seed(coordinate, domain)
    # New coordinates are namespaced, never silently decimal approximations.
    f = FRACTIONS[coordinate]
    payload = f'LUKA_FULLSTAT_FINE_TAIL/v1/{domain}/{coordinate}/{f.numerator}/{f.denominator}'
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'big')


def target_recipe(teacher):
    if teacher not in TEACHERS:
        raise ValueError('Unregistered embedding teacher')
    return artifact('TARGET_RECIPE', teacher=teacher, coordinate=TEACHERS[teacher],
        role='train', width=128, tap='native_mod.fc_input', teacher_eval=True,
        teacher_fp32=True, export_batch=128, minimum_scale=1e-3,
        normalization='TRAIN_population_moments_FP64_targets_FP32')


def kd_recipe(teacher):
    if teacher not in TEACHERS:
        raise ValueError('Unregistered logit teacher')
    return artifact('KD_RECIPE', teacher=teacher, coordinate=TEACHERS[teacher],
        role='train', temperature=2., class_count=11, export_batch=128,
        native_logits=True, cuda_forward='BF16', softmax='FP32_logits_divided_by_T',
        auxiliary_predictions_are_not_targets=True)


def recipe(stage):
    order = teachers(stage)
    base = auxiliary_recipe()['predictor_seed']
    seeds = dict(OFFLINE=base+1, UE100=base, DE066=base+2, DE033=base+3,
                 M0HLT=base+4, DE022=base+5, DE011=base+6)
    return artifact('RECIPE', parents={**{t: target_recipe(t)['content_hash'] for t in order},
        **{'kd_'+t: kd_recipe(t)['content_hash'] for t in order},
        'view': view_recipe(STAGES[stage])['content_hash']},
        configuration=CONFIGURATION, stage=stage, coordinate=STAGES[stage], base=base_recipe(128),
        base_usage='matched_optimizer_schedule_only', epochs=20, batch_size=128,
        ce_coefficient=.5, auxiliary_total=40., weights={t: 40./len(order) for t in order},
        logit_total=1., logit_weights={t: 1./len(order) for t in order}, temperature=2.,
        logit_heads={t: [128, 11] for t in order}, logit_seeds={t: seeds[t]+10000 for t in order},
        initialization='from_scratch_coordinate_paired', main_head_supervision='CE_only',
        kd_loss='T_squared_forward_KL_teacher_to_student_batchmean', teacher_order=order,
        predictors={t: [128, 'GELU', 128] for t in order}, predictor_seeds={t: seeds[t] for t in order},
        auxiliary_loss='per_teacher_mean_SmoothL1_beta1',
        inference='native_HLT_without_auxiliary_heads' if stage == 'DE000' else 'richer_input_oracle',
        attention=False, relational=False, warm_start=False, teacher_embedding_average=False)


def node(base, stage):
    if (base['coordinate'] != 'D000' or base['teacher'] is not None
            or base['context_coordinate'] is not None or not base['deployable']):
        raise ValueError('Expected original native M0HLT base')
    co = STAGES[stage]; f = 1-FRACTIONS[co]
    return dict(base, node_id=stage, coordinate=co, teacher=None, branch='fullstat_fine_tail',
        context_coordinate=None, deployable=stage == 'DE000', auxiliary_weight=40.,
        embedding_weights=recipe(stage)['weights'], view_recipe=view_recipe(co)['content_hash'],
        u=[1, 1], f=[f.numerator, f.denominator],
        initialization_seed=seed(co, 'initialization'), sampler_seed=seed(co, 'sampler'))


def tasks():
    return [dict(task_id='train_'+s, dependencies=['train_'+t for t in ancestors(s)]) for s in STAGES]
