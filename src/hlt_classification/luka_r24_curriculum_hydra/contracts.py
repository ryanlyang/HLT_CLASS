"""Twenty immutable curricula; encoder and decoder gradient weights are explicit."""
from copy import deepcopy
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS

DONOR = '58d0c0edf5f41cff2398fb168fc16fc92e169294'
BENCHMARK = '1094dea4ccfb36d08e29f740a97d90fc770335ec'
AUTHORIZATION = 'AUTHORIZE_R24_CURRICULUM_20_FOUR_1080TI_V1'
FITS = tuple(f'D{i:02}' for i in (2,1,18,13,7,15,4,3,20,5,6,8,9,10,11,12,14,16,17,19))


def cell(task):
    if task not in FITS: raise ValueError('Unknown R24 curriculum task')
    i = int(task[1:])
    value = dict(seed_index=0, control=False, delay=3., ramp=3., shape='linear',
                 maximum=.3, decoder_warm=False, fade=False)
    changes = {
        1: dict(control=True, maximum=0.), 3: dict(control=True, maximum=0., seed_index=1),
        4: dict(seed_index=1), 5: dict(delay=0.), 6: dict(delay=2.), 7: dict(delay=5.),
        8: dict(delay=8.), 9: dict(ramp=1.), 10: dict(ramp=6.), 11: dict(ramp=10.),
        12: dict(shape='cosine'), 13: dict(ramp=6., shape='cosine'),
        14: dict(maximum=.15), 15: dict(maximum=.5), 16: dict(maximum=1.), 17: dict(fade=True),
        18: dict(decoder_warm=True), 19: dict(decoder_warm=True, delay=5.),
        20: dict(decoder_warm=True, ramp=6., shape='cosine')}
    value.update(changes.get(i, {}))
    return value


def weights(task, position):
    if not math.isfinite(position) or not 0 <= position <= 30: raise ValueError('Invalid epoch position')
    row = cell(task)
    if row['control']: return 0., 0.
    u = min(1., max(0., (position-row['delay'])/row['ramp']))
    encoder = row['maximum'] * (u if row['shape']=='linear' else .5*(1-math.cos(math.pi*u)))
    if row['fade'] and position > 18: encoder = .3-(.2*(position-18)/12)
    decoder = row['maximum'] if row['decoder_warm'] else encoder
    return encoder, decoder


def control(task): return 'D03' if cell(task)['seed_index'] else 'D01'


def scientific_cfg(task):
    from hlt_classification.luka_next_supervision_hydra.contracts import setting
    row = cell(task)
    cfg = deepcopy(setting('C0' if row['control'] else 'R24'))
    cfg['seed_index'] = row['seed_index']
    return cfg


def artifact(kind, *, parents=None, **values):
    return with_content_hash(dict(contract=f'LUKA_R24_CURRICULUM_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **values))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_R24_CURRICULUM_{kind}/v1', expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value['parents']!=parents):
        raise ValueError('Curriculum scope/parents differ')


def recipe():
    return artifact('RECIPE', cells={t:cell(t) for t in FITS}, order=list(FITS), epochs=30,
        batch_size=128, ce_weight=1., embedding_weight=5., precision='fp32', tf32=False,
        normalization='SyncBatchNorm_global_padding', gradient_reduction='sample_weighted_SUM',
        decoder='original_R24_chunk8_fixed_TRAIN_support_equal_fields', optimizer_reset=False,
        delay_units='continuous_epoch_position', ramp='zero_through_delay_full_at_delay_plus_ramp',
        warm_decoder='decoder_constant_max_encoder_VJP_scaled_separately_CE_U100_always_active',
        diagnostics='first_batch_every_epoch_per_rank_condition_shuffles',
        selection='original_selection_key_AUC_CE_rejection_then_earliest_update',
        deployment='native_HLT_ParT_only', dropout='rank_offset_from_initialization_seed')
