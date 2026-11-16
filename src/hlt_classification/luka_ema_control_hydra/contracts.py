"""Three paired objectives; EMA05 is primary, single A6000 and FP32 throughout."""
from copy import deepcopy
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS

DONOR = '58d0c0edf5f41cff2398fb168fc16fc92e169294'
PARENT_COMMIT = '3ccc35ae709c63d8833cb0388c618b53dcde5b9e'
AUTHORIZATION = 'AUTHORIZE_EMA_CONTROL_NINE_SINGLE_A6000_V1'
QUALIFICATION_PROTOCOL = 'ema_three_objectives_independent_full_graph_and_native_exports/v1'
LANES = ('a6000',)
SEED_INDICES = (7, 8, 9)
RECIPES = ('CE', 'U', 'R24')
FITS = tuple(f'{r}_s{s}' for s in SEED_INDICES for r in RECIPES)
QUALIFICATION_TASKS = tuple(f'{r}_s7' for r in RECIPES)
READOUTS = ('RAW', 'RAW_BN', 'EMA05', 'EMA10')
PRIMARY = 'EMA05'
LATE_EPOCHS = tuple(range(21, 31))
RECOVERY_THRESHOLD = 52.
TRAIN_ROWS = 299234
MICROBATCH = 128


def cell(task):
    if task not in FITS:
        raise ValueError('Unknown EMA control task')
    name, index = task.rsplit('_s', 1)
    return dict(recipe=name, seed_index=int(index), control=name != 'R24',
        embedding_weight=0. if name == 'CE' else 5., lane='a6000', world_size=1,
        worker='ema_control', precision='fp32', delay=3., ramp=3., shape='linear',
        maximum=.3 if name == 'R24' else 0., token_layers=[8], decoder_warm=False,
        accumulation=1, encoder_lr_multiplier=1., gradient_rule='ordinary',
        decoder_extra_steps=0, soft_assignment=False)


def tasks(hardware, spec=None):
    if hardware not in LANES:
        raise ValueError('Only single-A6000 execution is registered')
    return FITS if spec is None else tuple(spec['tasks'])


def qualification_tasks(spec):
    return tuple(t for t in tasks('a6000', spec) if cell(t)['seed_index'] == 7)


def lane(task):
    if task in (*FITS, 'setup_a6000', 'release_a6000', 'report_a6000'):
        return 'a6000'
    raise ValueError('Unregistered task')


def control(task):
    c = cell(task)
    return f'{"U" if c["recipe"] == "R24" else "CE"}_s{c["seed_index"]}'


def scientific_cfg(task):
    from hlt_classification.luka_next_supervision_hydra.contracts import setting
    cfg = deepcopy(setting('C0' if cell(task)['control'] else 'R24'))
    cfg['seed_index'] = cell(task)['seed_index']
    # Explicit metadata; the CE kernel also branches before touching targets.
    cfg['embedding_weight'] = cell(task)['embedding_weight']
    return cfg


def seeds(spec, task):
    from hlt_classification.luka_next_supervision_hydra.contracts import seed
    cfg = scientific_cfg(task)
    init = seed(spec['imported'], cfg, 'initialization_seed')
    return dict(native_initialization=init, dropout_base=init,
        sampler=seed(spec['imported'], cfg, 'sampler_seed'))


def weights(task, position):
    if not math.isfinite(position) or not 0 <= position <= 30:
        raise ValueError('Invalid epoch position')
    c = cell(task)
    weight = c['maximum'] * min(1., max(0., (position - 3.) / 3.))
    return weight, weight


def encoder_multiplier(task, position):
    cell(task)
    return 1.


def artifact(kind, *, parents=None, **values):
    return with_content_hash(dict(contract=f'LUKA_EMA_CONTROL_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **values))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_EMA_CONTROL_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()) or (parents is not None and value['parents'] != parents):
        raise ValueError('EMA artifact scope or parents differ')


def recipe():
    return artifact('RECIPE', cells={t: cell(t) for t in FITS}, order=list(FITS), epochs=30,
        global_batch=128, ce_weight=1., scientific_configs={t: scientific_cfg(t) for t in FITS},
        decoder='original_R8_chunk8_equal_fields', original_worker=DONOR,
        readout_donor=PARENT_COMMIT, selection='original_selection_key_per_readout',
        deployment='native_HLT_only', precision='fp32_tf32_disabled', normalization='BatchNorm_physical128',
        clock='end_consumed_examples_over_TRAIN', validation='full_development_validation',
        statistical_scope='three_previously_used_development_seeds', primary_readout=PRIMARY,
        readouts=list(READOUTS), ema_half_lives=[.5, 1.], ema_start=3., bn_train_rows=8192,
        late_epochs=list(LATE_EPOCHS), recovery_thresholds=[50., 52.],
        initialization='canonical_untrained_setup_states_per_seed',
        dropout='reset_after_init_isolated_decoder_and_readouts',
        optimizer_resets=False, automatic_lr_scaling=False)
