"""The twenty registered students; no validation-driven changes to the grid."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import recipe as u100_recipe, COUNTS

AUTHORIZATION = 'AUTHORIZE_LUKA_U100_CONTRASTIVE_DEBUG_V1'
WIDTH, NEGATIVES, POOL, TEMPERATURE = 128, 1024, 4096, .1
ROWS = (
    ('mix_l0p3_s0', 'mix', .3, 0), ('u100_w5_s0', 'none', 0., 0),
    ('mix_l1_s0', 'mix', 1., 0), ('random_l0p3_s0', 'random', .3, 0),
    ('mix_l0p1_s0', 'mix', .1, 0), ('kinmix_l0p3_s0', 'kinmix', .3, 0),
    ('cross_l0p3_s0', 'cross', .3, 0), ('hard_l0p3_s0', 'hard', .3, 0),
    ('mix_l3_s0', 'mix', 3., 0), ('random_l1_s0', 'random', 1., 0),
    ('cross_l0p1_s0', 'cross', .1, 0), ('hard_l0p1_s0', 'hard', .1, 0),
    ('mix_l0p3_s1', 'mix', .3, 1), ('u100_w5_s1', 'none', 0., 1),
    ('random_l0p1_s0', 'random', .1, 0), ('cross_l1_s0', 'cross', 1., 0),
    ('hard_l1_s0', 'hard', 1., 0), ('random_l3_s0', 'random', 3., 0),
    ('mix_l0p03_s0', 'mix', .03, 0), ('mix_l10_s0', 'mix', 10., 0),
)
ARMS = {name: dict(strategy=s, coefficient=w, replicate=r, rank=i+1)
        for i, (name, s, w, r) in enumerate(ROWS)}


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_U100_CONTRASTIVE_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_U100_CONTRASTIVE_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope changed')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return result


def sampler_recipe():
    return artifact('SAMPLER', negatives=NEGATIVES, proposal_pool=POOL, hard_rank_interval=[257, 1024],
        strategy_counts=dict(random=dict(G=1024), mix=dict(G=512, S=256, H=256),
            cross=dict(X=1024), hard=dict(G=256, S=256, H=512), kinmix=dict(K=512, S=256, H=256)),
        assembly_order=['H', 'S', 'G_or_K'], temperature=TEMPERATURE,
        rng='sha256(version,root_seed,replicate,epoch,canonical_identity,stream,component)/PCG64',
        root_seed=2026103102, replacement=False, role='train', teacher='original_OFFLINE',
        key_transform='TRAIN_population_standardize_then_L2_eps1e-12',
        pt_quantile_bins=20, count_quantile_bins=10, quantile_method='linear',
        rank_tie_break='canonical_identity', mining='frozen_FP32_keys_chunk8',
        known_aliases='exclude_entire_group_when_authenticated', event_group='use_if_authenticated')


def recipe():
    base = u100_recipe()
    return artifact('RECIPE', parents=dict(u100=base['content_hash'], sampler=sampler_recipe()['content_hash']),
        u100=base, ce=1., embedding=5., arms=ARMS, matching_head_seed=2026103101,
        matching_ramp_epochs=3, temperature=TEMPERATURE, epochs=30, batch_size=128,
        native_inference=True, matching_input='native_preclassifier_128',
        zero_control='original_U100_training_kernel', teacher_frozen=True,
        final_test_access=False)


def seeds(imported, arm):
    r = ARMS[arm]['replicate']
    return dict(initialization=imported['node']['initialization_seed']+r,
                sampler=imported['node']['sampler_seed']+r,
                predictor=20261019+r, matching=2026103101+r, negatives=2026103102+r)


def node(imported, arm):
    s = seeds(imported, arm)
    return dict(imported['node'], initialization_seed=s['initialization'], sampler_seed=s['sampler'])


def ramp(coefficient, update, steps):
    if steps < 1 or update < 1 or coefficient < 0:
        raise ValueError('Invalid one-based ramp')
    return coefficient * min(1., update/(3*steps))
