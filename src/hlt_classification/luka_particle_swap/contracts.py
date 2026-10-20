"""Registered matched cold ladders and deterministic endpoint construction."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_attention.contracts import FLAGS, SURFACE, COUNTS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

ARMS = {'logit': 0., 'attn': .1}
COORDINATES = ('D066', 'D033', 'D000')
AUTHORIZATION = 'AUTHORIZE LUKA PARTICLE SWAP EXACT PLAN AND OWNED RETIREMENT'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_PARTICLE_SWAP_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_PARTICLE_SWAP_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Historical test/development scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent lineage differs')
    return digest


def views():
    return artifact('VIEWS', replacement='atomic_p4_charge_category_tracking_validity',
        slots='unchanged_native_hlt_shell', unmatched='native_hlt_at_all_rungs',
        groups=3, strata='native_hlt_charge_nonzero_cross_matched_pt_rank_tertile',
        capacities='balanced_count_remainders_in_hashed_batch_priority',
        order='descending_quantized_hlt_pt_share_then_hash', quantization=1000000,
        rounding='floor_scaled_plus_half', objective='equal_weight_delta_squared_count_pt_stratum_loads',
        ties='sha256_contract_jet_slot_batch', nested=True, data_dependent_tuning=False,
        endpoints=dict(U100='original_U100', D000='original_native_D000'),
        model_features='recompute_from_entire_physical_view', metadata_model_visible=False)


def input_identity(anchor):
    e = anchor['evidence']
    return artifact('INPUT_IDENTITY', parents=dict(prepared=e['prepared'], cache=e['cache'],
                                                  views=views()['content_hash']))


def recipe(coefficient):
    if type(coefficient) not in (int, float) or coefficient not in (0., .1):
        raise ValueError('Registered coefficients are exactly 0 and 0.1')
    return artifact('RECIPE', base=base_recipe(), surface=dict(SURFACE), views=views(),
        coefficient=float(coefficient), epochs=30, coordinates=list(COORDINATES),
        initialization='fresh', target_storage='job_local_fp32_resident_rolling',
        retirement='after_verified_last_consumer_v1', keep_validation_predictions=True)


def nodes(parent_nodes, arm):
    coefficient = ARMS[arm]
    old = {n['node_id']: n for n in parent_nodes}
    out, teacher = [], 'ANCHOR'
    for coordinate in COORDINATES:
        node = old['COARSE_'+coordinate]
        if node['coordinate'] != coordinate or node['context_coordinate'] is not None:
            raise ValueError('Expected regular fixed30 coordinate/seeds')
        name = 'SWAP_'+arm.upper()+'_'+coordinate
        out.append(dict(node, node_id=name, teacher=teacher, branch=arm.upper(),
                        view_name='SWAP'+coordinate[1:], attention_coefficient=coefficient))
        teacher = name
    return out
