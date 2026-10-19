"""Separate execution identities; original scientific recipe and bank unchanged."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, recipe

AUTHORIZATION = 'AUTHORIZE FULLSTAT FAST ATTENTION EXACT PLAN'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_FAST_ATTENTION_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_FAST_ATTENTION_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Evaluation scope changed')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Execution lineage differs')
    return result


def policy():
    return dict(recipe=recipe(), arm='ATTN_D000', reference='COARSE_D000',
        target_storage='resident_fp32_ragged', prefetch_batches=2,
        progress_updates=128, checkpoint='immutable_epoch_boundary_with_optimizer_and_rng',
        benchmark_batches=256, benchmark_warmup=8, maximum_minutes=2880,
        minimum_train_minutes=720, gate_minutes=180, cpus=6, memory_mb=180000,
        resident_cap_bytes=112*2**30, reserve_bytes=24*2**30,
        result_headroom_bytes=8*2**30, automatic_retry=False, automatic_followup=False)
