"""Explicit batch/BN ablation; never reinterpret the original batch256 recipe."""
from hlt_classification.data.cache_contracts import with_content_hash
from .campaign import recipe as original_recipe

CONTRACT = 'LUKA_MATCHED_BATCH128_RECIPE/v1'


def recipe():
    base = original_recipe()
    return with_content_hash(dict(base, contract=CONTRACT, schema_version=1,
        parents=dict(original_recipe=base['content_hash']), batch_size=128,
        inference_batch_size=128, accumulation=1))
