"""Registered fixed30 experiment; the measured batch128 recipe stays immutable."""
import math

from hlt_classification.data.cache_contracts import with_content_hash
from .batch128_recipe import recipe as original_recipe

CONTRACT = 'LUKA_MATCHED_BATCH128_30E_RECIPE/v1'


def recipe():
    original = original_recipe()
    return with_content_hash(dict(original, contract=CONTRACT,
        parents=dict(original_batch128_recipe=original['content_hash']),
        minimum_passes=30, maximum_passes=30, warmup_passes=3,
        hold_through=3, decay_through=30, patience=30,
        patience_clock='disabled_fixed_thirty_complete_passes',
        schedule='warmup3_cosine_to30', early_stopping=False))


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= 30:
        raise ValueError('Thirty-epoch schedule position must be in (0,30]')
    if position <= 3:
        return 3e-4 * position / 3
    return 1.5e-5 + (3e-4-1.5e-5)*.5*(1+math.cos(math.pi*(position-3)/27))
