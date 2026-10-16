"""Explicit twenty-pass alternative; legacy recipes are never modified."""
import math

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.jetclass2_delphes.campaign import recipe as legacy_recipe

CONTRACT = 'LUKA_FULLSTAT_20E_RECIPE/v1'


def recipe(batch_size=128):
    if type(batch_size) is not int or batch_size not in (128, 256):
        raise ValueError('Register batch128 or256, without accumulation/fallback')
    base = {k: v for k, v in legacy_recipe().items() if k != 'content_hash'}
    base.update(contract=CONTRACT, schema_version=1, minimum_passes=20,
                maximum_passes=20, warmup_passes=2, hold_through=2, decay_through=20,
                patience=20, patience_clock='disabled_fixed_twenty_complete_passes',
                batch_size=batch_size, inference_batch_size=batch_size,
                schedule='warmup2_cosine_to20', early_stopping=False)
    return with_content_hash(base)


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= 20:
        raise ValueError('Twenty-epoch schedule position must be in (0,20]')
    if position <= 2:
        return 3e-4 * position / 2
    return 1.5e-5 + (3e-4-1.5e-5)*.5*(1+math.cos(math.pi*(position-2)/18))
