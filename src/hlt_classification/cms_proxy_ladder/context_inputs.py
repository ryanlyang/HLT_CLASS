"""Explicit CONTEXT frontend binding; original physical input implementation is frozen."""
from hlt_classification.literature_context.transform import build_inputs, input_contract as encoding
from .inputs import input_contract as legacy_contract
from .contracts import artifact


def input_contract(*, capacity=512):
    old = legacy_contract(capacity=capacity)
    fields = {k: v for k, v in old.items() if k not in (
        'contract', 'schema_version', 'content_hash', 'parents', 'final_test_accessed')}
    fields.update(normalization='context_asinh_log1p_17_v1',
                  error_transform='log1p_error_over_scale_no_clipping',
                  encoding=encoding())
    return artifact('INPUTS', version=2, **fields)
