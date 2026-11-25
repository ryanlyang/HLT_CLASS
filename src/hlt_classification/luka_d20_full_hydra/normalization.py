"""Freeze TRAIN statistics; compare recomputed evidence without replacing them."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from . import contracts as c

RTOL=1e-10
ATOL=1e-12


def policy():
    return c.artifact('NORMALIZATION_EXECUTION_POLICY',statistics='selected_TRAIN_only_FP64_original_per_jet_order',
        application='one_authenticated_qualified_record_for_every_fit',
        exact=['content_hash','population','rows','tokens','support','maximum','role','frame'],
        recomputed_moment_check=dict(rtol=RTOL,atol=ATOL),
        comparison='diagnostic_tolerance_only_applied_statistics_remain_exact',mismatch='persist_both_then_fail')


def validate(record,spec,*,expected_hash=None):
    c.validate(record,'TARGET_NORMALIZATION',dict(population=spec['population']['content_hash']))
    n=spec['population']['counts']['train'];tokens=spec['population']['role_particles']['train']['offline']
    mean=np.asarray(record['mean']);scale=np.asarray(record['scale']);support=np.asarray(record['support'])
    if (record['role']!='train' or record['rows']!=n or record['tokens']!=tokens
            or record['maximum']!=spec['maximum'] or record['frame']!='paired_HLT_sum_p4_axis_and_vector_pt'
            or mean.shape!=(22,) or scale.shape!=(22,) or support.shape!=(22,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<.001).any()
            or support.dtype.kind not in 'iu' or np.any((support<0)|(support>tokens))):
        raise ValueError('Invalid TRAIN normalization population/shape/scope')
    if expected_hash is not None and record['content_hash']!=expected_hash:
        raise ValueError('Frozen normalization does not match the qualified hash')
    return record


def choose(spec,observed,frozen,*,evidence=None):
    try:
        validate(observed,spec)
        if frozen is not None:validate(frozen,spec)
    except ValueError as error:
        if evidence is not None:
            write_immutable_json(Path(evidence),c.artifact('NORMALIZATION_COMPARISON',
                parents=dict(spec=spec['content_hash'],policy=policy()['content_hash']),
                observed=observed,frozen=frozen,exact_scope=False,moments_within_tolerance=False,
                validation_error=str(error),applied_sha256=None))
        raise
    if frozen is None:return observed
    keys=('parents','role','rows','tokens','support','maximum','frame')
    exact=all(observed[k]==frozen[k] for k in keys)
    differences={k:float(np.max(np.abs(np.asarray(observed[k])-np.asarray(frozen[k])))) for k in ('mean','scale')}
    close=all(np.allclose(observed[k],frozen[k],rtol=RTOL,atol=ATOL) for k in differences)
    row=c.artifact('NORMALIZATION_COMPARISON',parents=dict(spec=spec['content_hash'],policy=policy()['content_hash']),
        observed=observed,frozen=frozen,exact_scope=exact,moments_within_tolerance=close,
        maximum_absolute_differences=differences,applied_sha256=frozen['content_hash'] if exact and close else None)
    if evidence is not None:write_immutable_json(Path(evidence),row)
    if not exact or not close:raise ValueError('TRAIN normalization differs substantively; inspect saved normalization comparison')
    return frozen


def qualified(spec):
    from . import resources
    q=resources.qualification(spec)
    if 'target_normalization' not in q:
        raise ValueError('Full-campaign qualified TRAIN normalizer is absent')
    record=load_json(validate_file_ref(q['target_normalization']))
    return validate(record,spec,expected_hash=q['target_normalization_sha256'])


def publish(spec,record):
    validate(record,spec);path=Path(spec['root'])/'normalization/qualified.json'
    if path.exists():
        if load_json(path)!=record:raise ValueError('Existing frozen normalization differs')
    else:write_immutable_json(path,record)
    return file_ref(path)
