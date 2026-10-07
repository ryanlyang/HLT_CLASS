"""Only the triangular shear amplitude changes; never mutate the frozen V1 code."""
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.literature_context import transform as v1
from .contracts import artifact


def recipe():
    return artifact('RECIPE', name='CONTEXT_V2', shear_amplitude=1.0, layers=3,
        construction='V2_forward(V1_inverse(authenticated_lossless_V1_bank))',
        base_noise='unchanged_V1_draws', context='unchanged_visible_float32_p4',
        phase_amplitude=.8, error_context_amplitude=.3, error_value_amplitude=.2,
        topology_pid_p4_charge_masks='byte_identical_V1',
        input_contract=v1.input_contract(), raw_inverse_tolerance=1e-10,
        physical_tracking_units='mm', physics_production_qualified=False)


def transform(p, *, inverse=False):
    if type(inverse) is not bool:
        raise ValueError('inverse must be boolean')
    valid = p.valid.all(axis=1)
    if not np.any(valid):
        return p
    c = v1.context(p)[valid]
    tr = p.tracking.copy()
    xy = np.arcsinh(tr[valid, :2] / v1.VALUE_SCALES)
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if inverse:
        tr[valid, 2:] *= np.exp(-v1._error_shift(c, x, y))
    for layer in reversed(range(3)) if inverse else range(3):
        q, r, rho = c.T
        theta = 2*np.pi*((layer+1)*rho+q) + np.pi*r
        a, b = np.sin(theta), np.cos(1.3*theta)  # fixed amplitude 1.0
        u, v = .8*np.sin(theta+.7), .8*np.cos(theta-.4)
        if inverse:
            y -= b*np.tanh(x+v)
            x -= a*np.tanh(y+u)
        else:
            x += a*np.tanh(y+u)
            y += b*np.tanh(x+v)
    if not inverse:
        tr[valid, 2:] *= np.exp(v1._error_shift(c, x, y))
    with np.errstate(over='raise', invalid='raise'):
        tr[valid, :2] = np.sinh(np.column_stack((x, y))) * v1.VALUE_SCALES
    return Particles(p.p4, p.charge, p.category, tr, p.valid, p.keys)


def _error(a, b):
    return float((np.abs(a-b)/np.maximum(1., np.abs(b))).max(initial=0.))


def reencode(item):
    identity, original = item
    base = v1.transform(original, inverse=True)
    result = transform(base)
    error = max(_error(v1.transform(base).tracking, original.tracking),
                _error(transform(result, inverse=True).tracking, base.tracking))
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError('V1/V2 reversible-map integrity differs')
    if any(getattr(result, k).tobytes() != getattr(original, k).tobytes()
           for k in ('p4', 'category', 'charge', 'valid')):
        raise ValueError('V2 changed frozen nontracking fields')
    invalid = ~original.valid.all(axis=1)
    if result.tracking[invalid].tobytes() != original.tracking[invalid].tobytes():
        raise ValueError('V2 changed missing tracking')
    features = v1.build_inputs(result, capacity=max(512, len(result))).features
    decoded = np.column_stack((np.sinh(features[:, [11, 13]].astype(float)*4)*v1.VALUE_SCALES,
                               np.expm1(features[:, [12, 14]].astype(float)*2)*v1.ERROR_SCALES))
    frontend_error = _error(decoded, result.tracking)
    if not np.isfinite(frontend_error):
        raise ValueError('V2 frontend is nonfinite')
    return identity, result, dict(inverse_max_scaled_error=error,
                                 frontend_max_scaled_error=frontend_error)
