"""Six triangular nonlinear couplings and two conditional error-scale maps.

The inverse needs only the transformed physical endpoint, never offline
particles, ancestry, labels or an identity key. This is a synthetic codec,
not a detector error/covariance model. Partial tracking remains unchanged.
"""
import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles, wrap_phi
from hlt_classification.cms_proxy_ladder.inputs import build_inputs as legacy_inputs
from .contracts import artifact

VALUE_SCALES = np.array([.1, .2])  # mm
ERROR_SCALES = np.array([.02, .05])  # mm
LAYERS = 3
AMPLITUDE = .65


def input_contract():
    return artifact("INPUTS", dimensions=17, tracking_columns=[11, 12, 13, 14],
        value_transform="asinh(value/[0.1,0.2])/4", error_transform="log1p(error/[0.02,0.05])/2",
        tracking_clipping=False, float32_lossless=False, geometry="unchanged_own_view",
        required_for_all_future_comparison_arms=True, final_test_accessed=False)


def build_inputs(p, *, capacity=512):
    out = legacy_inputs(p, capacity=capacity)
    tr = p.tracking
    out.features[:, [11, 13]] = (np.arcsinh(tr[:, :2]/VALUE_SCALES)/4).astype(np.float32)
    out.features[:, [12, 14]] = (np.log1p(tr[:, 2:]/ERROR_SCALES)/2).astype(np.float32)
    if not np.isfinite(out.features).all():
        raise ValueError("Nonfinite context input representation")
    return out


def context(p):
    """Derivable from exposed p4 alone, with permutation-stable reductions.

    Quantize p4 to the model-visible float32 first. Return bounded soft pT,
    jet-axis radius and smooth local density in original particle order.
    """
    if not len(p):
        return np.empty((0, 3), np.float64)
    v = p.p4.astype(np.float32).astype(np.float64)
    if not np.isfinite(v).all():
        raise ValueError("p4 not representable at model precision")
    order = np.lexsort(tuple(v[:, j] for j in (3, 2, 1, 0)))
    v = v[order]
    pt = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-8)
    eta, phi = np.arcsinh(v[:, 2]/pt), np.arctan2(v[:, 1], v[:, 0])
    total = v.sum(axis=0)
    axis_eta = np.arcsinh(total[2]/max(np.hypot(*total[:2]), 1e-8))
    axis_phi = np.arctan2(total[1], total[0])
    radius = np.hypot(eta-axis_eta, wrap_phi(phi-axis_phi))
    dr = np.hypot(eta[:, None]-eta, wrap_phi(phi[:, None]-phi))
    weights = np.exp(-(dr/.1)**2)
    np.fill_diagonal(weights, 0.)
    density = weights.sum(axis=1)
    result = np.empty((len(p), 3))
    result[order] = np.column_stack((np.tanh(np.log1p(pt/2)/2),
                                     radius/(.1+radius), density/(1+density)))
    return result


def _coefficients(c, layer):
    q, r, rho = c.T
    theta = 2*np.pi*((layer+1)*rho+q) + np.pi*r
    return (AMPLITUDE*np.sin(theta), AMPLITUDE*np.cos(1.3*theta),
            .8*np.sin(theta+.7), .8*np.cos(theta-.4))


def _error_shift(c, x, y):
    q, r, rho = c.T
    return np.column_stack((.3*np.sin(2*np.pi*(rho+r))+.2*np.tanh(y),
                            .3*np.cos(2*np.pi*(q-r))+.2*np.tanh(x)))


def transform(p, *, inverse=False):
    if type(inverse) is not bool:
        raise ValueError("inverse must be boolean")
    ok = p.valid.all(axis=1)
    if not np.any(ok):
        return p
    c = context(p)[ok]
    tr = p.tracking.copy()
    xy = np.arcsinh(tr[ok, :2]/VALUE_SCALES)
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if inverse:
        tr[ok, 2:] *= np.exp(-_error_shift(c, x, y))
        for layer in reversed(range(LAYERS)):
            a, b, u, v = _coefficients(c, layer)
            y -= b*np.tanh(x+v)
            x -= a*np.tanh(y+u)
    else:
        for layer in range(LAYERS):
            a, b, u, v = _coefficients(c, layer)
            x += a*np.tanh(y+u)
            y += b*np.tanh(x+v)
        tr[ok, 2:] *= np.exp(_error_shift(c, x, y))
    with np.errstate(over="raise", invalid="raise"):
        tr[ok, :2] = np.sinh(np.column_stack((x, y)))*VALUE_SCALES
    return Particles(p.p4, p.charge, p.category, tr, p.valid, p.keys)


def audit(base, coupled):
    recovered = transform(coupled, inverse=True)
    error = np.abs(recovered.tracking-base.tracking)/np.maximum(1., np.abs(base.tracking))
    maximum = float(error.max(initial=0.))
    if maximum > 1e-10:
        raise ValueError("Context inverse replay differs")
    new = build_inputs(coupled, capacity=max(512, len(coupled))).features
    decoded = np.column_stack((
        np.sinh(new[:, [11, 13]].astype(float)*4)*VALUE_SCALES,
        np.expm1(new[:, [12, 14]].astype(float)*2)*ERROR_SCALES))
    model_error = np.abs(decoded-coupled.tracking)/np.maximum(1., np.abs(coupled.tracking))
    decoded_p = Particles(coupled.p4, coupled.charge, coupled.category, decoded, coupled.valid, coupled.keys)
    model_recovered = transform(decoded_p, inverse=True)
    model_inverse_error = np.abs(model_recovered.tracking-base.tracking)/np.maximum(1., np.abs(base.tracking))
    old = legacy_inputs(coupled, capacity=max(512, len(coupled))).features
    base_old = legacy_inputs(base, capacity=max(512, len(base))).features
    return dict(eligible_particles=int(base.valid.all(axis=1).sum()),
        changed_particles=int(np.any(base.tracking != coupled.tracking, axis=1).sum()),
        inverse_max_scaled_error=maximum,
        new_frontend_max_scaled_error=float(model_error.max(initial=0.)),
        new_frontend_inverse_max_scaled_error=float(model_inverse_error.max(initial=0.)),
        base_legacy_value_saturated=int(((np.abs(base_old[:, [11, 13]]) == 1) & base.valid[:, :2]).sum()),
        base_legacy_error_clipped=int(((base.tracking[:, 2:] > 1) & base.valid[:, 2:]).sum()),
        legacy_value_saturated=int(((np.abs(old[:, [11, 13]]) == 1) & coupled.valid[:, :2]).sum()),
        legacy_error_clipped=int(((coupled.tracking[:, 2:] > 1) & coupled.valid[:, 2:]).sum()),
        tracking_slots=int(coupled.valid[:, :2].sum()),
        error_slots=int(coupled.valid[:, 2:].sum()))
