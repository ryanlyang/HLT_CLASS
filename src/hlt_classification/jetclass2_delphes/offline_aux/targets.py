"""28 deterministic summaries of native offline particles, not matched targets."""
from __future__ import annotations

import numpy as np
from ..reader import Particles
from ..inputs import eta_phi, wrap_phi
from .contracts import artifact

RADIAL = np.array([0, .05, .1, .15, .2, .3, .45, .65])
PAIR = np.array([0, .05, .1, .2, .3, .45, .65, .9])
SCALARS = np.array([0, 1, 2, 3, 4, 10, 11])


def definition():
    return artifact("TARGET_DEFINITION", columns=28, scalar_columns=SCALARS.tolist(),
                    species=["charged_hadron", "neutral_hadron", "photon", "electron", "muon"],
                    radial_edges=[*RADIAL.tolist(), "infinity"], pair_edges=[*PAIR.tolist(), "infinity"],
                    bins="left_closed_right_open_overflow_last", dtype="float32",
                    calculation="native_float32_promoted_float64_v1", epsilon_gev=1e-8,
                    mass_squared_relative_tolerance=1e-6, pair="i_lt_j_zizj_normalized",
                    scalar_targets=["log1p_species_counts", "log1p_mass_over_vector_pt", "log1p_pt_weighted_width"],
                    no_matching=True, no_error_significance=True)


def histogram(distance, weights, edges):
    return np.bincount(np.searchsorted(edges, distance, side="right") - 1,
                       weights=weights, minlength=8).astype(np.float64)


def summarize(particles: Particles):
    p = particles.values.astype(np.float64)
    pt = np.hypot(p[:, 0], p[:, 1])
    z = pt / pt.sum(dtype=np.float64)
    total = p[:, :4].sum(axis=0, dtype=np.float64)
    jet_pt = float(np.hypot(total[0], total[1]))
    eta, phi = eta_phi(p[:, :4])
    eta_j, phi_j = eta_phi(total)
    radii = np.hypot(eta - eta_j, wrap_phi(phi - phi_j))
    mass2 = total[3]**2 - np.dot(total[:3], total[:3])
    tolerance = 1e-6 * max(total[3]**2, float(np.dot(total[:3], total[:3])), 1.)
    if mass2 < -tolerance:
        raise ValueError("Invalid native jet mass squared")
    out = np.zeros(28, np.float64)
    out[:5] = np.log1p(p[:, 5:10].sum(axis=0))
    out[5:10] = (p[:, 5:10] * z[:, None]).sum(axis=0)
    out[10] = np.log1p(np.sqrt(max(mass2, 0.)) / max(jet_pt, 1e-8))
    out[11] = np.log1p(np.dot(z, radii))
    out[12:20] = histogram(radii, z, RADIAL)
    valid = len(p) >= 2
    if valid:
        i, j = np.triu_indices(len(p), 1)
        weights = z[i] * z[j]
        denominator = weights.sum(dtype=np.float64)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid pair normalization")
        distance = np.hypot(eta[i] - eta[j], wrap_phi(phi[i] - phi[j]))
        out[20:28] = histogram(distance, weights / denominator, PAIR)
    result = out.astype(np.float32)
    validate_targets(result[None], np.array([valid]))
    return result, valid, dict(mass_clamped=int(mass2 < 0), pt_floor=int(jet_pt < 1e-8))


def validate_targets(values, pair_valid):
    if (values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 28
            or pair_valid.dtype != np.bool_ or pair_valid.shape != (len(values),)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("Target shape/dtype/finiteness differs")
    for subset in (values[:, 5:10], values[:, 12:20], values[pair_valid, 20:28]):
        if not np.allclose(subset.sum(axis=1), 1., atol=2e-6, rtol=0):
            raise ValueError("Target distributions must sum to one")
    if np.any(values[~pair_valid, 20:28] != 0):
        raise ValueError("Invalid pair rows must be exactly zero")
