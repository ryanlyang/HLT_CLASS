"""Matched-marginal Gaussian tracking noise with explicit reference geometry."""
from dataclasses import dataclass
import hashlib

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles, wrap_phi
from .contracts import artifact

STRENGTHS = dict(LOW=.5, MID=1., HIGH=2.)
VARIANTS = {f"{mode}_{name}": (mode, value)
            for name, value in STRENGTHS.items() for mode in ("CORR", "INDEP")}
SIDES = ("OFFLINE", *VARIANTS)


def recipe():
    return artifact("RECIPE", name="CORRELATED_TRACKING_V1", strengths=dict(STRENGTHS),
        sides=list(SIDES), length_unit="mm", momentum_unit="GeV",
        vertex_sd_mm=[.020, .020, .050], independent_floor_mm=[.002, .005],
        density_radius=.1, independent_fraction="0.25+0.25*rho+0.25/(1+pt/2)",
        reference_jacobian=["sin(phi),-cos(phi),0", "cos(phi)*sinh(eta),sin(phi)*sinh(eta),-1"],
        convention="assumed_straight_line_perigee_not_verified_historical_producer",
        shared_unit="jet_not_event", error_policy="quadrature_original_and_added_variance",
        eligibility="valid_value_and_positive_valid_error_per_component",
        rng="SHA256_identity_domain_PCG64_canonical_particle_keys_common_epsilon_all_strengths",
        preserves=["p4", "charge", "category", "keys", "valid", "multiplicity"],
        classification="synthetic_mechanism_not_CMS_HLT", classifier_training_authorized=False)


def rng(identity, domain):
    if not isinstance(identity, str) or not identity:
        raise ValueError("Nonempty stable jet identity required")
    seed = hashlib.sha256(("CORRELATED_TRACKING_V1\0"+identity+"\0"+domain).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(seed[:16], "big")))


@dataclass(frozen=True)
class Geometry:
    eligible: np.ndarray
    density: np.ndarray
    projected_sd: np.ndarray  # n x 2 x 3; J diag(tau) at unit strength
    independent_sd: np.ndarray  # n x 2, unit strength

    @property
    def variance(self):
        return (self.projected_sd**2).sum(axis=2) + self.independent_sd**2


def geometry(p):
    # Canonical reductions make row permutation irrelevant, including density sums.
    order = np.argsort(np.asarray(p.keys, dtype=str), kind="stable")
    eta, phi = p.eta[order], p.phi[order]
    dr2 = (eta[:, None]-eta)**2 + wrap_phi(phi[:, None]-phi)**2
    weight = np.exp(-dr2/.1**2)
    np.fill_diagonal(weight, 0.)
    density = np.empty(len(p))
    total = weight.sum(axis=1)
    density[order] = total/(1+total)
    sn, cs, sh = np.sin(p.phi), np.cos(p.phi), np.sinh(p.eta)
    jac = np.stack((np.column_stack((sn, -cs, np.zeros(len(p)))),
                    np.column_stack((cs*sh, sn*sh, -np.ones(len(p))))), axis=1)
    fraction = .25+.25*density+.25/(1+p.pt/2)
    independent = np.sqrt((fraction[:, None]*p.tracking[:, 2:])**2 + np.array([.002, .005])**2)
    return Geometry(p.valid[:, :2] & p.valid[:, 2:], density,
                    jac*np.array([.020, .020, .050]), independent)


def generate_all(p, identity):
    """Return physical endpoints plus deterministic diagnostic geometry, no latents."""
    g = geometry(p)
    order = np.argsort(np.asarray(p.keys, dtype=str), kind="stable")
    independent_vertex, epsilon = np.empty((len(p), 3)), np.empty((len(p), 2))
    independent_vertex[order] = rng(identity, "independent_vertex").standard_normal((len(p), 3))
    epsilon[order] = rng(identity, "epsilon").standard_normal((len(p), 2))
    common_vertex = rng(identity, "common_vertex").standard_normal(3)
    residuals = dict(
        CORR=(g.projected_sd*common_vertex).sum(axis=2)+g.independent_sd*epsilon,
        INDEP=(g.projected_sd*independent_vertex[:, None, :]).sum(axis=2)+g.independent_sd*epsilon)
    sides = {"OFFLINE": p}
    for side, (mode, strength) in VARIANTS.items():
        tr = p.tracking.copy()
        tr[:, :2] = np.where(g.eligible, tr[:, :2]+strength*residuals[mode], tr[:, :2])
        tr[:, 2:] = np.where(g.eligible, np.sqrt(tr[:, 2:]**2+strength**2*g.variance), tr[:, 2:])
        sides[side] = Particles(p.p4, p.charge, p.category, tr, p.valid, p.keys)
        assert_structure(p, sides[side], g.eligible)
    return sides, g


def assert_structure(original, output, eligible):
    if original.keys != output.keys or any(not np.array_equal(getattr(original, k), getattr(output, k))
            for k in ("p4", "charge", "category", "valid")):
        raise ValueError("Tracking-only structure changed")
    mask = np.column_stack((eligible, eligible))
    if not np.array_equal(original.tracking[~mask], output.tracking[~mask]):
        raise ValueError("Ineligible tracking changed")
