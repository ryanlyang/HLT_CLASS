"""Frozen, label-blind physical response. Constants are benchmark choices."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates, wrap_phi
from hlt_classification.data.cache_contracts import canonical_json_bytes, with_content_hash

VARIANTS = {"MILD": .5, "NOMINAL": 1., "STRONG": 1.5}
SEED = 20261002
DOMAIN = "JC2_LITERATURE_PROXY_RANDOM/v1"


def recipe():
    return with_content_hash(dict(
        contract="JC2_LITERATURE_PROXY_RECIPE/v1", schema_version=1,
        classification="controlled_synthetic_benchmark_not_CMS_HLT", seed=SEED,
        variants=dict(VARIANTS), crowding_radius=.05,
        charged_to_neutral=dict(base=.03, soft=.05, pt_scale_gev=2., crowding=.04),
        electron_to_photon=dict(base=.01, crowding=.01),
        track_k=dict(base=1.5, crowding=.5), track_floor_mm=[.020, .050],
        relative_pt_sd=[.01, .10, .03], angle_sd=[.001, .010, .003],
        drop_probability=.05, drop_pt_max_gev=2., merge_probability=.05, merge_dr=.03,
        random_domain=DOMAIN, order="pid_track_p4_drop_merge_v1",
        units=dict(momentum="GeV", length="mm"), fitted_parameters=False,
    ))


def stream(identity, domain):
    if not isinstance(identity, str) or not identity:
        raise ValueError("Nonempty immutable jet identity required")
    seed = hashlib.sha256(canonical_json_bytes([DOMAIN, SEED, identity, domain])).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(seed, "big")))


def geometry(p):
    if len(p) < 2:
        return np.zeros(len(p)), np.full((len(p), len(p)), np.inf)
    dr = np.hypot(p.eta[:, None] - p.eta, wrap_phi(p.phi[:, None] - p.phi))
    np.fill_diagonal(dr, np.inf)
    return np.maximum(0., 1. - dr.min(axis=1) / .05), dr


@dataclass(frozen=True)
class Response:
    particles: Particles
    # Construction metadata is deliberately outside physical particle arrays.
    ancestry: tuple[tuple[str, ...], ...]
    counts: dict


def generate(p: Particles, identity: str, strength: float = 1.) -> Response:
    if strength not in (0., *VARIANTS.values()):
        raise ValueError("Unregistered response strength")
    if strength == 0:
        return Response(p, tuple((k,) for k in p.keys), {})
    p = p.take(sorted(range(len(p)), key=lambda i: p.keys[i]))
    n = len(p)
    c, dr = geometry(p)
    category, charge = p.category.copy(), p.charge.copy()
    tr, valid = p.tracking.copy(), p.valid.copy()
    prob = np.where(category == 0, .03 + .05 * np.exp(-p.pt / 2.) + .04 * c,
                    np.where(category == 3, .01 * (1. + c), 0.)) * strength
    converted = stream(identity, "pid").random(n) < prob
    category[converted & (p.category == 0)] = 1
    category[converted & (p.category == 3)] = 2
    charge[converted] = 0
    tr[converted], valid[converted] = 0., False
    ok = valid[:, :2] & valid[:, 2:]
    extra = strength**2 * (((1.5 + .5 * c[:, None])**2 - 1.) * tr[:, 2:]**2
                           + np.array([.020, .050])**2)
    noise = stream(identity, "tracking").standard_normal((n, 2))
    tr[:, :2] = np.where(ok, tr[:, :2] + np.sqrt(extra) * noise, tr[:, :2])
    tr[:, 2:] = np.where(ok, np.sqrt(tr[:, 2:]**2 + extra), tr[:, 2:])
    response_type = np.where(charge != 0, 0, np.where(category == 2, 2, 1))
    rel = np.array([.01, .10, .03])[response_type] * strength
    angular = np.array([.001, .010, .003])[response_type] * strength
    log_sd = np.sqrt(np.log1p(rel**2))
    scale = np.exp(log_sd * stream(identity, "pt").standard_normal(n) - .5 * log_sd**2)
    eta = p.eta + angular * stream(identity, "eta").standard_normal(n)
    phi = wrap_phi(p.phi + angular * stream(identity, "phi").standard_normal(n))
    p4 = p4_from_coordinates(p.pt * scale, eta, phi, p.mass)
    keep = np.ones(n, bool)
    drop = np.flatnonzero(np.isin(p.category, (0, 1, 2)) & (p.pt < 2.))
    dropped = 0
    if len(drop) and stream(identity, "drop_gate").random() < .05 * strength:
        keep[drop[stream(identity, "drop_choice").integers(len(drop))]] = False
        dropped = 1
    pairs = np.argwhere(np.triu((dr < .03) & (category[:, None] == category)
                                & np.isin(category[:, None], (1, 2))
                                & keep[:, None] & keep, k=1))
    ancestry = [(k,) for k in p.keys]
    merged = 0
    if len(pairs) and stream(identity, "merge_gate").random() < .05 * strength:
        i, j = pairs[stream(identity, "merge_choice").integers(len(pairs))]
        p4[i] += p4[j]
        keep[j] = False
        ancestry[i] = tuple(sorted((p.keys[i], p.keys[j])))
        merged = 1
    idx = np.flatnonzero(keep)
    ancestors = tuple(ancestry[i] for i in idx)
    result = Particles(p4[idx], charge[idx], category[idx], tr[idx], valid[idx],
                       tuple("+".join(a) for a in ancestors))
    counts = dict(jets=1, input_particles=n, output_particles=len(result),
                  charged_hadron_eligible=int((p.category == 0).sum()),
                  charged_to_neutral=int((converted & (p.category == 0)).sum()),
                  electron_eligible=int((p.category == 3).sum()),
                  electron_to_photon=int((converted & (p.category == 3)).sum()),
                  drop_eligible_jets=int(bool(len(drop))), dropped_particles=dropped,
                  merge_eligible_jets=int(bool(len(pairs))), merged_pairs=merged,
                  smeared_d0=int(ok[:, 0].sum()), smeared_dz=int(ok[:, 1].sum()),
                  empty_output=int(len(result) == 0))
    return Response(result, ancestors, counts)
