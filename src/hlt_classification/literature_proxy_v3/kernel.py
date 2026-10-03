"""Noise-only fork of count38 v2; original topology and streams are frozen."""
import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates, wrap_phi
from hlt_classification.literature_proxy.kernel import Response, stream
from hlt_classification.literature_proxy_v2.kernel import topology, probability, recipe as parent_recipe
from .contracts import artifact

TRACK_SCALE = 4.
KINEMATIC_SCALE = 2.


def recipe():
    return artifact("RECIPE", parent_recipe=parent_recipe(),
                    tracking_noise_scale=TRACK_SCALE, kinematic_noise_scale=KINEMATIC_SCALE,
                    calibration="reuse_exact_completed_v2_no_refit",
                    topology="identical_saved_v2_every_jet", random_streams="unchanged_v2",
                    classification="controlled_synthetic_benchmark_not_CMS_HLT")


def assert_structure(response, control, ancestry):
    p = response.particles
    if (response.ancestry != ancestry or p.keys != control.keys
            or any(not np.array_equal(getattr(p, k), getattr(control, k))
                   for k in ("charge", "category", "valid"))):
        raise ValueError("Noise-only structure differs from saved v2")


def generate(p, identity, p_drop, p_merge):
    p_merge = probability(p_merge)
    t = topology(p, identity, p_drop)
    p, c = t.original, t.crowding
    n = len(p)
    charge, tr, valid = p.charge.copy(), p.tracking.copy(), p.valid.copy()
    charge[t.converted] = 0
    tr[t.converted], valid[t.converted] = 0., False
    ok = valid[:, :2] & valid[:, 2:]
    extra = TRACK_SCALE**2 * (((1.5+.5*c[:, None])**2-1)*tr[:, 2:]**2 + np.array([.020, .050])**2)
    tr[:, :2] = np.where(ok, tr[:, :2]+np.sqrt(extra)*stream(identity, "tracking").standard_normal((n, 2)), tr[:, :2])
    tr[:, 2:] = np.where(ok, np.sqrt(tr[:, 2:]**2+extra), tr[:, 2:])
    kind = np.where(charge != 0, 0, np.where(t.category == 2, 2, 1))
    rel = np.array([.01, .10, .03])[kind]*KINEMATIC_SCALE
    angular = np.array([.001, .010, .003])[kind]*KINEMATIC_SCALE
    log_sd = np.sqrt(np.log1p(rel**2))
    scale = np.exp(log_sd*stream(identity, "pt").standard_normal(n)-.5*log_sd**2)
    p4 = p4_from_coordinates(p.pt*scale, p.eta+angular*stream(identity, "eta").standard_normal(n),
                              wrap_phi(p.phi+angular*stream(identity, "phi").standard_normal(n)), p.mass)
    ancestry = [(k,) for k in p.keys]
    dropped = int((~t.keep).sum())
    accepted = stream(identity, "count38_v2/merge").random(len(t.pairs)) < p_merge
    for (i, j), accept in zip(t.pairs, accepted):
        if accept:
            p4[i] += p4[j]
            t.keep[j] = False
            ancestry[i] = (p.keys[i], p.keys[j])
    idx = np.flatnonzero(t.keep)
    ancestors = tuple(ancestry[i] for i in idx)
    result = Particles(p4[idx], charge[idx], t.category[idx], tr[idx], valid[idx],
                       tuple("+".join(a) for a in ancestors))
    counts = dict(jets=1, input_particles=n, output_particles=len(result),
                  eligible_soft_particles=t.eligible_soft, dropped_particles=dropped,
                  disjoint_candidate_pairs=len(t.pairs), merged_pairs=int(accepted.sum()),
                  drop_eligible_jets=int(t.eligible_soft > 0), merge_eligible_jets=int(bool(t.pairs)),
                  charged_hadron_eligible=int((p.category == 0).sum()),
                  charged_to_neutral=int((t.converted & (p.category == 0)).sum()),
                  electron_eligible=int((p.category == 3).sum()),
                  electron_to_photon=int((t.converted & (p.category == 3)).sum()),
                  smeared_d0=int(ok[:, 0].sum()), smeared_dz=int(ok[:, 1].sum()), empty_output=int(not len(result)))
    return Response(result, ancestors, counts)
