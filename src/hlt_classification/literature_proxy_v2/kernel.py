"""Multiple eligible losses, disjoint merges and separately scaled noise."""
from dataclasses import dataclass

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates, wrap_phi
from hlt_classification.literature_proxy.kernel import Response, geometry, stream, recipe as old_recipe
from .contracts import artifact, validate

TARGET = 38.
DROP_BUDGET = 2.
PID_SCALE = TRACK_SCALE = 3.
KINEMATIC_SCALE = 1.5


def recipe():
    return artifact("RECIPE", base_recipe=old_recipe(), target_mean=TARGET,
                    intended_drops_per_jet=DROP_BUDGET, pid_scale=PID_SCALE,
                    tracking_noise_scale=TRACK_SCALE, kinematic_noise_scale=KINEMATIC_SCALE,
                    drop="independent_original_CH_NH_photon_pt_lt_2",
                    merge="random_priority_greedy_maximal_disjoint_post_PID_neutrals_dr_lt_0.03",
                    order="pid_tracking_p4_drop_disjoint_merge_v2",
                    new_random_domains=["count38_v2/drop", "count38_v2/pair_priority", "count38_v2/merge"],
                    calibration="two_pass_training_counts_realized_drop_then_expected_merge",
                    count_calibrated_on_training=True, fitted_to_CMS=False,
                    classification="controlled_synthetic_benchmark_not_CMS_HLT")


def probability(value):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability must be finite in [0,1]")
    return float(value)


def eligible(p):
    return np.isin(p.category, (0, 1, 2)) & (p.pt < 2.)


def drop_rate(jets, particles, soft):
    if any(type(v) is not int or v < 0 for v in (jets, particles, soft)) or not jets or soft > particles:
        raise ValueError("Invalid calibration counts")
    deficit = max(particles - TARGET * jets, 0.)
    budget = min(DROP_BUDGET * jets, deficit)
    return min(1., budget / soft) if soft else 0.


def calibration(spec_hash, *, jets, particles, soft, dropped, pairs):
    p_drop = drop_rate(jets, particles, soft)
    if any(type(v) is not int or v < 0 for v in (dropped, pairs)) or dropped > soft or 2*pairs > particles-dropped:
        raise ValueError("Invalid topology counts")
    remaining = max(particles - TARGET*jets - dropped, 0.)
    p_merge = min(1., remaining / pairs) if pairs else 0.
    return artifact("CALIBRATION", parents=dict(spec=spec_hash), jets=jets,
                    input_particles=particles, eligible_soft_particles=soft,
                    realized_drops=dropped, disjoint_candidate_pairs=pairs,
                    p_drop=p_drop, p_merge=p_merge, expected_drops=p_drop*soft,
                    expected_merges=p_merge*pairs,
                    predicted_mean=(particles-dropped-p_merge*pairs)/jets,
                    residual_capacity_shortfall=max(remaining-pairs, 0.),
                    target_mean=TARGET, final_test_accessed=False)


def validate_calibration(value, spec_hash):
    validate(value, "CALIBRATION")
    expected = calibration(spec_hash, jets=value["jets"], particles=value["input_particles"],
                           soft=value["eligible_soft_particles"], dropped=value["realized_drops"],
                           pairs=value["disjoint_candidate_pairs"])
    if value != expected:
        raise ValueError("Frozen calibration differs")


@dataclass
class Topology:
    original: Particles
    crowding: np.ndarray
    category: np.ndarray
    converted: np.ndarray
    keep: np.ndarray
    pairs: list
    eligible_soft: int


def topology(p, identity, p_drop):
    p_drop = probability(p_drop)
    p = p.take(sorted(range(len(p)), key=lambda i: p.keys[i]))
    c, dr = geometry(p)
    category = p.category.copy()
    prob = PID_SCALE * np.where(category == 0, .03+.05*np.exp(-p.pt/2.)+.04*c,
                                np.where(category == 3, .01*(1+c), 0.))
    converted = stream(identity, "pid").random(len(p)) < prob
    category[converted & (p.category == 0)] = 1
    category[converted & (p.category == 3)] = 2
    soft = eligible(p)
    keep = ~(soft & (stream(identity, "count38_v2/drop").random(len(p)) < p_drop))
    edges = np.argwhere(np.triu((dr < .03) & (category[:, None] == category)
                                & np.isin(category[:, None], (1, 2))
                                & keep[:, None] & keep, k=1))
    priority = stream(identity, "count38_v2/pair_priority").random(len(edges))
    used, pairs = set(), []
    # Canonical edge order supplies deterministic tie-breaking. Maximal, not maximum.
    for index in np.argsort(priority, kind="stable"):
        i, j = map(int, edges[index])
        if i not in used and j not in used:
            pairs.append((i, j))
            used.update((i, j))
    return Topology(p, c, category, converted, keep, pairs, int(soft.sum()))


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
