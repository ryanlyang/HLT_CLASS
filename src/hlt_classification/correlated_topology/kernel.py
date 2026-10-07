"""Tracking noise and topology are separate; no PID or extra p4 smearing."""
from dataclasses import dataclass
import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles, wrap_phi
from hlt_classification.correlated_tracking.kernel import generate_all, recipe as tracking_recipe
from hlt_classification.literature_proxy.kernel import stream
from hlt_classification.literature_proxy_v2.kernel import probability, validate_calibration
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash

HISTORICAL_REPLAY = dict(structure='exact', ineligible_tracking='exact',
    eligible_tracking='allclose', rtol=1e-12, atol_mm=1e-12,
    purpose='ARM_to_x86_roundoff_only_not_physics_tolerance')


def recipe(calibration):
    validate_calibration(calibration, calibration['parents']['spec'])
    return with_content_hash(dict(contract='JC2_CORR_HIGH_TOPO_RECIPE/v1', schema_version=1,
        name='CORR_HIGH_TOPO', parents=dict(calibration=calibration['content_hash']),
        tracking_recipe=tracking_recipe(), strength=2., endpoint='CORR_HIGH',
        historical_replay=dict(HISTORICAL_REPLAY),
        p_drop=calibration['p_drop'], p_merge=calibration['p_merge'],
        drop='original_CH_NH_photon_pt_lt_2', merge='original_same_neutral_dr_lt_0.03_disjoint',
        pid_changes=False, extra_kinematic_smearing=False, merged_tracking='invalid_zero',
        random_domains='literature_count38_v2_without_pid', redraw_per_epoch=False,
        classification='exploratory_synthetic_mechanism_not_CMS_HLT', final_test_accessed=False))


def validate_recipe(value):
    validate_content_hash(value, expected_contract='JC2_CORR_HIGH_TOPO_RECIPE/v1', expected_schema_version=1)
    probability(value['p_drop']); probability(value['p_merge'])
    if (value['tracking_recipe'] != tracking_recipe() or value['strength'] != 2.
            or value['historical_replay'] != HISTORICAL_REPLAY
            or value['endpoint'] != 'CORR_HIGH' or value['pid_changes'] is not False
            or value['extra_kinematic_smearing'] is not False or value['final_test_accessed'] is not False
            or value['name'] != 'CORR_HIGH_TOPO' or value['redraw_per_epoch'] is not False
            or value['drop'] != 'original_CH_NH_photon_pt_lt_2'
            or value['merge'] != 'original_same_neutral_dr_lt_0.03_disjoint'
            or value['merged_tracking'] != 'invalid_zero'
            or value['random_domains'] != 'literature_count38_v2_without_pid'
            or value['classification'] != 'exploratory_synthetic_mechanism_not_CMS_HLT'):
        raise ValueError('CORR_HIGH_TOPO recipe differs')


def equal(a, b):
    return all(np.array_equal(getattr(a, k), getattr(b, k))
               for k in ('p4', 'charge', 'category', 'tracking', 'valid'))


@dataclass(frozen=True)
class Response:
    particles: Particles
    mapping: np.ndarray
    counts: dict
    historical_replay: dict


def check_historical(expected, saved, original):
    if len(expected) != len(saved) or any(not np.array_equal(getattr(expected, k), getattr(saved, k))
                                           for k in ('p4', 'charge', 'category', 'valid')):
        raise ValueError('Historical CORR_MID structure differs')
    eligible = original.valid[:, :2] & original.valid[:, 2:]
    mask = np.column_stack((eligible, eligible))
    if not np.array_equal(expected.tracking[~mask], saved.tracking[~mask]):
        raise ValueError('Historical CORR_MID ineligible tracking differs')
    delta = np.abs(expected.tracking-saved.tracking)
    scaled = delta/(HISTORICAL_REPLAY['atol_mm'] + HISTORICAL_REPLAY['rtol']*np.abs(saved.tracking))
    if np.any(scaled > 1):
        raise ValueError('Historical CORR_MID replay differs beyond roundoff tolerance')
    return dict(jets=1, bitwise_exact_jets=int(np.array_equal(expected.tracking, saved.tracking)),
                max_abs_tracking_difference_mm=float(np.max(delta, initial=0.)),
                max_tolerance_fraction=float(np.max(scaled, initial=0.)))


def generate(offline, identity, frozen, *, historical_mid=None):
    validate_recipe(frozen)
    sides, _ = generate_all(offline, identity)
    replay = (check_historical(sides['CORR_MID'], historical_mid, offline)
              if historical_mid is not None else {})
    order = np.argsort(np.asarray(offline.keys, dtype=str), kind='stable')
    p = sides['CORR_HIGH'].take(order)
    n = len(p)
    soft = np.isin(p.category, (0, 1, 2)) & (p.pt < 2.)
    keep = ~(soft & (stream(identity, 'count38_v2/drop').random(n) < frozen['p_drop']))
    dropped = int((~keep).sum())
    dr = np.sqrt((p.eta[:, None]-p.eta)**2 + wrap_phi(p.phi[:, None]-p.phi)**2)
    edges = np.argwhere(np.triu((dr < .03) & (p.category[:, None] == p.category)
        & np.isin(p.category[:, None], (1, 2)) & keep[:, None] & keep, k=1))
    priorities = stream(identity, 'count38_v2/pair_priority').random(len(edges))
    pairs, used = [], set()
    for index in np.argsort(priorities, kind='stable'):
        i, j = map(int, edges[index])
        if i not in used and j not in used:
            pairs.append((i, j)); used.update((i, j))
    accepted = stream(identity, 'count38_v2/merge').random(len(pairs)) < frozen['p_merge']
    p4, tracking, valid = p.p4.copy(), p.tracking.copy(), p.valid.copy()
    keys = list(p.keys)
    for (i, j), yes in zip(pairs, accepted):
        if yes:
            p4[i] += p4[j]
            tracking[i], valid[i] = 0., False
            keys[i] = p.keys[i]+'+'+p.keys[j]
            keep[j] = False
    idx = np.flatnonzero(keep)
    out = Particles(p4[idx], p.charge[idx], p.category[idx], tracking[idx], valid[idx],
                    tuple(keys[i] for i in idx))
    counts = dict(jets=1, input_particles=n, output_particles=len(out), dropped_particles=dropped,
        eligible_soft_particles=int(soft.sum()), disjoint_candidate_pairs=len(pairs),
        merged_pairs=int(accepted.sum()), empty_output=int(len(out) == 0))
    if n - dropped - int(accepted.sum()) != len(out):
        raise ValueError('Topology accounting differs')
    return Response(out, order[idx].astype(np.int32), counts, replay)
