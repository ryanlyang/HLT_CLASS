"""Frozen response semantics, no physics-quality acceptance thresholds."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest

from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates, JC2_FIELDS
from hlt_classification.literature_proxy.kernel import generate, geometry, stream, recipe, VARIANTS
from hlt_classification.literature_proxy.population import from_columns
from hlt_classification.literature_proxy.worker import digest_response
from hlt_classification.literature_proxy import diagnostics as d


def particles(categories=(0, 1, 1, 2, 3, 4, 5), pt=None):
    n = len(categories)
    pt = np.array(pt if pt is not None else np.arange(n) + 1.)
    charge = np.where(np.isin(categories, (0, 3, 4)), 1, 0)
    valid = np.repeat((charge != 0)[:, None], 4, axis=1)
    tracking = np.tile([.1, -.2, .03, .08], (n, 1)) * valid
    return Particles(p4_from_coordinates(pt, np.arange(n)*.003, np.arange(n)*.003, .14),
                     charge, np.array(categories), tracking, valid, tuple(f"p:{i}" for i in range(n)))


def equal(a, b):
    return a.keys == b.keys and all(np.array_equal(getattr(a, f), getattr(b, f))
                                   for f in ("p4", "charge", "category", "tracking", "valid"))


def test_identity_and_replay_input_order():
    p = particles()
    assert equal(generate(p, "jet", 0).particles, p)
    for strength in VARIANTS.values():
        a = generate(p, "jet", strength)
        b = generate(p.take(list(reversed(range(len(p))))), "jet", strength)
        assert equal(a.particles, b.particles)
        assert a.ancestry == b.ancestry and a.counts == b.counts
    with pytest.raises(ValueError, match="Unregistered"):
        generate(p, "jet", 2.)


def test_real_spawn_parity():
    inputs = [(f"jet:{i}", particles()) for i in range(8)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert list(pool.map(digest_response, inputs)) == list(map(digest_response, inputs))


def test_crowding_wrapped_phi_empty_single():
    assert geometry(particles(()))[0].size == 0
    assert geometry(particles((0,)))[0].tolist() == [0.]
    p = particles((1, 1))
    p = Particles(p4_from_coordinates([2, 2], [0, 0], [np.pi-.001, -np.pi+.001], 0),
                  p.charge, p.category, p.tracking, p.valid, p.keys)
    assert geometry(p)[0] == pytest.approx([.96, .96])
    assert len(generate(particles(()), "empty").particles) == 0


def test_coupled_tracking_same_noise_strength_and_unavailable():
    p = particles((4, 4), [10., 15.])  # muons do not convert or drop
    c, _ = geometry(p)
    for strength in VARIANTS.values():
        r = generate(p, "tracking", strength).particles
        variance = strength**2 * (((1.5 + .5*c[:, None])**2 - 1) * p.tracking[:, 2:]**2 + [.02**2, .05**2])
        z = stream("tracking", "tracking").standard_normal((2, 2))
        assert r.tracking[:, :2] == pytest.approx(p.tracking[:, :2] + variance**.5*z)
        assert r.tracking[:, 2:]**2 == pytest.approx(p.tracking[:, 2:]**2 + variance)
    tr, mask = p.tracking.copy(), p.valid.copy()
    tr[:, 2:], mask[:, 2:] = 0., False
    invalid = Particles(p.p4, p.charge, p.category, tr, mask, p.keys)
    out = generate(invalid, "invalid").particles
    assert np.array_equal(out.tracking, invalid.tracking)
    assert np.array_equal(out.valid, invalid.valid)


def test_pid_coherence_loss_caps_and_physical_p4():
    p = particles((0, 1, 1, 2, 3, 4, 5), [.5, .7, .8, 1., 3., 4., 5.])
    totals = dict(converted=0, dropped=0, merged=0)
    for i in range(200):
        r = generate(p, f"coherence:{i}", 1.5)
        assert 0 <= len(p) - len(r.particles) <= 2
        assert r.counts["dropped_particles"] <= 1 and r.counts["merged_pairs"] <= 1
        for j, ancestry in enumerate(r.ancestry):
            assert len(ancestry) in (1, 2)
            if r.particles.charge[j] == 0:
                assert not r.particles.valid[j].any()
            if ancestry == ("p:5",):
                assert r.particles.category[j] == 4  # retained muon
        totals["converted"] += r.counts["charged_to_neutral"]
        totals["dropped"] += r.counts["dropped_particles"]
        totals["merged"] += r.counts["merged_pairs"]
    assert all(n > 0 for n in totals.values())


def test_merge_conserves_smeared_four_vector():
    p = particles((1, 1), [10., 12.])
    identity = next(f"m:{i}" for i in range(1000) if stream(f"m:{i}", "merge_gate").random() < .05)
    r = generate(p, identity)
    sd = np.sqrt(np.log1p(.1**2))
    scale = np.exp(sd*stream(identity, "pt").standard_normal(2) - .5*sd**2)
    expected = p4_from_coordinates(p.pt*scale, p.eta+.01*stream(identity, "eta").standard_normal(2),
                                    p.phi+.01*stream(identity, "phi").standard_normal(2), p.mass)
    assert len(r.particles) == 1
    assert r.particles.p4[0] == pytest.approx(expected.sum(axis=0))


def test_offline_adapter_units_unknown_and_masks():
    p = particles((0, 4, 1))
    columns = dict(zip(JC2_FIELDS[:4], p.p4.T))
    columns["charge"] = p.charge
    for i, name in enumerate(JC2_FIELDS[5:10]):
        columns[name] = (p.category == i).astype(float)
    for name, values in zip(("d0val", "dzval", "d0err", "dzerr"), p.tracking.T):
        columns[name] = values.copy()
    columns["dzval"][0] = np.nan
    columns["d0err"][1] = 0.
    columns["isElectron"][0] = 1.  # ambiguous -> explicit unknown, never argmax guess
    out = from_columns(columns)
    assert out.category.tolist() == [5, 4, 1]
    assert out.tracking[1, 0] == p.tracking[1, 0]  # mm unchanged
    assert out.tracking[0, 1] == 0. and not out.valid[0, 1]
    assert not out.valid[0, 3] and not out.valid[1, 2]
    columns["d0err"][1] = -1.
    with pytest.raises(ValueError, match="Negative"):
        from_columns(columns)


def test_diagnostic_histogram_tails_and_merge():
    a = d.summarize([-1e9, 0., 1e9], "d0")
    assert a["underflow"] == a["overflow"] == 1
    assert d.quantile(a, .01).startswith("<") and d.quantile(a, .99).startswith(">")
    merged = d.merge([{ "d0": a}, {"d0": a}])["d0"]
    assert merged["count"] == 6 and d.moments(merged) == d.moments(a)
    assert recipe()["fitted_parameters"] is False


def test_registered_conversion_rate_and_noise_distribution():
    # Implementation calibration on synthetic inputs, not tuning to physics data.
    p = particles((0,), [3.])
    muon = particles((4,), [3.])
    n = 1500
    converted = 0
    pulls = []
    scales = []
    variance = (1.5**2-1.) * muon.tracking[0, 2]**2 + .02**2
    for i in range(n):
        converted += generate(p, f"rate:{i}").counts["charged_to_neutral"]
        r = generate(muon, f"noise:{i}").particles
        pulls.append((r.tracking[0, 0] - muon.tracking[0, 0]) / np.sqrt(variance))
        scales.append(r.pt[0] / muon.pt[0])
    probability = .03 + .05*np.exp(-1.5)
    assert abs(converted - n*probability) < 6*np.sqrt(n*probability*(1-probability))
    assert abs(np.mean(pulls)) < .12 and .85 < np.std(pulls) < 1.15
    assert abs(np.mean(scales)-1.) < .0015 and .0085 < np.std(scales) < .0115
