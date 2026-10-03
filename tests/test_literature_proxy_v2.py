"""V2 count targeting without labels, physical eligibility or v1 regressions."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest

from test_literature_proxy import particles, equal
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.literature_proxy import kernel as old, diagnostics as d
from hlt_classification.literature_proxy_v2 import kernel as k, worker as w
from hlt_classification.literature_proxy_v2.contracts import with_content_hash


def test_multiple_soft_drops_not_electrons_muons_unknowns():
    p = particles((0, 1, 2, 0, 1, 3, 4, 5), [.5]*8)
    r = k.generate(p, "loss", 1., 0.)
    assert r.counts["dropped_particles"] == 5
    assert r.counts["output_particles"] == 3
    assert {a[0] for a in r.ancestry} == {"p:5", "p:6", "p:7"}
    hard = particles((0, 1, 2), [2., 4., 8.])
    assert k.generate(hard, "loss", 1., 0.).counts["dropped_particles"] == 0


def test_disjoint_merges_neutral_conservation_and_no_cascade():
    p = particles((1,)*6, [10.]*6)
    original = k.generate(p, "merge", 0., 0.)
    merged = k.generate(p, "merge", 0., 1.)
    assert merged.counts["merged_pairs"] == 3
    assert merged.counts["output_particles"] == 3
    assert all(len(a) == 2 for a in merged.ancestry)
    assert len({v for a in merged.ancestry for v in a}) == 6
    assert merged.particles.p4.sum(axis=0) == pytest.approx(original.particles.p4.sum(axis=0))
    assert not merged.particles.charge.any() and not merged.particles.valid.any()
    for categories in ((1, 2), (4, 4)):
        assert k.generate(particles(categories, [10., 10.]), "none", 0., 1.).counts["merged_pairs"] == 0
    p = particles((1,)*30, [10.]*30)
    t = k.topology(p, "many", 0.)
    for i, j in t.pairs:
        assert old.geometry(t.original)[1][i, j] < .03


def test_tracking_and_kinematic_scales_with_common_random_numbers():
    p = particles((4,), [8.])  # No PID conversion, drop or merge.
    baseline = old.generate(p, "noise", 1.).particles
    out = k.generate(p, "noise", 0., 0.).particles
    assert out.tracking[0, :2]-p.tracking[0, :2] == pytest.approx(3*(baseline.tracking[0, :2]-p.tracking[0, :2]))
    assert out.tracking[0, 2:]**2-p.tracking[0, 2:]**2 == pytest.approx(9*(baseline.tracking[0, 2:]**2-p.tracking[0, 2:]**2))
    assert out.eta-p.eta == pytest.approx(1.5*(baseline.eta-p.eta))
    assert out.mass == pytest.approx(p.mass)
    tr, valid = p.tracking.copy(), p.valid.copy()
    tr[:, 2:] = 0
    valid[:, 2:] = False
    missing = Particles(p.p4, p.charge, p.category, tr, valid, p.keys)
    r = k.generate(missing, "noise", 0., 0.).particles
    assert np.array_equal(r.tracking, missing.tracking) and np.array_equal(r.valid, missing.valid)


def test_pid_probabilities_and_coherent_neutral_masks():
    p = particles((0, 3, 4), [5., 5., 5.])
    t = k.topology(p, "pid-test", 0.)
    c, _ = old.geometry(p)
    expected = old.stream("pid-test", "pid").random(3) < np.array([
        3*(.03+.05*np.exp(-2.5)+.04*c[0]), 3*.01*(1+c[1]), 0.])
    assert np.array_equal(t.converted, expected)
    for i in range(60):
        r = k.generate(p, f"convert:{i}", 0., 0.)
        neutrals = r.particles.charge == 0
        assert not r.particles.valid[neutrals].any()
        assert not r.particles.tracking[neutrals].any()


def test_calibration_formulas_shortfall_and_low_input_mean():
    rate = k.drop_rate(20_000, 818_458, 200_000)
    assert rate == .2
    r = k.calibration("a"*64, jets=20_000, particles=818_458, soft=200_000, dropped=40_000, pairs=50_000)
    assert r["p_merge"] == 18_458/50_000 and r["predicted_mean"] == 38.
    k.validate_calibration(r, "a"*64)
    bad = with_content_hash(dict(r, p_merge=.9))
    with pytest.raises(ValueError, match="calibration"):
        k.validate_calibration(bad, "a"*64)
    shortage = k.calibration("s", jets=10, particles=410, soft=10, dropped=10, pairs=2)
    assert shortage["p_drop"] == shortage["p_merge"] == 1.
    assert shortage["residual_capacity_shortfall"] == 18 and shortage["predicted_mean"] == 39.8
    low = k.calibration("l", jets=10, particles=30, soft=0, dropped=0, pairs=0)
    assert low["p_drop"] == low["p_merge"] == 0 and low["predicted_mean"] == 3.


@pytest.mark.parametrize("p", [-.1, 1.1, float("nan"), float("inf"), True])
def test_bad_probability_rejected(p):
    with pytest.raises(ValueError):
        k.generate(particles(), "bad", p, 0.)
    with pytest.raises(ValueError):
        k.generate(particles(), "bad", 0., p)


def test_frozen_replay_input_order_process_count_and_empty():
    p = particles((1,)*8, [.5]*8)
    a = k.generate(p, "order", .2, .5)
    b = k.generate(p.take(list(reversed(range(len(p))))), "order", .2, .5)
    assert equal(a.particles, b.particles) and a.ancestry == b.ancestry and a.counts == b.counts
    rows = [(f"j:{i}", p, dict(p_drop=.2, p_merge=.5)) for i in range(8)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert list(pool.map(w.replay_task, rows)) == list(map(w.replay_task, rows))
    assert k.generate(particles(()), "empty", 1., 1.).counts["empty_output"] == 1
    assert k.generate(p, "all-dropped", 1., 1.).counts["empty_output"] == 1


def test_large_count_target_and_not_per_jet_forced():
    p = particles((1,)*41, [1.]*41)
    n = 300
    p_drop = k.drop_rate(n, 41*n, 41*n)
    top = [k.topology(p, f"target:{i}", p_drop) for i in range(n)]
    rates = k.calibration("sample", jets=n, particles=41*n, soft=41*n,
                          dropped=sum(int((~t.keep).sum()) for t in top), pairs=sum(len(t.pairs) for t in top))
    counts = [len(k.generate(p, f"target:{i}", rates["p_drop"], rates["p_merge"]).particles) for i in range(n)]
    assert abs(np.mean(counts)-38) < .4
    assert len(set(counts)) > 3  # Statistical mean, never truncate every jet to 38.


def test_count_histogram_and_v1_default_unchanged():
    col = w.Collector()
    col.add("COUNT38_V2/paired/count_delta", [-10, -4, 0])
    row = col.finish()["COUNT38_V2/paired/count_delta"]
    assert row["underflow"] == row["overflow"] == 0 and sum(row["histogram"]) == 3
    assert d.summarize([-4, 0], "count_delta")["underflow"] == 1
    assert old.recipe()["drop_probability"] == .05
    assert k.recipe()["fitted_to_CMS"] is False
