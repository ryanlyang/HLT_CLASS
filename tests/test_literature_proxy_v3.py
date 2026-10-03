"""Only noise changes: freeze count38 decisions, full rates and random draws."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest

from test_literature_proxy import particles, equal
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.literature_proxy.kernel import Response, stream
from hlt_classification.literature_proxy_v2 import kernel as v2
from hlt_classification.literature_proxy_v3 import kernel as k, worker as w, diagnostics as d


@pytest.mark.parametrize("rates", [(0., 0.), (.197498487058, .246029999971), (1., 1.)])
def test_exact_structure_despite_stronger_noise(rates):
    p = particles((0, 1, 2, 3, 4, 5, 1, 1, 2, 2)*4, [.5, 1., 4., 5., 8., 9., 10., 10., 10., 10.]*4)
    for i in range(50):
        a, b = (fn(p, f"noise:{i}", *rates) for fn in (v2.generate, k.generate))
        k.assert_structure(b, a.particles, a.ancestry)
        assert a.counts == b.counts
        assert len(a.particles) == len(b.particles)
    broken = list(a.ancestry)
    broken[0] = ("wrong",)
    with pytest.raises(ValueError, match="structure"):
        k.assert_structure(b, a.particles, tuple(broken))


def test_common_tracking_draws_coupled_errors_and_missing_values():
    p = particles((4,), [8.])
    a, b = (fn(p, "tracking", 0., 0.).particles for fn in (v2.generate, k.generate))
    assert b.tracking[:, :2]-p.tracking[:, :2] == pytest.approx((4/3)*(a.tracking[:, :2]-p.tracking[:, :2]))
    assert b.tracking[:, 2:]**2-p.tracking[:, 2:]**2 == pytest.approx((16/9)*(a.tracking[:, 2:]**2-p.tracking[:, 2:]**2))
    tr, valid = p.tracking.copy(), p.valid.copy()
    tr[:, 2:], valid[:, 2:] = 0., False
    missing = Particles(p.p4, p.charge, p.category, tr, valid, p.keys)
    result = k.generate(missing, "tracking", 0., 0.)
    assert np.array_equal(result.particles.tracking, tr)
    assert np.array_equal(result.particles.valid, valid)
    assert result.counts["smeared_d0"] == 0


@pytest.mark.parametrize("pid,rel,angle", [(4, .02, .002), (2, .06, .006), (1, .20, .020)])
def test_mean_one_lognormal_and_angle_scales(pid, rel, angle):
    p = particles((pid,), [10.])
    a, b = (fn(p, "p4", 0., 0.).particles for fn in (v2.generate, k.generate))
    z = stream("p4", "pt").standard_normal(1)
    sigma = np.sqrt(np.log1p(rel**2))
    assert b.pt/p.pt == pytest.approx(np.exp(sigma*z-.5*sigma**2))
    assert b.eta-p.eta == pytest.approx(angle*stream("p4", "eta").standard_normal(1))
    assert b.eta-p.eta == pytest.approx((4/3)*(a.eta-p.eta))
    assert b.mass == pytest.approx(p.mass)


def test_disjoint_merge_p4_conservation_and_drops_empty():
    p = particles((1,)*6, [10.]*6)
    merged, unmerged = (k.generate(p, "merge", 0., rate) for rate in (1., 0.))
    assert merged.counts["merged_pairs"] == 3
    assert merged.particles.p4.sum(axis=0) == pytest.approx(unmerged.particles.p4.sum(axis=0))
    for p in (particles(()), particles((1, 2), [.5, .5])):
        r = k.generate(p, "empty", 1., 1.)
        assert not len(r.particles)
        col = d.Collector()
        col.paired("NOISE_V3", p, r)
        assert col.finish()["NOISE_V3/paired/delta_r"]["count"] == 0


def test_exact_spawn_and_input_order_replay():
    p = particles((1,)*8, [.5]*8)
    a, b = (k.generate(inp, "order", .2, .5) for inp in (p, p.take(list(reversed(range(8))))))
    assert equal(a.particles, b.particles) and a.ancestry == b.ancestry and a.counts == b.counts
    jobs = [(f"j:{i}", p, dict(p_drop=.197498487058, p_merge=.246029999971)) for i in range(8)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert list(pool.map(w.replay_task, jobs)) == list(map(w.replay_task, jobs))


def test_paired_diagnostics_wrapped_phi_survivors_and_width():
    p = particles((4, 1, 1), [10.]*3)
    p = Particles(p4_from_coordinates(p.pt, [0.]*3, [np.pi-.001]*3, .14),
                  p.charge, p.category, p.tracking, p.valid, p.keys)
    col = d.Collector()
    for side, fn in (("COUNT38_V2", v2.generate), ("NOISE_V3", k.generate)):
        col.paired(side, p, fn(p, "wrap", 0., 1.))
    rows = col.finish()
    assert rows["NOISE_V3/single_parent/pt_ratio"]["count"] == 1  # Merge excluded.
    assert abs(rows["NOISE_V3/paired/delta_phi"]["sum"]) < .1
    for field in ("d0_delta", "dz_delta", "delta_eta", "delta_phi"):
        a, b = (rows[f"{s}/single_parent/{field}"] for s in d.SIDES[1:])
        assert b["sum"] == pytest.approx(a["sum"]*(4/3))
    c = d.Collector()
    c.add("NOISE_V3/paired/count_delta", [-30, -3, 0])
    assert c.finish()["NOISE_V3/paired/count_delta"]["underflow"] == 0


@pytest.mark.parametrize("rate", [True, -.01, 1.01, float("nan"), float("inf")])
def test_bad_probabilities(rate):
    with pytest.raises(ValueError):
        k.generate(particles(), "bad", rate, 0.)
    with pytest.raises(ValueError):
        k.generate(particles(), "bad", 0., rate)


def test_recipe_versions_do_not_alias():
    assert v2.recipe()["tracking_noise_scale"] == 3
    assert v2.recipe()["kinematic_noise_scale"] == 1.5
    assert k.recipe()["tracking_noise_scale"] == 4
    assert k.recipe()["kinematic_noise_scale"] == 2
    assert k.recipe()["parent_recipe"] == v2.recipe()
    assert k.recipe()["schema_version"] == 3
