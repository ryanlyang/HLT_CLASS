"""Mechanism isolation, visible-context inverse and unchanged old recipes."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest

from test_literature_proxy import particles, equal
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.literature_proxy_v3 import kernel as v3
from hlt_classification.literature_context import kernel as k, transform as t, worker as w


@pytest.mark.parametrize("rates", [(0., 0.), (.197498487058, .246029999971), (1., 1.)])
def test_frozen_topology_and_smaller_common_noise(rates):
    p = particles((0, 1, 2, 3, 4, 5, 1, 1, 2, 2)*3, [.5, 1., 4., 5., 8., 9., 10., 10., 10., 10.]*3)
    for i in range(20):
        old, low = [f(p, f"context:{i}", *rates) for f in (v3.generate, k.generate)]
        k.assert_structure(low, old.particles, old.ancestry)
        assert old.counts == low.counts
        c = t.transform(low.particles)
        for field in ("p4", "category", "charge", "valid"):
            np.testing.assert_array_equal(getattr(c, field), getattr(low.particles, field))
        assert t.audit(low.particles, c)["inverse_max_scaled_error"] < 1e-10


def test_noise_reduction_not_rescaling_displacement():
    p = particles((4,), [10.])
    old, low = [f(p, "common-draw", 0., 0.).particles for f in (v3.generate, k.generate)]
    np.testing.assert_allclose(low.tracking[:, :2]-p.tracking[:, :2],
                               (old.tracking[:, :2]-p.tracking[:, :2])/4)
    np.testing.assert_allclose(low.tracking[:, 2:]**2-p.tracking[:, 2:]**2,
                               (old.tracking[:, 2:]**2-p.tracking[:, 2:]**2)/16)
    np.testing.assert_allclose(low.eta-p.eta, (old.eta-p.eta)/2)
    assert k.recipe()["tracking_noise_scale"] == 1 and v3.recipe()["tracking_noise_scale"] == 4
    assert k.recipe()["sides"] == list(w.SIDES)


def test_inverse_and_real_float32_frontend_with_tails():
    p = particles((4,)*25)
    rng = np.random.default_rng(33)
    tr = p.tracking.copy()
    tr[:, :2] = rng.normal(size=(25, 2))*np.geomspace(1e-6, 1e5, 25)[:, None]
    tr[:, 2:] = np.exp(rng.uniform(-10, 3, (25, 2)))
    p = Particles(p.p4, p.charge, p.category, tr, p.valid, p.keys)
    c = t.transform(p)
    evidence = t.audit(p, c)
    assert evidence["changed_particles"] == 25
    assert evidence["inverse_max_scaled_error"] < 1e-10
    assert evidence["new_frontend_inverse_max_scaled_error"] < 1e-4
    assert evidence["legacy_value_saturated"] > 0 and evidence["legacy_error_clipped"] > 0
    assert np.isfinite(t.build_inputs(c).features).all()


def test_only_visible_context_no_identity_keys_and_permutation():
    p = particles((4,)*8)
    order = [7, 2, 4, 0, 6, 3, 1, 5]
    a = t.transform(p).take(order)
    b = t.transform(p.take(order))
    assert equal(a, b)
    renamed = Particles(p.p4, p.charge, p.category, p.tracking, p.valid, tuple(f"new:{i}" for i in range(8)))
    np.testing.assert_array_equal(t.transform(p).tracking, t.transform(renamed).tracking)
    changed = Particles(p4_from_coordinates(p.pt, np.arange(8)*.3, np.arange(8)*.3, p.mass),
                        p.charge, p.category, p.tracking, p.valid, p.keys)
    assert not np.allclose(t.transform(changed).tracking, t.transform(p).tracking)
    np.testing.assert_allclose(t.transform(t.transform(changed), inverse=True).tracking, p.tracking)


def test_partial_neutral_empty_and_single_particle():
    for p in (particles(()), particles((1, 2)), particles((4,))):
        q = t.transform(p)
        assert equal(p, q) if not np.any(p.valid) else not equal(p, q)
        np.testing.assert_allclose(t.transform(q, inverse=True).tracking, p.tracking)
    p = particles((4, 4))
    tr, valid = p.tracking.copy(), p.valid.copy()
    tr[0, 2], valid[0, 2] = 0., False
    p = Particles(p.p4, p.charge, p.category, tr, valid, p.keys)
    q = t.transform(p)
    np.testing.assert_array_equal(q.tracking[0], p.tracking[0])
    assert t.audit(p, q)["eligible_particles"] == 1
    with pytest.raises(ValueError, match="boolean"):
        t.transform(p, inverse=1)


def test_exact_spawn_replay_and_order_stability():
    p = particles((4, 1, 1, 0, 2, 2), [8, .5, 8, 1, 10, 10])
    jobs = [(f"j:{i}", p, dict(p_drop=.197498487058, p_merge=.246029999971)) for i in range(8)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert list(pool.map(w.replay_task, jobs)) == list(map(w.replay_task, jobs))
    assert w.replay_task(jobs[0]) == w.replay_task((jobs[0][0], p.take([5, 4, 3, 2, 1, 0]), jobs[0][2]))
