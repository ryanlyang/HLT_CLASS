"""Geometry, conditional marginals and cross-track covariance of the new toy."""
import numpy as np
import pytest

from test_literature_proxy import particles, equal
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.correlated_tracking import kernel as k
from hlt_classification.correlated_tracking.diagnostics import pair_moments


def test_projected_reference_geometry_and_uncertainty():
    p = particles((0, 4), [10., 4.])
    p = Particles(p4_from_coordinates(p.pt, [0., np.arcsinh(2.)], [0., np.pi/2], p.mass),
                  p.charge, p.category, p.tracking, p.valid, p.keys)
    sides, g = k.generate_all(p, "geometry")
    np.testing.assert_allclose(g.projected_sd[0], [[0., -.02, 0.], [0., 0., -.05]], atol=1e-16)
    np.testing.assert_allclose(g.projected_sd[1], [[.02, 0., 0.], [0., .04, -.05]], atol=1e-16)
    for side, (_, strength) in k.VARIANTS.items():
        np.testing.assert_allclose(sides[side].tracking[:, 2:]**2-p.tracking[:, 2:]**2,
                                   strength**2*g.variance)
        k.assert_structure(p, sides[side], g.eligible)
    np.testing.assert_array_equal(sides["CORR_MID"].tracking[:, 2:], sides["INDEP_MID"].tracking[:, 2:])


def test_replay_permutation_and_common_strength_draws():
    p = particles((0, 4, 3, 1, 2, 5), [2., 8., 9., .4, 5., 3.])
    order = [3, 2, 1, 5, 4, 0]
    original, _ = k.generate_all(p, "frozen")
    repeated, _ = k.generate_all(p, "frozen")
    permuted, _ = k.generate_all(p.take(order), "frozen")
    other, _ = k.generate_all(p, "different")
    for side in k.SIDES:
        assert equal(original[side], repeated[side])
        assert equal(original[side].take(order), permuted[side])
    for mode in ("CORR", "INDEP"):
        mid = original[f"{mode}_MID"].tracking[:, :2]-p.tracking[:, :2]
        for name, strength in k.STRENGTHS.items():
            np.testing.assert_allclose(original[f"{mode}_{name}"].tracking[:, :2]-p.tracking[:, :2], strength*mid, atol=1e-14)
    assert not equal(original["CORR_MID"], other["CORR_MID"])
    assert k.recipe()["sides"] == list(k.SIDES)


@pytest.mark.parametrize("kinds", [(), (1, 2), (4,)])
def test_empty_neutral_single(kinds):
    p = particles(kinds)
    sides, _ = k.generate_all(p, "edge")
    for out in sides.values():
        assert out.keys == p.keys
        if not np.any(p.valid):
            assert equal(out, p)


def test_partial_tracking_never_invented_and_bad_inputs():
    p = particles((0, 4))
    tr, valid = p.tracking.copy(), p.valid.copy()
    tr[0, 2], valid[0, 2] = 0., False
    tr[1, [1, 3]], valid[1, [1, 3]] = 0., False
    p = Particles(p.p4, p.charge, p.category, tr, valid, p.keys)
    sides, g = k.generate_all(p, "partial")
    assert g.eligible.tolist() == [[False, True], [True, False]]
    for out in sides.values():
        np.testing.assert_array_equal(out.tracking[0, [0, 2]], p.tracking[0, [0, 2]])
        np.testing.assert_array_equal(out.tracking[1, [1, 3]], p.tracking[1, [1, 3]])
    with pytest.raises(ValueError, match="identity"):
        k.generate_all(p, "")


def test_pair_moments_against_dense_covariance():
    delta = np.array([.1, -.2, .5])
    projected = np.array([[1., 2, 3], [-2., 2, 0], [3., 5, 1]])*.01
    variance = np.sum(projected**2, axis=1)+.001
    mask = ~np.eye(3, dtype=bool)
    for correlated in (False, True):
        row = pair_moments(delta, projected, variance, correlated)
        cov = projected@projected.T if correlated else np.zeros((3, 3))
        expected = variance[:, None]+variance[None, :]-2*cov
        assert row["pair_product"] == pytest.approx((delta[:, None]*delta)[mask].mean())
        assert row["pair_difference_sq"] == pytest.approx(((delta[:, None]-delta)**2)[mask].mean())
        assert row["expected_pair_product"] == pytest.approx(cov[mask].mean())
        assert row["expected_pair_difference_sq"] == pytest.approx(expected[mask].mean())
    assert pair_moments(delta[:1], projected[:1], variance[:1], True) is None


def test_empirical_covariance_matches_shared_and_independent_models():
    p = particles((4, 4), [8., 8.])
    # Geometry is similar across these tracks: shared residuals should correlate.
    g = k.geometry(p)
    samples = {m: [] for m in ("CORR", "INDEP")}
    for i in range(3000):
        sides, _ = k.generate_all(p, f"covariance:{i}")
        for mode in samples:
            samples[mode].append((sides[f"{mode}_MID"].tracking[:, :2]-p.tracking[:, :2]).reshape(-1))
    b = g.projected_sd.reshape(4, 3)
    expected_shared = b@b.T+np.diag(g.independent_sd.reshape(-1)**2)
    expected_independent = expected_shared.copy()
    expected_independent[:2, 2:] = expected_independent[2:, :2] = 0
    std = np.sqrt(g.variance.reshape(-1))
    for mode, expected in (("CORR", expected_shared), ("INDEP", expected_independent)):
        samples_array = np.asarray(samples[mode])/std
        np.testing.assert_allclose(samples_array.mean(axis=0), 0, atol=.065)
        np.testing.assert_allclose(np.cov(samples_array, rowvar=False), expected/(std[:, None]*std), atol=.085)
    for mode in samples:
        np.testing.assert_allclose(np.var(np.asarray(samples[mode]), axis=0), g.variance.reshape(-1), rtol=.09)
