from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.cms2jc2_response.association import associate, policy
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility, physical_status
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.cms2jc2_response.contracts import validate_compatibility, with_content_hash
from hlt_classification.cms2jc2_response.readers import Pair
from hlt_classification.cms2jc2_response.records import Reservoir
from hlt_classification.cms2jc2_response.topology import (Record, calibration_records, emission_state,
    emission_coordinates, decode_emission, reference)
from hlt_classification.cms2jc2_response.features import features
from hlt_classification.cms2jc2_response.response import collect, fit_response, Generator


def particles(scale=1., keys=("a", "b")):
    return Particles(p4_from_coordinates(np.array([2., 10.])*scale, [0., .3], [0., .2], [.1, .2]),
                     np.array([1, 0]), np.array([0, 1]), np.array([[.2, -.1, .01, .02], [0, 0, 0, 0]]),
                     np.array([[1, 1, 1, 1], [0, 0, 0, 0]], bool), keys)


def pairs(group, n=20):
    for i in range(n):
        p = particles(1+i/100)
        yield Pair(f"{group}:{i}", group, p, particles((1+i/100)*.9, ("h0", "h1")))


def test_provisional_is_explicit_immutable_and_not_verified():
    value = provisional_compatibility(inventory_hash="a"*64)
    validate_compatibility(value, inventory_hash="a"*64)
    assert physical_status(value)["provisional"]
    assert not physical_status(value)["physically_qualified_transfer_allowed"]
    value["cms_length_to_mm"] = 1.
    with pytest.raises(ValueError, match="Unregistered provisional"):
        validate_compatibility(with_content_hash(value))


def test_emission_roundtrip_preserves_physical_states():
    p = particles()
    origin = reference(p, (0, 1))
    result = decode_emission(emission_coordinates(p, origin), emission_state(p), origin, key="g")
    np.testing.assert_allclose(result.p4, p.p4, atol=1e-12)
    np.testing.assert_allclose(result.tracking, p.tracking)
    np.testing.assert_array_equal(result.valid, p.valid)


def test_topology_exposures_do_not_duplicate_consumed_merge():
    p = particles()
    # One exact two-input merged output, no singleton exposure after acceptance.
    p = Particles(p4_from_coordinates([2., 10.], [0, .01], [0, .01], [0., 0.]),
                  p.charge, p.category, p.tracking, p.valid, p.keys)
    h = Particles(p.p4.sum(axis=0)[None], np.array([0]), np.array([1]),
                  np.zeros((1, 4)), np.zeros((1, 4), bool), ("h",))
    rules = policy()
    assignment = associate(p, h, rules)
    records, info = calibration_records(p, h, assignment, rules)
    assert info["merges"] == 1
    assert [r.module for r in records] == ["merge", "emission1_state", "emission1_value", "additional_count"]
    assert records[0].target == (1,)


def test_reservoir_replays_with_correct_inclusion_weights():
    p = particles()
    records = [(f"j{i}", Record("singleton", "a", features(p)[0], (i%3,))) for i in range(200)]
    a, b = Reservoir(720), Reservoir(720)
    for jet, record in records:
        a.add(jet, record)
    for jet, record in reversed(records):
        b.add(jet, record)
    assert a.report() == b.report()
    xa, ya, wa, ia = a.arrays("singleton")
    xb, yb, wb, ib = b.arrays("singleton")
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(ia, ib)
    assert ya == yb and len(ya) == 10
    assert wa.sum() == 200 and np.all(wa == wb)


@pytest.mark.parametrize("candidate", ["A_L", "B_L", "C_L"])
def test_complete_response_three_families_and_keyed_replay(candidate):
    rules = policy()
    loc, lr = collect(pairs("location"), rules, cap=7200)
    res, rr = collect(pairs("residual"), rules, cap=7200)
    fitted = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id=candidate,
                          review=provisional_compatibility(inventory_hash="a"*64), rules=rules,
                          budget="MINIATURE", source_hash="b"*64)
    gen = Generator(fitted)
    p = particles()
    first, audit = gen(p, jet="held", trace=True)
    second, _ = gen(p.take([1, 0]), jet="held", trace=True)
    assert first.p4.tobytes() == second.p4.tobytes()
    assert first.tracking.tobytes() == second.tracking.tobytes()
    assert len(first) == 2 and audit["physical_status"]["provisional"]
    np.testing.assert_allclose(np.sort(first.pt), np.sort(p.pt*.9), rtol=.1)
    assert all("input_keys" in row for row in audit["operations"])
    with pytest.raises(PermissionError, match="disjoint"):
        fit_response(loc, loc, location_report=lr, residual_report=lr, candidate_id=candidate,
                     review=provisional_compatibility(inventory_hash="a"*64), rules=rules,
                     budget="MINIATURE", source_hash="b"*64)
