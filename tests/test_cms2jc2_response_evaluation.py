from __future__ import annotations

import pytest

from hlt_classification.cms2jc2_response.association import policy
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
from hlt_classification.cms2jc2_response.contracts import artifact
from hlt_classification.cms2jc2_response.evaluation import ObservationSample, fit_metric_lock, evaluate
from hlt_classification.cms2jc2_response.readers import Pair
from hlt_classification.cms2jc2_response.response import collect, fit_response
from hlt_classification.cms2jc2_response.transfer import evaluate_transfer
from test_cms2jc2_response_science import pairs


@pytest.fixture
def fitted():
    rules = policy()
    loc, lr = collect(pairs("loc", 12), rules, cap=720)
    res, rr = collect(pairs("res", 12), rules, cap=720)
    return fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id="A_L",
                         review=provisional_compatibility(inventory_hash="a"*64), rules=rules,
                         budget="DEVELOPMENT", source_hash="b"*64)


def test_metric_sampling_is_bounded_and_order_invariant():
    a, b = ObservationSample(32), ObservationSample(32)
    for i in range(100):
        a.add("x", str(i), [i])
    for i in reversed(range(100)):
        b.add("x", str(i), [i])
    assert a.values("x") == b.values("x") and len(a.values("x")) == 32
    assert a.counts == b.counts == {"x": 100}


def test_three_replica_paired_evaluation_is_not_three_times_the_data(fitted):
    lock = fit_metric_lock(pairs("fit", 5), policy(), membership_hash="c"*64, bins=32, cap=32)
    held = [p for group in ("s0", "s1", "s2") for p in pairs(group, 2)]
    result = evaluate(held, fitted, lock, membership_hash="d"*64, role="development")
    row = result["comparison"]
    assert row["real_jets"] == 6 and row["replicas"] == [0, 1, 2]
    assert len(row["bootstrap_scores"]) == 200
    assert row["independent_source_groups"] == 3 and row["weak_source_group_coverage"]
    assert not result["durable_proxy_arrays"]
    assert not result["final_test_accessed"]


def test_transfer_requires_locked_identity_and_offline_only_input(fitted):
    selection = artifact("SELECTION", response_hash=fitted["content_hash"])
    claim = artifact("TRANSFER_CLAIM", parents={"response": fitted["content_hash"],
                     "selection": selection["content_hash"], "profile": "a"*64}, role="train", separately_authorized=True)
    offline = [Pair(p.identity, p.source_group, p.offline, None) for p in pairs("jc2", 3)]
    result = evaluate_transfer(offline, fitted, selection=selection, claim=claim,
                               profile_hash="a"*64, role="train")
    assert result["counts"]["jets"] == 3
    assert not result["native_jc2_hlt_accessed"] and not result["detector_truth_validation"]
    assert result["physical_status"]["provisional"]
    with pytest.raises(PermissionError, match="native JC2"):
        evaluate_transfer(list(pairs("wrong", 1)), fitted, selection=selection, claim=claim,
                          profile_hash="a"*64, role="train")
    with pytest.raises(PermissionError, match="role"):
        evaluate_transfer(offline, fitted, selection=selection, claim=claim,
                          profile_hash="a"*64, role="final_test")
