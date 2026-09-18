from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr, wasserstein_distance

from hlt_classification.cms2jc2_response.metrics import (
    BLOCKS, Distribution, Summary, fit_registry, observations,
)
from hlt_classification.cms2jc2_response.association import policy
from test_cms2jc2_response_science import particles


def registry():
    obs = {}
    for i in range(30):
        row, _ = observations(particles(1+i/30), particles((1+i/30)*.9), policy())
        for name, values in row.items():
            obs.setdefault(name, []).extend(values)
    return fit_registry(obs, membership_hash="a"*64, bins=32)


def test_six_block_identity_and_missing_coverage():
    reg = registry()
    a, b, c = Summary(reg), Summary(reg), Summary(reg)
    for i in range(5):
        p = particles(1+i/10)
        row, info = observations(p, p, policy())
        a.add(row, info)
        b.add(row, info)
        missing, audit = observations(p, p.take([]), policy())
        c.add(missing, audit)
    same = a.compare(b)
    assert same["primary_score"] == pytest.approx(0.)
    assert set(same["blocks"]) == set(BLOCKS)
    assert a.compare(c)["primary_score"] > 0
    assert a.compare(c)["observables"]["d0"]["discrepancy"] >= 1.


def test_summary_merge_reproduces_streamed_statistics():
    reg = registry()
    whole, even, odd = Summary(reg), Summary(reg), Summary(reg)
    for i in range(10):
        obs, audit = observations(particles(), particles(1+i/10), policy())
        whole.add(obs, audit)
        (odd if i % 2 else even).add(obs, audit)
    even.merge(odd)
    assert whole.compare(even)["primary_score"] == pytest.approx(0, abs=1e-12)
    assert even.jets == 10
    with pytest.raises(ValueError):
        even.merge(odd, weight=.5)


def test_wasserstein_approximation_has_truth_inside_error_bound():
    reg = fit_registry({"delta_eta": list(np.linspace(-1, 1, 101))}, membership_hash="a"*64, bins=32)
    definition = next(r for r in reg["observables"] if r["name"] == "delta_eta")
    a, b = Distribution(definition), Distribution(definition)
    rng = np.random.default_rng(2)
    va, vb = rng.normal(size=1000), rng.normal(.3, 1.4, size=1000)
    a.add(va.tolist()); b.add(vb.tolist())
    result = a.compare(b)
    truth = wasserstein_distance(va, vb)/definition["scale"][0]
    assert abs(result["discrepancy"]-truth) <= result["error_bound"]+1e-12


def test_correlation_bound_and_constant_state():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1000, 2))
    y = x.copy(); y[:, 1] += .8*y[:, 0]
    name = "jet_corr_multiplicity_log_sum_pt"
    reg = fit_registry({name: x.tolist()}, membership_hash="a"*64, bins=32)
    definition = next(r for r in reg["observables"] if r["name"] == name)
    a, b = Distribution(definition), Distribution(definition)
    a.add(x); b.add(y)
    report = a.compare(b)
    truth = abs(spearmanr(x).statistic-spearmanr(y).statistic)
    assert abs(report["discrepancy"]-truth) <= report["error_bound"]+1e-12
    reg = fit_registry({"delta_eta": [0.]*20}, membership_hash="a"*64, bins=32)
    definition = next(r for r in reg["observables"] if r["name"] == "delta_eta")
    a, b = Distribution(definition), Distribution(definition)
    a.add([1.]*20); b.add([2.]*20)
    assert a.compare(b)["discrepancy"] == 1.
