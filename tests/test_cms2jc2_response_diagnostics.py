from __future__ import annotations

import pytest

from hlt_classification.cms2jc2_response.diagnostics import Diagnostics, TwoSample
from hlt_classification.cms2jc2_response.plots import panels, example_svg
from hlt_classification.cms2jc2_response.evaluation import fit_metric_lock
from hlt_classification.cms2jc2_response.association import policy
from hlt_classification.cms2jc2_response.metrics import observations
from hlt_classification.cms2jc2_response.readers import Pair
from test_cms2jc2_response_science import pairs
from test_cms2jc2_response_evaluation import fitted


def test_nonselecting_quantiles_full_population_tails_and_rare_categories():
    rows = list(pairs("s", 4))
    lock = fit_metric_lock(rows, policy(), membership_hash="a"*64, cap=32, bins=32)
    diag = Diagnostics(lock, replicas=[0, 1, 2], role="response_select", cap=32)
    for p in rows:
        obs, _ = observations(p.offline, p.hlt, policy())
        for side in diag.samples:
            diag.add(side, p.identity, p.hlt, obs)
        diag.two_sample.add(p, p.hlt)
    report = diag.report()
    assert report["nonselecting"] and not report["two_sample"]["available"]
    assert report["population_counters"]["real"]["jets"] == 4
    assert report["quantiles"]["real"] == report["quantiles"]["0"]
    assert report["tails_use_all_observations"] and not report["raw_records_exported"]
    assert report["fit_joint_thresholds"]
    assert report["population_counters"]["real"]["joint_tail_exceedances"] == report["population_counters"]["0"]["joint_tail_exceedances"]
    assert lock["fit_only_bridge_audit"]["counts"]["jets"] == 4


def test_two_sample_split_is_file_disjoint_order_invariant_and_not_confirmation_fit():
    a, b = TwoSample(32), TwoSample(32)
    population = [p for i in range(6) for p in pairs(str(i), 40)]
    for p in population:
        a.add(p, p.hlt)
    for p in reversed(population):
        b.add(p, p.hlt)
    x, y = a.report("response_select"), b.report("response_select")
    assert x == y and x["auc"] == .5
    assert not set(x["train_files"]) & set(x["test_files"])
    assert x["test_paired_jets"] == 96
    assert not a.report("response_confirm")["available"]


def test_visual_example_selection_and_traces_are_bounded(fitted):
    population = list(pairs("visual", 7))
    a = panels(population, fitted, role="response_select", selection_hash="a"*64)
    b = panels(reversed(population), fitted, role="response_select", selection_hash="a"*64)
    assert a == b and len(a["examples"]) == 7 and a["small_population"]
    assert len(a["reverse_tie_order_sensitivity"]["rows"]) == 7
    assert a["reverse_tie_order_sensitivity"]["primary_output_unchanged"]
    example = next(iter(a["examples"].values()))
    svg = example_svg(example, response_hash=fitted["content_hash"], role=a["role"], selection_rule=a["rule"])
    assert b"<svg" in svg and b"Real CMS HLT" in svg and b"replica 0" in svg
    offline = [Pair(p.identity, p.source_group, p.offline, None) for p in population]
    jc2 = panels(offline, fitted, role="jc2_train", selection_hash="a"*64, worst=0)
    assert not jc2["native_jc2_hlt_accessed"] and not jc2["sets"]["worst50_not_random"]
    with pytest.raises(ValueError, match="Worst"):
        panels(population, fitted, role="response_confirm", selection_hash="a"*64)
