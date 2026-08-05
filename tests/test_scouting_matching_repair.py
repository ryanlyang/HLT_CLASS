from __future__ import annotations

import numpy as np

from hlt_classification.scouting.inputs import build_hlt_inputs
from hlt_classification.scouting.matching import (
    ParticleSet, build_candidate_graph, decode_exclusive_categories,
    hungarian_with_dustbins, optimal_transport_with_dustbins, wrapped_delta_phi,
)
from hlt_classification.scouting.repair import build_alpha_repaired_inputs
from hlt_classification.scouting.matcher_validation import synthetic_particle_pair, wilson_interval
from hlt_classification.scouting.matcher_training import (
    MatcherTrainingConfig, likelihood_scores, load_contextual_matcher,
    train_contextual_matcher,
)
from hlt_classification.scouting.matcher_validation import select_matcher_variant
from hlt_classification.scouting.match_model import build_contextual_edge_matcher
import torch
from hlt_classification.scouting.schema import HLT_FEATURE_SPECS, HLT_VECTOR_BRANCHES
from hlt_classification.scouting.assignment import corrupt_assignment


def _hlt_arrays():
    arrays = {}
    for spec in HLT_FEATURE_SPECS:
        arrays[spec.branch] = [np.zeros(2, np.float32)]
    arrays["scoutpfcand_px"] = [np.array([10, 0], np.float32)]
    arrays["scoutpfcand_py"] = [np.array([0, 5], np.float32)]
    arrays["scoutpfcand_pz"] = [np.array([0, 0], np.float32)]
    arrays["scoutpfcand_energy"] = [np.array([10.1, 5.1], np.float32)]
    arrays["scoutpfcand_pt_log"] = [np.log(np.array([10, 5], np.float32))]
    arrays["scoutpfcand_e_log"] = [np.log(np.array([10.1, 5.1], np.float32))]
    return arrays


def test_category_decode_and_global_solvers_are_exclusive():
    assert decode_exclusive_categories(np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0]])).tolist() == [0, -1]
    hlt = ParticleSet(
        np.array([[10, 0, 0, 10.1], [9.9, .01, 0, 10.0]]),
        np.array([2, 2]), np.array([1, 1]), np.array([False, False]),
    )
    offline = ParticleSet(
        np.array([[10.05, 0, 0, 10.2]]), np.array([2]), np.array([1]), np.array([False]),
    )
    graph = build_candidate_graph(hlt, offline)
    for result in (
        hungarian_with_dustbins(graph, graph.manual_scores),
        optimal_transport_with_dustbins(graph, graph.manual_scores),
    ):
        accepted = result.hlt_to_offline[result.hlt_to_offline >= 0]
        assert len(accepted) == len(set(accepted.tolist())) == 1
    assert np.isclose(wrapped_delta_phi(-np.pi + .1, np.pi - .1), .2)


def test_alpha_zero_is_byte_identical_and_alpha_one_reaches_endpoint():
    arrays = _hlt_arrays()
    canonical = build_hlt_inputs(arrays)
    assignment = np.full((1, 200), -1, np.int16); assignment[0, 0] = 0
    offline = [np.array([[12, 0, 0, 12.1]], np.float32)]
    zero = build_alpha_repaired_inputs(arrays, offline, assignment, alpha=0.0)
    assert zero.features.tobytes() == canonical.features.tobytes()
    assert zero.vectors.tobytes() == canonical.vectors.tobytes()
    repaired = build_alpha_repaired_inputs(arrays, offline, assignment, alpha=1.0)
    assert np.array_equal(repaired.vectors[0, :, 0], offline[0][0])
    assert np.array_equal(repaired.vectors[0, :, 1], canonical.vectors[0, :, 1])
    assert np.array_equal(repaired.mask, canonical.mask)
    confidence = np.ones((1, 200), np.float32); confidence[0, 0] = .5
    weighted = build_alpha_repaired_inputs(
        arrays, offline, assignment, alpha=1.0,
        repair_family="CONFIDENCE_WEIGHTED", confidence_weights=confidence,
    )
    assert np.allclose(weighted.vectors[0, :, 0], .5 * canonical.vectors[0, :, 0] + .5 * offline[0][0])
    for family in ("RANDOM_DIRECTION", "LOG_ANGULAR"):
        control = build_alpha_repaired_inputs(arrays, offline, assignment, alpha=.5, repair_family=family)
        assert np.isfinite(control.vectors).all() and np.array_equal(control.mask, canonical.mask)


def test_synthetic_correspondence_and_wilson_bounds_are_deterministic():
    first = synthetic_particle_pair(seed=7); second = synthetic_particle_pair(seed=7)
    assert np.array_equal(first[0].p4, second[0].p4)
    assert np.array_equal(first[2], second[2])
    lower, upper = wilson_interval(1000, 1000)
    assert .99 < lower <= upper == 1.0


def test_contextual_matcher_consumes_competing_edge_sets():
    torch.manual_seed(4); model = build_contextual_edge_matcher()
    features = torch.randn(3, 13)
    scores = model(features, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]))
    assert scores.shape == (3,)
    with torch.no_grad():
        changed = model(features[[0, 2]], torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert not torch.equal(scores[[0, 2]], changed)


def test_fitted_likelihood_is_distinct_and_report_is_reloadable(tmp_path):
    graphs = []
    for seed in range(24):
        hlt, offline, truth = synthetic_particle_pair(seed=100 + seed, particles=12)
        graph = build_candidate_graph(hlt, offline)
        labels = (graph.offline_index == truth[graph.hlt_index]).astype(np.float32)
        graphs.append((graph, labels))
    parents = {"split": "0" * 64, "source": "1" * 64}
    report = train_contextual_matcher(
        graphs, config=MatcherTrainingConfig(epochs=1, hidden_dim=8),
        output_dir=tmp_path, parents=parents, device="cpu",
    )
    loaded = load_contextual_matcher(report, tmp_path)
    fitted = likelihood_scores(loaded, graphs[0][0])
    assert fitted.shape == graphs[0][0].manual_scores.shape
    assert not np.allclose(fitted, graphs[0][0].manual_scores)


def test_matching_only_selector_never_promotes_m0():
    variants = {}
    for index in range(6):
        variants[f"M{index}"] = {
            "passes_initial_99pct_lcb": index in {3, 5},
            "event_mixing_false_positive_rate": 0.0,
            "rotation_stability": 1.0,
            "stress_false_match_interval": [0.0, .01 + index * .001],
            "native_coverage": .5,
            "synthetic": {"by_category": {
                str(category): {"precision_interval": [.995, 1.0]} for category in range(5)
            }},
        }
    selection = select_matcher_variant([{"variants": variants}] * 5)
    assert selection["selected_variant"] == "M3"


def test_match_corruption_is_identity_bound_compatible_and_exclusive():
    hlt, offline, _ = synthetic_particle_pair(seed=91, particles=20, loss_fraction=0)
    graph = build_candidate_graph(hlt, offline, charged_dr_gate=10, neutral_dr_gate=10, log_response_gate=10)
    assignment = np.arange(20, dtype=np.int16)
    first = corrupt_assignment(graph, assignment, fraction=1.0, identity_key="a::tree::1")
    second = corrupt_assignment(graph, assignment, fraction=1.0, identity_key="a::tree::1")
    assert np.array_equal(first[0], second[0])
    retained = first[0][first[0] >= 0]
    assert len(retained) == len(set(retained.tolist()))
    assert first[1] == 20 and 0 <= first[2] <= first[1]
