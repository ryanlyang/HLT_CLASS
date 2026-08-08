from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.scouting.inputs import build_hlt_inputs
from hlt_classification.scouting.matching import (
    ParticleSet, build_candidate_graph, decode_exclusive_categories,
    hungarian_with_dustbins, optimal_transport_with_dustbins, wrapped_delta_phi,
)
from hlt_classification.scouting.repair import (
    FULL_ENDPOINT_FIELDS, HIGHCOV_HC_EXACT_FAMILY, HIGHCOV_SHELL_EXACT_FAMILY,
    HIGHCOV_SHELL_SOFT_FAMILY, SELECTIVE_FULL_REPAIR_FAMILY,
    build_alpha_repaired_inputs,
    build_full_offline_endpoint_inputs,
    build_selective_matched_offline_endpoint_inputs,
)
from hlt_classification.scouting.matcher_validation import synthetic_particle_pair, wilson_interval
from hlt_classification.scouting.matcher_training import (
    MatcherTrainingConfig, bootstrap_edge_labels, contextual_scores, contextual_scores_many,
    likelihood_scores, load_contextual_matcher,
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


def _full_endpoint_fixture():
    arrays = _hlt_arrays()
    hlt_values = {
        "scoutpfcand_quality": [1, 2], "scoutpfcand_charge": [1, 0],
        "scoutpfcand_isEl": [0, 0], "scoutpfcand_isMu": [0, 0],
        "scoutpfcand_isChargedHad": [1, 0], "scoutpfcand_isGamma": [0, 1],
        "scoutpfcand_isNeutralHad": [0, 0], "scoutpfcand_phirel": [3.1, -2.8],
        "scoutpfcand_etarel": [.1, -.2], "scoutpfcand_abseta": [.1, .2],
        "scoutpfcand_pt_log": [1.0, .5], "scoutpfcand_normchi2": [2, 0],
        "scoutpfcand_dz": [.01, 0], "scoutpfcand_dxy": [-.02, 0],
        "scoutpfcand_dxysig": [-2, 0], "scoutpfcand_btagEtaRel": [.4, 0],
        "scoutpfcand_btagPtRatio": [.6, 0], "scoutpfcand_btagPParRatio": [.7, 0],
        "scoutpfcand_dzsig": [1.5, 0], "scoutpfcand_e_log": [1.1, .6],
        "scoutpfcand_lostInnerHits": [1, 0],
    }
    for branch, values in hlt_values.items():
        arrays[branch] = [np.asarray(values, np.float32)]

    charged = {
        "px": 20, "py": 0, "pz": 1, "energy": 20.1,
        "quality": 5, "charge": -1, "isEl": 1, "isMu": 0,
        "isChargedHad": 0, "phirel": -3.0, "etarel": .4, "abseta": .8,
        "pt_log_nopuppi": 2.3, "normchi2": 4, "dz": .03, "dxy": -.04,
        "dxysig": -3, "btagEtaRel": .9, "btagPtRatio": .8,
        "btagPParRatio": .6, "dzsig": 2.5, "e_log_nopuppi": 2.5,
        "lostInnerHits": 2,
    }
    neutral = {
        "px": 0, "py": 12, "pz": -1, "energy": 12.1,
        "isGamma": 0, "isNeutralHad": 1, "phirel": 3.0, "etarel": -.3,
        "abseta": .7, "pt_log_nopuppi": 1.7, "e_log_nopuppi": 1.8,
    }
    for suffix, value in charged.items():
        arrays[f"cpfcandlt_{suffix}"] = [np.asarray([value], np.float32)]
    for suffix, value in neutral.items():
        arrays[f"npfcand_{suffix}"] = [np.asarray([value], np.float32)]
    offline_p4 = [np.asarray([
        [charged[name] for name in ("px", "py", "pz", "energy")],
        [neutral[name] for name in ("px", "py", "pz", "energy")],
    ], np.float32)]
    assignment = np.full((1, 200), -1, np.int16)
    assignment[0, :2] = [1, 0]
    return arrays, offline_p4, assignment, charged, neutral


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


def test_full_particle_endpoint_replaces_all_21_channels_and_preserves_hlt_skeleton():
    arrays, offline_p4, assignment, charged, neutral = _full_endpoint_fixture()
    canonical = build_hlt_inputs(arrays)
    zero = build_alpha_repaired_inputs(
        arrays, offline_p4, np.full_like(assignment, -1), alpha=0.0,
        repair_family="FULL_PARTICLE_ENDPOINT",
    )
    assert zero.features.tobytes() == canonical.features.tobytes()
    assert zero.vectors.tobytes() == canonical.vectors.tobytes()

    endpoint = build_full_offline_endpoint_inputs(arrays, arrays, offline_p4, assignment)
    expected_raw = {name: [np.asarray(row).copy() for row in value] for name, value in arrays.items()}
    for field in FULL_ENDPOINT_FIELDS:
        charged_value = 0 if field.charged_suffix is None else charged[field.charged_suffix]
        neutral_value = 0 if field.neutral_suffix is None else neutral[field.neutral_suffix]
        expected_raw[field.hlt_branch][0][:2] = [neutral_value, charged_value]
    for channel, branch in enumerate(HLT_VECTOR_BRANCHES):
        expected_raw[branch][0][:2] = offline_p4[0][[1, 0], channel]
    expected = build_hlt_inputs(expected_raw)
    assert endpoint.features.tobytes() == expected.features.tobytes()
    assert endpoint.vectors.tobytes() == expected.vectors.tobytes()
    assert np.array_equal(endpoint.mask, canonical.mask)
    assert np.array_equal(endpoint.raw_lengths, canonical.raw_lengths)


def test_full_endpoint_requires_complete_exclusive_in_range_assignments():
    arrays, offline_p4, assignment, _, _ = _full_endpoint_fixture()
    cases = []
    missing = assignment.copy(); missing[0, 1] = -1; cases.append(missing)
    duplicate = assignment.copy(); duplicate[0, :2] = 0; cases.append(duplicate)
    out_of_range = assignment.copy(); out_of_range[0, 1] = 2; cases.append(out_of_range)
    for invalid in cases:
        with pytest.raises(ValueError):
            build_alpha_repaired_inputs(
                arrays, offline_p4, invalid, alpha=1.0,
                repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=arrays,
            )


def test_selective_endpoint_replaces_complete_matched_record_and_keeps_unmatched_bytes():
    arrays, offline_p4, assignment, charged, _ = _full_endpoint_fixture()
    assignment[0, 0] = -1
    canonical = build_hlt_inputs(arrays)
    endpoint = build_selective_matched_offline_endpoint_inputs(
        arrays, arrays, offline_p4, assignment,
    )
    versioned_endpoint = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=1.0,
        repair_family=SELECTIVE_FULL_REPAIR_FAMILY, offline_arrays=arrays,
    )
    assert versioned_endpoint.features.tobytes() == endpoint.features.tobytes()
    assert versioned_endpoint.vectors.tobytes() == endpoint.vectors.tobytes()
    assert endpoint.features[0, :, 0].tobytes() == canonical.features[0, :, 0].tobytes()
    assert endpoint.vectors[0, :, 0].tobytes() == canonical.vectors[0, :, 0].tobytes()
    for field in FULL_ENDPOINT_FIELDS:
        expected = 0 if field.charged_suffix is None else charged[field.charged_suffix]
        expected_raw = {name: [np.asarray(value[0]).copy()] for name, value in arrays.items()}
        expected_raw[field.hlt_branch][0][1] = expected
        expected_view = build_hlt_inputs(expected_raw)
        assert endpoint.features[0, field.channel, 1] == expected_view.features[0, field.channel, 1]
    assert np.array_equal(endpoint.vectors[0, :, 1], offline_p4[0][0])
    assert np.array_equal(endpoint.mask, canonical.mask)


def test_selective_endpoint_ignores_invalid_identity_on_unmatched_hlt_token():
    arrays, offline_p4, assignment, _, _ = _full_endpoint_fixture()
    assignment[0, 0] = -1
    arrays["scoutpfcand_isGamma"][0][0] = 1
    canonical = build_hlt_inputs(arrays)

    endpoint = build_selective_matched_offline_endpoint_inputs(
        arrays, arrays, offline_p4, assignment,
    )

    assert endpoint.features[0, :, 0].tobytes() == canonical.features[0, :, 0].tobytes()
    assert endpoint.vectors[0, :, 0].tobytes() == canonical.vectors[0, :, 0].tobytes()
    complete = assignment.copy(); complete[0, 0] = 1
    with pytest.raises(ValueError, match="invalid matched HLT particle identity"):
        build_alpha_repaired_inputs(
            arrays, offline_p4, complete, alpha=1.0,
            repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=arrays,
        )


def test_full_endpoint_discrete_switches_are_nested_deterministic_and_coherent():
    arrays, offline_p4, assignment, _, _ = _full_endpoint_fixture()
    canonical = build_hlt_inputs(arrays)
    endpoint = build_full_offline_endpoint_inputs(arrays, arrays, offline_p4, assignment)
    views = {}
    for alpha in (.1, .5):
        views[alpha] = build_alpha_repaired_inputs(
            arrays, offline_p4, assignment, alpha=alpha,
            repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=arrays,
            identity_keys=("file.root::tree::17",), discrete_seed=91,
        )
    repeated = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=.5,
        repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=arrays,
        identity_keys=("file.root::tree::17",), discrete_seed=91,
    )
    assert repeated.features.tobytes() == views[.5].features.tobytes()
    for token in range(2):
        lower_is_endpoint = np.array_equal(
            views[.1].features[0, 1:7, token], endpoint.features[0, 1:7, token],
        )
        upper_is_endpoint = np.array_equal(
            views[.5].features[0, 1:7, token], endpoint.features[0, 1:7, token],
        )
        assert not lower_is_endpoint or upper_is_endpoint
        assert np.array_equal(
            views[.5].features[0, 1:7, token],
            endpoint.features[0, 1:7, token] if upper_is_endpoint
            else canonical.features[0, 1:7, token],
        )
        # Both fixture matches cross charged/neutral applicability. Their track
        # channels move atomically with identity, never through a hybrid midpoint.
        assert np.array_equal(
            views[.5].features[0, 11:19, token],
            endpoint.features[0, 11:19, token] if upper_is_endpoint
            else canonical.features[0, 11:19, token],
        )


def test_full_endpoint_missing_continuous_values_switch_instead_of_blending():
    arrays, offline_p4, assignment, _, _ = _full_endpoint_fixture()
    # Keep slot 0 charged at both endpoints so this specifically exercises the
    # stored-value validity transition rather than chargedness applicability.
    assignment[0, :2] = [0, 1]
    arrays["cpfcandlt_normchi2"][0][0] = np.nan
    canonical = build_hlt_inputs(arrays)
    endpoint = build_full_offline_endpoint_inputs(arrays, arrays, offline_p4, assignment)
    middle = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=.5,
        repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=arrays,
        identity_keys=("file.root::tree::18",), discrete_seed=4,
    )
    assert middle.features[0, 11, 0] in {
        canonical.features[0, 11, 0], endpoint.features[0, 11, 0],
    }
    assert np.isfinite(middle.features).all()


def test_full_endpoint_switches_are_identity_bound_not_chunk_bound():
    arrays, offline_p4, assignment, _, _ = _full_endpoint_fixture()
    doubled = {
        name: [np.asarray(values[0]).copy(), np.asarray(values[0]).copy()]
        for name, values in arrays.items()
    }
    joint = build_alpha_repaired_inputs(
        doubled, [offline_p4[0].copy(), offline_p4[0].copy()],
        np.concatenate((assignment, assignment)), alpha=.25,
        repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=doubled,
        identity_keys=("file.root::tree::31", "file.root::tree::32"),
        discrete_seed=77,
    )
    for row, key in enumerate(("file.root::tree::31", "file.root::tree::32")):
        single_arrays = {name: [np.asarray(values[row]).copy()] for name, values in doubled.items()}
        single = build_alpha_repaired_inputs(
            single_arrays, [offline_p4[0].copy()], assignment.copy(), alpha=.25,
            repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=single_arrays,
            identity_keys=(key,), discrete_seed=77,
        )
        assert single.features[0].tobytes() == joint.features[row].tobytes()
        assert single.vectors[0].tobytes() == joint.vectors[row].tobytes()


def test_highcov_shell_exact_all_field_endpoints_and_confidence_warp():
    arrays, offline_p4, assignment, _, _ = _full_endpoint_fixture()
    canonical = build_hlt_inputs(arrays)
    selective = build_selective_matched_offline_endpoint_inputs(
        arrays, arrays, offline_p4, assignment,
    )
    confidence = np.zeros((1, 200), np.float32)
    confidence[0, :2] = [0.0, 1.0]

    zero = build_alpha_repaired_inputs(
        arrays, offline_p4, np.full_like(assignment, -1), alpha=0.0,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
    )
    assert zero.features.tobytes() == canonical.features.tobytes()
    assert zero.vectors.tobytes() == canonical.vectors.tobytes()

    endpoint = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=1.0,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
        confidence_weights=confidence, offline_arrays=arrays,
    )
    assert endpoint.features.tobytes() == selective.features.tobytes()
    assert endpoint.vectors.tobytes() == selective.vectors.tobytes()

    quarter = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=.25,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
        confidence_weights=confidence, offline_arrays=arrays,
        identity_keys=("file.root::tree::shell",), discrete_seed=8,
    )
    strengths = np.asarray([.25 ** 2.0, .25 ** .7])
    for token, strength in enumerate(strengths):
        expected = (
            (1 - strength) * canonical.vectors[0, :, token]
            + strength * offline_p4[0][assignment[0, token]]
        )
        np.testing.assert_allclose(quarter.vectors[0, :, token], expected, rtol=1e-6)


def test_highcov_soft_and_core_families_keep_declared_hlt_tokens():
    arrays, offline_p4, assignment, _, _ = _full_endpoint_fixture()
    canonical = build_hlt_inputs(arrays)
    confidence = np.zeros((1, 200), np.float32)
    confidence[0, :2] = [.5, .99]
    soft = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=1.0,
        repair_family=HIGHCOV_SHELL_SOFT_FAMILY,
        confidence_weights=confidence, offline_arrays=arrays,
        identity_keys=("file.root::tree::soft",), discrete_seed=3,
    )
    np.testing.assert_allclose(
        soft.vectors[0, :, 0],
        .5 * canonical.vectors[0, :, 0] + .5 * offline_p4[0][assignment[0, 0]],
        rtol=1e-6,
    )
    core = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=1.0,
        repair_family=HIGHCOV_HC_EXACT_FAMILY,
        confidence_weights=confidence, offline_arrays=arrays,
    )
    assert core.features[0, :, 0].tobytes() == canonical.features[0, :, 0].tobytes()
    assert core.vectors[0, :, 0].tobytes() == canonical.vectors[0, :, 0].tobytes()
    assert np.array_equal(core.vectors[0, :, 1], offline_p4[0][assignment[0, 1]])


def test_synthetic_correspondence_and_wilson_bounds_are_deterministic():
    first = synthetic_particle_pair(seed=7); second = synthetic_particle_pair(seed=7)
    assert np.array_equal(first[0].p4, second[0].p4)
    assert np.array_equal(first[2], second[2])
    lower, upper = wilson_interval(1000, 1000)
    assert .99 < lower <= upper == 1.0


def test_contextual_matcher_consumes_competing_edge_sets():
    torch.manual_seed(4); model = build_contextual_edge_matcher()
    features = torch.randn(3, 31)
    hlt_index = torch.tensor([0, 0, 1]); offline_index = torch.tensor([0, 1, 1])
    hlt_nodes = torch.randn(2, 28); offline_nodes = torch.randn(2, 28)
    scores = model(features, hlt_index, offline_index, hlt_nodes, offline_nodes)
    assert scores.shape == (3,)
    with torch.no_grad():
        changed = model(
            features[[0, 2]], torch.tensor([0, 1]), torch.tensor([0, 1]),
            hlt_nodes, offline_nodes,
        )
    assert not torch.equal(scores[0], changed[0])


def test_batched_sparse_matcher_inference_matches_single_graph_calls():
    graphs = []
    for seed in (31, 32):
        hlt, offline, _ = synthetic_particle_pair(seed=seed, particles=8)
        graphs.append(build_candidate_graph(hlt, offline))
    torch.manual_seed(3); model = build_contextual_edge_matcher()
    single = [contextual_scores(model, graph) for graph in graphs]
    batched = contextual_scores_many(model, graphs, maximum_edges_per_forward=100000)
    assert all(np.allclose(left, right, atol=1e-6) for left, right in zip(single, batched, strict=True))


def test_bootstrap_is_positive_unlabeled_and_confidence_is_edge_probability():
    hlt = ParticleSet(
        np.array([[10, 0, 0, 10.1], [9.9, .02, 0, 10.0]]),
        np.array([2, 2]), np.array([1, 1]), np.array([False, False]),
    )
    offline = ParticleSet(
        np.array([[10, 0, 0, 10.1], [9.8, .03, 0, 9.9]]),
        np.array([2, 2]), np.array([1, 1]), np.array([False, False]),
    )
    graph = build_candidate_graph(hlt, offline, charged_dr_gate=1, log_response_gate=2)
    labels = bootstrap_edge_labels(graph)
    assert 1 in labels and -1 in labels
    assert np.count_nonzero(labels == 0) < np.count_nonzero(labels != 1)
    raw_logit = np.log(.1 / .9)
    result = hungarian_with_dustbins(graph, np.full(len(graph.hlt_index), raw_logit))
    assert np.allclose(result.confidence[result.hlt_to_offline >= 0], .1, atol=1e-6)


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
            "independent_perturbation_stability": 1.0,
            "solver_consensus_fraction": 1.0,
            "stress_false_match_interval": [0.0, .01 + index * .001],
            "native_coverage": .5,
            "synthetic": {"confidence_brier": .001, "by_category": {
                str(category): {"precision_interval": [.995, 1.0]} for category in range(5)
            }},
        }
    selection = select_matcher_variant([{"variants": variants}] * 5)
    assert selection["selected_variant"] == "M3"


def test_match_corruption_is_identity_bound_compatible_and_exclusive():
    hlt, offline, _ = synthetic_particle_pair(
        seed=91, particles=20, loss_fraction=0, fake_fraction=0,
        split_fraction=0, merge_fraction=0, category_confusion=0,
    )
    graph = build_candidate_graph(hlt, offline, charged_dr_gate=10, neutral_dr_gate=10, log_response_gate=10)
    assignment = np.arange(20, dtype=np.int16)
    first = corrupt_assignment(graph, assignment, fraction=1.0, identity_key="a::tree::1")
    second = corrupt_assignment(graph, assignment, fraction=1.0, identity_key="a::tree::1")
    assert np.array_equal(first[0], second[0])
    retained = first[0][first[0] >= 0]
    assert len(retained) == len(set(retained.tolist()))
    assert first[1] == 20 and 0 <= first[2] <= first[1]
