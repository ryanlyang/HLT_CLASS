from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from hlt_classification.scouting.hcwdl_mhpe_roc import (
    MAIN_LADDER,
    PROGRESSION_LADDER,
    SIGNALS,
    _plot,
    _plot_progression,
    _metric_deltas,
    _probability_metrics,
    _recovered_fraction,
    _validate_same_component_checkpoints,
    qcd_rejection_curve,
)
from hlt_classification.scouting.hcwdl_mhpe_graph import (
    PROFILE_DENSE_C25P75_300K60,
    ensemble_components,
    ensemble_weight_rationals,
)
from hlt_classification.scouting.hcwdl_mhpe_targets import (
    weighted_probability_ensemble,
)

matplotlib.use("Agg")


def test_qcd_rejection_curve_uses_signal_over_signal_plus_qcd_and_ties():
    probabilities = np.zeros((6, 15), dtype=np.float64)
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    # Scores are 0.9, 0.8, 0.5 for signal and 0.8, 0.4, 0.1 for QCD.
    scores = (0.9, 0.8, 0.5, 0.8, 0.4, 0.1)
    for row, score in enumerate(scores):
        probabilities[row, 1] = score
        probabilities[row, 0] = 1.0 - score
    curve = qcd_rejection_curve(probabilities, labels, signal_index=1)
    np.testing.assert_allclose(
        curve["signal_efficiency"],
        [0, 1 / 3, 2 / 3, 1, 1, 1],
    )
    np.testing.assert_allclose(curve["qcd_rejection"], [3, 3, 3, 3, 1.5, 1])
    assert curve["signal_rows"] == 3
    assert curve["qcd_rows"] == 3


def test_qcd_rejection_curve_ignores_other_classes():
    probabilities = np.zeros((4, 15), dtype=np.float64)
    probabilities[:, 0] = (0.1, 0.9, 0.0, 0.5)
    probabilities[:, 2] = (0.9, 0.1, 1.0, 0.5)
    labels = np.array([2, 0, 7, 7], dtype=np.int64)
    curve = qcd_rejection_curve(probabilities, labels, signal_index=2)
    assert curve["signal_rows"] == 1
    assert curve["qcd_rows"] == 1
    np.testing.assert_allclose(curve["signal_efficiency"], [0, 1, 1])


@pytest.mark.parametrize("signal_index", [0, 15, -1])
def test_qcd_rejection_curve_rejects_invalid_signal(signal_index):
    with pytest.raises(ValueError, match="non-QCD"):
        qcd_rejection_curve(
            np.full((2, 15), 1 / 15, dtype=np.float64),
            np.array([0, 1]),
            signal_index=signal_index,
        )


def test_dense_roc_plot_writes_both_formats(tmp_path):
    curve = {
        "signal_efficiency": np.array([0.0, 0.5, 1.0]),
        "qcd_rejection": np.array([100.0, 10.0, 1.0]),
    }
    node_ids = {
        node_id for ladder in (MAIN_LADDER, PROGRESSION_LADDER)
        for node_id, _ in ladder
    }
    curves = {
        node_id: {signal: curve for signal in SIGNALS}
        for node_id in node_ids
    }
    paths = _plot(curves, tmp_path)
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    assert paths["png"].read_bytes().startswith(b"\x89PNG")
    progression = _plot_progression(curves, tmp_path)
    assert progression["progression_pdf"].read_bytes().startswith(b"%PDF")
    assert progression["progression_png"].read_bytes().startswith(b"\x89PNG")


def test_d100_hlt_transfer_uses_registered_dense_ensemble_weights():
    profile = PROFILE_DENSE_C25P75_300K60
    components = ensemble_components(profile)["U100E"]
    logits = {
        name: np.full((3, 15), index / 10, dtype=np.float32)
        for index, name in enumerate(components)
    }
    for index, name in enumerate(components):
        logits[name][:, index] += 1.0
    actual = weighted_probability_ensemble(
        logits,
        temperature=1,
        weights=ensemble_weight_rationals(profile, "U100E"),
    )
    assert actual.shape == (3, 15)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=2e-6)
    local = "U100_from_U066E"
    assert ensemble_weight_rationals(profile, "U100E")[local] == [1, 2]


def test_d100_hlt_transfer_metric_helpers_are_directional_and_finite():
    probabilities = np.full((30, 15), 0.01, dtype=np.float32)
    labels = np.arange(30, dtype=np.int64) % 15
    probabilities[np.arange(30), labels] = 0.86
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics = _probability_metrics(probabilities, labels)
    assert metrics["accuracy"] == 1.0
    deltas = _metric_deltas(metrics, metrics)
    assert set(deltas) == {
        "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal", "top_label_ece_15_bin",
    }
    assert all(value == 0 for value in deltas.values())
    assert _recovered_fraction(0.95, 0.90, 1.00) == pytest.approx(0.5)
    assert _recovered_fraction(1.0, 1.0, 1.0) is None


def test_d100_hlt_transfer_requires_exact_native_component_checkpoints():
    order = ("a", "b")
    native = {
        "a": {"report_sha256": "1" * 64, "checkpoint_sha256": "2" * 64},
        "b": {"report_sha256": "3" * 64, "checkpoint_sha256": "4" * 64},
    }
    _validate_same_component_checkpoints(native, dict(native), order)
    changed = {key: dict(value) for key, value in native.items()}
    changed["b"]["checkpoint_sha256"] = "5" * 64
    with pytest.raises(ValueError, match="checkpoint differs"):
        _validate_same_component_checkpoints(native, changed, order)
    with pytest.raises(ValueError, match="lineage set"):
        _validate_same_component_checkpoints(native, {"a": native["a"]}, order)
