from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.evaluation.metrics import (
    ECE_EDGES,
    accuracy_statistic,
    classification_metrics,
    paired_class_balanced_bootstrap,
    qcd_signal_rejection,
    top_label_ece,
)


def _balanced_labels(repeats: int = 2) -> np.ndarray:
    return np.repeat(np.arange(10, dtype=np.int64), repeats)


def test_exact_top_label_ece_edges_and_empty_bins() -> None:
    probabilities = np.zeros((3, 10), dtype=np.float64)
    probabilities[0, :2] = [0.5, 0.5]
    probabilities[1, :2] = [1.0 / 15.0, 14.0 / 15.0]
    probabilities[2, 2] = 1.0
    labels = np.asarray([0, 0, 2], dtype=np.int64)
    report = top_label_ece(probabilities, labels)
    assert report["edges"] == list(ECE_EDGES)
    assert report["bins"][7]["count"] == 1
    assert report["bins"][14]["count"] == 2
    assert report["bins"][14]["upper_inclusive"] is True
    assert report["bins"][0]["count"] == 0
    assert report["bins"][0]["weighted_absolute_gap"] == 0.0
    final_bin_mean_confidence = (14.0 / 15.0 + 1.0) / 2.0
    expected = (
        abs(1.0 - 0.5) / 3
        + abs(0.5 - final_bin_mean_confidence) * 2 / 3
    )
    assert report["value"] == pytest.approx(expected)


def test_qcd_rejection_uses_inclusive_ties_and_zero_background_rule() -> None:
    probabilities = np.zeros((6, 10), dtype=np.float64)
    probabilities[:, 0] = [0.99, 0.98, 0.8, 0.4, 0.4, 0.1]
    probabilities[:, 1] = 1.0 - probabilities[:, 0]
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    report = qcd_signal_rejection(
        probabilities,
        labels,
        target_signal_efficiency=0.5,
    )
    assert report["threshold"] == pytest.approx(0.6)
    assert report["achieved_signal_efficiency"] == pytest.approx(0.75)
    assert report["background_efficiency"] == 0.0
    assert report["qcd_rejection"] is None
    assert report["zero_background_selected"] is True


def test_classification_metrics_handle_auc_ties_deterministically() -> None:
    labels = _balanced_labels()
    logits = np.zeros((len(labels), 10), dtype=np.float32)
    report = classification_metrics(logits, labels)
    assert report["accuracy"] == pytest.approx(0.1)
    assert report["cross_entropy"] == pytest.approx(np.log(10.0))
    assert report["ovr_auc"] == pytest.approx([0.5] * 10)
    assert report["macro_ovr_auc"] == pytest.approx(0.5)
    assert report["class_counts"] == [2] * 10


def test_nonfinite_metrics_fail_closed() -> None:
    labels = _balanced_labels()
    logits = np.zeros((len(labels), 10), dtype=np.float32)
    logits[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        classification_metrics(logits, labels)


def test_paired_bootstrap_is_seeded_paired_and_class_balanced() -> None:
    labels = _balanced_labels(3)
    left = np.zeros((len(labels), 10), dtype=np.float32)
    right = np.zeros_like(left)
    left[np.arange(len(labels)), labels] = 2.0
    first = paired_class_balanced_bootstrap(
        left,
        right,
        labels,
        statistic=accuracy_statistic,
        samples=31,
    )
    second = paired_class_balanced_bootstrap(
        left,
        right,
        labels,
        statistic=accuracy_statistic,
        samples=31,
    )
    assert first == second
    assert first["observed_difference"] == pytest.approx(0.9)
    assert first["confidence_interval"] == pytest.approx([0.9, 0.9])
    assert first["sampling_unit"] == "paired_jet_within_class"
