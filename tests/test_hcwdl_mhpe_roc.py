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
    qcd_rejection_curve,
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
