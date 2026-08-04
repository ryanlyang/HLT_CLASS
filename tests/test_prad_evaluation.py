from __future__ import annotations

import numpy as np

from hlt_classification.prad.evaluation import (
    binary_auc,
    prad_classification_metrics,
    stratified_prad_metrics,
)
from hlt_classification.prad.reporting import build_paired_bootstrap_evidence
from hlt_classification.prad.plotting import save_loss_curves


def test_prad_primary_metric_is_finite_deterministic_and_per_class() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    logits = np.zeros((40, 10), dtype=np.float32)
    logits[np.arange(40), labels] = 5.0
    first = prad_classification_metrics(logits, labels)
    second = prad_classification_metrics(logits, labels)
    assert first == second
    assert np.isfinite(first["macro_log_rejection"])
    assert len(first["per_class_rejection"]) == 10
    assert all(
        item["zero_background_interpolation"]
        for item in first["per_class_rejection"].values()
    )
    assert sum(map(sum, first["confusion_matrix"])) == 40


def test_tied_scores_interpolate_to_rejection_two() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    logits = np.zeros((20, 10), dtype=np.float32)
    metrics = prad_classification_metrics(logits, labels)
    assert all(
        np.isclose(item["background_rejection"], 2.0)
        for item in metrics["per_class_rejection"].values()
    )
    assert np.isclose(metrics["macro_log_rejection"], np.log(2.0))


def test_binary_auc_handles_ties_with_average_ranks() -> None:
    assert binary_auc(
        np.asarray([0.0, 0.5, 0.5, 1.0]),
        np.asarray([False, False, True, True]),
    ) == 0.875


def test_stratified_metric_marks_absent_classes_unavailable() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    logits = np.zeros((20, 10), dtype=np.float32)
    result = stratified_prad_metrics(
        logits,
        labels,
        {
            "all": np.ones(20, dtype=np.bool_),
            "partial": labels < 5,
        },
    )
    assert result["all"]["available"]
    assert not result["partial"]["available"]


def test_five_seed_bootstrap_surface_covers_every_graph_and_exact_baseline() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    baseline = np.zeros((40, 10), dtype=np.float32)
    improved = baseline.copy()
    improved[np.arange(40), labels] = 5.0

    def load(graph: str, seed: int) -> np.ndarray:
        del seed
        return baseline if graph == "E0" else improved

    rows = build_paired_bootstrap_evidence(
        graphs=("E0", "E9"),
        seeds=(11, 22),
        labels=labels,
        logits_loader=load,
        samples=20,
    )
    assert [row["graph_id"] for row in rows] == ["E0", "E9"]
    assert all(
        interval["confidence_interval"] == [0.0, 0.0]
        for interval in rows[0]["paired_test_bootstrap"]
    )
    assert len(rows[1]["paired_test_bootstrap"]) == 2
    assert all(
        interval["observed_difference"] > 0.0
        for interval in rows[1]["paired_test_bootstrap"]
    )


def test_required_loss_curve_plot_is_materialized(tmp_path) -> None:
    output = tmp_path / "loss_curves.png"
    save_loss_curves(
        {
            "E0": [
                {"kind": "train", "update": 1, "loss": 2.0},
                {"kind": "train", "update": 2, "loss": 1.5},
            ],
            "E9": [{"kind": "train", "update": 1, "loss": 1.8}],
        },
        output,
    )
    assert output.is_file() and output.stat().st_size > 0
