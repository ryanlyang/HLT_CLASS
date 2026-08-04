"""Frozen PRAD selection metric and stratified evaluation helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hlt_classification.data.schema import CLASS_LABELS
from hlt_classification.evaluation.metrics import (
    classification_metrics,
    softmax_probabilities,
)

PRAD_METRICS_CONTRACT = "hlt_classification_prad_metrics_v1"
PRAD_METRICS_SCHEMA_VERSION = 1
TARGET_SIGNAL_EFFICIENCY = 0.5
ZERO_BACKGROUND_RULE = "empirical_floor_one_over_background_count"


def binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Exact rank AUC with average ranks for deterministic score ties."""

    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positive, dtype=np.bool_)
    if values.ndim != 1 or labels.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("binary AUC inputs differ")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("binary AUC requires positive and negative examples")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and ordered_values[stop] == ordered_values[cursor]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * ((cursor + 1) + stop)
        cursor = stop
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _ovr_rejection_at_efficiency(
    scores: np.ndarray,
    positive: np.ndarray,
    *,
    target_efficiency: float = TARGET_SIGNAL_EFFICIENCY,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    signal = np.asarray(positive, dtype=np.bool_)
    if values.ndim != 1 or signal.shape != values.shape:
        raise ValueError("PRAD OVR score and label shapes differ")
    if not np.isfinite(values).all():
        raise ValueError("PRAD OVR scores are nonfinite")
    signal_count = int(signal.sum())
    background_count = len(signal) - signal_count
    if signal_count == 0 or background_count == 0:
        raise ValueError("PRAD OVR rejection requires both classes")
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_signal = signal[order]
    tpr = [0.0]
    fpr = [0.0]
    cursor = 0
    cumulative_signal = 0
    cumulative_background = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[cursor]:
            stop += 1
        group = sorted_signal[cursor:stop]
        cumulative_signal += int(group.sum())
        cumulative_background += len(group) - int(group.sum())
        tpr.append(cumulative_signal / signal_count)
        fpr.append(cumulative_background / background_count)
        cursor = stop
    tpr_array = np.asarray(tpr, dtype=np.float64)
    fpr_array = np.asarray(fpr, dtype=np.float64)
    interpolated = float(np.interp(target_efficiency, tpr_array, fpr_array))
    empirical_floor = 1.0 / background_count
    effective = max(interpolated, empirical_floor)
    return {
        "target_signal_efficiency": target_efficiency,
        "background_efficiency_interpolated": interpolated,
        "background_efficiency_effective": effective,
        "background_rejection": float(1.0 / effective),
        "signal_count": signal_count,
        "background_count": background_count,
        "zero_background_interpolation": interpolated == 0.0,
        "zero_background_rule": ZERO_BACKGROUND_RULE,
        "tie_rule": "threshold_groups_are_indivisible_then_roc_segments_interpolated",
        "interpolation": "linear_in_fpr_between_neighboring_tpr_points",
    }


def prad_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute the primary macro log-rejection plus secondary metrics."""

    scores = np.asarray(logits)
    targets = np.asarray(labels)
    base = classification_metrics(scores, targets)
    probabilities = softmax_probabilities(scores)
    rejections = {
        CLASS_LABELS[index]: _ovr_rejection_at_efficiency(
            probabilities[:, index], targets == index
        )
        for index in range(len(CLASS_LABELS))
    }
    values = np.asarray(
        [item["background_rejection"] for item in rejections.values()],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise FloatingPointError("PRAD rejection values are invalid")
    predictions = np.argmax(scores, axis=1)
    confusion = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    return {
        "contract": PRAD_METRICS_CONTRACT,
        "schema_version": PRAD_METRICS_SCHEMA_VERSION,
        "primary_metric": "macro_mean_log_ovr_background_rejection_at_50pct_signal",
        "macro_log_rejection": float(np.log(values).mean()),
        "per_class_rejection": rejections,
        "confusion_matrix": confusion.tolist(),
        "secondary": base,
    }


def stratified_prad_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    strata: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate predeclared boolean strata; each must contain all ten classes."""

    scores = np.asarray(logits)
    targets = np.asarray(labels)
    result: dict[str, Any] = {}
    for name, raw_mask in strata.items():
        mask = np.asarray(raw_mask)
        if mask.dtype != np.bool_ or mask.shape != (len(targets),):
            raise ValueError(f"PRAD stratum {name!r} mask differs")
        present = set(targets[mask].tolist())
        if present != set(range(len(CLASS_LABELS))):
            result[name] = {
                "available": False,
                "reason": "stratum_does_not_contain_all_classes",
                "rows": int(mask.sum()),
            }
        else:
            result[name] = {
                "available": True,
                "rows": int(mask.sum()),
                "metrics": prad_classification_metrics(scores[mask], targets[mask]),
            }
    return result


__all__ = [
    "PRAD_METRICS_CONTRACT",
    "binary_auc",
    "prad_classification_metrics",
    "stratified_prad_metrics",
]
