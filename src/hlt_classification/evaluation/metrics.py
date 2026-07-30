"""Frozen multiclass classification metrics and paired statistics."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

METRICS_CONTRACT = "hlt_classification_metrics_v1"
METRICS_SCHEMA_VERSION = 1
NUM_CLASSES = 10
ECE_BIN_COUNT = 15
ECE_EDGES = tuple(index / ECE_BIN_COUNT for index in range(ECE_BIN_COUNT + 1))
QCD_CLASS_INDEX = 0
QCD_REJECTION_SIGNAL_EFFICIENCY = 0.5
PAIRED_BOOTSTRAP_SEED = 8041
PAIRED_BOOTSTRAP_QUANTILES = (0.025, 0.975)
PAIRED_BOOTSTRAP_QUANTILE_METHOD = "linear"


def _validate_logits_labels(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(logits)
    targets = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must have shape [rows,{NUM_CLASSES}]")
    if scores.dtype not in (np.float32, np.float64):
        raise ValueError("logits must have float32 or float64 dtype")
    if targets.shape != (len(scores),) or targets.dtype != np.int64:
        raise ValueError("labels must be int64 [rows]")
    if not np.isfinite(scores).all():
        raise ValueError("logits contain nonfinite values")
    if np.any((targets < 0) | (targets >= NUM_CLASSES)):
        raise ValueError("labels lie outside the ten-class contract")
    if len(scores) == 0:
        raise ValueError("metrics require at least one row")
    return scores.astype(np.float64, copy=False), targets


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must have shape [rows,{NUM_CLASSES}]")
    if not np.isfinite(scores).all():
        raise ValueError("logits contain nonfinite values")
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("softmax produced nonfinite probabilities")
    return probabilities


def _ovr_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Mann-Whitney AUC with exact average ranks for score ties."""

    positives = int(np.sum(positive))
    negatives = len(positive) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = float(np.sum(ranks[positive]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def top_label_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Exact 15-bin top-label multiclass ECE.

    Bins are ``[i/15,(i+1)/15)`` except the final bin, which includes 1.0.
    Empty bins have zero weight and contribute zero.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels)
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise ValueError("probabilities have the wrong shape")
    if targets.shape != (len(probs),):
        raise ValueError("ECE labels have the wrong shape")
    if targets.dtype != np.int64 or np.any(
        (targets < 0) | (targets >= NUM_CLASSES)
    ):
        raise ValueError("ECE labels differ from the ten-class contract")
    if not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    predictions = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), predictions]
    correct = predictions == targets
    indices = np.searchsorted(
        np.asarray(ECE_EDGES),
        confidence,
        side="right",
    ) - 1
    indices = np.clip(indices, 0, ECE_BIN_COUNT - 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(ECE_BIN_COUNT):
        selected = indices == index
        count = int(np.sum(selected))
        if count:
            mean_confidence = float(np.mean(confidence[selected]))
            accuracy = float(np.mean(correct[selected]))
            contribution = count / len(probs) * abs(accuracy - mean_confidence)
        else:
            mean_confidence = None
            accuracy = None
            contribution = 0.0
        ece += contribution
        bins.append(
            {
                "index": index,
                "lower": ECE_EDGES[index],
                "upper": ECE_EDGES[index + 1],
                "lower_inclusive": True,
                "upper_inclusive": index == ECE_BIN_COUNT - 1,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "weighted_absolute_gap": contribution,
            }
        )
    return {
        "value": float(ece),
        "bin_count": ECE_BIN_COUNT,
        "edges": list(ECE_EDGES),
        "definition": "top_label_multiclass",
        "empty_bin_contribution": 0.0,
        "bins": bins,
    }


def qcd_signal_rejection(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    target_signal_efficiency: float = QCD_REJECTION_SIGNAL_EFFICIENCY,
) -> dict[str, Any]:
    """Lock ``1-P(QCD) >= threshold`` at the requested inclusive efficiency."""

    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels)
    if (
        probs.ndim != 2
        or probs.shape[1] != NUM_CLASSES
        or targets.shape != (len(probs),)
        or targets.dtype != np.int64
        or not np.isfinite(probs).all()
        or np.any(probs < 0.0)
        or not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("QCD rejection probabilities or labels differ")
    if not 0.0 < target_signal_efficiency <= 1.0:
        raise ValueError("target signal efficiency must lie in (0,1]")
    signal = targets != QCD_CLASS_INDEX
    background = ~signal
    if not np.any(signal) or not np.any(background):
        raise ValueError("QCD rejection requires signal and QCD rows")
    discriminant = 1.0 - probs[:, QCD_CLASS_INDEX]
    required = int(math.ceil(target_signal_efficiency * int(np.sum(signal))))
    descending = np.sort(discriminant[signal], kind="mergesort")[::-1]
    threshold = float(descending[required - 1])
    accepted = discriminant >= threshold
    achieved_signal = float(np.mean(accepted[signal]))
    background_efficiency = float(np.mean(accepted[background]))
    zero_background = background_efficiency == 0.0
    return {
        "discriminant": "1_minus_p_qcd",
        "tie_rule": "signal_if_discriminant_greater_than_or_equal_to_threshold",
        "target_signal_efficiency": float(target_signal_efficiency),
        "threshold": threshold,
        "achieved_signal_efficiency": achieved_signal,
        "background_efficiency": background_efficiency,
        "qcd_rejection": (
            None if zero_background else float(1.0 / background_efficiency)
        ),
        "zero_background_behavior": "null_rejection_with_explicit_flag",
        "zero_background_selected": zero_background,
        "selected_signal_count": int(np.sum(accepted & signal)),
        "selected_qcd_count": int(np.sum(accepted & background)),
    }


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    scores, targets = _validate_logits_labels(logits, labels)
    probabilities = softmax_probabilities(scores)
    predictions = np.argmax(scores, axis=1)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(
        scores, axis=1
    )
    cross_entropy = float(
        np.mean(logsumexp - scores[np.arange(len(scores)), targets])
    )
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[targets]
    class_efficiencies: list[float | None] = []
    aucs: list[float | None] = []
    class_counts: list[int] = []
    for class_index in range(NUM_CLASSES):
        class_mask = targets == class_index
        class_count = int(np.sum(class_mask))
        class_counts.append(class_count)
        class_efficiencies.append(
            None
            if class_count == 0
            else float(np.mean(predictions[class_mask] == class_index))
        )
        aucs.append(_ovr_auc(probabilities[:, class_index], class_mask))
    return {
        "contract": METRICS_CONTRACT,
        "schema_version": METRICS_SCHEMA_VERSION,
        "rows": len(scores),
        "accuracy": float(np.mean(predictions == targets)),
        "cross_entropy": cross_entropy,
        "class_counts": class_counts,
        "per_class_efficiency": class_efficiencies,
        "ovr_auc": aucs,
        "macro_ovr_auc": (
            None
            if any(value is None for value in aucs)
            else float(np.mean(np.asarray(aucs, dtype=np.float64)))
        ),
        "multiclass_brier": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "ece": top_label_ece(probabilities, targets),
        "qcd_vs_signal": qcd_signal_rejection(probabilities, targets),
    }


def paired_class_balanced_bootstrap(
    left_logits: np.ndarray,
    right_logits: np.ndarray,
    labels: np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    samples: int = 2000,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired jet bootstrap, independently balanced within every class."""

    left, targets = _validate_logits_labels(left_logits, labels)
    right, right_targets = _validate_logits_labels(right_logits, labels)
    if not np.array_equal(targets, right_targets):
        raise ValueError("paired bootstrap labels differ")
    if left.shape != right.shape:
        raise ValueError("paired bootstrap logits shapes differ")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    class_indices = [np.flatnonzero(targets == index) for index in range(NUM_CLASSES)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("class-balanced bootstrap requires every class")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled = np.concatenate(
            [
                indices[rng.integers(0, len(indices), size=len(indices))]
                for indices in class_indices
            ]
        )
        sampled_labels = targets[sampled]
        differences[sample_index] = statistic(
            left[sampled], sampled_labels
        ) - statistic(right[sampled], sampled_labels)
        if not np.isfinite(differences[sample_index]):
            raise ValueError("bootstrap statistic produced a nonfinite value")
    lower, upper = np.quantile(
        differences,
        PAIRED_BOOTSTRAP_QUANTILES,
        method=PAIRED_BOOTSTRAP_QUANTILE_METHOD,
    )
    return {
        "sampling_unit": "paired_jet_within_class",
        "balanced_class_handling": (
            "sample_each_class_with_replacement_preserving_observed_class_count"
        ),
        "seed": int(seed),
        "samples": int(samples),
        "quantiles": list(PAIRED_BOOTSTRAP_QUANTILES),
        "quantile_method": PAIRED_BOOTSTRAP_QUANTILE_METHOD,
        "observed_difference": float(
            statistic(left, targets) - statistic(right, targets)
        ),
        "confidence_interval": [float(lower), float(upper)],
    }


def accuracy_statistic(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == labels))


__all__ = [
    "ECE_BIN_COUNT",
    "ECE_EDGES",
    "METRICS_CONTRACT",
    "PAIRED_BOOTSTRAP_SEED",
    "QCD_REJECTION_SIGNAL_EFFICIENCY",
    "accuracy_statistic",
    "classification_metrics",
    "paired_class_balanced_bootstrap",
    "qcd_signal_rejection",
    "softmax_probabilities",
    "top_label_ece",
]
