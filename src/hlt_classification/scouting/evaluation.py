"""Fifteen-class PMARD metrics, paired bootstrap, and deterministic selectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy.special import logsumexp

from .schema import CLASS_NAMES
from .training import BOOTSTRAP_SEED


def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    if value.ndim != 2 or value.shape[1] != 15 or not np.isfinite(value).all():
        raise ValueError("logits must be finite [rows,15]")
    return np.exp(value - logsumexp(value, axis=1, keepdims=True))


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    value = np.asarray(logits, np.float64); target = np.asarray(labels, np.int64)
    if target.shape != (len(value),) or np.any((target < 0) | (target >= 15)):
        raise ValueError("labels differ from PMARD task")
    return float(np.mean(logsumexp(value, axis=1) - value[np.arange(len(value)), target]))


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from scipy.stats import rankdata
    ranks = rankdata(scores, method="average")
    positive = labels.astype(bool); p = positive.sum(); n = len(labels) - p
    if p == 0 or n == 0:
        return float("nan")
    return float((ranks[positive].sum() - p * (p + 1) / 2) / (p * n))


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    probabilities = softmax(logits); target = np.asarray(labels, np.int64)
    per_class = {}
    aucs = []
    qcd = target == 0
    predictions = np.argmax(logits, axis=1)
    confusion = np.zeros((15, 15), np.int64)
    np.add.at(confusion, (target, predictions), 1)
    log_rejections = []
    for index, name in enumerate(CLASS_NAMES):
        auc = _binary_auc(target == index, probabilities[:, index])
        if np.isfinite(auc): aucs.append(auc)
        true_positive = confusion[index, index]
        row: dict[str, object] = {
            "ovr_auc": None if not np.isfinite(auc) else auc,
            "recall": float(true_positive / max(1, confusion[index].sum())),
            "precision": float(true_positive / max(1, confusion[:, index].sum())),
        }
        if index > 0 and qcd.any() and np.any(target == index):
            signal_scores = probabilities[:, index]
            rejections = {}
            for efficiency in (.3, .5, .8):
                ordered = np.sort(signal_scores[target == index])[::-1]
                rank = max(0, min(len(ordered) - 1, int(np.ceil(efficiency * len(ordered))) - 1))
                threshold = ordered[rank]
                background_pass = int(np.count_nonzero(signal_scores[qcd] >= threshold))
                rejection = int(qcd.sum()) / max(1, background_pass)
                rejections[f"{int(efficiency * 100)}pct"] = {
                    "threshold": float(threshold), "qcd_pass": background_pass,
                    "zero_background": background_pass == 0,
                    "rejection": float(rejection),
                    "achieved_signal_efficiency": float(np.mean(signal_scores[target == index] >= threshold)),
                }
            row["qcd_rejection"] = rejections
            log_rejections.append(np.log(max(rejections["50pct"]["rejection"], 1.0)))
        per_class[name] = row
    confidence = probabilities.max(axis=1); correct = predictions == target
    ece = 0.0
    for lower in np.linspace(0, 1, 16)[:-1]:
        upper = lower + 1 / 15
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    one_hot = np.eye(15, dtype=np.float64)[target]
    return {
        "rows": len(target), "cross_entropy": cross_entropy(logits, target),
        "accuracy": float(np.mean(predictions == target)),
        "macro_ovr_auc": float(np.mean(aucs)) if aucs else None,
        "macro_mean_log_qcd_rejection_at_50pct_signal": float(np.mean(log_rejections)) if log_rejections else None,
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "top_label_ece_15_bin": float(ece),
        "confusion_matrix": confusion.tolist(), "per_class": per_class,
    }


STRATIFICATION_BINS = {
    "scoutfj_pt": (0, 300, 500, 750, 1000, 1500, float("inf")),
    "scoutfj_sdmass": (0, 25, 50, 100, 150, 250, float("inf")),
    "fj_pt": (0, 300, 500, 750, 1000, 1500, float("inf")),
    "fj_mass": (0, 25, 50, 100, 150, 250, float("inf")),
    "n_scoutpfcands": (0, 25, 50, 100, 150, 200, float("inf")),
    "offline_multiplicity": (0, 25, 50, 100, 150, 250, float("inf")),
    "hlt_truncated": (-.5, .5, 1.5),
}


def diagnostic_metrics(
    logits: np.ndarray, labels: np.ndarray, observers: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Frozen observer-only stratification and QCD mass-sculpting diagnostics."""
    rows = len(labels)
    values = {name: np.asarray(value) for name, value in observers.items()}
    if any(value.shape != (rows,) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("diagnostic observers must be finite row-aligned scalars")
    if all(name in values for name in ("n_cpfcands", "n_lts", "n_npfcands")):
        values["offline_multiplicity"] = values["n_cpfcands"] + values["n_lts"] + values["n_npfcands"]
    stratified = {}
    for name, edges in STRATIFICATION_BINS.items():
        if name not in values: continue
        groups = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            selected = (values[name] >= lower) & (values[name] < upper)
            groups.append({
                "lower": lower, "upper": None if np.isinf(upper) else upper,
                "rows": int(selected.sum()),
                "metrics": classification_metrics(logits[selected], labels[selected]) if selected.any() else None,
            })
        stratified[name] = groups
    probabilities = softmax(logits); qcd = labels == 0; sculpting = {}
    if qcd.any() and "scoutfj_sdmass" in values:
        mass_edges = np.asarray(STRATIFICATION_BINS["scoutfj_sdmass"][:-1] + (500.0,))
        reference = np.histogram(values["scoutfj_sdmass"][qcd], bins=mass_edges)[0].astype(np.float64)
        reference /= max(1.0, reference.sum())
        for signal in range(1, 15):
            signal_rows = labels == signal
            if not signal_rows.any(): continue
            threshold = np.quantile(probabilities[signal_rows, signal], .5, method="inverted_cdf")
            selected = qcd & (probabilities[:, signal] >= threshold)
            shifted = np.histogram(values["scoutfj_sdmass"][selected], bins=mass_edges)[0].astype(np.float64)
            shifted /= max(1.0, shifted.sum()); midpoint = .5 * (reference + shifted)
            def kl(left, right):
                active = left > 0
                return float(np.sum(left[active] * np.log(left[active] / np.maximum(right[active], 1e-15))))
            sculpting[str(signal)] = {
                "qcd_rows_passing": int(selected.sum()),
                "jensen_shannon_divergence": .5 * (kl(reference, midpoint) + kl(shifted, midpoint)),
            }
    qcd_sublabels = {}
    if "fj_label" in values:
        predictions = np.argmax(logits, axis=1)
        for sublabel in range(309, 314):
            selected = (labels == 0) & (values["fj_label"] == sublabel)
            qcd_sublabels[str(sublabel)] = {
                "rows": int(selected.sum()),
                "qcd_recall": float(np.mean(predictions[selected] == 0)) if selected.any() else None,
            }
    return {
        "contract": "hlt_classification_pmard_diagnostics_v1",
        "stratification_bins": {name: [None if np.isinf(value) else value for value in bins]
                                for name, bins in STRATIFICATION_BINS.items()},
        "stratified": stratified, "qcd_mass_sculpting": sculpting,
        "qcd_sublabels_309_313": qcd_sublabels,
    }


def select_validation_report(reports: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not reports:
        raise ValueError("selector requires reports")
    return min(reports, key=lambda row: (
        float(row["cross_entropy"]), -float(row["accuracy"]), str(row["experiment_id"]),
    ))


def paired_bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, samples: int = 2000, seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired bootstrap inputs must be finite, aligned vectors")
    rng = np.random.default_rng(seed); delta = a - b
    estimates = np.empty(samples, np.float64)
    for index in range(samples):
        estimates[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {
        "difference": float(delta.mean()),
        "lower_95": float(np.quantile(estimates, .025)),
        "upper_95": float(np.quantile(estimates, .975)),
    }


def privilege_recovery_ratio(student_gain: float, teacher_gain: float) -> float | None:
    if teacher_gain <= 0 or not np.isfinite([student_gain, teacher_gain]).all():
        return None
    return float(student_gain / teacher_gain)


__all__ = [
    "classification_metrics", "cross_entropy", "diagnostic_metrics", "paired_bootstrap_difference",
    "privilege_recovery_ratio", "select_validation_report", "softmax",
]
