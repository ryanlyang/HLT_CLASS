"""Eleven-class metrics and recovery on fresh, same-population references."""
from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import roc_auc_score

from .contracts import artifact
from .schema import CLASS_NAMES


def metrics_contract() -> dict:
    return artifact(
        "METRICS", classes=list(CLASS_NAMES), macro_auc="unweighted_one_vs_rest",
        rejection="signal_vs_QCD_score_pSignal_over_pSignal_plus_pQCD",
        signal_efficiency=.5, threshold="descending_signal_ceil_half_rank_include_ties",
        zero_probability_pair="score_zero", zero_background_passing="null_rejection_with_one_event_resolution_bound",
        macro_R50="geometric_mean_ten_signal_rejections_null_if_any_censored",
        recovery="linear_metric_difference_new_HLT_zero_new_U000_one",
    )


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels, p = np.asarray(labels), np.asarray(probabilities, np.float64)
    if (labels.ndim != 1 or labels.dtype.kind not in "iu" or p.shape != (len(labels), len(CLASS_NAMES))
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1, atol=2e-6, rtol=0)
            or set(np.unique(labels)) != set(range(len(CLASS_NAMES)))):
        raise ValueError("Metrics need finite normalized probabilities and every registered class")
    qcd = labels == 0
    per_class = {}
    for cls in range(1, len(CLASS_NAMES)):
        denominator = p[:, cls] + p[:, 0]
        score = np.divide(p[:, cls], denominator, out=np.zeros(len(p)), where=denominator > 0)
        signal_scores = np.sort(score[labels == cls])[::-1]
        index = math.ceil(.5 * len(signal_scores)) - 1
        threshold = float(signal_scores[index])
        count = int(np.count_nonzero(score[qcd] >= threshold))
        rejection = int(qcd.sum()) / count if count else None
        per_class[CLASS_NAMES[cls]] = dict(
            rejection_at_50pct=rejection, threshold=threshold, background_count=int(qcd.sum()),
            background_passing=count, achieved_signal_efficiency=float(np.mean(signal_scores >= threshold)),
            zero_fpr_censored=count == 0,
            one_event_resolution_rejection=int(qcd.sum()),
        )
    rejections = [r["rejection_at_50pct"] for r in per_class.values()]
    log_r50 = float(np.mean(np.log(rejections))) if all(v is not None for v in rejections) else None
    return dict(
        rows=len(labels), accuracy=float(np.mean(p.argmax(axis=1) == labels)),
        macro_ovr_auc=float(roc_auc_score(labels, p / p.sum(axis=1, keepdims=True), multi_class="ovr", average="macro")),
        cross_entropy=float(-np.mean(np.log(np.maximum(p[np.arange(len(p)), labels], 1e-30)))),
        macro_mean_log_qcd_rejection_at_50pct_signal=log_r50,
        macro_r50=math.exp(log_r50) if log_r50 is not None else None,
        per_class=per_class,
    )


def recovery(metrics: dict, baseline: dict, oracle: dict) -> dict:
    def ratio(value, low, high):
        return None if value is None or low is None or high is None or high == low else 100 * (value - low) / (high - low)
    result = {key: ratio(metrics[key], baseline[key], oracle[key])
              for key in ("accuracy", "macro_ovr_auc", "macro_r50")}
    result["per_class_r50"] = {
        name: ratio(metrics["per_class"][name]["rejection_at_50pct"],
                    baseline["per_class"][name]["rejection_at_50pct"],
                    oracle["per_class"][name]["rejection_at_50pct"])
        for name in CLASS_NAMES[1:]
    }
    return result
