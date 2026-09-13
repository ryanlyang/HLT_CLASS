"""Paired confirmation statistics and source-file cluster bootstrap."""
from __future__ import annotations

import hashlib
import math
import numpy as np
from ..schema import CLASS_NAMES
from .targets import SCALARS
from .contracts import artifact


def metrics_contract():
    return artifact("METRICS", classes=list(CLASS_NAMES), macro_auc="unweighted_one_vs_rest",
        rejection="pSignal/(pSignal+pQCD)_zero_denominator_zero",
        threshold="descending_signal_ceil_half_include_ties", zero_background="null_with_resolution_bound",
        macro_r50="geometric_mean_ten_signals_null_if_any_censored", recovery="not_evaluated_no_oracle",
        report_role="VAL_REPORT", bootstrap="1000_paired_source_file_clusters",
        interval="percentile_2.5_97.5_require_900_valid", seed_variance="three_paired_fits_ddof1")

CONTRASTS = (("COMP", "CE"), ("STRUCT", "CE"), ("BOTH", "CE"),
             ("BOTH", "COMP"), ("BOTH", "STRUCT"), ("STRUCT", "COMP"))
BOOTSTRAP_SEED = int.from_bytes(hashlib.sha256(
    b"JC2/offline-aux/v1/report-bootstrap/20260912").digest()[:4], "big")


def flatten_metrics(metrics):
    result = {k: metrics[k] for k in ("accuracy", "macro_ovr_auc", "macro_r50")}
    result.update({k+"_r50": v["rejection_at_50pct"] for k, v in metrics["per_class"].items()})
    return result


def paired_statistics(metrics):
    result = {}
    for arm, base in CONTRASTS:
        by_seed = [{k: None if a[k] is None or b[k] is None else a[k]-b[k] for k in a}
                   for a, b in [(flatten_metrics(metrics[f"CONFIRM_{i:02d}_{arm}"]),
                                 flatten_metrics(metrics[f"CONFIRM_{i:02d}_{base}"])) for i in range(1, 4)]]
        result[arm+"_minus_"+base] = {
            k: dict(paired_deltas=[v[k] for v in by_seed],
                    mean=float(np.mean([v[k] for v in by_seed])) if all(v[k] is not None for v in by_seed) else None,
                    seed_std=float(np.std([v[k] for v in by_seed], ddof=1)) if all(v[k] is not None for v in by_seed) else None,
                    seeds_improved=sum(v[k] > 0 for v in by_seed if v[k] is not None)) for k in by_seed[0]}
    return result


class WeightedMetrics:
    """Cache sort/tie groups once. Draws are exact integer file multiplicities;
    equivalence to explicit row repetition is tested independently.
    """
    def __init__(self, labels, probabilities):
        self.y = np.asarray(labels, np.int64)
        p = np.asarray(probabilities, np.float64)
        if (p.shape != (len(self.y), 11) or not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(1), 1, atol=2e-6, rtol=0)):
            raise ValueError("Invalid bootstrap probabilities")
        self.correct = p.argmax(1) == self.y
        self.ce = -np.log(np.maximum(p[np.arange(len(p)), self.y], 1e-30))
        normalized = p/p.sum(1, keepdims=True)
        self.auc = [self._groups(normalized[:, k], np.arange(len(p))) for k in range(11)]
        self.r50 = []
        for k in range(1, 11):
            idx = np.flatnonzero((self.y == 0) | (self.y == k))
            denom = p[idx, k]+p[idx, 0]
            scores = np.divide(p[idx, k], denom, out=np.zeros(len(idx)), where=denom > 0)
            self.r50.append(self._groups(scores, idx))

    @staticmethod
    def _groups(scores, indices):
        permutation = np.argsort(scores, kind="stable")
        ordered = scores[permutation]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1])+1]
        return indices[permutation].astype(np.int32), starts.astype(np.int32)

    def evaluate(self, weights):
        weights = np.asarray(weights, np.float64)
        if weights.shape != self.y.shape or np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("Invalid bootstrap weights")
        counts = np.bincount(self.y, weights=weights, minlength=11)
        if np.any(counts == 0):
            return None
        auc = []
        for k, (order, starts) in enumerate(self.auc):
            w = weights[order]
            positive = np.add.reduceat(w*(self.y[order] == k), starts)
            negative = np.add.reduceat(w*(self.y[order] != k), starts)
            auc.append(float(np.dot(positive, np.cumsum(negative)-.5*negative) /
                             (counts[k]*(weights.sum()-counts[k]))))
        per_class = {}
        for k, (order, starts) in enumerate(self.r50, 1):
            w = weights[order]
            signal = np.add.reduceat(w*(self.y[order] == k), starts)[::-1]
            background = np.add.reduceat(w*(self.y[order] == 0), starts)[::-1]
            at = np.searchsorted(np.cumsum(signal), counts[k]*.5, side="left")
            passing = float(background[:at+1].sum())
            per_class[CLASS_NAMES[k]] = dict(rejection_at_50pct=float(counts[0]/passing) if passing else None)
        values = [v["rejection_at_50pct"] for v in per_class.values()]
        log_r = float(np.log(values).mean()) if all(v is not None for v in values) else None
        return dict(accuracy=float(np.dot(weights, self.correct)/weights.sum()),
                    cross_entropy=float(np.dot(weights, self.ce)/weights.sum()),
                    macro_ovr_auc=float(np.mean(auc)), macro_r50=None if log_r is None else math.exp(log_r),
                    macro_mean_log_qcd_rejection_at_50pct_signal=log_r, per_class=per_class)


def bootstrap_draw(file_index, draw):
    files, inverse = np.unique(file_index, return_inverse=True)
    # A single frozen PRNG stream; advancing by complete draws makes sharding
    # invariant and never redraws missing-class/censored samples.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selected = rng.integers(0, len(files), size=(draw+1, len(files)))[-1]
    return np.bincount(selected, minlength=len(files))[inverse]


def bootstrap(models, file_index, *, start=0, end=1000):
    if not 0 <= start < end <= 1000:
        raise ValueError("Unregistered bootstrap draw range")
    results = []
    for draw in range(start, end):
        weights = bootstrap_draw(file_index, draw)
        metrics = {k: m.evaluate(weights) for k, m in models.items()}
        stats = None if any(v is None for v in metrics.values()) else paired_statistics(metrics)
        results.append(dict(draw=draw, contrasts=None if stats is None else {
            c: {k: v["mean"] for k, v in row.items()} for c, row in stats.items()}))
    return results


def bootstrap_intervals(draws):
    if [r["draw"] for r in draws] != list(range(1000)):
        raise ValueError("Bootstrap must cover exactly 1000 draws")
    sample = next((r["contrasts"] for r in draws if r["contrasts"] is not None), {
        a+"_minus_"+b: {k: None for k in ["accuracy", "macro_ovr_auc", "macro_r50"]+
                       [c+"_r50" for c in CLASS_NAMES[1:]]} for a, b in CONTRASTS})
    result = {}
    for contrast, fields in sample.items():
        result[contrast] = {}
        for key in fields:
            valid = [r["contrasts"][contrast][key] for r in draws
                     if r["contrasts"] is not None and r["contrasts"][contrast][key] is not None]
            result[contrast][key] = dict(valid_draws=len(valid),
                interval=np.quantile(valid, [.025, .975], method="linear").tolist() if len(valid) >= 900 else None)
    return result


def auxiliary_metrics(heads, targets, pair_valid, normalizer, *, hlt=None):
    means, scales = np.array(normalizer["mean"]), np.array(normalizer["scale"])
    predicted = np.zeros_like(targets, dtype=np.float64)
    active = []
    for head, cols in (("counts", slice(0, 5)), ("scalars", slice(5, 7))):
        if head in heads:
            predicted[:, SCALARS[cols]] = heads[head]*scales[cols]+means[cols]
            active.extend(SCALARS[cols].tolist())
    for head, lo, hi in (("fractions", 5, 10), ("radial", 12, 20), ("pair", 20, 28)):
        if head in heads:
            x = heads[head].astype(np.float64)
            e = np.exp(x-x.max(1, keepdims=True))
            predicted[:, lo:hi] = e/e.sum(1, keepdims=True)

    def metrics(p, valid, model_log_probabilities=None):
        result = {}
        if active:
            error = p[:, active]-targets[:, active]
            result["scalar_mae"] = np.abs(error).mean(0).tolist()
            result["scalar_rmse"] = np.sqrt((error**2).mean(0)).tolist()
        if "counts" in heads:
            with np.errstate(over="ignore", invalid="ignore"):
                error = np.abs(np.maximum(np.expm1(p[:, :5]), 0)-np.expm1(targets[:, :5].astype(np.float64)))
            result["physical_count_mae"] = [float(error[:, i].mean()) if np.isfinite(error[:, i]).all() else None for i in range(5)]
            result["physical_count_overflow_rows"] = (~np.isfinite(error)).sum(0).tolist()
        for head, lo, hi in (("fractions", 5, 10), ("radial", 12, 20), ("pair", 20, 28)):
            if head not in heads:
                continue
            mask = valid if head == "pair" else np.ones(len(targets), bool)
            q, prob = targets[mask, lo:hi].astype(np.float64), p[mask, lo:hi]
            # HLT or TRAIN-constant predictors may have true zero support.
            # Preserve infinite KL as an explicit null+count, not pseudocounts.
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = np.log(prob) if model_log_probabilities is None else model_log_probabilities[head][mask]
                terms = np.where(q > 0, q*(np.log(q)-lp), 0.)
            sums = terms.sum(1)
            result[head+"_kl"] = float(sums.mean()) if mask.any() and np.isfinite(sums).all() else None
            result[head+"_infinite_kl_rows"] = int(np.isinf(sums).sum())
            result[head+"_rows"] = int(mask.sum())
        return result
    constants = np.broadcast_to(normalizer["constant_targets"], targets.shape)
    log_probs = {}
    for head in ("fractions", "radial", "pair"):
        if head in heads:
            x = heads[head].astype(np.float64)
            x = x-x.max(1, keepdims=True)
            log_probs[head] = x-np.log(np.exp(x).sum(1, keepdims=True))
    return dict(model=metrics(predicted, pair_valid, log_probs),
                train_mean=metrics(constants, pair_valid & normalizer["constant_pair_valid"]),
                hlt_observable=None if hlt is None else metrics(hlt["targets"], pair_valid & hlt["pair_valid"]),
                scalar_columns=active, zero_support_kl="null_with_infinite_row_count_no_pseudocount")
