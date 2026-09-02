"""Pure calibration, mixing, bootstrap, and selection semantics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .evaluation import classification_metrics
from .hcwdl_adjacent_output_handoff_contracts import (
    BOOTSTRAP_REPORT_CONTRACT, MIXTURE_CURVE_CONTRACT,
    SELECTED_MIXTURE_CONTRACT, TEMPERATURE_CALIBRATION_CONTRACT, artifact,
)


PROBABILITY_FLOOR = 2.0 ** -126
TEMPERATURE_BOUNDS = (.25, 4.0)
# Canonical schema names; reports may render these as Hbb/Hcc/Hqq.
REQUIRED_R50_CLASSES = ("Xbb", "Xcc", "Xqq")


def validate_probabilities(value: np.ndarray, *, name: str = "probabilities") -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if (
        result.ndim != 2 or result.shape[1] != 15 or len(result) == 0
        or not np.isfinite(result).all() or np.any(result < 0)
        or not np.allclose(result.sum(1, dtype=np.float64), 1, rtol=0, atol=2e-6)
    ):
        raise ValueError(f"{name} must be finite normalized [rows,15]")
    return result


def centered_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = validate_probabilities(probabilities).astype(np.float64)
    logits = np.log(np.maximum(values, PROBABILITY_FLOOR))
    return logits - logits.mean(axis=1, keepdims=True)


def temperature_calibrate(
    probabilities: np.ndarray, labels: np.ndarray, *, model_id: str,
    parents: Mapping[str, str], role: str = "V_blend",
) -> tuple[dict[str, Any], float]:
    logits = centered_log_probabilities(probabilities)
    target = np.asarray(labels, dtype=np.int64)
    if target.shape != (len(logits),) or np.any((target < 0) | (target >= 15)):
        raise ValueError("temperature calibration labels differ")

    def nll(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled = logits / temperature
        return float(np.mean(
            logsumexp(scaled, axis=1) - scaled[np.arange(len(target)), target]
        ))

    lower, upper = map(np.log, TEMPERATURE_BOUNDS)
    optimized = minimize_scalar(
        nll, method="bounded", bounds=(lower, upper),
        options={"xatol": 1e-10, "maxiter": 256},
    )
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError("temperature calibration failed")
    temperature = float(np.clip(np.exp(optimized.x), *TEMPERATURE_BOUNDS))
    report = artifact({
        "parents": dict(sorted(parents.items())), "model_id": model_id,
        "role": role, "rows": len(target), "bounds": list(TEMPERATURE_BOUNDS),
        "objective": "multiclass_negative_log_likelihood_v1",
        "temperature": temperature, "nll_before": nll(0.0),
        "nll_after": nll(np.log(temperature)), "labels_used_for_targets": False,
        "final_test_accessed": False,
    }, contract=TEMPERATURE_CALIBRATION_CONTRACT)
    return report, temperature


def mix_probabilities(
    rich: np.ndarray, poor: np.ndarray, *, alpha: float, family: str,
    rich_temperature: float = 1.0, poor_temperature: float = 1.0,
) -> np.ndarray:
    left = validate_probabilities(rich, name="rich probabilities")
    right = validate_probabilities(poor, name="poor probabilities")
    if left.shape != right.shape or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("output mixture alignment/alpha differs")
    if family == "arithmetic_probability":
        result = (1.0 - alpha) * left.astype(np.float64) + alpha * right
    elif family == "calibrated_centered_logit":
        if (
            not TEMPERATURE_BOUNDS[0] <= rich_temperature <= TEMPERATURE_BOUNDS[1]
            or not TEMPERATURE_BOUNDS[0] <= poor_temperature <= TEMPERATURE_BOUNDS[1]
        ):
            raise ValueError("output mixture temperature differs")
        logits = (
            (1.0 - alpha) * centered_log_probabilities(left) / rich_temperature
            + alpha * centered_log_probabilities(right) / poor_temperature
        )
        result = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    else:
        raise ValueError("unknown output mixture family")
    result = np.ascontiguousarray(result, dtype=np.float32)
    validate_probabilities(result, name="mixed probabilities")
    return result


def _binary_auc_influences(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(labels, dtype=bool)
    p = int(positive.sum()); n = len(positive) - p
    if p == 0 or n == 0:
        raise ValueError("AUC influence requires both classes")
    pos = np.asarray(scores[positive], dtype=np.float64)
    neg = np.asarray(scores[~positive], dtype=np.float64)
    neg_sorted = np.sort(neg); pos_sorted = np.sort(pos)
    pos_left = np.searchsorted(neg_sorted, pos, side="left")
    pos_right = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (pos_left + .5 * (pos_right - pos_left)) / n
    neg_left = np.searchsorted(pos_sorted, neg, side="left")
    neg_right = np.searchsorted(pos_sorted, neg, side="right")
    v01 = ((p - neg_right) + .5 * (neg_right - neg_left)) / p
    return v10, v01


def paired_stratified_macro_auc_bootstrap(
    candidate: np.ndarray, rich: np.ndarray, labels: np.ndarray, *,
    samples: int, seed: int,
) -> dict[str, float | int | str]:
    """Paired class-stratified Gaussian-multiplier influence bootstrap.

    Exact paired DeLong-style AUC influences determine the point estimate and
    its class-stratified variance.  Registered Gaussian multiplier draws then
    produce the one-sided interval without materializing an impossible
    ``samples x full-validation-population`` index matrix.  Every candidate
    uses the same seed and therefore the same multiplier stream.
    """
    cand = validate_probabilities(candidate, name="candidate probabilities")
    base = validate_probabilities(rich, name="rich probabilities")
    target = np.asarray(labels, dtype=np.int64)
    if cand.shape != base.shape or target.shape != (len(cand),):
        raise ValueError("paired bootstrap inputs differ")
    if samples <= 0 or seed < 0 or set(np.unique(target)) != set(range(15)):
        raise ValueError("paired bootstrap strata/samples differ")
    rows = len(target); classes = 15
    contribution = np.zeros(rows, dtype=np.float64)
    for class_id in range(classes):
        selected = target == class_id
        c_pos, c_neg = _binary_auc_influences(selected, cand[:, class_id])
        r_pos, r_neg = _binary_auc_influences(selected, base[:, class_id])
        contribution[selected] += .5 * (c_pos - r_pos) / classes
        negative_indexes = np.flatnonzero(~selected)
        negative_classes = target[negative_indexes]
        for true_class in range(classes):
            if true_class == class_id:
                continue
            in_stratum = negative_classes == true_class
            stratum_rows = int(np.sum(target == true_class))
            contribution[negative_indexes[in_stratum]] += (
                .5 * (c_neg[in_stratum] - r_neg[in_stratum]) / classes
                * stratum_rows / len(negative_indexes)
            )
    point = float(sum(
        contribution[target == class_id].mean() for class_id in range(classes)
    ))
    variance = 0.0
    for class_id in range(classes):
        values = contribution[target == class_id]
        variance += float(np.var(values, ddof=1)) / len(values)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    rng = np.random.default_rng(seed)
    estimates = point + standard_error * rng.standard_normal(samples)
    return {
        "method": "paired_class_stratified_gaussian_multiplier_auc_influence_bootstrap_v1",
        "samples": samples, "seed": seed, "difference": point,
        "standard_error": standard_error,
        "lower_95": float(np.quantile(estimates, .05)),
        "upper_95": float(np.quantile(estimates, .95)),
    }


def _r50(metrics: Mapping[str, Any], class_name: str) -> float:
    try:
        value = float(metrics["per_class"][class_name]["qcd_rejection"]["50pct"]["rejection"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"required {class_name} R50 is unavailable") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"required {class_name} R50 is nonfinite")
    return value


def evaluate_mixture_curve(
    *, rich_probabilities: np.ndarray, poor_probabilities: np.ndarray,
    labels: np.ndarray, rich_id: str, poor_id: str, transition_id: str,
    parents: Mapping[str, str], bootstrap_seed: int, bootstrap_samples: int = 2000,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    rich = validate_probabilities(rich_probabilities, name="rich probabilities")
    poor = validate_probabilities(poor_probabilities, name="poor probabilities")
    target = np.asarray(labels, dtype=np.int64)
    rich_calibration, rich_temperature = temperature_calibrate(
        rich, target, model_id=rich_id, parents=parents,
    )
    poor_calibration, poor_temperature = temperature_calibrate(
        poor, target, model_id=poor_id, parents=parents,
    )
    base_metrics = classification_metrics(np.log(np.maximum(rich, PROBABILITY_FLOOR)), target)
    candidates = []
    arrays: dict[tuple[str, int], np.ndarray] = {}
    for family in ("calibrated_centered_logit", "arithmetic_probability"):
        for numerator in range(41):
            alpha = numerator / 40
            mixed = mix_probabilities(
                rich, poor, alpha=alpha, family=family,
                rich_temperature=rich_temperature, poor_temperature=poor_temperature,
            )
            arrays[(family, numerator)] = mixed
            metrics = classification_metrics(
                np.log(np.maximum(mixed, PROBABILITY_FLOOR)), target,
            )
            bootstrap = paired_stratified_macro_auc_bootstrap(
                mixed, rich, target, samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            ratios = {
                name: _r50(metrics, name) / _r50(base_metrics, name)
                for name in REQUIRED_R50_CLASSES
            }
            auc_delta = float(metrics["macro_ovr_auc"]) - float(base_metrics["macro_ovr_auc"])
            feasible = bool(
                np.isfinite(auc_delta) and auc_delta >= -1e-4
                and float(bootstrap["lower_95"]) >= -3e-4
                and all(np.isfinite(x) and x >= .95 for x in ratios.values())
            )
            candidates.append({
                "family": family, "alpha_numerator": numerator,
                "alpha_denominator": 40, "alpha": alpha,
                "metrics": metrics, "auc_delta_from_rich": auc_delta,
                "r50_ratios_from_rich": ratios, "bootstrap": bootstrap,
                "feasible": feasible,
            })
    admissible = [row for row in candidates if row["feasible"]]
    if not admissible:
        raise RuntimeError("alpha zero must provide a noninferiority fallback")
    selected = max(admissible, key=lambda row: (
        row["alpha"], float(row["metrics"]["macro_ovr_auc"]),
        float(row["metrics"]["macro_mean_log_qcd_rejection_at_50pct_signal"]),
        int(row["family"] == "calibrated_centered_logit"),
        -len(row["family"]), row["family"],
    ))
    bootstrap_report = artifact({
        "parents": dict(sorted(parents.items())),
        "transition_id": transition_id, "role": "V_blend",
        "method": "paired_class_stratified_gaussian_multiplier_auc_influence_bootstrap_v1",
        "samples": bootstrap_samples, "seed": bootstrap_seed,
        "same_multiplier_stream_for_every_candidate": True,
        "one_sided_confidence_level": .95,
        "candidate_results": [{
            "family": row["family"],
            "alpha_numerator": row["alpha_numerator"],
            "alpha_denominator": row["alpha_denominator"],
            **row["bootstrap"],
        } for row in candidates],
        "final_test_accessed": False,
    }, contract=BOOTSTRAP_REPORT_CONTRACT)
    curve = artifact({
        "parents": dict(sorted(parents.items())), "transition_id": transition_id,
        "role": "V_blend", "rich_id": rich_id, "poor_id": poor_id,
        "alpha_grid": [[i, 40] for i in range(41)],
        "temperature_calibrations": {
            "rich": rich_calibration, "poor": poor_calibration,
        },
        "bootstrap_report": bootstrap_report,
        "rich_metrics": base_metrics, "candidates": candidates,
        "poor_metrics_do_not_control_graph": True, "final_test_accessed": False,
    }, contract=MIXTURE_CURVE_CONTRACT)
    selected_report = artifact({
        "parents": {**dict(sorted(parents.items())), "mixture_curve": curve["content_hash"]},
        "transition_id": transition_id, "role": "V_blend",
        "selected_family": selected["family"],
        "selected_alpha_numerator": selected["alpha_numerator"],
        "selected_alpha_denominator": 40,
        "selected_rich_temperature": rich_temperature,
        "selected_poor_temperature": poor_temperature,
        "selection_order": ["maximum_alpha", "macro_auc", "macro_r50", "calibrated_logit", "lexical"],
        "alpha_zero_fallback_registered": True,
        "selected_candidate": selected, "final_test_accessed": False,
    }, contract=SELECTED_MIXTURE_CONTRACT)
    chosen = arrays[(selected["family"], selected["alpha_numerator"])]
    return curve, selected_report, chosen


def distillation_target(probabilities: np.ndarray, *, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("distillation temperature differs")
    logits = centered_log_probabilities(probabilities) / temperature
    return np.ascontiguousarray(
        np.exp(logits - logsumexp(logits, axis=1, keepdims=True)), dtype=np.float32,
    )


__all__ = [
    "PROBABILITY_FLOOR", "REQUIRED_R50_CLASSES", "TEMPERATURE_BOUNDS",
    "centered_log_probabilities", "distillation_target", "evaluate_mixture_curve",
    "mix_probabilities", "paired_stratified_macro_auc_bootstrap",
    "temperature_calibrate", "validate_probabilities",
]
