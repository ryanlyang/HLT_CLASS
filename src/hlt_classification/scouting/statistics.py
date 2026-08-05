"""Seed-aware, class-stratified paired PMARD final-test intervals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np

from hlt_classification.data.cache_contracts import load_npz_arrays
from .evaluation import softmax
from .training import BOOTSTRAP_SEED


@dataclass(frozen=True)
class PredictionSet:
    identities: np.ndarray
    logits: np.ndarray
    labels: np.ndarray


def load_prediction_set(path: str | Path) -> PredictionSet:
    arrays = load_npz_arrays(path)
    if set(arrays) != {"identity_keys", "logits", "labels"}:
        raise ValueError("PMARD prediction arrays differ")
    if arrays["logits"].shape != (len(arrays["labels"]), 15):
        raise ValueError("PMARD prediction shape differs")
    return PredictionSet(arrays["identity_keys"], arrays["logits"], arrays["labels"])


def _ce_rows(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from scipy.special import logsumexp
    return logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]


def _macro_log_rejection(probabilities: np.ndarray, labels: np.ndarray, indexes: np.ndarray) -> float:
    selected_labels = labels[indexes]; qcd_indexes = indexes[selected_labels == 0]
    values = []
    for signal in range(1, 15):
        signal_indexes = indexes[selected_labels == signal]
        scores = probabilities[signal_indexes, signal]
        ordered = np.sort(scores)[::-1]
        threshold = ordered[max(0, int(np.ceil(.5 * len(ordered))) - 1)]
        qcd_pass = np.count_nonzero(probabilities[qcd_indexes, signal] >= threshold)
        values.append(np.log(len(qcd_indexes) / max(1, qcd_pass)))
    return float(np.mean(values))


def nested_paired_intervals(
    candidate: Mapping[int, PredictionSet], baseline: Mapping[int, PredictionSet], *,
    replicates: int = 10_000, seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    seeds = sorted(candidate)
    if seeds != sorted(baseline) or len(seeds) != 5:
        raise ValueError("nested PMARD bootstrap requires the same five seeds")
    prepared = {}
    for master in seeds:
        left, right = candidate[master], baseline[master]
        if not np.array_equal(left.identities, right.identities) or not np.array_equal(left.labels, right.labels):
            raise ValueError("paired final predictions are not identity/label aligned")
        prepared[master] = {
            "labels": left.labels,
            "left_prob": softmax(left.logits), "right_prob": softmax(right.logits),
            "left_ce": _ce_rows(left.logits, left.labels), "right_ce": _ce_rows(right.logits, right.labels),
            "classes": [np.flatnonzero(left.labels == value) for value in range(15)],
        }
        if any(len(value) == 0 for value in prepared[master]["classes"]):
            raise ValueError("final bootstrap class is absent")
    rng = np.random.default_rng(seed)
    rejection_delta = np.empty(replicates); ce_delta = np.empty(replicates)
    for replicate in range(replicates):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        rep_rejection = []; rep_ce = []
        for master in chosen:
            data = prepared[int(master)]
            indexes = np.concatenate([
                rng.choice(group, size=len(group), replace=True) for group in data["classes"]
            ])
            rep_rejection.append(
                _macro_log_rejection(data["left_prob"], data["labels"], indexes)
                - _macro_log_rejection(data["right_prob"], data["labels"], indexes)
            )
            rep_ce.append(float(np.mean(data["left_ce"][indexes] - data["right_ce"][indexes])))
        rejection_delta[replicate] = np.mean(rep_rejection); ce_delta[replicate] = np.mean(rep_ce)
    def interval(values):
        return {
            "mean": float(np.mean(values)),
            "lower_95": float(np.quantile(values, .025, method="linear")),
            "upper_95": float(np.quantile(values, .975, method="linear")),
            "lower_bonferroni_97_5": float(np.quantile(values, .0125, method="linear")),
            "upper_bonferroni_97_5": float(np.quantile(values, .9875, method="linear")),
        }
    rejection = interval(rejection_delta); ce = interval(ce_delta)
    return {
        "replicates": replicates, "seed": seed,
        "macro_mean_log_qcd_rejection_difference": rejection,
        "cross_entropy_difference": ce,
        "positive_primary_evidence": rejection["lower_95"] > 0 and ce["upper_95"] < 0,
    }


def dependence_sensitivity(
    candidate: Mapping[int, PredictionSet], baseline: Mapping[int, PredictionSet], *,
    block_size: int = 1024, block_replicates: int = 1000, seed: int = BOOTSTRAP_SEED + 1,
) -> dict[str, object]:
    seeds = sorted(candidate)
    prepared = {}
    for master in seeds:
        left, right = candidate[master], baseline[master]
        if not np.array_equal(left.identities, right.identities):
            raise ValueError("dependence diagnostic identities differ")
        files = np.asarray([str(value).split("::tree::", 1)[0] for value in left.identities])
        prepared[master] = (
            left.labels, softmax(left.logits), softmax(right.logits),
            _ce_rows(left.logits, left.labels), _ce_rows(right.logits, right.labels), files,
        )
    all_files = sorted(set.intersection(*(set(row[-1]) for row in prepared.values())))
    delete_one = []
    for file_name in all_files:
        rejection = []; ce = []
        for labels, left_prob, right_prob, left_ce, right_ce, files in prepared.values():
            indexes = np.flatnonzero(files != file_name)
            rejection.append(_macro_log_rejection(left_prob, labels, indexes) - _macro_log_rejection(right_prob, labels, indexes))
            ce.append(float(np.mean(left_ce[indexes] - right_ce[indexes])))
        delete_one.append({"source_file": file_name, "macro_log_rejection_difference": float(np.mean(rejection)),
                           "cross_entropy_difference": float(np.mean(ce))})
    rng = np.random.default_rng(seed); rejection_values = []; ce_values = []
    for _ in range(block_replicates):
        seed_rejection = []; seed_ce = []
        for labels, left_prob, right_prob, left_ce, right_ce, files in prepared.values():
            sampled = []
            for file_name in sorted(set(files)):
                native = np.flatnonzero(files == file_name)
                blocks = [native[start:start + block_size] for start in range(0, len(native), block_size)]
                sampled.extend(blocks[index] for index in rng.integers(0, len(blocks), len(blocks)))
            indexes = np.concatenate(sampled)
            seed_rejection.append(_macro_log_rejection(left_prob, labels, indexes) - _macro_log_rejection(right_prob, labels, indexes))
            seed_ce.append(float(np.mean(left_ce[indexes] - right_ce[indexes])))
        rejection_values.append(np.mean(seed_rejection)); ce_values.append(np.mean(seed_ce))
    return {
        "source_file_delete_one": delete_one, "block_size": block_size,
        "block_replicates": block_replicates, "block_seed": seed,
        "block_macro_log_rejection_range": [float(np.min(rejection_values)), float(np.max(rejection_values))],
        "block_cross_entropy_range": [float(np.min(ce_values)), float(np.max(ce_values))],
    }


__all__ = ["PredictionSet", "dependence_sensitivity", "load_prediction_set", "nested_paired_intervals"]
