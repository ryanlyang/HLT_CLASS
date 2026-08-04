"""Training-split-only semantic balancing statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json

from .cache import PradCacheDataset

PRAD_SEMANTIC_WEIGHTS_CONTRACT = "hlt_classification_prad_semantic_weights_v1"


def compute_training_input_statistics(
    paired_cache: PradCacheDataset,
) -> dict[str, Any]:
    """Fit masked raw-token moments from the authenticated train role only."""

    if (
        paired_cache.manifest.get("cache_kind") != "paired_views"
        or paired_cache.manifest.get("logical_role") != "train"
    ):
        raise ValueError("input statistics require the PRAD train paired cache")
    result: dict[str, Any] = {}
    for view in ("offline", "hlt"):
        count = np.zeros(14, dtype=np.int64)
        total = np.zeros(14, dtype=np.float64)
        total_square = np.zeros(14, dtype=np.float64)
        for arrays in paired_cache.iter_shards():
            tokens = np.asarray(arrays[f"{view}_tokens"], dtype=np.float64)
            mask = np.asarray(arrays[f"{view}_mask"], dtype=np.bool_)
            values = tokens[mask]
            if not np.isfinite(values).all():
                raise FloatingPointError("PRAD train input statistics are nonfinite")
            count += len(values)
            total += values.sum(axis=0)
            total_square += np.square(values).sum(axis=0)
        if np.any(count == 0):
            raise ValueError("PRAD train input statistics have an empty feature")
        mean = total / count
        variance = np.maximum(total_square / count - np.square(mean), 0.0)
        result[view] = {
            "particle_count": int(count[0]),
            "mean": mean.tolist(),
            "standard_deviation": np.sqrt(variance).tolist(),
        }
    return result


def compute_semantic_positive_weights(
    target_cache: PradCacheDataset | Sequence[PradCacheDataset],
) -> dict[str, Any]:
    """Count matched K=2,3,4 pair labels using only the train role."""

    caches = (
        (target_cache,)
        if isinstance(target_cache, PradCacheDataset)
        else tuple(target_cache)
    )
    if not caches or any(
        cache.manifest.get("cache_kind") != "structural_targets"
        or cache.manifest.get("logical_role") != "train"
        for cache in caches
    ):
        raise ValueError("semantic weights require PRAD train target caches")
    positives = np.zeros(3, dtype=np.int64)
    negatives = np.zeros(3, dtype=np.int64)
    teacher_positives = np.zeros(3, dtype=np.int64)
    teacher_negatives = np.zeros(3, dtype=np.int64)
    for cache in caches:
        for arrays in cache.iter_shards():
            mapping = arrays["hlt_to_offline"].astype(np.int64)
            assignments = arrays["ca_assignments"].astype(np.int64)
            for row in range(len(mapping)):
                for scale in range(3):
                    offline_labels = assignments[row, scale]
                    offline_valid = offline_labels >= 0
                    pair_valid = offline_valid[:, None] & offline_valid[None, :]
                    np.fill_diagonal(pair_valid, False)
                    same = offline_labels[:, None] == offline_labels[None, :]
                    teacher_positives[scale] += int(np.sum(pair_valid & same))
                    teacher_negatives[scale] += int(np.sum(pair_valid & ~same))
                valid_hlt = np.flatnonzero(mapping[row] >= 0)
                if len(valid_hlt) < 2:
                    continue
                offline_indices = mapping[row, valid_hlt]
                for scale in range(3):
                    labels = assignments[row, scale, offline_indices]
                    pair_valid = (labels[:, None] >= 0) & (labels[None, :] >= 0)
                    np.fill_diagonal(pair_valid, False)
                    same = labels[:, None] == labels[None, :]
                    positives[scale] += int(np.sum(pair_valid & same))
                    negatives[scale] += int(np.sum(pair_valid & ~same))
    if (
        np.any(positives == 0)
        or np.any(negatives == 0)
        or np.any(teacher_positives == 0)
        or np.any(teacher_negatives == 0)
    ):
        raise ValueError("semantic training split has an empty positive/negative class")
    weights = negatives.astype(np.float64) / positives.astype(np.float64)
    teacher_weights = teacher_negatives.astype(np.float64) / teacher_positives.astype(np.float64)
    return with_content_hash(
        {
            "contract": PRAD_SEMANTIC_WEIGHTS_CONTRACT,
            "schema_version": 1,
            "target_cache_manifest_sha256": (
                caches[0].manifest_sha256 if len(caches) == 1 else None
            ),
            "target_cache_manifest_sha256s": [
                cache.manifest_sha256 for cache in caches
            ],
            "fit_role": "train",
            "multiplicities": [2, 3, 4],
            "positive_pairs": positives.tolist(),
            "negative_pairs": negatives.tolist(),
            "positive_weights": weights.tolist(),
            "student_matched": {
                "positive_pairs": positives.tolist(),
                "negative_pairs": negatives.tolist(),
                "positive_weights": weights.tolist(),
            },
            "teacher_offline": {
                "positive_pairs": teacher_positives.tolist(),
                "negative_pairs": teacher_negatives.tolist(),
                "positive_weights": teacher_weights.tolist(),
            },
            "pair_directionality": "ordered_symmetric_pairs_excluding_diagonal",
        }
    )


def save_semantic_positive_weights(
    target_cache: PradCacheDataset | Sequence[PradCacheDataset],
    output: str | Path,
    *,
    paired_cache: PradCacheDataset | None = None,
) -> dict[str, Any]:
    report = compute_semantic_positive_weights(target_cache)
    if paired_cache is not None:
        target_caches = (
            (target_cache,)
            if isinstance(target_cache, PradCacheDataset)
            else tuple(target_cache)
        )
        if any(
            paired_cache.manifest.get("identity_order_sha256")
            != cache.manifest.get("identity_order_sha256")
            for cache in target_caches
        ):
            raise ValueError("PRAD statistics train populations differ")
        report = with_content_hash(
            {
                **{key: value for key, value in report.items() if key != "content_hash"},
                "paired_cache_manifest_sha256": paired_cache.manifest_sha256,
                "input_statistics": compute_training_input_statistics(paired_cache),
                "normalization_policy": (
                    "statistics_fitted_on_train_only; canonical Weaver fixed "
                    "feature transforms retained for baseline parity"
                ),
                "hard_class_balancing": (
                    "none; deterministic class-stratified train population and "
                    "canonical unweighted baseline objective"
                ),
            }
        )
    write_immutable_json(output, report)
    return report


__all__ = [
    "compute_semantic_positive_weights",
    "compute_training_input_statistics",
    "save_semantic_positive_weights",
]
