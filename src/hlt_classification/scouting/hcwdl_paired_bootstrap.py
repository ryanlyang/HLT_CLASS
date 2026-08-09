"""Deterministic fifteen-class paired bootstrap for HCWDL-RKD reports.

The resampling unit is a jet identity.  Each replicate independently samples
with replacement *within every one of the fifteen classes*, preserving the
observed class counts exactly.  A single index stream is shared by both models
in a comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    deterministic_npz_bytes,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

from .evaluation import classification_metrics
from .hcwdl_representation_artifacts import (
    CommittedBinaryEnvelope,
    FailureHook,
    publish_binary_envelope,
)
from .hcwdl_representation_contracts import logical_array_sha256
from .schema import CLASS_NAMES


PAIRED_BOOTSTRAP_CONTRACT: Final = "HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1"
BOOTSTRAP_SEED: Final = 8041
BOOTSTRAP_REPLICATES: Final = 2_000
CLASS_COUNT: Final = 15
BASE_METRICS: Final = (
    "cross_entropy",
    "accuracy",
    "balanced_accuracy",
    "macro_ovr_auc",
    "macro_mean_log_qcd_rejection_at_50pct_signal",
    "multiclass_brier",
    "top_label_ece_15_bin",
)
DEFAULT_METRICS: Final = (
    *BASE_METRICS,
    *(f"per_class.{name}.ovr_auc" for name in CLASS_NAMES),
    *(
        field
        for name in CLASS_NAMES[1:]
        for field in (
            f"per_class.{name}.qcd_pass_50pct",
            f"per_class.{name}.qcd_fpr_50pct",
            f"per_class.{name}.qcd_rejection_50pct",
        )
    ),
)


def _labels(value: np.ndarray) -> np.ndarray:
    labels = np.asarray(value, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("paired bootstrap labels must be a nonempty vector")
    if np.any((labels < 0) | (labels >= CLASS_COUNT)):
        raise ValueError("paired bootstrap labels lie outside 0..14")
    counts = np.bincount(labels, minlength=CLASS_COUNT)
    if np.any(counts == 0):
        raise ValueError("paired bootstrap requires all fifteen classes")
    return labels


def iter_stratified_indices(
    labels: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
):
    """Yield PCG64 class-stratified replicate indices in canonical order."""

    target = _labels(labels)
    if replicates <= 0:
        raise ValueError("paired bootstrap replicate count must be positive")
    groups = tuple(np.flatnonzero(target == class_index) for class_index in range(CLASS_COUNT))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for _ in range(int(replicates)):
        yield np.concatenate(
            tuple(group[rng.integers(0, len(group), size=len(group))] for group in groups)
        ).astype(np.int64, copy=False)


def bootstrap_index_sha256(
    labels: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> str:
    target = _labels(labels)
    raw_digest = hashlib.sha256()
    for indices in iter_stratified_indices(target, replicates=replicates, seed=seed):
        raw_digest.update(indices.astype(">i8", copy=False).tobytes(order="C"))
    return _bootstrap_index_logical_sha256(
        labels=target,
        replicates=replicates,
        seed=seed,
        index_stream_byte_sha256=raw_digest.hexdigest(),
    )


def _bootstrap_index_logical_sha256(
    *, labels: np.ndarray, replicates: int, seed: int,
    index_stream_byte_sha256: str,
) -> str:
    return canonical_sha256({
        "bit_generator": "PCG64",
        "class_counts": np.bincount(labels, minlength=CLASS_COUNT).tolist(),
        "contract": "HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP_INDICES/v1",
        "index_dtype": ">i8",
        "index_shape": [int(replicates), int(len(labels))],
        "index_stream_byte_sha256": require_sha256(
            index_stream_byte_sha256,
            name="paired-bootstrap index-stream byte SHA-256",
        ),
        "replicates": int(replicates),
        "rows": int(len(labels)),
        "seed": int(seed),
    })


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    if name in metrics:
        value = metrics.get(name)
    else:
        pieces = name.split(".")
        if len(pieces) != 3 or pieces[0] != "per_class":
            raise KeyError(f"unknown bootstrap metric {name!r}")
        class_name, field = pieces[1:]
        per_class = metrics.get("per_class")
        if not isinstance(per_class, Mapping) or class_name not in per_class:
            raise KeyError(f"bootstrap class metric {name!r} is absent")
        row = per_class[class_name]
        if not isinstance(row, Mapping):
            raise ValueError(f"bootstrap class metric {name!r} differs")
        if field == "ovr_auc":
            value = row.get("ovr_auc")
        else:
            qcd = row.get("qcd_rejection")
            if not isinstance(qcd, Mapping) or not isinstance(qcd.get("50pct"), Mapping):
                raise ValueError(f"bootstrap QCD metric {name!r} differs")
            qcd50 = qcd["50pct"]
            key = {
                "qcd_pass_50pct": "qcd_pass",
                "qcd_fpr_50pct": "qcd_fpr",
                "qcd_rejection_50pct": "rejection",
            }.get(field)
            if key is None:
                raise KeyError(f"unknown bootstrap class metric {name!r}")
            value = qcd50.get(key)
    if value is None or not math.isfinite(float(value)):
        raise FloatingPointError(f"bootstrap metric {name!r} is absent or nonfinite")
    return float(value)


def _validate_qcd_policy(metrics: Mapping[str, Any], labels: np.ndarray) -> None:
    qcd_rows = int(np.count_nonzero(labels == 0))
    if qcd_rows <= 0:
        raise ValueError("paired bootstrap QCD population is empty")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        return  # bounded custom metric fixtures carry no QCD rows
    for class_name in CLASS_NAMES[1:]:
        row = per_class.get(class_name)
        if not isinstance(row, Mapping):
            raise ValueError("paired bootstrap per-class metric registry differs")
        qcd = row.get("qcd_rejection")
        if not isinstance(qcd, Mapping) or not isinstance(qcd.get("50pct"), Mapping):
            raise ValueError("paired bootstrap 50pct QCD metric is absent")
        qcd50 = qcd["50pct"]
        passed = qcd50.get("qcd_pass")
        if isinstance(passed, bool) or not isinstance(passed, (int, np.integer)):
            raise ValueError("paired bootstrap raw QCD passing count differs")
        if passed < 0 or passed > qcd_rows:
            raise ValueError("paired bootstrap raw QCD passing count is out of range")
        expected_fpr = passed / qcd_rows
        expected_rejection = qcd_rows / max(1, passed)
        if (
            float(qcd50.get("qcd_fpr")) != float(expected_fpr)
            or float(qcd50.get("rejection")) != float(expected_rejection)
        ):
            raise ValueError("paired bootstrap QCD cap convention differs")


def paired_classification_bootstrap(
    *,
    left_logits: np.ndarray,
    right_logits: np.ndarray,
    labels: np.ndarray,
    identity_digests: np.ndarray,
    left_id: str,
    right_id: str,
    comparison_id: str,
    parent_hashes: Mapping[str, str],
    metrics: Sequence[str] = DEFAULT_METRICS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    metric_function: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]] = classification_metrics,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return the authenticated sidecar and compact replicate arrays."""

    target = _labels(labels)
    left = np.asarray(left_logits)
    right = np.asarray(right_logits)
    if left.shape != right.shape or left.shape != (len(target), CLASS_COUNT):
        raise ValueError("paired bootstrap logits must be aligned float arrays [rows,15]")
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise ValueError("paired bootstrap logits must remain exact FP32 arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise FloatingPointError("paired bootstrap logits contain nonfinite values")
    identities = np.asarray(identity_digests)
    if identities.dtype != np.uint8 or identities.shape != (len(target), 32):
        raise ValueError("paired bootstrap identities must be uint8 [rows,32]")
    identity_rows = [bytes(row).hex() for row in identities]
    if len(identity_rows) != len(set(identity_rows)):
        raise ValueError("paired bootstrap repeats a joined identity")
    identity_order_sha256 = canonical_sha256(identity_rows)
    if not comparison_id or not left_id or not right_id or left_id == right_id:
        raise ValueError("paired bootstrap comparison identities differ")
    names = tuple(str(name) for name in metrics)
    if not names or len(names) != len(set(names)):
        raise ValueError("paired bootstrap metric registry differs")

    left_values = np.empty((int(replicates), len(names)), dtype=np.float64)
    right_values = np.empty_like(left_values)
    index_byte_digest = hashlib.sha256()
    counts = np.bincount(target, minlength=CLASS_COUNT)
    for replicate, indices in enumerate(
        iter_stratified_indices(target, replicates=replicates, seed=seed)
    ):
        if not np.array_equal(np.bincount(target[indices], minlength=CLASS_COUNT), counts):
            raise RuntimeError("paired bootstrap failed class-count conservation")
        index_byte_digest.update(indices.astype(">i8", copy=False).tobytes(order="C"))
        left_metrics = metric_function(left[indices], target[indices])
        right_metrics = metric_function(right[indices], target[indices])
        _validate_qcd_policy(left_metrics, target[indices])
        _validate_qcd_policy(right_metrics, target[indices])
        left_values[replicate] = [_finite_metric(left_metrics, name) for name in names]
        right_values[replicate] = [_finite_metric(right_metrics, name) for name in names]

    point_left = metric_function(left, target)
    point_right = metric_function(right, target)
    _validate_qcd_policy(point_left, target)
    _validate_qcd_policy(point_right, target)
    left_point = np.asarray(
        [_finite_metric(point_left, name) for name in names], dtype=np.float64,
    )
    right_point = np.asarray(
        [_finite_metric(point_right, name) for name in names], dtype=np.float64,
    )
    deltas = left_values - right_values
    point = left_point - right_point
    arrays = {
        "replicate_left": left_values,
        "replicate_right": right_values,
        "replicate_deltas": deltas,
        "point_left": left_point,
        "point_right": right_point,
        "point_deltas": point,
    }
    intervals = {
        name: {
            "left": {
                "point": float(left_point[index]),
                "bootstrap_median": float(np.quantile(
                    left_values[:, index], 0.5, method="linear",
                )),
                "lower_95": float(np.quantile(
                    left_values[:, index], 0.025, method="linear",
                )),
                "upper_95": float(np.quantile(
                    left_values[:, index], 0.975, method="linear",
                )),
            },
            "right": {
                "point": float(right_point[index]),
                "bootstrap_median": float(np.quantile(
                    right_values[:, index], 0.5, method="linear",
                )),
                "lower_95": float(np.quantile(
                    right_values[:, index], 0.025, method="linear",
                )),
                "upper_95": float(np.quantile(
                    right_values[:, index], 0.975, method="linear",
                )),
            },
            "difference": {
                "point": float(point[index]),
                "bootstrap_median": float(np.quantile(
                    deltas[:, index], 0.5, method="linear",
                )),
                "lower_95": float(np.quantile(
                    deltas[:, index], 0.025, method="linear",
                )),
                "upper_95": float(np.quantile(
                    deltas[:, index], 0.975, method="linear",
                )),
            },
        }
        for index, name in enumerate(names)
    }
    normalized_parents = {
        str(name): require_sha256(value, name=f"paired bootstrap parent {name}")
        for name, value in sorted(parent_hashes.items())
    }
    if not normalized_parents:
        raise ValueError("paired bootstrap parent registry is empty")
    scientific_authorization = bool(
        names == DEFAULT_METRICS
        and int(replicates) == BOOTSTRAP_REPLICATES
        and int(seed) == BOOTSTRAP_SEED
    )
    sidecar = with_content_hash(
        {
            "contract": PAIRED_BOOTSTRAP_CONTRACT,
            "schema_version": 1,
            "comparison_id": comparison_id,
            "left_id": left_id,
            "right_id": right_id,
            "rows": int(len(target)),
            "joined_identity_order_sha256": identity_order_sha256,
            "class_counts": counts.tolist(),
            "replicates": int(replicates),
            "seed": int(seed),
            "bit_generator": "PCG64",
            "metric_order": list(names),
            "scientific_authorization": scientific_authorization,
            "replicate_index_sha256": _bootstrap_index_logical_sha256(
                labels=target,
                replicates=replicates,
                seed=seed,
                index_stream_byte_sha256=index_byte_digest.hexdigest(),
            ),
            "arrays": {
                name: logical_array_sha256(name, value)
                for name, value in arrays.items()
            },
            "intervals": intervals,
            "quantile_method": "linear",
            "zero_qcd_pass_policy": {
                "raw_qcd_pass_stored": True,
                "fpr": "qcd_pass/N_qcd",
                "rejection": "N_qcd/max(1,qcd_pass)",
                "log_rejection": "log(max(rejection,1.0))",
            },
            "parent_hashes": normalized_parents,
        }
    )
    return sidecar, arrays


def publish_paired_bootstrap_envelope(
    root: str | Path,
    *,
    bootstrap_report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    producer_task_id: str,
    registered_output_row: Mapping[str, Any],
    campaign_or_recovery_owner: Mapping[str, Any],
    failure_hook: FailureHook | None = None,
) -> CommittedBinaryEnvelope:
    """Publish the paired-bootstrap arrays under the mandatory shared envelope."""

    validate_content_hash(
        bootstrap_report,
        expected_contract=PAIRED_BOOTSTRAP_CONTRACT,
        expected_schema_version=1,
    )
    expected_arrays = bootstrap_report.get("arrays")
    if not isinstance(expected_arrays, Mapping) or set(arrays) != set(expected_arrays):
        raise ValueError("paired-bootstrap publication array registry differs")
    normalized = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    for name, array in normalized.items():
        if (
            array.dtype != np.float64
            or logical_array_sha256(name, array) != expected_arrays[name]
        ):
            raise ValueError(f"paired-bootstrap publication array differs: {name}")
    parents = bootstrap_report.get("parent_hashes")
    if not isinstance(parents, Mapping) or not parents:
        raise ValueError("paired-bootstrap publication parents differ")
    payload_bytes = deterministic_npz_bytes(normalized)
    payload = {
        key: value
        for key, value in bootstrap_report.items()
        if key not in {"contract", "schema_version", "content_hash", "parent_hashes"}
    }
    return publish_binary_envelope(
        root,
        artifact_contract=PAIRED_BOOTSTRAP_CONTRACT,
        producer_task_id=producer_task_id,
        schema={
            "container": "deterministic_npz",
            "arrays": {
                name: {"dtype": array.dtype.str, "shape": list(array.shape)}
                for name, array in sorted(normalized.items())
            },
        },
        immutable_parent_hashes=parents,
        registered_output_row=registered_output_row,
        campaign_or_recovery_owner=campaign_or_recovery_owner,
        payloads={"bootstrap_arrays.npz": payload_bytes},
        member_metadata={
            "bootstrap_arrays.npz": {
                "logical_sha256": canonical_sha256(dict(sorted(expected_arrays.items()))),
            },
        },
        sidecar_payload={
            **payload,
            "source_bootstrap_report_sha256": bootstrap_report["content_hash"],
        },
        failure_hook=failure_hook,
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CLASS_COUNT",
    "DEFAULT_METRICS",
    "PAIRED_BOOTSTRAP_CONTRACT",
    "bootstrap_index_sha256",
    "iter_stratified_indices",
    "paired_classification_bootstrap",
    "publish_paired_bootstrap_envelope",
]
