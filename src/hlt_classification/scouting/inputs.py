"""Bounded, streaming tensor builders for HLT and native-offline ParT views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np

from .schema import (
    HLT_FEATURE_SPECS, HLT_MAX_LENGTH, HLT_VECTOR_BRANCHES,
    OFFLINE_CHARGED_MAX_LENGTH, OFFLINE_NEUTRAL_MAX_LENGTH,
    TOFF_CHARGED_FEATURES, TOFF_NEUTRAL_FEATURES, TOFF_VECTOR_FIELDS,
)


@dataclass(frozen=True)
class ParticleInputs:
    features: np.ndarray
    vectors: np.ndarray
    mask: np.ndarray
    raw_lengths: np.ndarray


@dataclass(frozen=True)
class NativeOfflineInputs:
    charged: ParticleInputs
    neutral: ParticleInputs


_TOFF_NORMALIZATION = {
    "pt_log_nopuppi": (1.0, .5), "e_log_nopuppi": (1.3, .5),
    "abseta": (.6, 1.6), "normchi2": (5.0, .2), "quality": (0.0, .2),
    "dz": (0.0, 180.0), "dzsig": (0.0, .9), "dxy": (0.0, 300.0),
    "dxysig": (0.0, 1.0), "btagEtaRel": (1.5, .5),
    "btagPtRatio": (0.0, 1.0), "btagPParRatio": (0.0, 1.0),
}


def _rows(value: object) -> list[np.ndarray]:
    try:
        import awkward as ak
        if isinstance(value, ak.Array):
            return [np.asarray(row) for row in ak.to_list(value)]
    except ImportError:
        pass
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return [np.asarray(row) for row in value]
    return [np.asarray(row) for row in value]  # type: ignore[arg-type]


def _sanitize_transform(
    value: np.ndarray, median: float, factor: float, lower: float, upper: float,
) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float32)
    valid = np.isfinite(raw) & (raw >= -1.0e32) & (raw <= 1.0e32)
    clean = np.where(valid, raw, np.float32(0.0))
    return np.clip((clean - np.float32(median)) * np.float32(factor), lower, upper)


def _infer_rows(arrays: Mapping[str, object], branch: str) -> int:
    rows = _rows(arrays[branch])
    return len(rows)


def build_hlt_inputs(
    arrays: Mapping[str, object], *, max_length: int = HLT_MAX_LENGTH,
) -> ParticleInputs:
    if max_length <= 0 or max_length > HLT_MAX_LENGTH:
        raise ValueError(f"HLT max_length must lie in [1,{HLT_MAX_LENGTH}]")
    feature_rows = {spec.branch: _rows(arrays[spec.branch]) for spec in HLT_FEATURE_SPECS}
    vector_rows = {name: _rows(arrays[name]) for name in HLT_VECTOR_BRANCHES}
    row_count = len(next(iter(feature_rows.values())))
    all_rows = [*feature_rows.values(), *vector_rows.values()]
    if any(len(value) != row_count for value in all_rows):
        raise ValueError("HLT branch row counts differ")
    features = np.zeros((row_count, len(HLT_FEATURE_SPECS), max_length), np.float32)
    vectors = np.zeros((row_count, 4, max_length), np.float32)
    mask = np.zeros((row_count, 1, max_length), np.bool_)
    lengths = np.zeros(row_count, np.int32)
    for row in range(row_count):
        native_lengths = [len(value[row]) for value in all_rows]
        if len(set(native_lengths)) != 1:
            raise ValueError(f"HLT collection length mismatch in row {row}")
        lengths[row] = native_lengths[0]
        visible = min(native_lengths[0], max_length)
        mask[row, 0, :visible] = True
        for channel, spec in enumerate(HLT_FEATURE_SPECS):
            features[row, channel, :visible] = _sanitize_transform(
                feature_rows[spec.branch][row][:visible], spec.median, spec.factor,
                spec.lower, spec.upper,
            )
        for channel, branch in enumerate(HLT_VECTOR_BRANCHES):
            vectors[row, channel, :visible] = _sanitize_transform(
                vector_rows[branch][row][:visible], 0, 1, -1.0e32, 1.0e32,
            )
    return ParticleInputs(features, vectors, mask, lengths)


def transform_hlt_endpoint_features(raw_features: np.ndarray) -> np.ndarray:
    """Apply the canonical 21-channel HLT transform without a length cap.

    This helper only transforms an already authenticated raw endpoint matrix.
    It does not change the ordinary deployable 200-token ``build_hlt_inputs``
    contract; privileged training-only views can use it when they must retain
    every raw HLT particle.
    """

    raw = np.asarray(raw_features)
    if raw.ndim != 2 or raw.shape[1] != len(HLT_FEATURE_SPECS):
        raise ValueError("raw HLT endpoint features must be [tokens,21]")
    transformed = np.empty(raw.shape, dtype=np.float32)
    for channel, spec in enumerate(HLT_FEATURE_SPECS):
        transformed[:, channel] = _sanitize_transform(
            raw[:, channel], spec.median, spec.factor, spec.lower, spec.upper,
        )
    return np.ascontiguousarray(transformed)


def transform_hlt_endpoint_vectors(raw_vectors: np.ndarray) -> np.ndarray:
    """Apply the canonical finite-value sanitation to an unbounded HLT p4."""

    raw = np.asarray(raw_vectors)
    if raw.ndim != 2 or raw.shape[1] != len(HLT_VECTOR_BRANCHES):
        raise ValueError("raw HLT endpoint vectors must be [tokens,4]")
    transformed = np.empty(raw.shape, dtype=np.float32)
    for channel in range(raw.shape[1]):
        transformed[:, channel] = _sanitize_transform(
            raw[:, channel], 0, 1, -1.0e32, 1.0e32,
        )
    return np.ascontiguousarray(transformed)


def _build_native_group(
    arrays: Mapping[str, object], *, prefix: str, feature_names: Sequence[str],
    max_length: int,
) -> ParticleInputs:
    branches = [f"{prefix}_{name}" for name in feature_names]
    vector_branches = [f"{prefix}_{name}" for name in TOFF_VECTOR_FIELDS]
    rows_by_branch = {name: _rows(arrays[name]) for name in branches + vector_branches}
    row_count = len(next(iter(rows_by_branch.values())))
    if any(len(value) != row_count for value in rows_by_branch.values()):
        raise ValueError(f"{prefix} branch row counts differ")
    features = np.zeros((row_count, len(feature_names), max_length), np.float32)
    vectors = np.zeros((row_count, 4, max_length), np.float32)
    mask = np.zeros((row_count, 1, max_length), np.bool_)
    lengths = np.zeros(row_count, np.int32)
    for row in range(row_count):
        native_lengths = [len(value[row]) for value in rows_by_branch.values()]
        if len(set(native_lengths)) != 1:
            raise ValueError(f"{prefix} collection length mismatch in row {row}")
        lengths[row] = native_lengths[0]
        visible = min(native_lengths[0], max_length)
        mask[row, 0, :visible] = True
        for channel, name in enumerate(feature_names):
            center, scale = _TOFF_NORMALIZATION.get(name, (0.0, 1.0))
            features[row, channel, :visible] = _sanitize_transform(
                rows_by_branch[f"{prefix}_{name}"][row][:visible], center, scale,
                -5.0 if name in _TOFF_NORMALIZATION else -1.0e32,
                5.0 if name in _TOFF_NORMALIZATION else 1.0e32,
            )
        for channel, name in enumerate(TOFF_VECTOR_FIELDS):
            vectors[row, channel, :visible] = _sanitize_transform(
                rows_by_branch[f"{prefix}_{name}"][row][:visible], 0, 1,
                -1.0e32, 1.0e32,
            )
    return ParticleInputs(features, vectors, mask, lengths)


def build_native_offline_inputs(arrays: Mapping[str, object]) -> NativeOfflineInputs:
    return NativeOfflineInputs(
        charged=_build_native_group(
            arrays, prefix="cpfcandlt", feature_names=TOFF_CHARGED_FEATURES,
            max_length=OFFLINE_CHARGED_MAX_LENGTH,
        ),
        neutral=_build_native_group(
            arrays, prefix="npfcand", feature_names=TOFF_NEUTRAL_FEATURES,
            max_length=OFFLINE_NEUTRAL_MAX_LENGTH,
        ),
    )


def deployment_length(raw_length: int) -> int:
    from .schema import HLT_MIN_DEPLOYMENT_LENGTH
    return min(HLT_MAX_LENGTH, max(HLT_MIN_DEPLOYMENT_LENGTH, int(raw_length)))


__all__ = [
    "NativeOfflineInputs", "ParticleInputs", "build_hlt_inputs",
    "build_native_offline_inputs", "deployment_length",
    "transform_hlt_endpoint_features",
    "transform_hlt_endpoint_vectors",
]
