"""Production inference for the calibrated ``fitted_strict`` matcher.

The implementation is a direct, standalone port of the canonical selective
matcher.  It deliberately does not refit either learned model at inference
time and never turns an abstention into an arbitrary offline assignment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
from scipy.optimize import linear_sum_assignment

from hlt_classification.data.cache_contracts import canonical_sha256

from .matching import ParticleSet, p4_kinematics, wrapped_delta_phi


FITTED_STRICT_CONTRACT: Final = "hlt_classification_fitted_strict_matcher_v1"
FITTED_STRICT_VARIANT: Final = "fitted_strict"
FITTED_STRICT_THRESHOLD: Final = 0.9828147479721088
FITTED_STRICT_DUMMY_SCORE: Final = -20.0
FITTED_STRICT_FORBIDDEN_SCORE: Final = -1.0e6
FITTED_STRICT_MAX_DR: Final = 0.15
FITTED_STRICT_MAX_ABS_LOG_RESPONSE: Final = 2.5
FITTED_STRICT_FEATURES: Final = (
    "dr", "log_pt", "log_energy", "pid_transition", "charge_transition",
)
FITTED_STRICT_DIAGNOSTICS: Final = (
    "score", "row_margin", "column_margin", "negative_dr_over_0p02",
    "negative_abs_log_pt", "negative_abs_log_energy", "pid_equal",
    "charge_equal", "mutual_geometry", "consensus",
)
FITTED_STRICT_ARTIFACT_DIR: Final = Path(__file__).with_name("resources") / "fitted_strict_v1"

# Canonical hashes are over parsed JSON and are stable across checkout line
# endings.  Original donor byte hashes remain recorded in ARTIFACT_PROVENANCE.
_EDGE_SEMANTIC_SHA256: Final = "a3630c6f52b359a2bb74000dde96da70f1a648eecb2101d3cea19c468a8a6c07"
_CONFIDENCE_SEMANTIC_SHA256: Final = "e32475865c58330b5035175f342cfd793676ec0b419249e4acd8b958725afa37"
_AUDIT_SEMANTIC_SHA256: Final = "f501b55dabd7f803443646bce29d7ab90bb27fade8170d38b3c4e8a360818f1f"


@dataclass(frozen=True)
class FittedStrictAssignment:
    """All real Hungarian assignments, before confidence selection."""

    hlt_index: np.ndarray
    offline_index: np.ndarray
    score: np.ndarray
    row_margin: np.ndarray
    column_margin: np.ndarray
    dr: np.ndarray
    abs_log_pt: np.ndarray
    abs_log_energy: np.ndarray
    pid_equal: np.ndarray
    charge_equal: np.ndarray
    mutual_geometry: np.ndarray

    def diagnostic_matrix(self) -> np.ndarray:
        if not len(self.hlt_index):
            return np.empty((0, len(FITTED_STRICT_DIAGNOSTICS)), np.float64)
        return np.column_stack((
            self.score,
            np.clip(self.row_margin, -20.0, 20.0),
            np.clip(self.column_margin, -20.0, 20.0),
            -self.dr / 0.02,
            -self.abs_log_pt,
            -self.abs_log_energy,
            self.pid_equal,
            self.charge_equal,
            self.mutual_geometry,
            np.ones(len(self.hlt_index), np.float64),
        ))


@dataclass(frozen=True)
class FittedStrictMatchResult:
    """Selective per-HLT-token result and assignment-level audit values."""

    match_index: np.ndarray
    match_mask: np.ndarray
    match_confidence: np.ndarray
    assignment: FittedStrictAssignment
    assignment_confidence: np.ndarray
    threshold: float

    @property
    def accepted_count(self) -> int:
        return int(np.count_nonzero(self.match_mask))


def _load_json(path: str | Path) -> dict[str, object]:
    location = Path(path)
    try:
        value = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid fitted_strict artifact {location}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"fitted_strict artifact {location} is not an object")
    return value


def _finite_vector(value: object, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _validate_table(name: str, raw: object) -> tuple[str, np.ndarray, np.ndarray]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"edge table {name} is not an object")
    kind = str(raw.get("kind"))
    categories = {"pid_transition": 36, "charge_transition": 9}
    if name in categories:
        if kind != "categorical":
            raise ValueError(f"edge table {name} must be categorical")
        llr = _finite_vector(raw.get("llr"), name=f"{name}.llr", length=categories[name])
        return kind, np.empty(0, np.float64), llr
    if kind != "continuous":
        raise ValueError(f"edge table {name} must be continuous")
    edges = _finite_vector(raw.get("edges"), name=f"{name}.edges", length=63)
    llr = _finite_vector(raw.get("llr"), name=f"{name}.llr", length=64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"edge table {name} edges are not strictly increasing")
    return kind, edges, llr


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _charge_codes(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value, np.float64)
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} charge is nonfinite")
    result = np.rint(raw).astype(np.int8)
    if not np.isin(result, (-1, 0, 1)).all():
        raise ValueError(f"{name} charge does not round to -1, 0, or +1")
    return result


def _pid_codes(categories: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(categories)
    if raw.ndim != 1 or not np.isin(raw, (-1, 0, 1, 2, 3, 4)).all():
        raise ValueError(f"{name} PID encoding differs from the five-category contract")
    return np.where(raw < 0, 5, raw).astype(np.int8)


class ConstituentMatcher:
    """Immutable calibrated ``fitted_strict`` inference engine."""

    def __init__(
        self, *, tables: Mapping[str, tuple[str, np.ndarray, np.ndarray]],
        intercept: float, weights: Mapping[str, float], confidence_mean: np.ndarray,
        confidence_scale: np.ndarray, confidence_parameters: np.ndarray,
        threshold: float, artifact_identity: str,
    ) -> None:
        frozen_tables = {}
        for name, (kind, edges, llr) in tables.items():
            edge_values = np.asarray(edges, np.float64).copy()
            llr_values = np.asarray(llr, np.float64).copy()
            edge_values.setflags(write=False); llr_values.setflags(write=False)
            frozen_tables[name] = (kind, edge_values, llr_values)
        self.tables = MappingProxyType(frozen_tables)
        self.intercept = float(intercept)
        self.weights = MappingProxyType(dict(weights))
        self.confidence_mean = confidence_mean.copy()
        self.confidence_scale = confidence_scale.copy()
        self.confidence_parameters = confidence_parameters.copy()
        self.confidence_mean.setflags(write=False)
        self.confidence_scale.setflags(write=False)
        self.confidence_parameters.setflags(write=False)
        self.threshold = float(threshold)
        self.artifact_identity = str(artifact_identity)

    @classmethod
    def from_artifacts(
        cls, edge_model_json: str | Path, confidence_model_json: str | Path, *,
        variant: str = FITTED_STRICT_VARIANT,
        independent_audit_json: str | Path | None = None,
        require_canonical: bool = True,
    ) -> "ConstituentMatcher":
        if variant != FITTED_STRICT_VARIANT:
            raise ValueError("this production module supports only fitted_strict")
        edge = _load_json(edge_model_json)
        confidence = _load_json(confidence_model_json)
        edge_hash = canonical_sha256(edge)
        confidence_hash = canonical_sha256(confidence)
        if require_canonical and edge_hash != _EDGE_SEMANTIC_SHA256:
            raise ValueError("edge model differs from the canonical 7,500-jet artifact")
        if require_canonical and confidence_hash != _CONFIDENCE_SEMANTIC_SHA256:
            raise ValueError("confidence model differs from the canonical 7,500-jet artifact")

        if tuple(edge.get("features", ())) != (*FITTED_STRICT_FEATURES, "rank_delta"):
            raise ValueError("edge model feature order differs from the fitted model")
        raw_tables = edge.get("tables")
        raw_meta = edge.get("meta")
        if not isinstance(raw_tables, Mapping) or not isinstance(raw_meta, Mapping):
            raise ValueError("edge model lacks tables or meta model")
        tables = {name: _validate_table(name, raw_tables.get(name)) for name in FITTED_STRICT_FEATURES}
        intercept = float(raw_meta.get("intercept", np.nan))
        raw_weights = raw_meta.get("weights")
        if not np.isfinite(intercept) or not isinstance(raw_weights, Mapping):
            raise ValueError("edge meta model is invalid")
        weights = {name: float(raw_weights.get(name, np.nan)) for name in FITTED_STRICT_FEATURES}
        if not np.isfinite(tuple(weights.values())).all() or np.any(np.asarray(tuple(weights.values())) < 0):
            raise ValueError("fitted_strict edge weights must be finite and nonnegative")

        models = confidence.get("models")
        thresholds = confidence.get("thresholds")
        if not isinstance(models, Mapping) or not isinstance(thresholds, Mapping):
            raise ValueError("confidence artifact lacks models or thresholds")
        model = models.get(variant)
        if not isinstance(model, Mapping):
            raise ValueError("confidence artifact lacks fitted_strict")
        mean = _finite_vector(model.get("mean"), name="confidence.mean", length=10)
        scale = _finite_vector(model.get("scale"), name="confidence.scale", length=10)
        parameters = _finite_vector(model.get("parameters"), name="confidence.parameters", length=11)
        if np.any(scale <= 0):
            raise ValueError("confidence scales must be positive")
        threshold = float(thresholds.get(variant, np.nan))
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("fitted_strict confidence threshold is invalid")
        if require_canonical and threshold != FITTED_STRICT_THRESHOLD:
            raise ValueError("fitted_strict threshold differs from the canonical operating point")

        audit_hash = None
        if independent_audit_json is not None:
            audit = _load_json(independent_audit_json)
            audit_hash = canonical_sha256(audit)
            if require_canonical and audit_hash != _AUDIT_SEMANTIC_SHA256:
                raise ValueError("independent matcher audit differs from the canonical artifact")
            if audit.get("all_checks_pass") is not True:
                raise ValueError("independent matcher audit did not pass")
            if audit.get("recommended_safety_first_candidate") != FITTED_STRICT_VARIANT:
                raise ValueError("independent matcher audit does not recommend fitted_strict")

        identity = canonical_sha256({
            "contract": FITTED_STRICT_CONTRACT,
            "variant": variant,
            "edge_model_semantic_sha256": edge_hash,
            "confidence_model_semantic_sha256": confidence_hash,
            "independent_audit_semantic_sha256": audit_hash,
            "threshold": threshold,
        })
        return cls(
            tables=tables, intercept=intercept, weights=weights,
            confidence_mean=mean, confidence_scale=scale,
            confidence_parameters=parameters, threshold=threshold,
            artifact_identity=identity,
        )

    @classmethod
    def canonical(cls) -> "ConstituentMatcher":
        return cls.from_artifacts(
            FITTED_STRICT_ARTIFACT_DIR / "fitted_edge_model.json",
            FITTED_STRICT_ARTIFACT_DIR / "confidence_models.json",
            independent_audit_json=FITTED_STRICT_ARTIFACT_DIR / "independent_validation.json",
        )

    def _table_score(self, name: str, values: np.ndarray) -> np.ndarray:
        kind, edges, llr = self.tables[name]
        if kind == "categorical":
            indexes = np.asarray(values, dtype=np.int64)
            if np.any(indexes < 0) or np.any(indexes >= len(llr)):
                raise ValueError(f"{name} category is out of bounds")
            return llr[indexes]
        output = np.zeros(np.shape(values), np.float64)
        valid = np.isfinite(values)
        output[valid] = llr[np.searchsorted(edges, np.asarray(values)[valid])]
        return output

    @staticmethod
    def _empty_assignment() -> FittedStrictAssignment:
        integer = np.empty(0, np.int64)
        floating = np.empty(0, np.float64)
        return FittedStrictAssignment(
            integer.copy(), integer.copy(), floating.copy(), floating.copy(), floating.copy(),
            floating.copy(), floating.copy(), floating.copy(), floating.copy(), floating.copy(),
            floating.copy(),
        )

    def _assign(
        self, score: np.ndarray, features: Mapping[str, np.ndarray],
        h_pid: np.ndarray, o_pid: np.ndarray, h_charge: np.ndarray, o_charge: np.ndarray,
    ) -> FittedStrictAssignment:
        nh, no = score.shape
        if nh == 0 or no == 0:
            return self._empty_assignment()
        augmented = np.full((nh, no + nh), 1.0e6, np.float64)
        augmented[:, :no] = -score
        augmented[np.arange(nh), no + np.arange(nh)] = -FITTED_STRICT_DUMMY_SCORE
        rows, columns = linear_sum_assignment(augmented)
        real = columns < no
        rows, columns = rows[real], columns[real]
        if not len(rows):
            return self._empty_assignment()

        chosen = score[rows, columns]
        alternative = score[rows].copy()
        alternative[np.arange(len(rows)), columns] = FITTED_STRICT_DUMMY_SCORE
        row_second = np.maximum(np.max(alternative, axis=1), FITTED_STRICT_DUMMY_SCORE)
        column_margin = np.empty(len(rows), np.float64)
        for index, (row, column) in enumerate(zip(rows, columns, strict=True)):
            competitors = np.delete(score[:, column], row)
            alternative_score = (
                max(float(np.max(competitors)), FITTED_STRICT_DUMMY_SCORE)
                if len(competitors) else FITTED_STRICT_DUMMY_SCORE
            )
            column_margin[index] = chosen[index] - alternative_score
        dr = features["dr"]
        h_best = np.argmin(dr, axis=1)
        o_best = np.argmin(dr, axis=0)
        return FittedStrictAssignment(
            rows.astype(np.int64, copy=False), columns.astype(np.int64, copy=False), chosen,
            chosen - row_second, column_margin, dr[rows, columns],
            np.abs(features["log_pt"][rows, columns]),
            np.abs(features["log_energy"][rows, columns]),
            (h_pid[rows] == o_pid[columns]).astype(np.float64),
            (h_charge[rows] == o_charge[columns]).astype(np.float64),
            ((h_best[rows] == columns) & (o_best[columns] == rows)).astype(np.float64),
        )

    def match_jet(self, hlt: ParticleSet, offline: ParticleSet) -> FittedStrictMatchResult:
        hpt, heta, hphi, henergy = p4_kinematics(hlt.p4)
        opt_all, oeta_all, ophi_all, oenergy_all = p4_kinematics(offline.p4)
        if np.any(hpt <= 0) or np.any(henergy <= 0):
            raise ValueError("fitted_strict requires positive HLT pT and energy")
        offline_population = np.flatnonzero(~np.asarray(offline.lost_track, dtype=np.bool_))
        opt = opt_all[offline_population]
        oeta = oeta_all[offline_population]
        ophi = ophi_all[offline_population]
        oenergy = oenergy_all[offline_population]
        if np.any(opt <= 0) or np.any(oenergy <= 0):
            raise ValueError("fitted_strict requires positive regular-offline pT and energy")
        h_pid = _pid_codes(hlt.categories, name="HLT")
        o_pid = _pid_codes(np.asarray(offline.categories)[offline_population], name="offline")
        h_charge = _charge_codes(hlt.charge, name="HLT")
        o_charge = _charge_codes(np.asarray(offline.charge)[offline_population], name="offline")

        nh, no = len(hpt), len(opt)
        match_index = np.full(nh, -1, np.int64)
        match_confidence = np.zeros(nh, np.float32)
        if nh == 0 or no == 0:
            empty = self._empty_assignment()
            return FittedStrictMatchResult(
                match_index, match_index >= 0, match_confidence, empty,
                np.empty(0, np.float64), self.threshold,
            )

        deta = heta[:, None] - oeta[None, :]
        dphi = wrapped_delta_phi(hphi[:, None], ophi[None, :])
        features = {
            "dr": np.hypot(deta, dphi),
            "log_pt": np.log(hpt[:, None] / opt[None, :]),
            "log_energy": np.log(henergy[:, None] / oenergy[None, :]),
            "pid_transition": h_pid[:, None].astype(np.int16) * 6 + o_pid[None, :].astype(np.int16),
            "charge_transition": (h_charge[:, None].astype(np.int16) + 1) * 3 + (o_charge[None, :].astype(np.int16) + 1),
        }
        gate = (
            (features["dr"] <= FITTED_STRICT_MAX_DR)
            & (np.abs(features["log_pt"]) <= FITTED_STRICT_MAX_ABS_LOG_RESPONSE)
            & (np.abs(features["log_energy"]) <= FITTED_STRICT_MAX_ABS_LOG_RESPONSE)
            & (h_pid[:, None] == o_pid[None, :])
            & (h_charge[:, None] == o_charge[None, :])
        )
        score = np.full((nh, no), self.intercept, np.float64)
        for name in FITTED_STRICT_FEATURES:
            score += self.weights[name] * self._table_score(name, features[name])
        score = np.where(gate, score, FITTED_STRICT_FORBIDDEN_SCORE)
        compact = self._assign(score, features, h_pid, o_pid, h_charge, o_charge)
        diagnostics = compact.diagnostic_matrix()
        confidence = _sigmoid(
            self.confidence_parameters[0]
            + ((diagnostics - self.confidence_mean) / self.confidence_scale)
            @ self.confidence_parameters[1:]
        )
        accepted = confidence >= self.threshold
        accepted_hlt = compact.hlt_index[accepted]
        accepted_offline = offline_population[compact.offline_index[accepted]]
        if len(np.unique(accepted_hlt)) != len(accepted_hlt):
            raise RuntimeError("fitted_strict produced duplicate HLT assignments")
        if len(np.unique(accepted_offline)) != len(accepted_offline):
            raise RuntimeError("fitted_strict produced duplicate offline assignments")
        match_index[accepted_hlt] = accepted_offline
        match_confidence[accepted_hlt] = confidence[accepted].astype(np.float32)
        match_mask = match_index >= 0
        if np.any(match_index[~match_mask] != -1):
            raise RuntimeError("fitted_strict abstention encoding is inconsistent")
        return FittedStrictMatchResult(
            match_index, match_mask, match_confidence, compact, confidence, self.threshold,
        )

    def match_batch(
        self, hlt_batch: Iterable[ParticleSet], offline_batch: Iterable[ParticleSet],
    ) -> list[FittedStrictMatchResult]:
        left, right = list(hlt_batch), list(offline_batch)
        if len(left) != len(right):
            raise ValueError("HLT and offline matcher batches differ in length")
        return [self.match_jet(hlt, offline) for hlt, offline in zip(left, right, strict=True)]


def fitted_strict_artifact_report(matcher: ConstituentMatcher) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract": FITTED_STRICT_CONTRACT,
        "schema_version": 1,
        "variant": FITTED_STRICT_VARIANT,
        "threshold": matcher.threshold,
        "artifact_identity": matcher.artifact_identity,
        "edge_model_semantic_sha256": _EDGE_SEMANTIC_SHA256,
        "confidence_model_semantic_sha256": _CONFIDENCE_SEMANTIC_SHA256,
        "independent_audit_semantic_sha256": _AUDIT_SEMANTIC_SHA256,
        "selective": True,
        "unmatched_index": -1,
        "offline_lost_tracks_excluded": True,
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


__all__ = [
    "ConstituentMatcher", "FITTED_STRICT_ARTIFACT_DIR", "FITTED_STRICT_CONTRACT",
    "FITTED_STRICT_DIAGNOSTICS", "FITTED_STRICT_THRESHOLD", "FITTED_STRICT_VARIANT",
    "FittedStrictAssignment", "FittedStrictMatchResult", "fitted_strict_artifact_report",
]
