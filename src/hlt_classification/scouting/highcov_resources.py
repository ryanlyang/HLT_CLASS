"""Strict validation for the frozen high-coverage matcher resources."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.data.cache_contracts import with_content_hash

from .highcov_assignment import DIAGNOSTIC_NAMES, SCORE_QUANTUM
from .highcov_hashing import canonical_sha256
from .highcov_scorers import EMPIRICAL_FEATURES


RESOURCE_CONTRACT = "HIGHCOV_MATCHER_RESOURCES/v1"
RESOURCE_SCHEMA_VERSION = 1
SELECTED_CONFIG_CONTENT_HASH = "ea7dde63b66f9dc07d9f7532a320d560e83a20885c2580478627d68fee1a68d3"
EMPIRICAL_CONTENT_HASH = "b09d4ff84049f9646d3521e00cf6838d69ef62e0876535a52ad7981dba29b6bb"
CALIBRATION_CONTENT_HASH = "7db644933ceb6541abd2e8869dccf8874d84753b46199cf2ab334082c1c3f53f"
MODEL_KEYS = (
    "full_development_for_audit",
    "holdout_0",
    "holdout_1",
    "holdout_2",
    "holdout_3",
)
HC_CONFIDENCE = 0.958730161190033
P99_CONFIDENCE = 0.9998835921287537


def _resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources" / "highcov_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"highcov resource must be a JSON object: {path}")
    return value


def _validate_content_hash(payload: Mapping[str, Any], expected: str, name: str) -> None:
    supplied = payload.get("content_hash")
    if supplied != expected:
        raise ValueError(f"{name} selected content hash differs")
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    if canonical_sha256(unhashed) != expected:
        raise ValueError(f"{name} semantic content hash differs")


def _finite_scalar(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_empirical(payload: Mapping[str, Any]) -> None:
    if payload.get("contract") != "highcov_empirical_edge_models_v1" or payload.get("schema_version") != 1:
        raise ValueError("empirical resource contract differs")
    if payload.get("final_test_opened") is not False:
        raise PermissionError("empirical resource does not attest sealed final test")
    models = payload.get("models")
    if not isinstance(models, Mapping) or tuple(models.keys()) != MODEL_KEYS:
        raise ValueError("empirical scorer model order or identities differ")
    for model_key in MODEL_KEYS:
        model = models[model_key]
        if not isinstance(model, Mapping) or set(model) < {"meta", "tables"}:
            raise ValueError(f"empirical model {model_key} is incomplete")
        meta = model["meta"]
        tables = model["tables"]
        if not isinstance(meta, Mapping) or not isinstance(tables, Mapping):
            raise ValueError(f"empirical model {model_key} structure differs")
        _finite_scalar(meta.get("intercept"), f"{model_key} intercept")
        weights = meta.get("weights")
        if not isinstance(weights, Mapping) or set(weights) != set(EMPIRICAL_FEATURES):
            raise ValueError(f"empirical model {model_key} feature order differs")
        for feature in EMPIRICAL_FEATURES:
            weight = _finite_scalar(weights[feature], f"{model_key}.{feature} weight")
            if weight < 0:
                raise ValueError("empirical weights must be nonnegative")
            table = tables.get(feature)
            if not isinstance(table, Mapping):
                raise ValueError(f"missing empirical table {model_key}.{feature}")
            llr = np.asarray(table.get("llr"), np.float64)
            if llr.ndim != 1 or not len(llr) or not np.isfinite(llr).all():
                raise ValueError(f"invalid LLR table {model_key}.{feature}")
            if feature in ("pid_transition", "charge_transition"):
                expected = 36 if feature == "pid_transition" else 9
                if table.get("kind") != "categorical" or len(llr) != expected:
                    raise ValueError(f"categorical table size differs for {feature}")
            else:
                edges = np.asarray(table.get("edges"), np.float64)
                if table.get("kind") != "continuous" or len(llr) != len(edges) + 1:
                    raise ValueError(f"continuous table size differs for {feature}")
                if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                    raise ValueError(f"continuous edges are not strictly increasing for {feature}")


def _validate_calibration(payload: Mapping[str, Any]) -> None:
    if payload.get("contract") != "highcov_final_confidence_calibration_v1" or payload.get("schema_version") != 1:
        raise ValueError("confidence resource contract differs")
    if payload.get("final_test_opened") is not False:
        raise PermissionError("confidence resource does not attest sealed final test")
    if payload.get("empirical_models_sha256") != EMPIRICAL_CONTENT_HASH:
        raise ValueError("confidence-to-empirical parent differs")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("method") != "standardized_logistic_then_isotonic_v1":
        raise ValueError("confidence calibration method differs")
    dimension = len(DIAGNOSTIC_NAMES)
    for name in ("mean", "scale", "coefficient"):
        value = np.asarray(model.get(name), np.float64)
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(f"confidence {name} dimension or finiteness differs")
        if name == "scale" and np.any(value <= 0):
            raise ValueError("confidence scales must be positive")
    _finite_scalar(model.get("intercept"), "confidence intercept")
    x = np.asarray(model.get("isotonic_x"), np.float64)
    y = np.asarray(model.get("isotonic_y"), np.float64)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("isotonic calibration shape differs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("isotonic calibration is nonfinite")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0) or np.any((y < 0) | (y > 1)):
        raise ValueError("isotonic calibration is not monotone in [0,1]")


def _validate_selected(payload: Mapping[str, Any]) -> None:
    if payload.get("contract") != "highcov_selected_matcher_v1" or payload.get("schema_version") != 1:
        raise ValueError("selected matcher contract differs")
    algorithm = payload.get("algorithm")
    artifacts = payload.get("artifacts")
    operating = payload.get("operating_points")
    if not isinstance(algorithm, Mapping) or not isinstance(artifacts, Mapping) or not isinstance(operating, Mapping):
        raise ValueError("selected matcher structure differs")
    gate = algorithm.get("candidate_gate")
    if not isinstance(gate, Mapping) or (
        gate.get("max_delta_r") != 0.3
        or gate.get("max_abs_log_pt_response") != 4.0
        or gate.get("max_abs_log_energy_response") != 4.0
        or algorithm.get("private_dustbins") is not True
        or algorithm.get("hard_anchor_lock") is not False
        or algorithm.get("tie_quantum") != SCORE_QUANTUM
    ):
        raise ValueError("selected matcher algorithm constants differ")
    if artifacts.get("empirical_models_content_hash") != EMPIRICAL_CONTENT_HASH or artifacts.get("confidence_calibration_content_hash") != CALIBRATION_CONTENT_HASH:
        raise ValueError("selected matcher artifact parents differ")
    if operating.get("completion_shell", {}).get("minimum_confidence") != 0.0:
        raise ValueError("completion-shell operating point differs")
    if operating.get("high_confidence_core", {}).get("minimum_confidence") != HC_CONFIDENCE:
        raise ValueError("HC operating point differs")
    if operating.get("p99_core", {}).get("minimum_confidence") != P99_CONFIDENCE:
        raise ValueError("P99 operating point differs")


@dataclass(frozen=True)
class HighCovResources:
    selected: Mapping[str, Any]
    empirical: Mapping[str, Any]
    calibration: Mapping[str, Any]
    selected_path: Path
    empirical_path: Path
    calibration_path: Path

    @property
    def content_hashes(self) -> dict[str, str]:
        return {
            "selected": SELECTED_CONFIG_CONTENT_HASH,
            "empirical": EMPIRICAL_CONTENT_HASH,
            "calibration": CALIBRATION_CONTENT_HASH,
        }


def load_highcov_resources(root: str | Path | None = None) -> HighCovResources:
    directory = _resource_root() if root is None else Path(root)
    selected_path = directory / "selected_matcher.json"
    empirical_path = directory / "empirical_models.json"
    calibration_path = directory / "final_confidence_calibration.json"
    selected = _load(selected_path)
    empirical = _load(empirical_path)
    calibration = _load(calibration_path)
    _validate_content_hash(selected, SELECTED_CONFIG_CONTENT_HASH, "selected matcher")
    _validate_content_hash(empirical, EMPIRICAL_CONTENT_HASH, "empirical models")
    _validate_content_hash(calibration, CALIBRATION_CONTENT_HASH, "confidence calibration")
    _validate_selected(selected)
    _validate_empirical(empirical)
    _validate_calibration(calibration)
    return HighCovResources(
        selected, empirical, calibration,
        selected_path, empirical_path, calibration_path,
    )


def resource_validation_report(root: str | Path | None = None) -> dict[str, Any]:
    resources = load_highcov_resources(root)
    paths = (resources.selected_path, resources.empirical_path, resources.calibration_path)
    return with_content_hash({
        "contract": RESOURCE_CONTRACT,
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "donor_commit": "64be1a82f11f42949fdffa639a869ccea2528bfa",
        "semantic_hashes": resources.content_hashes,
        "byte_hashes": {path.name: sha256_file(path) for path in paths},
        "model_keys": list(MODEL_KEYS),
        "diagnostic_names": list(DIAGNOSTIC_NAMES),
        "selected_constants": {
            "max_delta_r": 0.3,
            "max_abs_log_response": 4.0,
            "score_quantum": SCORE_QUANTUM,
            "hc_confidence": HC_CONFIDENCE,
            "p99_confidence": P99_CONFIDENCE,
        },
    })


__all__ = [
    "CALIBRATION_CONTENT_HASH", "EMPIRICAL_CONTENT_HASH", "HC_CONFIDENCE",
    "HighCovResources", "MODEL_KEYS", "P99_CONFIDENCE", "RESOURCE_CONTRACT",
    "SELECTED_CONFIG_CONTENT_HASH", "load_highcov_resources",
    "resource_validation_report",
]
