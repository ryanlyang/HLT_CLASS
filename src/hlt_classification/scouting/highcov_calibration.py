"""Compact logistic-plus-isotonic post-assignment confidence calibration."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ConfidenceCalibrator:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    isotonic_x: np.ndarray
    isotonic_y: np.ndarray

    def predict(self, diagnostics: np.ndarray) -> np.ndarray:
        z = (np.asarray(diagnostics, np.float64) - self.mean) / self.scale
        logit = self.intercept + z @ self.coefficient
        raw = 1 / (1 + np.exp(-np.clip(logit, -40, 40)))
        return np.interp(raw, self.isotonic_x, self.isotonic_y).astype(np.float32)

    def payload(self) -> dict[str, object]:
        return {
            "method": "standardized_logistic_then_isotonic_v1",
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(), "intercept": self.intercept,
            "isotonic_x": self.isotonic_x.tolist(), "isotonic_y": self.isotonic_y.tolist(),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "ConfidenceCalibrator":
        if value.get("method") != "standardized_logistic_then_isotonic_v1":
            raise ValueError("confidence calibrator method differs")
        return cls(
            np.asarray(value["mean"], np.float64), np.asarray(value["scale"], np.float64),
            np.asarray(value["coefficient"], np.float64), float(value["intercept"]),
            np.asarray(value["isotonic_x"], np.float64), np.asarray(value["isotonic_y"], np.float64),
        )


def fit_confidence_calibrator(
    logistic_x: np.ndarray, logistic_y: np.ndarray,
    isotonic_x: np.ndarray, isotonic_y: np.ndarray, *, seed: int = 20260807,
) -> ConfidenceCalibrator:
    x = np.asarray(logistic_x, np.float64); y = np.asarray(logistic_y, np.int8)
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-6] = 1
    model = LogisticRegression(
        C=10.0, max_iter=1000, solver="lbfgs", random_state=seed,
    ).fit((x - mean) / scale, y)
    base = ConfidenceCalibrator(
        mean, scale, model.coef_[0], float(model.intercept_[0]),
        np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]),
    )
    raw = base.predict(isotonic_x)
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, isotonic_y)
    return ConfidenceCalibrator(
        mean, scale, model.coef_[0], float(model.intercept_[0]),
        np.asarray(isotonic.X_thresholds_), np.asarray(isotonic.y_thresholds_),
    )


def brier(probability: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((np.asarray(probability) - np.asarray(label)) ** 2))


def expected_calibration_error(
    probability: np.ndarray, label: np.ndarray, *, bins: int = 20,
) -> float:
    p = np.asarray(probability); y = np.asarray(label)
    edges = np.linspace(0, 1, bins + 1); total = len(p); value = 0.0
    for index in range(bins):
        selected = (p >= edges[index]) & (p < edges[index + 1] if index + 1 < bins else p <= 1)
        if np.any(selected):
            value += np.count_nonzero(selected) / total * abs(float(p[selected].mean() - y[selected].mean()))
    return float(value)


__all__ = [
    "ConfidenceCalibrator", "brier", "expected_calibration_error", "fit_confidence_calibrator",
]
