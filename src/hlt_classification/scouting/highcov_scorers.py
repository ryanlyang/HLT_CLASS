"""Geometry and empirical likelihood edge scorers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from scipy.optimize import minimize

from .highcov_features import EdgeMatrices


EMPIRICAL_FEATURES = (
    "dr", "log_pt", "log_energy", "pid_transition", "charge_transition", "rank_delta",
)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def table_score(values: np.ndarray, table: Mapping[str, object]) -> np.ndarray:
    llr = np.asarray(table["llr"], np.float64)
    if table["kind"] == "categorical":
        return llr[np.asarray(values, np.int64)]
    edges = np.asarray(table["edges"], np.float64)
    output = np.zeros(np.shape(values), np.float64)
    valid = np.isfinite(values)
    output[valid] = llr[np.searchsorted(edges, np.asarray(values)[valid])]
    return output


@dataclass(frozen=True)
class EmpiricalScorer:
    tables: Mapping[str, Mapping[str, object]]
    intercept: float
    weights: Mapping[str, float]
    features: tuple[str, ...] = EMPIRICAL_FEATURES

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], *, include_rank: bool = True) -> "EmpiricalScorer":
        meta = payload["meta"]
        assert isinstance(meta, Mapping)
        names = EMPIRICAL_FEATURES if include_rank else EMPIRICAL_FEATURES[:-1]
        return cls(
            tables=payload["tables"], intercept=float(meta["intercept"]),
            weights={name: float(meta["weights"][name]) for name in names},
            features=names,
        )

    def score(self, matrices: EdgeMatrices) -> np.ndarray:
        result = np.full(matrices.shape, self.intercept, np.float64)
        for name in self.features:
            result += self.weights[name] * table_score(getattr(matrices, name), self.tables[name])
        return result


def geometry_score(matrices: EdgeMatrices) -> np.ndarray:
    return -(matrices.dr / .02) ** 2


def geometry_response_score(matrices: EdgeMatrices) -> np.ndarray:
    return -(
        (matrices.dr / .02) ** 2
        + .45 * (matrices.log_pt / .45) ** 2
        + .15 * ((matrices.log_energy - matrices.log_pt) / .30) ** 2
    )


def fit_tables(
    positive: Mapping[str, np.ndarray], negative: Mapping[str, np.ndarray], *,
    bins: int = 64, smoothing: float = 2.0,
) -> dict[str, dict[str, object]]:
    tables: dict[str, dict[str, object]] = {}
    for name in ("dr", "log_pt", "log_energy", "rank_delta"):
        pos = positive[name][np.isfinite(positive[name])]
        neg = negative[name][np.isfinite(negative[name])]
        edges = np.unique(np.quantile(np.r_[pos, neg], np.linspace(0, 1, bins + 1)[1:-1]))
        pc = np.bincount(np.searchsorted(edges, pos), minlength=len(edges) + 1)
        nc = np.bincount(np.searchsorted(edges, neg), minlength=len(edges) + 1)
        pp = (pc + smoothing) / (len(pos) + smoothing * len(pc))
        np_ = (nc + smoothing) / (len(neg) + smoothing * len(nc))
        tables[name] = {
            "kind": "continuous", "edges": edges.tolist(),
            "llr": np.clip(np.log(pp / np_), -6, 6).tolist(),
        }
    for name, categories in (("pid_transition", 36), ("charge_transition", 9)):
        pc = np.bincount(positive[name].astype(int), minlength=categories)
        nc = np.bincount(negative[name].astype(int), minlength=categories)
        pp = (pc + smoothing) / (len(positive[name]) + smoothing * categories)
        np_ = (nc + smoothing) / (len(negative[name]) + smoothing * categories)
        tables[name] = {
            "kind": "categorical", "llr": np.clip(np.log(pp / np_), -6, 6).tolist(),
        }
    return tables


def fit_nonnegative_meta(
    positive: Mapping[str, np.ndarray], negative: Mapping[str, np.ndarray],
    tables: Mapping[str, Mapping[str, object]], *, seed: int = 20260807,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    npos = min(len(positive["dr"]), 300_000)
    nneg = min(len(negative["dr"]), 450_000)
    pi = rng.choice(len(positive["dr"]), npos, replace=False)
    ni = rng.choice(len(negative["dr"]), nneg, replace=False)
    xp = np.column_stack([table_score(positive[name][pi], tables[name]) for name in EMPIRICAL_FEATURES])
    xn = np.column_stack([table_score(negative[name][ni], tables[name]) for name in EMPIRICAL_FEATURES])
    x = np.vstack((xp, xn)); y = np.r_[np.ones(npos), np.zeros(nneg)]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = parameters[0] + x @ parameters[1:]
        loss = np.mean(np.logaddexp(0, logits) - y * logits) + .005 * np.sum(parameters[1:] ** 2)
        residual = sigmoid(logits) - y
        gradient = np.r_[np.mean(residual), x.T @ residual / len(y) + .01 * parameters[1:]]
        return float(loss), gradient

    result = minimize(
        objective, np.r_[0.0, np.ones(len(EMPIRICAL_FEATURES))], jac=True,
        method="L-BFGS-B", bounds=[(-12, 12), *[(0, 3)] * len(EMPIRICAL_FEATURES)],
        options={"maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"empirical meta fit failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "weights": dict(zip(EMPIRICAL_FEATURES, map(float, result.x[1:]), strict=True)),
    }


__all__ = [
    "EMPIRICAL_FEATURES", "EmpiricalScorer", "fit_nonnegative_meta", "fit_tables",
    "geometry_response_score", "geometry_score", "sigmoid", "table_score",
]
