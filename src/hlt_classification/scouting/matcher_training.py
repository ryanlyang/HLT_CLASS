"""Fold-local seed bootstrap, contextual edge training, and matcher inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np

from hlt_classification.data.cache_contracts import atomic_publish_bytes, sha256_file, validate_content_hash, with_content_hash, write_immutable_json
from .match_model import build_contextual_edge_matcher
from .matching import CandidateGraph

MATCHER_REPORT_CONTRACT = "hlt_classification_pmard_matcher_report_v1"


@dataclass(frozen=True)
class MatcherTrainingConfig:
    feature_dim: int = 13
    hidden_dim: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 5
    seed: int = 1337
    positive_dr_charged: float = .005
    positive_dr_neutral: float = .010
    positive_log_response: float = .10
    calibration_fraction: float = .20


def bootstrap_edge_labels(graph: CandidateGraph) -> np.ndarray:
    """Ultra-tight mutual-nearest positives; other plausible edges are negatives."""
    labels = np.zeros(len(graph.hlt_index), np.float32)
    if not len(labels):
        return labels
    distance = graph.features[:, 0]
    for row, (i, j) in enumerate(zip(graph.hlt_index, graph.offline_index, strict=True)):
        h_rows = np.flatnonzero(graph.hlt_index == i); o_rows = np.flatnonzero(graph.offline_index == j)
        mutual = row == h_rows[np.argmin(distance[h_rows])] and row == o_rows[np.argmin(distance[o_rows])]
        response = max(abs(float(graph.features[row, 3])), abs(float(graph.features[row, 4])))
        category = int(np.argmax(graph.features[row, 8:13]))
        dr_threshold = .005 if category < 3 else .010
        if mutual and distance[row] < dr_threshold and response < .10:
            labels[row] = 1
    return labels


def train_contextual_matcher(
    graphs: Iterable[CandidateGraph | tuple[CandidateGraph, np.ndarray]], *, config: MatcherTrainingConfig,
    output_dir: str | Path, parents: dict[str, str], device: str = "cuda",
) -> dict[str, object]:
    import torch
    torch.manual_seed(config.seed); target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA matcher training requested but unavailable")
    feature_blocks: list[np.ndarray] = []; label_blocks: list[np.ndarray] = []
    hlt_index_blocks: list[np.ndarray] = []; offline_index_blocks: list[np.ndarray] = []
    pending_features: list[np.ndarray] = []; pending_labels: list[np.ndarray] = []
    pending_hlt: list[np.ndarray] = []; pending_offline: list[np.ndarray] = []
    edge_starts: list[int] = []; edge_counts: list[int] = []
    edge_cursor = hlt_cursor = offline_cursor = pending_edges = 0
    positive_count = 0
    def flush() -> None:
        nonlocal pending_edges
        if not pending_features: return
        feature_blocks.append(np.concatenate(pending_features)); label_blocks.append(np.concatenate(pending_labels))
        hlt_index_blocks.append(np.concatenate(pending_hlt)); offline_index_blocks.append(np.concatenate(pending_offline))
        pending_features.clear(); pending_labels.clear(); pending_hlt.clear(); pending_offline.clear(); pending_edges = 0
    for item in graphs:
        graph, supplied = item if isinstance(item, tuple) else (item, None)
        if not len(graph.hlt_index): continue
        labels = bootstrap_edge_labels(graph) if supplied is None else np.asarray(supplied, np.float32)
        if labels.shape != (len(graph.hlt_index),):
            raise ValueError("supplied matcher supervision shape differs")
        edge_starts.append(edge_cursor); edge_counts.append(len(labels)); edge_cursor += len(labels)
        positive_count += int(labels.sum())
        pending_features.append(graph.features); pending_labels.append(labels)
        pending_hlt.append(graph.hlt_index.astype(np.int64) + hlt_cursor)
        pending_offline.append(graph.offline_index.astype(np.int64) + offline_cursor)
        hlt_cursor += graph.hlt_count; offline_cursor += graph.offline_count
        pending_edges += len(labels)
        if pending_edges >= 1_000_000: flush()
    flush()
    if not edge_counts or not positive_count:
        raise ValueError("matcher bootstrap produced no positive seeds")
    all_features = np.concatenate(feature_blocks); all_labels = np.concatenate(label_blocks)
    all_hlt_index = np.concatenate(hlt_index_blocks); all_offline_index = np.concatenate(offline_index_blocks)
    edge_starts_array = np.asarray(edge_starts, np.int64); edge_counts_array = np.asarray(edge_counts, np.int64)
    def edge_range(index: int) -> np.ndarray:
        return np.arange(edge_starts_array[index], edge_starts_array[index] + edge_counts_array[index])
    model = build_contextual_edge_matcher(feature_dim=config.feature_dim, hidden_dim=config.hidden_dim).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    positives = int(all_labels.sum()); negatives = int(len(all_labels) - positives)
    positive_weight = torch.tensor(max(1.0, negatives / max(1, positives)), device=target)
    model.train()
    generator = np.random.default_rng(config.seed)
    graph_permutation = generator.permutation(len(edge_counts_array))
    calibration_graph_count = max(1, int(len(graph_permutation) * config.calibration_fraction))
    calibration_graphs = graph_permutation[:calibration_graph_count]
    fit_graphs = graph_permutation[calibration_graph_count:]
    calibration_indexes = np.concatenate([edge_range(int(index)) for index in calibration_graphs])
    fit_indexes = np.concatenate([edge_range(int(index)) for index in fit_graphs])
    calibration_count = len(calibration_indexes)
    if not np.any(all_labels[calibration_indexes]) or not np.any(all_labels[fit_indexes]):
        raise ValueError("matcher fit/calibration split lacks positive seeds")
    feature_mean = all_features[fit_indexes].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = all_features[fit_indexes].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.where(feature_scale > 1.0e-6, feature_scale, 1.0).astype(np.float32)
    likelihood = torch.nn.Linear(config.feature_dim, 1).to(target)
    optimizer.add_param_group({"params": likelihood.parameters()})
    for _ in range(config.epochs):
        graph_order = generator.permutation(fit_graphs)
        batches: list[np.ndarray] = []; current: list[np.ndarray] = []; current_edges = 0
        for graph_index in graph_order:
            rows = edge_range(int(graph_index))
            if current and current_edges + len(rows) > 65_536:
                batches.append(np.concatenate(current)); current = []; current_edges = 0
            current.append(rows); current_edges += len(rows)
        if current: batches.append(np.concatenate(current))
        for indexes in batches:
            x = torch.as_tensor(
                (all_features[indexes] - feature_mean) / feature_scale, device=target,
            )
            y = torch.as_tensor(all_labels[indexes], device=target)
            h_index = torch.as_tensor(all_hlt_index[indexes], device=target)
            o_index = torch.as_tensor(all_offline_index[indexes], device=target)
            _, h_index = torch.unique(h_index, sorted=True, return_inverse=True)
            _, o_index = torch.unique(o_index, sorted=True, return_inverse=True)
            optimizer.zero_grad(set_to_none=True); logits = model(x, h_index, o_index)
            likelihood_logits = likelihood(x).squeeze(-1)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=positive_weight)
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    likelihood_logits, y, pos_weight=positive_weight,
                )
            )
            if not torch.isfinite(loss): raise FloatingPointError("matcher loss is nonfinite")
            loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        calibration_h = torch.as_tensor(all_hlt_index[calibration_indexes], device=target)
        calibration_o = torch.as_tensor(all_offline_index[calibration_indexes], device=target)
        _, calibration_h = torch.unique(calibration_h, sorted=True, return_inverse=True)
        _, calibration_o = torch.unique(calibration_o, sorted=True, return_inverse=True)
        calibration_features = torch.as_tensor(
            (all_features[calibration_indexes] - feature_mean) / feature_scale,
            device=target,
        )
        raw_calibration = model(
            calibration_features,
            calibration_h, calibration_o,
        ).float().cpu().numpy()
        raw_likelihood_calibration = likelihood(calibration_features).squeeze(-1).float().cpu().numpy()
    calibration_labels = all_labels[calibration_indexes].astype(np.float64)
    from scipy.optimize import minimize
    def objective(parameters, raw_scores):
        slope = np.exp(parameters[0]); logits = slope * raw_scores + parameters[1]
        return float(np.mean(np.logaddexp(0, logits) - calibration_labels * logits))
    fit = minimize(lambda value: objective(value, raw_calibration), np.array([0.0, 0.0]), method="BFGS")
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError("matcher probability calibration failed")
    calibration_slope, calibration_intercept = float(np.exp(fit.x[0])), float(fit.x[1])
    likelihood_fit = minimize(
        lambda value: objective(value, raw_likelihood_calibration),
        np.array([0.0, 0.0]), method="BFGS",
    )
    if not likelihood_fit.success or not np.isfinite(likelihood_fit.x).all():
        raise RuntimeError("physics likelihood probability calibration failed")
    likelihood_calibration_slope = float(np.exp(likelihood_fit.x[0]))
    likelihood_calibration_intercept = float(likelihood_fit.x[1])
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    stream = BytesIO(); torch.save(model.state_dict(), stream)
    weights_path = root / "matcher_weights.pt"; atomic_publish_bytes(weights_path, stream.getvalue())
    report = with_content_hash({
        "contract": MATCHER_REPORT_CONTRACT, "schema_version": 1,
        "config": asdict(config), "parents": dict(parents),
        "bootstrap_positive_edges": positives, "bootstrap_negative_edges": negatives,
        "calibration": {
            "method": "heldout_platt_log_odds_v1", "rows": calibration_count,
            "slope": calibration_slope, "intercept": calibration_intercept,
        },
        "feature_standardization": {
            "mean": feature_mean.tolist(), "scale": feature_scale.tolist(),
        },
        "physics_likelihood": {
            "method": "standardized_linear_log_likelihood_ratio_v1",
            "weight": likelihood.weight.detach().float().cpu().numpy().reshape(-1).tolist(),
            "bias": float(likelihood.bias.detach().float().cpu().item()),
            "calibration_slope": likelihood_calibration_slope,
            "calibration_intercept": likelihood_calibration_intercept,
        },
        "weights_file": weights_path.name, "weights_sha256": sha256_file(weights_path),
        "downstream_labels_used": False, "complete": True,
    })
    write_immutable_json(root / "matcher_report.json", report); return report


@dataclass(frozen=True)
class LoadedContextualMatcher:
    model: object
    calibration_slope: float
    calibration_intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    likelihood_weight: np.ndarray
    likelihood_bias: float
    likelihood_calibration_slope: float
    likelihood_calibration_intercept: float
    device: str

    def to(self, device: str):
        self.model.to(device)
        return LoadedContextualMatcher(
            self.model, self.calibration_slope, self.calibration_intercept,
            self.feature_mean, self.feature_scale, self.likelihood_weight,
            self.likelihood_bias, self.likelihood_calibration_slope,
            self.likelihood_calibration_intercept, device,
        )


def load_contextual_matcher(report: dict[str, object], root: str | Path, *, device: str = "cpu"):
    import torch
    if report.get("contract") != MATCHER_REPORT_CONTRACT:
        raise ValueError("matcher report contract differs")
    validate_content_hash(report, expected_contract=MATCHER_REPORT_CONTRACT)
    config = MatcherTrainingConfig(**report["config"])
    path = Path(root) / str(report["weights_file"])
    if sha256_file(path) != report["weights_sha256"]:
        raise ValueError("matcher weights hash differs")
    model = build_contextual_edge_matcher(feature_dim=config.feature_dim, hidden_dim=config.hidden_dim)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)); model.eval()
    calibration = report.get("calibration", {})
    if calibration.get("method") != "heldout_platt_log_odds_v1":
        raise ValueError("matcher calibration contract differs")
    standardization = report.get("feature_standardization", {})
    feature_mean = np.asarray(standardization.get("mean"), np.float32)
    feature_scale = np.asarray(standardization.get("scale"), np.float32)
    likelihood = report.get("physics_likelihood", {})
    likelihood_weight = np.asarray(likelihood.get("weight"), np.float32)
    if (feature_mean.shape != (config.feature_dim,) or feature_scale.shape != (config.feature_dim,)
            or likelihood_weight.shape != (config.feature_dim,) or np.any(feature_scale <= 0)
            or likelihood.get("method") != "standardized_linear_log_likelihood_ratio_v1"):
        raise ValueError("matcher score-normalization/likelihood contract differs")
    return LoadedContextualMatcher(
        model.to(device), float(calibration["slope"]), float(calibration["intercept"]),
        feature_mean, feature_scale, likelihood_weight, float(likelihood["bias"]),
        float(likelihood["calibration_slope"]), float(likelihood["calibration_intercept"]), device,
    )


def contextual_scores(model, graph: CandidateGraph, *, device: str = "cpu") -> np.ndarray:
    import torch
    if not len(graph.hlt_index): return np.empty(0, np.float64)
    active = model.model if isinstance(model, LoadedContextualMatcher) else model
    target_device = model.device if isinstance(model, LoadedContextualMatcher) else device
    features = graph.features
    if isinstance(model, LoadedContextualMatcher):
        features = (features - model.feature_mean) / model.feature_scale
    with torch.inference_mode():
        value = active(
            torch.as_tensor(features, device=target_device),
            torch.as_tensor(graph.hlt_index, dtype=torch.long, device=target_device),
            torch.as_tensor(graph.offline_index, dtype=torch.long, device=target_device),
        ).float().cpu().numpy()
    if isinstance(model, LoadedContextualMatcher):
        value = model.calibration_slope * value + model.calibration_intercept
    if not np.isfinite(value).all(): raise FloatingPointError("matcher inference is nonfinite")
    return value.astype(np.float64)


def likelihood_scores(model: LoadedContextualMatcher, graph: CandidateGraph) -> np.ndarray:
    """Return the separately fitted and calibrated M2/M3 physics LLR."""
    if not isinstance(model, LoadedContextualMatcher):
        raise TypeError("fitted likelihood scores require a loaded matcher report")
    if not len(graph.hlt_index): return np.empty(0, np.float64)
    features = (graph.features - model.feature_mean) / model.feature_scale
    raw = features @ model.likelihood_weight + model.likelihood_bias
    value = model.likelihood_calibration_slope * raw + model.likelihood_calibration_intercept
    if not np.isfinite(value).all(): raise FloatingPointError("physics likelihood inference is nonfinite")
    return np.asarray(value, np.float64)


__all__ = [
    "LoadedContextualMatcher", "MatcherTrainingConfig", "bootstrap_edge_labels", "contextual_scores",
    "likelihood_scores", "load_contextual_matcher", "train_contextual_matcher",
]
