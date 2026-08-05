"""Fold-local seed bootstrap, contextual edge training, and matcher inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np

from hlt_classification.data.cache_contracts import atomic_publish_bytes, sha256_file, validate_content_hash, with_content_hash, write_immutable_json
from .match_model import build_contextual_edge_matcher
from .matching import (
    ASSIGNMENT_DIAGNOSTIC_DIM, MATCH_EDGE_FEATURE_DIM, MATCH_NODE_FEATURE_DIM,
    CandidateGraph, MatchResult, assignment_diagnostics, hungarian_with_dustbins,
    optimal_transport_with_dustbins,
)

MATCHER_REPORT_CONTRACT = "hlt_classification_pmard_matcher_report_v2"
MATCHER_REPORT_VERSION = 2


@dataclass(frozen=True)
class MatcherTrainingConfig:
    feature_dim: int = MATCH_EDGE_FEATURE_DIM
    node_dim: int = MATCH_NODE_FEATURE_DIM
    hidden_dim: int = 64
    message_passing_rounds: int = 3
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 5
    seed: int = 1337
    positive_dr_charged: float = .005
    positive_dr_neutral: float = .010
    positive_log_response: float = .10
    calibration_fraction: float = .20
    pseudo_label_rounds: int = 2
    pseudo_label_probability: float = .999

    def __post_init__(self) -> None:
        if self.feature_dim != MATCH_EDGE_FEATURE_DIM or self.node_dim != MATCH_NODE_FEATURE_DIM:
            raise ValueError("matcher feature dimensions differ from the v2 graph contract")
        if self.message_passing_rounds not in {2, 3} or self.pseudo_label_rounds != 2:
            raise ValueError("matcher rounds differ from the locked v2 design")
        if not .99 <= self.pseudo_label_probability < 1:
            raise ValueError("pseudo-label probability is not ultra-conservative")


def bootstrap_edge_labels(graph: CandidateGraph) -> np.ndarray:
    """Return 1 for pure seeds, 0 for definite negatives, and -1 for unlabeled edges."""
    labels = np.full(len(graph.hlt_index), -1, np.float32)
    if not len(labels):
        return labels
    distance = graph.features[:, 0]
    for row, (i, j) in enumerate(zip(graph.hlt_index, graph.offline_index, strict=True)):
        h_rows = np.flatnonzero(graph.hlt_index == i); o_rows = np.flatnonzero(graph.offline_index == j)
        mutual = row == h_rows[np.argmin(distance[h_rows])] and row == o_rows[np.argmin(distance[o_rows])]
        response = max(abs(float(graph.features[row, 3])), abs(float(graph.features[row, 4])))
        category = int(np.argmax(graph.features[row, -5:]))
        dr_threshold = .005 if category < 3 else .010
        if mutual and distance[row] < dr_threshold and response < .10:
            labels[row] = 1
        elif (distance[row] > (0.075 if category < 3 else 0.11)
              or response > 1.0):
            labels[row] = 0
    return labels


def train_contextual_matcher(
    graphs: Iterable[CandidateGraph | tuple[CandidateGraph, np.ndarray]], *, config: MatcherTrainingConfig,
    output_dir: str | Path, parents: dict[str, str], device: str = "cuda",
    sampling_config: dict[str, object] | None = None,
) -> dict[str, object]:
    import torch
    torch.manual_seed(config.seed); target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA matcher training requested but unavailable")
    feature_blocks: list[np.ndarray] = []; label_blocks: list[np.ndarray] = []
    hlt_node_blocks: list[np.ndarray] = []; offline_node_blocks: list[np.ndarray] = []
    hlt_index_blocks: list[np.ndarray] = []; offline_index_blocks: list[np.ndarray] = []
    pending_features: list[np.ndarray] = []; pending_labels: list[np.ndarray] = []
    pending_hlt: list[np.ndarray] = []; pending_offline: list[np.ndarray] = []
    edge_starts: list[int] = []; edge_counts: list[int] = []
    graph_records: list[CandidateGraph] = []
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
        if np.any(~np.isin(labels, (-1.0, 0.0, 1.0))):
            raise ValueError("matcher supervision must use positive/negative/unlabeled values")
        graph_records.append(graph)
        edge_starts.append(edge_cursor); edge_counts.append(len(labels)); edge_cursor += len(labels)
        positive_count += int(np.count_nonzero(labels == 1))
        hlt_node_blocks.append(graph.hlt_node_features)
        offline_node_blocks.append(graph.offline_node_features)
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
    all_hlt_nodes = np.concatenate(hlt_node_blocks); all_offline_nodes = np.concatenate(offline_node_blocks)
    edge_starts_array = np.asarray(edge_starts, np.int64); edge_counts_array = np.asarray(edge_counts, np.int64)
    def edge_range(index: int) -> np.ndarray:
        return np.arange(edge_starts_array[index], edge_starts_array[index] + edge_counts_array[index])
    model = build_contextual_edge_matcher(
        feature_dim=config.feature_dim, node_dim=config.node_dim,
        hidden_dim=config.hidden_dim, message_passing_rounds=config.message_passing_rounds,
    ).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    positives = int(np.count_nonzero(all_labels == 1)); negatives = int(np.count_nonzero(all_labels == 0))
    unlabeled = int(np.count_nonzero(all_labels < 0))
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
    for name, indexes in (("fit", fit_indexes), ("calibration", calibration_indexes)):
        if not np.any(all_labels[indexes] == 1) or not np.any(all_labels[indexes] == 0):
            raise ValueError(f"matcher {name} split lacks known positive/negative supervision")
    feature_mean = all_features[fit_indexes].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = all_features[fit_indexes].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.where(feature_scale > 1.0e-6, feature_scale, 1.0).astype(np.float32)
    likelihood = torch.nn.Linear(config.feature_dim, 1).to(target)
    optimizer.add_param_group({"params": likelihood.parameters()})
    def tensors_for(indexes: np.ndarray):
        x = torch.as_tensor(
            (all_features[indexes] - feature_mean) / feature_scale, device=target,
        )
        raw_h = torch.as_tensor(all_hlt_index[indexes], device=target)
        raw_o = torch.as_tensor(all_offline_index[indexes], device=target)
        unique_h, h_index = torch.unique(raw_h, sorted=True, return_inverse=True)
        unique_o, o_index = torch.unique(raw_o, sorted=True, return_inverse=True)
        h_nodes = torch.as_tensor(all_hlt_nodes[unique_h.cpu().numpy()], device=target)
        o_nodes = torch.as_tensor(all_offline_nodes[unique_o.cpu().numpy()], device=target)
        return x, h_index, o_index, h_nodes, o_nodes

    def inference_scores(graph_indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score many disjoint sparse graphs without one GPU launch per jet."""
        contextual_output = np.full(len(all_labels), np.nan, np.float32)
        likelihood_output = np.full(len(all_labels), np.nan, np.float32)
        batches: list[np.ndarray] = []; current: list[np.ndarray] = []; current_edges = 0
        for graph_index in graph_indexes:
            rows = edge_range(int(graph_index))
            if current and current_edges + len(rows) > 65_536:
                batches.append(np.concatenate(current)); current = []; current_edges = 0
            current.append(rows); current_edges += len(rows)
        if current: batches.append(np.concatenate(current))
        model.eval(); likelihood.eval()
        with torch.inference_mode():
            for indexes in batches:
                x, h_index, o_index, h_nodes, o_nodes = tensors_for(indexes)
                contextual_output[indexes] = model(
                    x, h_index, o_index, h_nodes, o_nodes,
                ).float().cpu().numpy()
                likelihood_output[indexes] = likelihood(x).squeeze(-1).float().cpu().numpy()
        return contextual_output, likelihood_output

    pseudo_label_counts: list[int] = []
    for pseudo_round in range(config.pseudo_label_rounds + 1):
        positive_weight = torch.tensor(
            max(1.0, np.count_nonzero(all_labels == 0) / max(1, np.count_nonzero(all_labels == 1))),
            device=target,
        )
        round_epochs = config.epochs if pseudo_round == 0 else max(1, config.epochs // 2)
        model.train()
        for _ in range(round_epochs):
            graph_order = generator.permutation(fit_graphs)
            batches: list[np.ndarray] = []; current: list[np.ndarray] = []; current_edges = 0
            for graph_index in graph_order:
                rows = edge_range(int(graph_index))
                if current and current_edges + len(rows) > 65_536:
                    batches.append(np.concatenate(current)); current = []; current_edges = 0
                current.append(rows); current_edges += len(rows)
            if current: batches.append(np.concatenate(current))
            for indexes in batches:
                x, h_index, o_index, h_nodes, o_nodes = tensors_for(indexes)
                y = torch.as_tensor(all_labels[indexes], device=target)
                known = y >= 0
                if not known.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, h_index, o_index, h_nodes, o_nodes)
                likelihood_logits = likelihood(x).squeeze(-1)
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[known], y[known], pos_weight=positive_weight,
                    )
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        likelihood_logits[known], y[known], pos_weight=positive_weight,
                    )
                )
                if not torch.isfinite(loss): raise FloatingPointError("matcher loss is nonfinite")
                loss.backward(); optimizer.step()
        if pseudo_round == config.pseudo_label_rounds:
            break
        additions = 0
        fit_contextual, _ = inference_scores(fit_graphs)
        for graph_index in fit_graphs:
            indexes = edge_range(int(graph_index)); graph = graph_records[int(graph_index)]
            raw = fit_contextual[indexes]
            hungarian = hungarian_with_dustbins(graph, raw)
            transport = optimal_transport_with_dustbins(graph, raw)
            for hlt_index in np.flatnonzero(
                (hungarian.hlt_to_offline >= 0)
                & (hungarian.hlt_to_offline == transport.hlt_to_offline)
            ):
                offline_index = int(hungarian.hlt_to_offline[hlt_index])
                local = np.flatnonzero(
                    (graph.hlt_index == hlt_index) & (graph.offline_index == offline_index)
                )
                if len(local) != 1:
                    continue
                edge = int(indexes[local[0]])
                probability = 1.0 / (1.0 + np.exp(-np.clip(raw[local[0]], -80, 80)))
                mutually_first = graph.features[local[0], 5] == 0 and graph.features[local[0], 6] == 0
                if all_labels[edge] < 0 and probability >= config.pseudo_label_probability and mutually_first:
                    all_labels[edge] = 1; additions += 1
        pseudo_label_counts.append(additions)
    calibration_contextual, calibration_likelihood = inference_scores(calibration_graphs)
    raw_calibration = calibration_contextual[calibration_indexes]
    raw_likelihood_calibration = calibration_likelihood[calibration_indexes]
    calibration_known = all_labels[calibration_indexes] >= 0
    calibration_count = int(np.count_nonzero(calibration_known))
    raw_calibration = raw_calibration[calibration_known]
    raw_likelihood_calibration = raw_likelihood_calibration[calibration_known]
    calibration_labels = all_labels[calibration_indexes][calibration_known].astype(np.float64)
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
    assignment_features: list[np.ndarray] = []
    assignment_labels: list[float] = []
    with torch.inference_mode():
        for graph_index in calibration_graphs:
            indexes = edge_range(int(graph_index)); graph = graph_records[int(graph_index)]
            raw = calibration_contextual[indexes]
            edge_logits = calibration_slope * raw + calibration_intercept
            hungarian = hungarian_with_dustbins(graph, edge_logits)
            transport = optimal_transport_with_dustbins(graph, edge_logits)
            diagnostics = assignment_diagnostics(
                graph, hungarian, edge_logits, independent=transport,
            )
            for hlt_index in np.flatnonzero(hungarian.hlt_to_offline >= 0):
                offline_index = int(hungarian.hlt_to_offline[hlt_index])
                local = np.flatnonzero(
                    (graph.hlt_index == hlt_index) & (graph.offline_index == offline_index)
                )
                if len(local) != 1:
                    continue
                truth = all_labels[indexes[local[0]]]
                if truth < 0:
                    continue
                assignment_features.append(diagnostics[hlt_index])
                assignment_labels.append(float(truth == 1))
    assignment_x = np.asarray(assignment_features, np.float64)
    assignment_y = np.asarray(assignment_labels, np.float64)
    if not np.any(assignment_y == 0):
        # A clean holdout can yield no naturally selected mistake. Add known
        # false candidate assignments as explicit calibration negatives rather
        # than weakening the post-assignment target or fabricating confidence.
        for graph_index in calibration_graphs:
            indexes = edge_range(int(graph_index)); graph = graph_records[int(graph_index)]
            known_false = np.flatnonzero(all_labels[indexes] == 0)
            for local in known_false[:max(1, 64 - len(assignment_labels))]:
                hlt_index = int(graph.hlt_index[local]); offline_index = int(graph.offline_index[local])
                assignment = np.full(graph.hlt_count, -1, np.int16); assignment[hlt_index] = offline_index
                synthetic_result = MatchResult(
                    assignment, np.zeros(graph.hlt_count, np.float32),
                    np.zeros(graph.hlt_count, np.float32), assignment >= 0,
                    "known_false_calibration_control",
                )
                diagnostics = assignment_diagnostics(graph, synthetic_result, graph.manual_scores)
                assignment_features.append(diagnostics[hlt_index]); assignment_labels.append(0.0)
            if any(value == 0 for value in assignment_labels):
                break
        assignment_x = np.asarray(assignment_features, np.float64)
        assignment_y = np.asarray(assignment_labels, np.float64)
    if (assignment_x.ndim != 2 or assignment_x.shape[1] != ASSIGNMENT_DIAGNOSTIC_DIM
            or not np.any(assignment_y == 0) or not np.any(assignment_y == 1)):
        raise ValueError("held-out global assignments cannot fit correctness calibration")
    assignment_mean = assignment_x.mean(axis=0)
    assignment_scale = assignment_x.std(axis=0)
    assignment_scale = np.where(assignment_scale > 1.0e-6, assignment_scale, 1.0)
    standardized_assignment = (assignment_x - assignment_mean) / assignment_scale
    def assignment_objective(parameters):
        logits = standardized_assignment @ parameters[:-1] + parameters[-1]
        return float(
            np.mean(np.logaddexp(0, logits) - assignment_y * logits)
            + 1.0e-4 * np.square(parameters[:-1]).mean()
        )
    assignment_fit = minimize(
        assignment_objective, np.zeros(ASSIGNMENT_DIAGNOSTIC_DIM + 1), method="BFGS",
    )
    if not assignment_fit.success or not np.isfinite(assignment_fit.x).all():
        raise RuntimeError("post-assignment correctness calibration failed")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    stream = BytesIO(); torch.save(model.state_dict(), stream)
    weights_path = root / "matcher_weights.pt"; atomic_publish_bytes(weights_path, stream.getvalue())
    report = with_content_hash({
        "contract": MATCHER_REPORT_CONTRACT, "schema_version": MATCHER_REPORT_VERSION,
        "config": asdict(config), "parents": dict(parents),
        "bootstrap_positive_edges": positives, "bootstrap_negative_edges": negatives,
        "unlabeled_plausible_edges": unlabeled,
        "pseudo_label_positive_additions_by_round": pseudo_label_counts,
        "supervision_policy": "positive_unlabeled_with_definite_and_event_mixed_negatives_v2",
        "architecture": "three_round_sparse_bipartite_message_passing_v2",
        "sampling_config": dict(sampling_config or {"mode": "caller_supplied_graphs"}),
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
        "assignment_calibration": {
            "method": "post_assignment_logistic_v1",
            "rows": len(assignment_y),
            "mean": assignment_mean.tolist(), "scale": assignment_scale.tolist(),
            "weight": assignment_fit.x[:-1].tolist(),
            "intercept": float(assignment_fit.x[-1]),
            "target": "global_assignment_correctness",
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
    assignment_calibration: dict[str, object]
    device: str

    def to(self, device: str):
        self.model.to(device)
        return LoadedContextualMatcher(
            self.model, self.calibration_slope, self.calibration_intercept,
            self.feature_mean, self.feature_scale, self.likelihood_weight,
            self.likelihood_bias, self.likelihood_calibration_slope,
            self.likelihood_calibration_intercept, self.assignment_calibration, device,
        )


def load_contextual_matcher(report: dict[str, object], root: str | Path, *, device: str = "cpu"):
    import torch
    if report.get("contract") != MATCHER_REPORT_CONTRACT:
        raise ValueError("matcher report contract differs")
    validate_content_hash(
        report, expected_contract=MATCHER_REPORT_CONTRACT,
        expected_schema_version=MATCHER_REPORT_VERSION,
    )
    config = MatcherTrainingConfig(**report["config"])
    path = Path(root) / str(report["weights_file"])
    if sha256_file(path) != report["weights_sha256"]:
        raise ValueError("matcher weights hash differs")
    model = build_contextual_edge_matcher(
        feature_dim=config.feature_dim, node_dim=config.node_dim,
        hidden_dim=config.hidden_dim, message_passing_rounds=config.message_passing_rounds,
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)); model.eval()
    calibration = report.get("calibration", {})
    if calibration.get("method") != "heldout_platt_log_odds_v1":
        raise ValueError("matcher calibration contract differs")
    standardization = report.get("feature_standardization", {})
    feature_mean = np.asarray(standardization.get("mean"), np.float32)
    feature_scale = np.asarray(standardization.get("scale"), np.float32)
    likelihood = report.get("physics_likelihood", {})
    assignment_calibration = report.get("assignment_calibration", {})
    likelihood_weight = np.asarray(likelihood.get("weight"), np.float32)
    if (feature_mean.shape != (config.feature_dim,) or feature_scale.shape != (config.feature_dim,)
            or likelihood_weight.shape != (config.feature_dim,) or np.any(feature_scale <= 0)
            or likelihood.get("method") != "standardized_linear_log_likelihood_ratio_v1"
            or assignment_calibration.get("method") != "post_assignment_logistic_v1"):
        raise ValueError("matcher score-normalization/likelihood contract differs")
    return LoadedContextualMatcher(
        model.to(device), float(calibration["slope"]), float(calibration["intercept"]),
        feature_mean, feature_scale, likelihood_weight, float(likelihood["bias"]),
        float(likelihood["calibration_slope"]), float(likelihood["calibration_intercept"]),
        dict(assignment_calibration), device,
    )


def contextual_scores(model, graph: CandidateGraph, *, device: str = "cpu") -> np.ndarray:
    return contextual_scores_many(model, (graph,), device=device)[0]


def contextual_scores_many(
    model, graphs: Iterable[CandidateGraph], *, device: str = "cpu",
    maximum_edges_per_forward: int = 65_536,
) -> list[np.ndarray]:
    """Contextually score sparse jets in bounded multi-graph GPU forwards."""
    import torch
    materialized = list(graphs)
    if maximum_edges_per_forward <= 0:
        raise ValueError("matcher inference edge budget must be positive")
    active = model.model if isinstance(model, LoadedContextualMatcher) else model
    target_device = model.device if isinstance(model, LoadedContextualMatcher) else device
    result: list[np.ndarray | None] = [None] * len(materialized)
    pending: list[int] = []; pending_edges = 0

    def score_pending() -> None:
        nonlocal pending_edges
        if not pending: return
        features = []; hlt_index = []; offline_index = []; hlt_nodes = []; offline_nodes = []
        hlt_offset = offline_offset = 0
        for graph_index in pending:
            graph = materialized[graph_index]
            block = graph.features
            if isinstance(model, LoadedContextualMatcher):
                block = (block - model.feature_mean) / model.feature_scale
            features.append(block)
            hlt_index.append(graph.hlt_index.astype(np.int64) + hlt_offset)
            offline_index.append(graph.offline_index.astype(np.int64) + offline_offset)
            hlt_nodes.append(graph.hlt_node_features); offline_nodes.append(graph.offline_node_features)
            hlt_offset += graph.hlt_count; offline_offset += graph.offline_count
        with torch.inference_mode():
            values = active(
                torch.as_tensor(np.concatenate(features), device=target_device),
                torch.as_tensor(np.concatenate(hlt_index), dtype=torch.long, device=target_device),
                torch.as_tensor(np.concatenate(offline_index), dtype=torch.long, device=target_device),
                torch.as_tensor(np.concatenate(hlt_nodes), device=target_device),
                torch.as_tensor(np.concatenate(offline_nodes), device=target_device),
            ).float().cpu().numpy()
        if isinstance(model, LoadedContextualMatcher):
            values = model.calibration_slope * values + model.calibration_intercept
        cursor = 0
        for graph_index in pending:
            edges = len(materialized[graph_index].hlt_index)
            block = values[cursor:cursor + edges].astype(np.float64)
            if not np.isfinite(block).all():
                raise FloatingPointError("matcher inference is nonfinite")
            result[graph_index] = block; cursor += edges
        pending.clear(); pending_edges = 0

    for index, graph in enumerate(materialized):
        edges = len(graph.hlt_index)
        if not edges:
            result[index] = np.empty(0, np.float64); continue
        if pending and pending_edges + edges > maximum_edges_per_forward:
            score_pending()
        pending.append(index); pending_edges += edges
        if pending_edges >= maximum_edges_per_forward:
            score_pending()
    score_pending()
    if any(value is None for value in result):
        raise RuntimeError("matcher multi-graph inference did not fill every graph")
    return [value for value in result if value is not None]


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
    "contextual_scores_many",
    "likelihood_scores", "load_contextual_matcher", "train_contextual_matcher",
]
