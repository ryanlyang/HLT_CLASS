"""Physics-informed sparse constituent matching with abstaining global solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCHER_VARIANTS: Final = ("M0", "M1", "M2", "M3", "M4", "M5")
MATCHER_OPERATING_POINTS: Final = {
    "ultra_pure": 0.99, "high_purity": 0.95, "high_coverage": 0.80,
}
CATEGORY_NAMES: Final = ("electron", "muon", "charged_hadron", "photon", "neutral_hadron")
CHARGED_CATEGORIES: Final = frozenset((0, 1, 2))
TRACK_FIELDS: Final = ("dxy", "dxysig", "dz", "dzsig", "normchi2", "quality", "lostInnerHits")
MATCH_NODE_FEATURE_DIM: Final = 28
MATCH_EDGE_FEATURE_DIM: Final = 31
ASSIGNMENT_DIAGNOSTIC_DIM: Final = 13
SCORE_QUANTUM: Final = 1.0e-5


@dataclass(frozen=True)
class ParticleSet:
    p4: np.ndarray
    categories: np.ndarray
    charge: np.ndarray
    lost_track: np.ndarray
    measurements: np.ndarray | None = None
    measurement_validity: np.ndarray | None = None

    def __post_init__(self) -> None:
        p4 = np.asarray(self.p4)
        count = len(p4)
        if p4.shape != (count, 4) or not np.isfinite(p4).all():
            raise ValueError("particle p4 must be finite [particles,4]")
        if np.asarray(self.categories).shape != (count,):
            raise ValueError("particle categories shape differs")
        if np.asarray(self.charge).shape != (count,) or np.asarray(self.lost_track).shape != (count,):
            raise ValueError("particle charge/lost-track shape differs")
        measurements = (
            np.zeros((count, len(TRACK_FIELDS)), np.float64)
            if self.measurements is None else np.asarray(self.measurements, np.float64)
        )
        validity = (
            np.zeros_like(measurements, np.bool_)
            if self.measurement_validity is None else np.asarray(self.measurement_validity, np.bool_)
        )
        if measurements.shape != (count, len(TRACK_FIELDS)) or validity.shape != measurements.shape:
            raise ValueError("particle measurement/value validity shape differs")
        if not np.isfinite(measurements).all():
            raise ValueError("invalid particle measurements must be finite-filled and masked")
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "measurement_validity", validity)


@dataclass(frozen=True)
class CandidateGraph:
    hlt_count: int
    offline_count: int
    hlt_index: np.ndarray
    offline_index: np.ndarray
    features: np.ndarray
    manual_scores: np.ndarray
    hlt_node_features: np.ndarray
    offline_node_features: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    hlt_to_offline: np.ndarray
    score: np.ndarray
    confidence: np.ndarray
    accepted: np.ndarray
    solver: str


def decode_exclusive_categories(flags: np.ndarray) -> np.ndarray:
    values = np.asarray(flags)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("category flags must be [particles,5]")
    binary = np.isfinite(values).all(axis=1) & np.all((values == 0) | (values == 1), axis=1)
    exclusive = binary & (values.sum(axis=1) == 1)
    result = np.full(len(values), -1, dtype=np.int8)
    result[exclusive] = values[exclusive].argmax(axis=1).astype(np.int8)
    return result


def wrapped_delta_phi(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    return (np.asarray(left) - np.asarray(right) + np.pi) % (2 * np.pi) - np.pi


def p4_kinematics(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(p4, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("p4 must have shape [particles,4]")
    px, py, pz, energy = value.T
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0))
    return pt, eta, phi, energy


def physical_p4_mask(p4: np.ndarray, *, tolerance: float = 1.0e-5) -> np.ndarray:
    value = np.asarray(p4, dtype=np.float64)
    momentum = np.linalg.norm(value[:, :3], axis=1)
    return np.isfinite(value).all(axis=1) & (value[:, 3] > 0) & (
        value[:, 3] + tolerance * np.maximum(1.0, momentum) >= momentum
    )


def compatible(hlt: ParticleSet, offline: ParticleSet, i: int, j: int) -> bool:
    category = int(hlt.categories[i])
    if category < 0 or category != int(offline.categories[j]) or bool(offline.lost_track[j]):
        return False
    if category in CHARGED_CATEGORIES:
        left, right = float(hlt.charge[i]), float(offline.charge[j])
        return np.isfinite(left) and np.isfinite(right) and left != 0 and left == right
    return True


def _scaled_measurements(particles: ParticleSet) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(particles.measurements, np.float64).copy()
    validity = np.asarray(particles.measurement_validity, np.bool_)
    scales = np.asarray((300.0, 1.0, 180.0, .9, .2, .2, 1.0))
    offsets = np.asarray((0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0))
    values = np.clip(values * scales + offsets, -8.0, 8.0)
    values[~validity] = 0.0
    return values, validity.astype(np.float64)


def _node_features(particles: ParticleSet) -> np.ndarray:
    pt, eta, phi, energy = p4_kinematics(particles.p4)
    count = len(pt); eps = 1.0e-12
    categories = np.zeros((count, 5), np.float64)
    valid_categories = (particles.categories >= 0) & (particles.categories < 5)
    categories[np.flatnonzero(valid_categories), particles.categories[valid_categories].astype(int)] = 1
    measurements, validity = _scaled_measurements(particles)
    local_density = np.zeros(count, np.float64)
    for index in range(count):
        same = particles.categories == particles.categories[index]
        distance = np.hypot(eta - eta[index], wrapped_delta_phi(phi, phi[index]))
        local_density[index] = np.log1p(np.count_nonzero(same & (distance < .10)) - 1)
    rank = np.arange(count, dtype=np.float64) / max(1, count - 1)
    competitor_placeholder = np.zeros(count, np.float64)
    return np.column_stack((
        np.log(np.maximum(pt, eps)), np.log(np.maximum(energy, eps)), eta,
        np.sin(phi), np.cos(phi), categories, np.asarray(particles.charge, np.float64),
        measurements, validity, rank, local_density, competitor_placeholder,
    )).astype(np.float32)


def build_candidate_graph(
    hlt: ParticleSet, offline: ParticleSet, *, charged_dr_gate: float = .10,
    neutral_dr_gate: float = .15, log_response_gate: float = 1.5,
) -> CandidateGraph:
    hpt, heta, hphi, henergy = p4_kinematics(hlt.p4)
    opt, oeta, ophi, oenergy = p4_kinematics(offline.p4)
    hvalid, ovalid = physical_p4_mask(hlt.p4), physical_p4_mask(offline.p4)
    edges: list[tuple[int, int]] = []
    primitive: list[list[float]] = []
    scores: list[float] = []
    eps = 1.0e-12
    for i in range(len(hlt.p4)):
        if not hvalid[i] or hlt.categories[i] < 0:
            continue
        for j in range(len(offline.p4)):
            if not ovalid[j] or not compatible(hlt, offline, i, j):
                continue
            dphi = float(wrapped_delta_phi(hphi[i], ophi[j]))
            deta = float(heta[i] - oeta[j])
            dr = math.hypot(deta, dphi)
            log_pt = math.log(max(opt[j], eps) / max(hpt[i], eps))
            log_energy = math.log(max(oenergy[j], eps) / max(henergy[i], eps))
            gate = charged_dr_gate if int(hlt.categories[i]) in CHARGED_CATEGORIES else neutral_dr_gate
            if dr > gate or abs(log_pt) > log_response_gate or abs(log_energy) > log_response_gate:
                continue
            sigma_r = .025 if int(hlt.categories[i]) in CHARGED_CATEGORIES else .05
            manual = -0.5 * ((dr / sigma_r) ** 2 + (log_pt / .25) ** 2 + (log_energy / .3) ** 2)
            edges.append((i, j)); primitive.append([dr, deta, dphi, log_pt, log_energy]); scores.append(manual)
    hlt_nodes = _node_features(hlt); offline_nodes = _node_features(offline)
    if edges:
        pairs = np.asarray(edges, dtype=np.int32)
        base = np.asarray(primitive, dtype=np.float64)
        feature_rows: list[list[float]] = []
        for row, (i, j) in enumerate(edges):
            h_rows = np.flatnonzero(pairs[:, 0] == i); o_rows = np.flatnonzero(pairs[:, 1] == j)
            h_order = h_rows[np.argsort(base[h_rows, 0], kind="stable")]
            o_order = o_rows[np.argsort(base[o_rows, 0], kind="stable")]
            h_mutual_rank = int(np.flatnonzero(h_order == row)[0]) / max(1, len(h_order) - 1)
            o_mutual_rank = int(np.flatnonzero(o_order == row)[0]) / max(1, len(o_order) - 1)
            row_competitors = len(h_rows); column_competitors = len(o_rows)
            track_valid = (
                np.asarray(hlt.measurement_validity[i], bool)
                & np.asarray(offline.measurement_validity[j], bool)
                & (int(hlt.categories[i]) in CHARGED_CATEGORIES)
            )
            track_residual = np.asarray(hlt.measurements[i]) - np.asarray(offline.measurements[j])
            track_scales = np.asarray((300.0, 1.0, 180.0, .9, .2, .2, 1.0))
            track_residual = np.clip(track_residual * track_scales, -8.0, 8.0)
            track_residual[~track_valid] = 0.0
            category_one_hot = [float(int(hlt.categories[i]) == value) for value in range(5)]
            feature_rows.append([
                *base[row], h_mutual_rank, o_mutual_rank,
                np.log1p(row_competitors), np.log1p(column_competitors),
                float(hlt_nodes[i, -2]), float(offline_nodes[j, -2]),
                float(scores[row]), *track_residual.tolist(),
                *track_valid.astype(np.float64).tolist(), *category_one_hot,
            ])
        feature_array = np.asarray(feature_rows, dtype=np.float32)
        degrees_h = np.bincount(pairs[:, 0], minlength=len(hlt_nodes))
        degrees_o = np.bincount(pairs[:, 1], minlength=len(offline_nodes))
        hlt_nodes[:, -1] = np.log1p(degrees_h); offline_nodes[:, -1] = np.log1p(degrees_o)
        score_array = np.asarray(scores, dtype=np.float64)
    else:
        pairs = np.empty((0, 2), dtype=np.int32)
        feature_array = np.empty((0, MATCH_EDGE_FEATURE_DIM), dtype=np.float32)
        score_array = np.empty(0, dtype=np.float64)
    if feature_array.shape[1] != MATCH_EDGE_FEATURE_DIM:
        raise RuntimeError("candidate edge feature construction differs from v2 contract")
    if hlt_nodes.shape[1] != MATCH_NODE_FEATURE_DIM or offline_nodes.shape[1] != MATCH_NODE_FEATURE_DIM:
        raise RuntimeError("candidate node feature construction differs from v2 contract")
    return CandidateGraph(
        len(hlt.p4), len(offline.p4), pairs[:, 0], pairs[:, 1], feature_array, score_array,
        hlt_nodes, offline_nodes,
    )


def canonicalize_scores(scores: np.ndarray, *, quantum: float = SCORE_QUANTUM) -> np.ndarray:
    if quantum <= 0 or not np.isfinite(scores).all():
        raise ValueError("score quantum must be positive and scores finite")
    return np.rint(np.asarray(scores, np.float64) / quantum).astype(np.int64)


def _dense_scores(graph: CandidateGraph, edge_scores: np.ndarray) -> np.ndarray:
    dense = np.full((graph.hlt_count, graph.offline_count), -np.inf, np.float64)
    fixed = canonicalize_scores(edge_scores)
    for row, (i, j) in enumerate(zip(graph.hlt_index, graph.offline_index, strict=True)):
        dense[int(i), int(j)] = fixed[row]
    return dense


def hungarian_with_dustbins(
    graph: CandidateGraph, edge_scores: np.ndarray, *, unmatched_score: float = -8.0,
) -> MatchResult:
    h, o = graph.hlt_count, graph.offline_count
    assignment = np.full(h, -1, np.int16)
    score = np.full(h, unmatched_score, np.float32)
    confidence = np.zeros(h, np.float32)
    if h == 0 or o == 0 or len(edge_scores) == 0:
        return MatchResult(assignment, score, confidence, assignment >= 0, "hungarian_dustbin_v1")
    dense = _dense_scores(graph, edge_scores).astype(np.float64) * SCORE_QUANTUM
    size = h + o
    utility = np.full((size, size), -1.0e12, np.float64)
    utility[:h, :o] = dense
    utility[np.arange(h), o + np.arange(h)] = unmatched_score
    utility[h + np.arange(o), np.arange(o)] = unmatched_score
    utility[h:, o:] = 0.0
    rows, columns = linear_sum_assignment(-utility)
    for row, column in zip(rows, columns, strict=True):
        if row < h and column < o and np.isfinite(dense[row, column]) and dense[row, column] > unmatched_score:
            assignment[row] = column
            score[row] = dense[row, column]
            # This is the calibrated edge probability when edge_scores are
            # calibrated logits. It is deliberately not compared with the
            # arbitrary solver dustbin utility.
            confidence[row] = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, dense[row, column]))))
    return MatchResult(assignment, score, confidence, assignment >= 0, "hungarian_dustbin_v1")


def sinkhorn_transport(
    graph: CandidateGraph, edge_scores: np.ndarray, *, dustbin_score: float = -8.0,
    temperature: float = .25, iterations: int = 100, tolerance: float = 1.0e-7,
) -> np.ndarray:
    if temperature <= 0 or iterations <= 0 or tolerance <= 0:
        raise ValueError("invalid Sinkhorn configuration")
    h, o = graph.hlt_count, graph.offline_count
    logits = np.full((h + 1, o + 1), -1.0e9, np.float64)
    if h and o:
        dense = _dense_scores(graph, edge_scores) * SCORE_QUANTUM
        logits[:h, :o] = dense
    logits[:h, o] = dustbin_score
    logits[h, :o] = dustbin_score
    logits[h, o] = 0.0
    logits /= temperature
    row_mass = np.ones(h + 1, np.float64); row_mass[-1] = max(1, o)
    col_mass = np.ones(o + 1, np.float64); col_mass[-1] = max(1, h)
    total = max(row_mass.sum(), col_mass.sum())
    row_mass /= total; col_mass /= total
    row_mass[-1] += 1.0 - row_mass.sum(); col_mass[-1] += 1.0 - col_mass.sum()
    kernel = np.exp(np.clip(logits, -80, 80))
    u = np.ones_like(row_mass); v = np.ones_like(col_mass)
    for _ in range(iterations):
        previous = u.copy()
        u = row_mass / np.maximum(kernel @ v, 1.0e-300)
        v = col_mass / np.maximum(kernel.T @ u, 1.0e-300)
        if np.max(np.abs(u - previous)) <= tolerance:
            break
    return (u[:, None] * kernel) * v[None, :]


def optimal_transport_with_dustbins(
    graph: CandidateGraph, edge_scores: np.ndarray, *, unmatched_score: float = -8.0,
) -> MatchResult:
    plan = sinkhorn_transport(graph, edge_scores, dustbin_score=unmatched_score)
    if graph.hlt_count == 0:
        empty = np.empty(0, np.float32)
        return MatchResult(np.empty(0, np.int16), empty, empty, np.empty(0, np.bool_), "uot_dustbin_v1")
    edge_probabilities = np.asarray([
        plan[int(i), int(j)] / max(plan[int(i)].sum(), 1.0e-30)
        for i, j in zip(graph.hlt_index, graph.offline_index, strict=True)
    ])
    result = hungarian_with_dustbins(
        graph, np.log(np.maximum(edge_probabilities, 1.0e-30)),
        unmatched_score=math.log(max(1.0e-30, 1.0 / (1.0 + math.exp(-unmatched_score)))),
    )
    confidence = np.zeros(graph.hlt_count, np.float32)
    edge_lookup = {
        (int(i), int(j)): float(score)
        for i, j, score in zip(graph.hlt_index, graph.offline_index, edge_scores, strict=True)
    }
    for i, j in enumerate(result.hlt_to_offline):
        if j >= 0:
            raw = max(-80.0, min(80.0, edge_lookup[(i, int(j))]))
            confidence[i] = 1.0 / (1.0 + math.exp(-raw))
    return MatchResult(result.hlt_to_offline, result.score, confidence, result.accepted, "uot_dustbin_v1")


def greedy_angular(graph: CandidateGraph) -> MatchResult:
    order = sorted(
        range(len(graph.hlt_index)),
        key=lambda row: (float(graph.features[row, 0]), int(graph.hlt_index[row]), int(graph.offline_index[row])),
    )
    assignment = np.full(graph.hlt_count, -1, np.int16)
    scores = np.full(graph.hlt_count, -np.inf, np.float32)
    used: set[int] = set()
    for row in order:
        i, j = int(graph.hlt_index[row]), int(graph.offline_index[row])
        if assignment[i] < 0 and j not in used:
            assignment[i] = j; used.add(j); scores[i] = -graph.features[row, 0]
    confidence = np.where(assignment >= 0, np.exp(np.maximum(scores, -80)), 0).astype(np.float32)
    return MatchResult(assignment, scores, confidence, assignment >= 0, "greedy_angular_diagnostic_v1")


def apply_acceptance(
    result: MatchResult, *, probability_threshold: float,
    independent: MatchResult | None = None,
) -> MatchResult:
    if not 0 <= probability_threshold <= 1:
        raise ValueError("probability threshold lies outside [0,1]")
    accepted = (result.hlt_to_offline >= 0) & (result.confidence >= probability_threshold)
    if independent is not None:
        accepted &= result.hlt_to_offline == independent.hlt_to_offline
    assignment = np.where(accepted, result.hlt_to_offline, -1).astype(np.int16)
    confidence = np.where(accepted, result.confidence, 0).astype(np.float32)
    return MatchResult(assignment, result.score.copy(), confidence, accepted, result.solver)


def _assignment_utility(assignment: np.ndarray, dense: np.ndarray, unmatched_score: float) -> float:
    used = set(int(value) for value in assignment if value >= 0)
    value = sum(
        dense[i, int(j)] if j >= 0 else unmatched_score
        for i, j in enumerate(assignment)
    )
    value += unmatched_score * (dense.shape[1] - len(used))
    return float(value)


def _component_for_edge(graph: CandidateGraph, seed_hlt: int, seed_offline: int):
    h_nodes = {seed_hlt}; o_nodes = {seed_offline}; changed = True
    while changed:
        changed = False
        for i, j in zip(graph.hlt_index, graph.offline_index, strict=True):
            i, j = int(i), int(j)
            if i in h_nodes or j in o_nodes:
                before = (len(h_nodes), len(o_nodes)); h_nodes.add(i); o_nodes.add(j)
                changed |= before != (len(h_nodes), len(o_nodes))
    h_order, o_order = sorted(h_nodes), sorted(o_nodes)
    h_map = {value: index for index, value in enumerate(h_order)}
    o_map = {value: index for index, value in enumerate(o_order)}
    rows = np.asarray([
        row for row, (i, j) in enumerate(zip(graph.hlt_index, graph.offline_index, strict=True))
        if int(i) in h_nodes and int(j) in o_nodes
    ], np.int64)
    subgraph = CandidateGraph(
        len(h_order), len(o_order),
        np.asarray([h_map[int(graph.hlt_index[row])] for row in rows], np.int32),
        np.asarray([o_map[int(graph.offline_index[row])] for row in rows], np.int32),
        graph.features[rows], graph.manual_scores[rows],
        graph.hlt_node_features[h_order], graph.offline_node_features[o_order],
    )
    return subgraph, rows, h_map[seed_hlt], o_map[seed_offline]


def assignment_diagnostics(
    graph: CandidateGraph, result: MatchResult, edge_scores: np.ndarray, *,
    independent: MatchResult | None = None, unmatched_score: float = -8.0,
) -> np.ndarray:
    """Build post-global-assignment correctness features for every HLT node."""
    scores = np.asarray(edge_scores, np.float64)
    if scores.shape != (len(graph.hlt_index),) or not np.isfinite(scores).all():
        raise ValueError("assignment diagnostic scores differ from candidate edges")
    dense = _dense_scores(graph, scores) * SCORE_QUANTUM
    diagnostics = np.zeros((graph.hlt_count, ASSIGNMENT_DIAGNOSTIC_DIM), np.float64)
    for i in np.flatnonzero(result.hlt_to_offline >= 0):
        j = int(result.hlt_to_offline[i])
        edge_rows = np.flatnonzero((graph.hlt_index == i) & (graph.offline_index == j))
        if len(edge_rows) != 1:
            continue
        selected = float(dense[i, j])
        row_alternatives = np.delete(dense[i], j)
        column_alternatives = np.delete(dense[:, j], i)
        row_alternative = max(
            unmatched_score,
            float(np.max(row_alternatives)) if len(row_alternatives) else -np.inf,
        )
        column_alternative = max(
            unmatched_score,
            float(np.max(column_alternatives)) if len(column_alternatives) else -np.inf,
        )
        mutual = float(selected >= np.max(dense[i]) and selected >= np.max(dense[:, j]))
        agreement = float(
            independent is None or int(independent.hlt_to_offline[i]) == j
        )
        subgraph, component_rows, local_i, local_j = _component_for_edge(graph, int(i), j)
        component_scores = scores[component_rows]
        component_dense = _dense_scores(subgraph, component_scores) * SCORE_QUANTUM
        base = hungarian_with_dustbins(
            subgraph, component_scores, unmatched_score=unmatched_score,
        )
        base_utility = _assignment_utility(base.hlt_to_offline, component_dense, unmatched_score)
        local_edge = np.flatnonzero(
            (subgraph.hlt_index == local_i) & (subgraph.offline_index == local_j)
        )
        global_margin = 0.0
        if len(local_edge) == 1:
            forbidden = component_scores.copy(); forbidden[local_edge[0]] = -1.0e12
            alternative = hungarian_with_dustbins(
                subgraph, forbidden, unmatched_score=unmatched_score,
            )
            global_margin = base_utility - _assignment_utility(
                alternative.hlt_to_offline, component_dense, unmatched_score,
            )
        category = int(np.argmax(graph.features[edge_rows[0], -5:]))
        diagnostics[i] = (
            selected, np.clip(global_margin, -20, 20),
            np.clip(selected - row_alternative, -20, 20),
            np.clip(selected - column_alternative, -20, 20), mutual, agreement,
            graph.features[edge_rows[0], 7], graph.features[edge_rows[0], 8],
            *[float(category == value) for value in range(5)],
        )
    return diagnostics


def calibrate_assignment_confidence(
    graph: CandidateGraph, result: MatchResult, edge_scores: np.ndarray, *,
    calibrator: Mapping[str, object], independent: MatchResult | None = None,
) -> MatchResult:
    """Predict P(global assignment is correct) after exclusivity is solved."""
    weight = np.asarray(calibrator.get("weight"), np.float64)
    mean = np.asarray(calibrator.get("mean"), np.float64)
    scale = np.asarray(calibrator.get("scale"), np.float64)
    if (calibrator.get("method") != "post_assignment_logistic_v1"
            or weight.shape != (ASSIGNMENT_DIAGNOSTIC_DIM,)
            or mean.shape != weight.shape or scale.shape != weight.shape
            or np.any(scale <= 0)):
        raise ValueError("post-assignment calibrator contract differs")
    diagnostics = assignment_diagnostics(
        graph, result, edge_scores, independent=independent,
    )
    logits = ((diagnostics - mean) / scale) @ weight + float(calibrator["intercept"])
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    confidence = np.where(result.hlt_to_offline >= 0, probability, 0).astype(np.float32)
    return MatchResult(
        result.hlt_to_offline.copy(), result.score.copy(), confidence,
        result.hlt_to_offline >= 0, result.solver + "+postcal",
    )


def certify_global_matches(
    graph: CandidateGraph, result: MatchResult, edge_scores: np.ndarray, *,
    independent: MatchResult, probability_threshold: float,
    minimum_global_margin: float = .25, guard_band: float = 1.0e-4,
) -> MatchResult:
    """Apply mutual preference, independent-solver, and exact forbidden-edge margins."""
    dense = _dense_scores(graph, edge_scores) * SCORE_QUANTUM
    accepted = (
        (result.hlt_to_offline >= 0)
        & (result.confidence > probability_threshold + guard_band)
        & (result.hlt_to_offline == independent.hlt_to_offline)
    )
    for i in np.flatnonzero(accepted):
        j = int(result.hlt_to_offline[i]); selected_score = dense[i, j]
        row_best = np.nanmax(dense[i])
        col_best = np.nanmax(dense[:, j])
        if selected_score != row_best or selected_score != col_best:
            accepted[i] = False; continue
        subgraph, component_rows, local_i, local_j = _component_for_edge(graph, int(i), j)
        edge_rows = np.flatnonzero(
            (subgraph.hlt_index == local_i) & (subgraph.offline_index == local_j)
        )
        if len(edge_rows) != 1:
            accepted[i] = False; continue
        component_scores = np.asarray(edge_scores, np.float64)[component_rows]
        component_dense = _dense_scores(subgraph, component_scores) * SCORE_QUANTUM
        component_base = hungarian_with_dustbins(subgraph, component_scores)
        base_utility = _assignment_utility(component_base.hlt_to_offline, component_dense, -8.0)
        forbidden_scores = component_scores.copy()
        forbidden_scores[edge_rows[0]] = -1.0e12
        alternative = hungarian_with_dustbins(subgraph, forbidden_scores)
        alternative_utility = _assignment_utility(alternative.hlt_to_offline, component_dense, -8.0)
        if base_utility - alternative_utility < minimum_global_margin:
            accepted[i] = False
    assignment = np.where(accepted, result.hlt_to_offline, -1).astype(np.int16)
    confidence = np.where(accepted, result.confidence, 0).astype(np.float32)
    return MatchResult(assignment, result.score.copy(), confidence, accepted, result.solver + "+certified")


def match_variant(
    graph: CandidateGraph, variant: str, *, contextual_scores: np.ndarray | None = None,
    likelihood_scores: np.ndarray | None = None, threshold: float = .99,
    assignment_calibrator: Mapping[str, object] | None = None,
) -> MatchResult:
    if variant not in MATCHER_VARIANTS:
        raise ValueError(f"unknown matcher variant {variant!r}")
    if variant == "M0":
        return greedy_angular(graph)
    likelihood = likelihood_scores if likelihood_scores is not None else graph.manual_scores
    contextual = contextual_scores if contextual_scores is not None else likelihood
    if variant == "M1":
        return apply_acceptance(hungarian_with_dustbins(graph, graph.manual_scores), probability_threshold=threshold)
    if variant == "M2":
        return apply_acceptance(hungarian_with_dustbins(graph, likelihood), probability_threshold=threshold)
    if variant == "M3":
        return apply_acceptance(optimal_transport_with_dustbins(graph, likelihood), probability_threshold=threshold)
    hungarian = hungarian_with_dustbins(graph, contextual)
    transport = optimal_transport_with_dustbins(graph, contextual)
    primary = (
        calibrate_assignment_confidence(
            graph, hungarian, contextual, calibrator=assignment_calibrator,
            independent=transport,
        ) if assignment_calibrator is not None else hungarian
    )
    if variant == "M4":
        return certify_global_matches(
            graph, primary, contextual,
            independent=transport,
            probability_threshold=threshold,
        )
    control = optimal_transport_with_dustbins(graph, likelihood)
    contextual_certified = certify_global_matches(
        graph, primary, contextual,
        independent=transport,
        probability_threshold=threshold,
    )
    return apply_acceptance(
        contextual_certified, probability_threshold=threshold, independent=control,
    )


__all__ = [
    "ASSIGNMENT_DIAGNOSTIC_DIM", "CandidateGraph", "MATCH_EDGE_FEATURE_DIM",
    "MATCH_NODE_FEATURE_DIM", "MatchResult", "ParticleSet", "SCORE_QUANTUM", "TRACK_FIELDS",
    "apply_acceptance", "assignment_diagnostics", "calibrate_assignment_confidence",
    "certify_global_matches",
    "build_candidate_graph", "canonicalize_scores", "compatible",
    "decode_exclusive_categories", "greedy_angular", "hungarian_with_dustbins",
    "match_variant", "optimal_transport_with_dustbins", "p4_kinematics",
    "physical_p4_mask", "sinkhorn_transport", "wrapped_delta_phi",
]
