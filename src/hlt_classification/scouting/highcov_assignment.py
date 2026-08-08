"""Deterministic one-to-one assignment, anchor completion, and transport controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .highcov_data import Particles
from .highcov_features import AnchorContext, EdgeMatrices


SCORE_QUANTUM = 1.0e-6
FORBIDDEN = -1.0e9


@dataclass(frozen=True)
class Assignment:
    offline_index: np.ndarray
    score: np.ndarray
    solver: str

    @property
    def accepted(self) -> np.ndarray:
        return self.offline_index >= 0

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.accepted))


def canonical_scores(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, np.float64)
    if not np.isfinite(value).all():
        raise ValueError("assignment scores must be finite")
    return np.rint(value / SCORE_QUANTUM) * SCORE_QUANTUM


def hungarian_dustbin(
    score: np.ndarray, gate: np.ndarray, *, dummy_score: float = -12.0,
    solver: str = "hungarian_private_dustbin_v1",
) -> Assignment:
    value = canonical_scores(np.where(gate, score, FORBIDDEN))
    nh, no = value.shape
    assignment = np.full(nh, -1, np.int32)
    selected_score = np.full(nh, dummy_score, np.float32)
    if nh == 0 or no == 0:
        return Assignment(assignment, selected_score, solver)
    cost = np.full((nh, no + nh), -FORBIDDEN, np.float64)
    cost[:, :no] = -value
    cost[np.arange(nh), no + np.arange(nh)] = -dummy_score
    rows, cols = linear_sum_assignment(cost)
    real = (cols < no) & gate[rows, np.minimum(cols, no - 1)]
    real &= value[rows, np.minimum(cols, no - 1)] > dummy_score
    rows, cols = rows[real], cols[real]
    assignment[rows] = cols.astype(np.int32)
    selected_score[rows] = value[rows, cols].astype(np.float32)
    return Assignment(assignment, selected_score, solver)


def anchor_completion(
    score: np.ndarray, gate: np.ndarray, anchors: AnchorContext, *,
    dummy_score: float = -12.0,
) -> Assignment:
    nh, no = score.shape
    assignment = np.full(nh, -1, np.int32)
    selected_score = np.full(nh, dummy_score, np.float32)
    ah = np.asarray(anchors.hlt_index, np.int64)
    ao = np.asarray(anchors.offline_index, np.int64)
    if len(np.unique(ah)) != len(ah) or len(np.unique(ao)) != len(ao):
        raise ValueError("anchors are not one-to-one")
    if len(ah):
        valid = gate[ah, ao]
        ah, ao = ah[valid], ao[valid]
        assignment[ah] = ao
        selected_score[ah] = np.asarray(score)[ah, ao]
    remaining_h = np.setdiff1d(np.arange(nh), ah, assume_unique=True)
    remaining_o = np.setdiff1d(np.arange(no), ao, assume_unique=True)
    if len(remaining_h) and len(remaining_o):
        residual = hungarian_dustbin(
            np.asarray(score)[np.ix_(remaining_h, remaining_o)],
            np.asarray(gate)[np.ix_(remaining_h, remaining_o)],
            dummy_score=dummy_score, solver="residual_hungarian_v1",
        )
        accepted = residual.offline_index >= 0
        assignment[remaining_h[accepted]] = remaining_o[residual.offline_index[accepted]]
        selected_score[remaining_h[accepted]] = residual.score[accepted]
    return Assignment(assignment, selected_score, "strict_anchor_completion_v1")


def sinkhorn_plan(
    score: np.ndarray, gate: np.ndarray, *, dustbin_score: float = -12.0,
    temperature: float = 0.25, iterations: int = 200, tolerance: float = 1e-8,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Sinkhorn temperature must be positive")
    value = canonical_scores(np.where(gate, score, FORBIDDEN))
    h, o = value.shape
    logits = np.full((h + 1, o + 1), -80.0, np.float64)
    logits[:h, :o] = np.where(gate, value / temperature, -80.0)
    logits[:h, o] = dustbin_score / temperature
    logits[h, :o] = dustbin_score / temperature
    logits[h, o] = 0.0
    row_mass = np.ones(h + 1, np.float64); row_mass[-1] = max(1, o)
    col_mass = np.ones(o + 1, np.float64); col_mass[-1] = max(1, h)
    row_mass /= row_mass.sum(); col_mass /= col_mass.sum()
    kernel = np.exp(np.clip(logits, -80, 80))
    u = np.ones_like(row_mass); v = np.ones_like(col_mass)
    for _ in range(iterations):
        previous_u = u.copy(); previous_v = v.copy()
        u = row_mass / np.maximum(kernel @ v, 1e-300)
        v = col_mass / np.maximum(kernel.T @ u, 1e-300)
        if max(np.max(np.abs(u - previous_u)), np.max(np.abs(v - previous_v))) <= tolerance:
            break
    plan = (u[:, None] * kernel) * v[None, :]
    if not np.isfinite(plan).all():
        raise RuntimeError("Sinkhorn plan is nonfinite")
    return plan


def transport_rounding(
    score: np.ndarray, gate: np.ndarray, *, dustbin_score: float = -12.0,
    temperature: float = 0.25,
) -> tuple[Assignment, np.ndarray]:
    plan = sinkhorn_plan(
        score, gate, dustbin_score=dustbin_score, temperature=temperature,
    )
    h, o = np.shape(score)
    row_probability = plan[:h, :o] / np.maximum(plan[:h].sum(axis=1, keepdims=True), 1e-30)
    dustbin_probability = plan[:h, o] / np.maximum(plan[:h].sum(axis=1), 1e-30)
    utility = np.log(np.maximum(row_probability, 1e-30))
    assignment = hungarian_dustbin(
        utility, gate,
        dummy_score=float(np.median(np.log(np.maximum(dustbin_probability, 1e-30)))) if h else -12.0,
        solver=f"sinkhorn_t{temperature:g}_hungarian_round_v1",
    )
    return assignment, plan


def forced_max_cardinality(score: np.ndarray) -> Assignment:
    value = canonical_scores(np.asarray(score, np.float64))
    nh, no = value.shape
    assignment = np.full(nh, -1, np.int32)
    selected = np.full(nh, -np.inf, np.float32)
    if nh and no:
        rows, cols = linear_sum_assignment(-value)
        assignment[rows] = cols.astype(np.int32)
        selected[rows] = value[rows, cols].astype(np.float32)
    return Assignment(assignment, selected, "forced_max_cardinality_oracle_v1")


def lexicographic_assignment(
    score: np.ndarray, gate: np.ndarray, *, solver: str = "lexicographic_cardinality_score_v1",
) -> Assignment:
    """Maximize plausible-edge cardinality first, then the supplied edge score."""
    value = canonical_scores(np.asarray(score, np.float64))
    allowed = np.asarray(gate, bool)
    nh, no = value.shape
    if allowed.shape != value.shape:
        raise ValueError("gate shape differs from score")
    assignment = np.full(nh, -1, np.int32)
    selected = np.full(nh, -np.inf, np.float32)
    if nh == 0 or no == 0 or not np.any(allowed):
        return Assignment(assignment, selected, solver)
    finite = value[allowed]
    lo, hi = float(np.min(finite)), float(np.max(finite))
    secondary = np.zeros_like(value)
    if hi > lo:
        secondary[allowed] = 2.0 * (value[allowed] - lo) / (hi - lo) - 1.0
    cardinality_unit = 2.0 * min(nh, no) + 1.0
    utility = np.where(allowed, cardinality_unit + secondary, FORBIDDEN)
    solved = hungarian_dustbin(utility, allowed, dummy_score=0.0, solver=solver)
    accepted = solved.accepted
    assignment[accepted] = solved.offline_index[accepted]
    selected[accepted] = value[np.flatnonzero(accepted), assignment[accepted]].astype(np.float32)
    return Assignment(assignment, selected, solver)


def anchor_lexicographic_assignment(
    score: np.ndarray, gate: np.ndarray, anchors: AnchorContext, *,
    solver: str = "anchor_then_lexicographic_completion_v1",
) -> Assignment:
    """Lock one-to-one ultra-tight anchors, then lexicographically complete."""
    value = canonical_scores(np.asarray(score, np.float64))
    allowed = np.asarray(gate, bool)
    nh, no = value.shape
    assignment = np.full(nh, -1, np.int32)
    selected = np.full(nh, -np.inf, np.float32)
    ah = np.asarray(anchors.hlt_index, np.int64)
    ao = np.asarray(anchors.offline_index, np.int64)
    keep = allowed[ah, ao] if len(ah) else np.empty(0, bool)
    ah, ao = ah[keep], ao[keep]
    if len(ah):
        assignment[ah] = ao
        selected[ah] = value[ah, ao]
    remaining_h = np.setdiff1d(np.arange(nh), ah, assume_unique=True)
    remaining_o = np.setdiff1d(np.arange(no), ao, assume_unique=True)
    if len(remaining_h) and len(remaining_o):
        residual = lexicographic_assignment(
            value[np.ix_(remaining_h, remaining_o)],
            allowed[np.ix_(remaining_h, remaining_o)], solver=solver + "_residual",
        )
        accepted = residual.accepted
        assignment[remaining_h[accepted]] = remaining_o[residual.offline_index[accepted]]
        selected[remaining_h[accepted]] = residual.score[accepted]
    return Assignment(assignment, selected, solver)


DIAGNOSTIC_NAMES = (
    "score", "row_margin", "column_margin", "negative_dr_over_0p02",
    "negative_abs_log_pt", "negative_abs_log_energy", "pid_equal", "charge_equal",
    "mutual_geometry", "solver_consensus", "anchor", "log_anchor_count",
    "abs_centered_deta", "abs_centered_dphi", "abs_centered_log_pt",
    "abs_centered_log_energy", "row_degree_log1p", "column_degree_log1p",
)


def assignment_diagnostics(
    assignment: Assignment, score: np.ndarray, gate: np.ndarray,
    matrices: EdgeMatrices, hlt: Particles, offline: Particles,
    context: AnchorContext, *, independent: Assignment | None = None,
    dummy_score: float = -12.0,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(assignment.accepted)
    if not len(rows):
        return rows.astype(np.int32), np.empty((0, len(DIAGNOSTIC_NAMES)), np.float32)
    cols = assignment.offline_index[rows].astype(np.int64)
    value = canonical_scores(np.where(gate, score, FORBIDDEN))
    selected = value[rows, cols]
    row_margin = np.empty(len(rows)); col_margin = np.empty(len(rows))
    for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
        row_alt = value[i].copy(); row_alt[j] = dummy_score
        row_margin[k] = selected[k] - max(dummy_score, float(np.max(row_alt)))
        col_alt = np.delete(value[:, j], i)
        col_margin[k] = selected[k] - max(dummy_score, float(np.max(col_alt)) if len(col_alt) else dummy_score)
    hbest = np.argmin(matrices.dr, axis=1)
    obest = np.argmin(matrices.dr, axis=0)
    anchor_pairs = set(zip(context.hlt_index.tolist(), context.offline_index.tolist()))
    consensus = np.ones(len(rows)) if independent is None else (
        independent.offline_index[rows] == cols
    ).astype(float)
    centered = np.column_stack((
        matrices.deta[rows, cols], matrices.dphi[rows, cols],
        matrices.log_pt[rows, cols], matrices.log_energy[rows, cols],
    )) - context.center
    diagnostics = np.column_stack((
        selected, np.clip(row_margin, -30, 30), np.clip(col_margin, -30, 30),
        -matrices.dr[rows, cols] / .02,
        -np.abs(matrices.log_pt[rows, cols]), -np.abs(matrices.log_energy[rows, cols]),
        (hlt.category[rows] == offline.category[cols]).astype(float),
        (hlt.charge[rows] == offline.charge[cols]).astype(float),
        ((hbest[rows] == cols) & (obest[cols] == rows)).astype(float), consensus,
        np.asarray([(int(i), int(j)) in anchor_pairs for i, j in zip(rows, cols, strict=True)], float),
        np.full(len(rows), np.log1p(context.count)), np.abs(centered),
        np.log1p(np.sum(gate[rows], axis=1)), np.log1p(np.sum(gate[:, cols], axis=0)),
    )).astype(np.float32)
    if diagnostics.shape[1] != len(DIAGNOSTIC_NAMES):
        raise RuntimeError("assignment diagnostic dimension changed")
    return rows.astype(np.int32), diagnostics


__all__ = [
    "Assignment", "DIAGNOSTIC_NAMES", "SCORE_QUANTUM", "anchor_completion",
    "anchor_lexicographic_assignment",
    "assignment_diagnostics", "canonical_scores", "forced_max_cardinality",
    "hungarian_dustbin", "lexicographic_assignment", "sinkhorn_plan", "transport_rounding",
]
