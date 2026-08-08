"""Compact selected empirical lexicographic matcher and confidence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .highcov_assignment import Assignment, assignment_diagnostics, lexicographic_assignment
from .highcov_calibration import ConfidenceCalibrator
from .highcov_data import Particles
from .highcov_features import broad_gate, edge_matrices, mutual_anchor_context
from .highcov_scorers import EmpiricalScorer, geometry_response_score


MAX_DR = 0.30
MAX_ABS_LOG_RESPONSE = 4.0


@dataclass(frozen=True)
class MatchResult:
    concatenated_offline_index: np.ndarray
    native_offline_index: np.ndarray
    confidence: np.ndarray
    assignment_score: np.ndarray
    accepted: np.ndarray


def selected_assignment_components(
    hlt: Particles, offline: Particles, scorer: EmpiricalScorer,
) -> tuple[Assignment, Assignment, np.ndarray, np.ndarray, object, object]:
    matrices = edge_matrices(hlt, offline)
    context = mutual_anchor_context(matrices, hlt, offline)
    gate = broad_gate(
        matrices, hlt, offline, max_dr=MAX_DR,
        max_abs_log_response=MAX_ABS_LOG_RESPONSE,
    )
    score = scorer.score(matrices)
    primary = lexicographic_assignment(
        score, gate, solver="highcov_empirical_lexicographic_dr0p30_v1",
    )
    independent = lexicographic_assignment(
        geometry_response_score(matrices), gate,
        solver="highcov_geometry_response_consensus_v1",
    )
    return primary, independent, score, gate, matrices, context


def selected_diagnostics(
    hlt: Particles, offline: Particles, scorer: EmpiricalScorer,
) -> tuple[Assignment, np.ndarray, np.ndarray]:
    primary, independent, score, gate, matrices, context = selected_assignment_components(
        hlt, offline, scorer,
    )
    rows, diagnostics = assignment_diagnostics(
        primary, score, gate, matrices, hlt, offline, context,
        independent=independent, dummy_score=-30.0,
    )
    return primary, rows, diagnostics


class HighCoverageMatcher:
    """Reusable inference object; no neural-network dependency at inference."""

    def __init__(
        self, empirical_payload: Mapping[str, object], calibration_payload: Mapping[str, object],
        *, model_key: str = "full_development_for_audit",
    ) -> None:
        models = empirical_payload.get("models")
        if not isinstance(models, Mapping) or model_key not in models:
            raise ValueError(f"unknown highcov empirical model {model_key!r}")
        model = models[model_key]
        self.scorer = EmpiricalScorer.from_payload(model, include_rank=True)
        self.calibrator = ConfidenceCalibrator.from_payload(calibration_payload["model"])
        self.model_key = model_key

    def match(self, hlt: Particles, offline: Particles) -> MatchResult:
        assignment, rows, diagnostics = selected_diagnostics(hlt, offline, self.scorer)
        confidence = np.zeros(len(hlt.p4), np.float32)
        confidence[rows] = self.calibrator.predict(diagnostics)
        native = np.full(len(hlt.p4), -1, np.int32)
        if offline.native_index is not None:
            native[rows] = offline.native_index[assignment.offline_index[rows]].astype(np.int32)
        else:
            native[rows] = assignment.offline_index[rows]
        return MatchResult(
            assignment.offline_index.copy(), native, confidence,
            assignment.score.copy(), assignment.accepted.copy(),
        )


def model_key_for_role(role: str, source_fold: int | None = None) -> str:
    """Select a scorer without permitting train-fold leakage."""

    if role == "train":
        if source_fold not in (0, 1, 2, 3):
            raise ValueError("train assignments require a cross-fitted source fold 0--3")
        return f"holdout_{source_fold}"
    if role == "matcher_audit":
        if source_fold != 4:
            raise ValueError("matcher audit role requires frozen fold 4")
        return "full_development_for_audit"
    if role in ("validation", "final_test", "inference"):
        if source_fold is not None:
            raise ValueError(f"{role} must not select a train-fold scorer")
        return "full_development_for_audit"
    raise ValueError(f"unknown highcov matcher role {role!r}")


def from_scouting_particles(value: object, *, offline: bool) -> Particles:
    """Adapt the repository ParticleSet while excluding appended lost tracks."""

    required = ("p4", "categories", "charge", "lost_track", "measurements", "measurement_validity")
    if any(not hasattr(value, name) for name in required):
        raise TypeError("value is not a Scouting ParticleSet")
    p4 = np.asarray(value.p4)
    category = np.asarray(value.categories)
    charge = np.asarray(value.charge)
    track = np.asarray(value.measurements)
    valid = np.asarray(value.measurement_validity)
    lost = np.asarray(value.lost_track, bool)
    if offline:
        keep = ~lost
        native = np.flatnonzero(keep).astype(np.int32)
    else:
        if np.any(lost):
            raise ValueError("HLT matcher population cannot contain lost tracks")
        keep = np.ones(len(p4), bool)
        native = None
    return Particles(
        p4[keep], category[keep], charge[keep], track[keep], valid[keep], native,
    )


__all__ = [
    "HighCoverageMatcher", "MAX_ABS_LOG_RESPONSE", "MAX_DR", "MatchResult",
    "from_scouting_particles", "model_key_for_role", "selected_assignment_components",
    "selected_diagnostics",
]
