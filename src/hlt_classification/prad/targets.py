"""Deterministic offline structural targets for PRAD."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .contracts import PRAD_CA_MULTIPLICITIES


def _wrap_phi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _four_vector(token: np.ndarray) -> np.ndarray:
    pt, eta, phi, energy = (float(token[index]) for index in range(4))
    return np.asarray(
        [pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), energy],
        dtype=np.float64,
    )


def _coordinates(vector: np.ndarray) -> tuple[float, float]:
    px, py, pz, _energy = vector
    pt = math.hypot(px, py)
    eta = math.asinh(pz / max(pt, 1.0e-12)) if pt > 0.0 else 0.0
    return eta, math.atan2(py, px) if pt > 0.0 else 0.0


def _exclusive_assignment(
    tokens: np.ndarray,
    valid_indices: np.ndarray,
    multiplicity: int,
) -> np.ndarray:
    clusters = [
        {"members": [int(index)], "vector": _four_vector(tokens[index])}
        for index in valid_indices.tolist()
    ]
    target = min(multiplicity, len(clusters))
    while len(clusters) > target:
        candidates = []
        for left in range(len(clusters)):
            eta_left, phi_left = _coordinates(clusters[left]["vector"])
            for right in range(left + 1, len(clusters)):
                eta_right, phi_right = _coordinates(clusters[right]["vector"])
                distance = (eta_left - eta_right) ** 2 + _wrap_phi(
                    phi_left - phi_right
                ) ** 2
                candidates.append(
                    (
                        distance,
                        min(clusters[left]["members"]),
                        min(clusters[right]["members"]),
                        left,
                        right,
                    )
                )
        _distance, _left_id, _right_id, left, right = min(candidates)
        merged = {
            "members": sorted(
                clusters[left]["members"] + clusters[right]["members"]
            ),
            "vector": clusters[left]["vector"] + clusters[right]["vector"],
        }
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: min(cluster["members"]))
    assignment = np.full((len(tokens),), -1, dtype=np.int16)
    for cluster_id, cluster in enumerate(clusters):
        assignment[np.asarray(cluster["members"], dtype=np.int64)] = cluster_id
    return assignment


def build_exclusive_ca_assignments(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    multiplicities: Sequence[int] = PRAD_CA_MULTIPLICITIES,
) -> np.ndarray:
    """Return deterministic exclusive C/A cluster IDs with no radius guess."""

    tokens = np.asarray(tokens)
    mask = np.asarray(mask)
    if tokens.dtype != np.float32 or tokens.ndim != 2 or tokens.shape[1] != 14:
        raise ValueError("offline tokens must have float32 shape [N,14]")
    if mask.dtype != np.bool_ or mask.shape != tokens.shape[:1]:
        raise ValueError("offline mask must have boolean shape [N]")
    if not np.isfinite(tokens).all() or np.any(tokens[~mask] != 0.0):
        raise ValueError("offline tokens are nonfinite or padding is nonzero")
    requested = tuple(int(value) for value in multiplicities)
    if requested != PRAD_CA_MULTIPLICITIES:
        raise ValueError(
            f"PRAD exclusive C/A multiplicities must be {PRAD_CA_MULTIPLICITIES}"
        )
    valid = np.flatnonzero(mask)
    return np.stack(
        [_exclusive_assignment(tokens, valid, value) for value in requested],
        axis=0,
    )


def same_cluster_targets(assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.asarray(assignments)
    if assignments.ndim != 2 or assignments.shape[0] != len(
        PRAD_CA_MULTIPLICITIES
    ):
        raise ValueError("C/A assignments must have shape [3,N]")
    valid = assignments >= 0
    targets = (
        assignments[:, :, None] == assignments[:, None, :]
    ) & valid[:, :, None] & valid[:, None, :]
    pair_mask = valid[:, :, None] & valid[:, None, :]
    diagonal = np.arange(assignments.shape[1])
    pair_mask[:, diagonal, diagonal] = False
    targets[:, diagonal, diagonal] = False
    return targets.astype(np.bool_), pair_mask.astype(np.bool_)


__all__ = ["build_exclusive_ca_assignments", "same_cluster_targets"]
