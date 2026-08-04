"""Training-only deterministic HLT-to-offline Hungarian association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import PRAD_MATCH_ALGORITHM, PRAD_MATCH_CONTRACT

CHARGED_CATEGORIES = frozenset({0, 3, 4})
NEUTRAL_CATEGORIES = frozenset({1, 2})


@dataclass(frozen=True)
class MatchResult:
    hlt_to_offline: np.ndarray
    costs: np.ndarray
    hlt_valid: np.ndarray
    offline_valid: np.ndarray
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.hlt_to_offline.dtype != np.int64:
            raise ValueError("PRAD match indices must use int64")
        if self.costs.dtype != np.float32:
            raise ValueError("PRAD match costs must use float32")
        if self.hlt_to_offline.shape != self.costs.shape:
            raise ValueError("PRAD match index and cost shapes differ")


def _categories(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pid = np.asarray(tokens[:, 5:10])
    categories = np.full((len(tokens),), 5, dtype=np.int8)
    if np.any(mask):
        valid = pid[mask]
        if not np.all((valid == 0.0) | (valid == 1.0)):
            raise ValueError("PRAD matching requires exact PID flags")
        counts = valid.sum(axis=1)
        if np.any(counts > 1.0):
            raise ValueError("PRAD matching PID flags are not exclusive")
        known = counts == 1.0
        valid_categories = np.full((len(valid),), 5, dtype=np.int8)
        valid_categories[known] = np.argmax(valid[known], axis=1).astype(np.int8)
        categories[np.flatnonzero(mask)] = valid_categories
    return categories


def _validate_view(tokens: np.ndarray, mask: np.ndarray, *, name: str) -> None:
    if tokens.dtype != np.float32 or tokens.ndim != 2 or tokens.shape[1] != 14:
        raise ValueError(f"{name} tokens must have float32 shape [N,14]")
    if mask.dtype != np.bool_ or mask.shape != tokens.shape[:1]:
        raise ValueError(f"{name} mask must have boolean shape [N]")
    if not np.isfinite(tokens).all() or np.any(tokens[~mask] != 0.0):
        raise ValueError(f"{name} tokens are nonfinite or have nonzero padding")


def _wrap_phi(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _assign_group(
    hlt_tokens: np.ndarray,
    offline_tokens: np.ndarray,
    hlt_indices: np.ndarray,
    offline_indices: np.ndarray,
    *,
    charged: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(hlt_indices) or not len(offline_indices):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise ImportError(
            "PRAD Hungarian matching requires scipy.optimize"
        ) from error
    hlt = hlt_tokens[hlt_indices]
    offline = offline_tokens[offline_indices]
    deta = hlt[:, None, 1] - offline[None, :, 1]
    dphi = _wrap_phi(hlt[:, None, 2] - offline[None, :, 2])
    dr = np.hypot(deta, dphi)
    if charged:
        ratio = hlt[:, None, 0] / np.maximum(offline[None, :, 0], 1.0e-12)
        compatible = (
            (hlt[:, None, 4] == offline[None, :, 4])
            & (dr < 0.03)
            & (ratio > 0.5)
            & (ratio < 2.0)
        )
        cost = (dr / 0.01) ** 2 + (np.log(np.maximum(ratio, 1.0e-12)) / 0.2) ** 2
    else:
        ratio = hlt[:, None, 3] / np.maximum(offline[None, :, 3], 1.0e-12)
        compatible = (
            (dr < 0.06)
            & (ratio > 0.25)
            & (ratio < 4.0)
        )
        cost = (dr / 0.02) ** 2 + (np.log(np.maximum(ratio, 1.0e-12)) / 0.35) ** 2
    finite_cost = np.where(compatible, cost, 1.0e12)
    row, column = linear_sum_assignment(finite_cost)
    selected = compatible[row, column] & (cost[row, column] <= 9.0)
    return (
        hlt_indices[row[selected]],
        offline_indices[column[selected]],
        cost[row[selected], column[selected]],
    )


def match_hlt_to_offline(
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
) -> MatchResult:
    """Apply the registered charged/neutral fallback without source indices."""

    hlt_tokens = np.asarray(hlt_tokens)
    hlt_mask = np.asarray(hlt_mask)
    offline_tokens = np.asarray(offline_tokens)
    offline_mask = np.asarray(offline_mask)
    _validate_view(hlt_tokens, hlt_mask, name="HLT")
    _validate_view(offline_tokens, offline_mask, name="offline")
    hlt_categories = _categories(hlt_tokens, hlt_mask)
    offline_categories = _categories(offline_tokens, offline_mask)
    mapping = np.full((len(hlt_tokens),), -1, dtype=np.int64)
    costs = np.full((len(hlt_tokens),), np.inf, dtype=np.float32)
    for category in range(5):
        charged = category in CHARGED_CATEGORIES
        if not charged and category not in NEUTRAL_CATEGORIES:
            continue
        hlt_indices = np.flatnonzero(hlt_mask & (hlt_categories == category))
        offline_indices = np.flatnonzero(
            offline_mask & (offline_categories == category)
        )
        hlt_selected, offline_selected, selected_cost = _assign_group(
            hlt_tokens,
            offline_tokens,
            hlt_indices,
            offline_indices,
            charged=charged,
        )
        mapping[hlt_selected] = offline_selected
        costs[hlt_selected] = selected_cost.astype(np.float32)
    matched = mapping >= 0
    hlt_pt = float(np.sum(hlt_tokens[hlt_mask, 0], dtype=np.float64))
    matched_pt = float(np.sum(hlt_tokens[matched, 0], dtype=np.float64))
    diagnostics = {
        "contract": PRAD_MATCH_CONTRACT,
        "algorithm": PRAD_MATCH_ALGORITHM,
        "hlt_particles": int(np.sum(hlt_mask)),
        "offline_particles": int(np.sum(offline_mask)),
        "matched_particles": int(np.sum(matched)),
        "matched_particle_fraction": (
            float(np.sum(matched) / np.sum(hlt_mask)) if np.any(hlt_mask) else 0.0
        ),
        "matched_pt_fraction": matched_pt / hlt_pt if hlt_pt > 0.0 else 0.0,
        "direct_source_indices_used": False,
    }
    return MatchResult(
        hlt_to_offline=mapping,
        costs=costs,
        hlt_valid=np.array(hlt_mask, copy=True),
        offline_valid=np.array(offline_mask, copy=True),
        diagnostics=diagnostics,
    )


def pair_supervision_mask(match: MatchResult) -> np.ndarray:
    matched = match.hlt_valid & (match.hlt_to_offline >= 0)
    result = matched[:, None] & matched[None, :]
    np.fill_diagonal(result, False)
    return result


__all__ = ["MatchResult", "match_hlt_to_offline", "pair_supervision_mask"]
