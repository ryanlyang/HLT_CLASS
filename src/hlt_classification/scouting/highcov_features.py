"""Common physics and anchor-conditioned edge features."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .highcov_data import Particles


EPS = 1.0e-12


def wrap_phi(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def kinematics(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(p4, np.float64)
    px, py, pz, energy = value.T
    pt = np.hypot(px, py)
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0))
    phi = np.arctan2(py, px)
    return pt, eta, phi, energy


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="stable")
    rank = np.empty(len(order), np.float64)
    rank[order] = np.arange(len(order), dtype=np.float64)
    return rank / max(1, len(order) - 1)


@dataclass(frozen=True)
class EdgeMatrices:
    dr: np.ndarray
    deta: np.ndarray
    dphi: np.ndarray
    log_pt: np.ndarray
    log_energy: np.ndarray
    rank_delta: np.ndarray
    pid_transition: np.ndarray
    charge_transition: np.ndarray
    hpt: np.ndarray
    opt: np.ndarray
    henergy: np.ndarray
    oenergy: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.dr.shape


@dataclass(frozen=True)
class AnchorContext:
    hlt_index: np.ndarray
    offline_index: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    @property
    def count(self) -> int:
        return len(self.hlt_index)


def edge_matrices(hlt: Particles, offline: Particles) -> EdgeMatrices:
    hpt, heta, hphi, he = kinematics(hlt.p4)
    opt, oeta, ophi, oe = kinematics(offline.p4)
    if np.any(hpt <= 0) or np.any(opt <= 0) or np.any(he <= 0) or np.any(oe <= 0):
        raise ValueError("matcher requires positive pT and energy")
    deta = heta[:, None] - oeta[None, :]
    dphi = wrap_phi(hphi[:, None] - ophi[None, :])
    hpid = np.where(hlt.category < 0, 5, hlt.category).astype(np.int16)
    opid = np.where(offline.category < 0, 5, offline.category).astype(np.int16)
    return EdgeMatrices(
        dr=np.hypot(deta, dphi), deta=deta, dphi=dphi,
        log_pt=np.log(hpt[:, None] / opt[None, :]),
        log_energy=np.log(he[:, None] / oe[None, :]),
        rank_delta=percentile_rank(hpt)[:, None] - percentile_rank(opt)[None, :],
        pid_transition=hpid[:, None] * 6 + opid[None, :],
        charge_transition=(hlt.charge[:, None].astype(np.int16) + 1) * 3
        + (offline.charge[None, :].astype(np.int16) + 1),
        hpt=hpt, opt=opt, henergy=he, oenergy=oe,
    )


def mutual_anchor_context(
    matrices: EdgeMatrices, hlt: Particles, offline: Particles, *,
    max_dr: float = 0.0015, max_abs_log_response: float = 0.7,
) -> AnchorContext:
    nh, no = matrices.shape
    if nh == 0 or no == 0:
        empty = np.empty(0, np.int64)
        return AnchorContext(empty, empty.copy(), np.zeros(4), np.ones(4))
    hbest = np.argmin(matrices.dr, axis=1)
    obest = np.argmin(matrices.dr, axis=0)
    hi = np.arange(nh, dtype=np.int64); oi = hbest
    valid_identity = (
        (hlt.category[hi] >= 0) & (offline.category[oi] >= 0)
        & np.isin(hlt.charge[hi], (-1, 0, 1))
        & np.isin(offline.charge[oi], (-1, 0, 1))
    )
    keep = (
        (obest[oi] == hi) & valid_identity
        & (matrices.dr[hi, oi] <= max_dr)
        & (np.abs(matrices.log_pt[hi, oi]) <= max_abs_log_response)
        & (np.abs(matrices.log_energy[hi, oi]) <= max_abs_log_response)
    )
    hi, oi = hi[keep], oi[keep]
    if len(hi):
        values = np.column_stack((
            matrices.deta[hi, oi], matrices.dphi[hi, oi],
            matrices.log_pt[hi, oi], matrices.log_energy[hi, oi],
        ))
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0) * 1.4826
        floor = np.asarray((5e-4, 5e-4, .03, .03))
        scale = np.maximum(mad, floor)
    else:
        center = np.zeros(4); scale = np.asarray((.01, .01, .25, .30))
    return AnchorContext(hi, oi, center, scale)


def broad_gate(
    matrices: EdgeMatrices, hlt: Particles, offline: Particles, *,
    max_dr: float = 0.20, max_abs_log_response: float = 3.0,
) -> np.ndarray:
    valid_h = (hlt.category >= 0) & np.isin(hlt.charge, (-1, 0, 1))
    valid_o = (offline.category >= 0) & np.isin(offline.charge, (-1, 0, 1))
    return (
        (matrices.dr <= max_dr)
        & (np.abs(matrices.log_pt) <= max_abs_log_response)
        & (np.abs(matrices.log_energy) <= max_abs_log_response)
        & valid_h[:, None] & valid_o[None, :]
    )


def contextual_edge_features(
    matrices: EdgeMatrices, hlt: Particles, offline: Particles,
    context: AnchorContext, gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hi, oi = np.nonzero(gate)
    if not len(hi):
        return hi.astype(np.int32), oi.astype(np.int32), np.empty((0, 57), np.float32)
    raw = np.column_stack((
        matrices.deta[hi, oi], matrices.dphi[hi, oi],
        matrices.log_pt[hi, oi], matrices.log_energy[hi, oi],
    ))
    centered = raw - context.center
    normalized = centered / context.scale
    hcat = np.eye(5, dtype=np.float64)[hlt.category[hi]]
    ocat = np.eye(5, dtype=np.float64)[offline.category[oi]]
    hcharge = np.eye(3, dtype=np.float64)[hlt.charge[hi].astype(int) + 1]
    ocharge = np.eye(3, dtype=np.float64)[offline.charge[oi].astype(int) + 1]
    both_valid = hlt.track_valid[hi] & offline.track_valid[oi]
    scales = np.asarray((300.0, 1.0, 180.0, .9, .2, .2, 1.0))
    track_residual = np.clip((hlt.track[hi] - offline.track[oi]) * scales, -8, 8)
    track_residual[~both_valid] = 0
    row_degree = np.bincount(hi, minlength=len(hlt.p4))[hi]
    col_degree = np.bincount(oi, minlength=len(offline.p4))[oi]
    row_rank = np.empty(len(hi), np.float64); col_rank = np.empty(len(hi), np.float64)
    for h in np.unique(hi):
        rows = np.flatnonzero(hi == h)
        order = rows[np.argsort(matrices.dr[hi[rows], oi[rows]], kind="stable")]
        row_rank[order] = np.arange(len(order)) / max(1, len(order) - 1)
    for o in np.unique(oi):
        rows = np.flatnonzero(oi == o)
        order = rows[np.argsort(matrices.dr[hi[rows], oi[rows]], kind="stable")]
        col_rank[order] = np.arange(len(order)) / max(1, len(order) - 1)
    scalar = np.column_stack((
        matrices.dr[hi, oi], np.abs(matrices.log_pt[hi, oi]),
        np.abs(matrices.log_energy[hi, oi]), matrices.rank_delta[hi, oi],
        *raw.T, *centered.T, *normalized.T,
        np.log(np.maximum(matrices.hpt[hi], EPS)),
        np.log(np.maximum(matrices.opt[oi], EPS)),
        percentile_rank(matrices.hpt)[hi], percentile_rank(matrices.opt)[oi],
        row_rank, col_rank, np.log1p(row_degree), np.log1p(col_degree),
        np.full(len(hi), np.log1p(context.count)),
        (hlt.category[hi] == offline.category[oi]).astype(float),
        (hlt.charge[hi] == offline.charge[oi]).astype(float),
    ))
    features = np.column_stack((
        scalar, hcat, ocat, hcharge, ocharge,
        track_residual, both_valid.astype(np.float64),
    )).astype(np.float32)
    if features.shape[1] != 57:
        raise RuntimeError(f"contextual feature dimension changed: {features.shape[1]}")
    return hi.astype(np.int32), oi.astype(np.int32), features


def contextual_node_features(particles: Particles, degree: np.ndarray) -> np.ndarray:
    pt, eta, phi, energy = kinematics(particles.p4)
    categories = np.zeros((len(pt), 5), np.float64)
    valid = (particles.category >= 0) & (particles.category < 5)
    categories[np.flatnonzero(valid), particles.category[valid]] = 1
    scales = np.asarray((300.0, 1.0, 180.0, .9, .2, .2, 1.0))
    track = np.clip(particles.track * scales, -8, 8)
    track[~particles.track_valid] = 0
    density = np.zeros(len(pt), np.float64)
    for index in range(len(pt)):
        distance = np.hypot(eta - eta[index], wrap_phi(phi - phi[index]))
        density[index] = np.log1p(np.count_nonzero(distance < .10) - 1)
    result = np.column_stack((
        np.log(np.maximum(pt, EPS)), np.log(np.maximum(energy, EPS)), eta,
        np.sin(phi), np.cos(phi), categories, particles.charge.astype(np.float64),
        track, particles.track_valid.astype(np.float64), percentile_rank(pt), density,
        np.log1p(np.asarray(degree, np.float64)),
    )).astype(np.float32)
    if result.shape[1] != 28:
        raise RuntimeError(f"contextual node feature dimension changed: {result.shape[1]}")
    return result


def contextual_graph_features(
    matrices: EdgeMatrices, hlt: Particles, offline: Particles,
    context: AnchorContext, gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hi, oi, edges = contextual_edge_features(matrices, hlt, offline, context, gate)
    hdegree = np.bincount(hi, minlength=len(hlt.p4))
    odegree = np.bincount(oi, minlength=len(offline.p4))
    return (
        hi, oi, edges,
        contextual_node_features(hlt, hdegree),
        contextual_node_features(offline, odegree),
    )


__all__ = [
    "AnchorContext", "EdgeMatrices", "broad_gate", "contextual_edge_features",
    "contextual_graph_features", "contextual_node_features",
    "edge_matrices", "kinematics", "mutual_anchor_context", "percentile_rank", "wrap_phi",
]
