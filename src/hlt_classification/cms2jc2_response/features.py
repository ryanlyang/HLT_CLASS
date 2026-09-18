"""Label-blind physical conditioning; fit-only preprocessing is explicit."""
from __future__ import annotations

import numpy as np

from .bridge import Particles, wrap_phi
from .association import distances
from .contracts import artifact, validate

BASIC_NAMES = ("category", "charge", "log_pt", "abs_eta", "log_energy", "d0", "dz",
               "log_d0err", "log_dzerr", "valid_d0", "valid_dz", "valid_d0err", "valid_dzerr")
NEIGHBOUR_NAMES = ("nearest_dr", "nearest_charged_dr", "count_002", "log_local_pt_002",
                   "count_005", "log_local_pt_005", "count_010", "log_local_pt_010",
                   "charged_fraction_005", "empty_005", "nearest_pt_ratio", "pt_fraction", "axis_dr",
                   "isolated", "no_charged_neighbour")
GROUP_NAMES = ("group_size", "n_charged_hadron", "n_neutral_hadron", "n_photon", "n_electron", "n_muon", "n_unknown")
EMISSION_NAMES = tuple(f"output_{i}_{name}" for i in range(2)
                       for name in ("category", "charge", "valid_state", "positive_mass"))
FEATURE_NAMES = BASIC_NAMES + NEIGHBOUR_NAMES + GROUP_NAMES + EMISSION_NAMES


def features(p: Particles) -> np.ndarray:
    n = len(p)
    output = np.zeros((n, len(FEATURE_NAMES)), np.float64)
    if not n:
        return output
    output[:, :5] = np.column_stack((p.category, p.charge, np.log(p.pt), abs(p.eta), np.log(p.p4[:, 3])))
    output[:, 5:7] = p.tracking[:, :2]
    output[:, 7:9] = np.where(p.valid[:, 2:], np.log(np.maximum(p.tracking[:, 2:], 1e-12)), 0)
    output[:, 9:13] = p.valid
    dr = distances(p); np.fill_diagonal(dr, np.inf)
    charged = p.charge != 0
    axis = p.p4.sum(axis=0); axis_pt = np.hypot(axis[0], axis[1])
    # A directionless constituent sum has a flagged finite coordinate convention.
    axis_eta = np.arcsinh(axis[2]/axis_pt) if axis_pt > 0 else 0.
    axis_phi = np.arctan2(axis[1], axis[0]) if axis_pt > 0 else 0.
    for i in range(n):
        near = sorted((j for j in range(n) if j != i), key=lambda j: (float(dr[i,j]), p.keys[j]))
        cnear = [j for j in near if charged[j]]
        row = [float(dr[i,near[0]]) if near else 0., float(dr[i,cnear[0]]) if cnear else 0.]
        for radius in (.02,.05,.1):
            keep = dr[i] < radius
            row += [float(keep.sum()), float(np.log1p(p.pt[keep].sum()))]
        keep = dr[i] < .05; local_pt = p.pt[keep].sum()
        row += [float(p.pt[keep & charged].sum()/local_pt) if local_pt > 0 else 0., float(local_pt == 0),
                float(p.pt[near[0]]/p.pt[i]) if near else 0., float(p.pt[i]/p.pt.sum()),
                float(np.hypot(p.eta[i]-axis_eta, wrap_phi(p.phi[i]-axis_phi))),
                float(not near), float(not cnear)]
        output[i,13:28] = row
        output[i,28] = 1
        output[i,29+int(p.category[i])] = 1
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite conditioning vector")
    return output


def group_features(p: Particles, indices: tuple[int, ...], precomputed: np.ndarray | None = None) -> np.ndarray:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Expected nonempty, nonoverlapping group")
    x = features(p) if precomputed is None else precomputed
    if len(indices) == 1:
        return x[indices[0]].copy()
    selected = p.take(indices); w = selected.pt/selected.pt.sum()
    row = np.average(x[list(indices)],axis=0,weights=w)
    vector = selected.p4.sum(axis=0); pt = np.hypot(vector[0],vector[1])
    composition = np.bincount(selected.category,minlength=6)
    row[0] = selected.category[0] if np.count_nonzero(composition) == 1 else 5
    row[1] = np.sign(selected.charge.sum())
    row[2:5] = (np.log(pt),abs(np.arcsinh(vector[2]/pt)),np.log(vector[3]))
    # Group measurements are only conditioning summaries, not an alleged
    # reconstructed track. Model targets remain real output measurements.
    for col,value_col in enumerate((5,6,7,8)):
        available = selected.valid[:,col]
        row[9+col] = int(available.any())
        row[value_col] = np.average(x[np.asarray(indices)[available],value_col],weights=w[available]) if available.any() else 0.
    row[28] = len(indices); row[29:35] = composition
    return row


def fit_preprocessing(x: np.ndarray, *, membership_hash: str) -> dict:
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Invalid fit-only predictor matrix")
    q = np.quantile(x, [.001,.25,.5,.75,.999], axis=0)
    scale = q[3]-q[1]
    scale[scale == 0] = 1.
    return artifact("PREPROCESSING", parents={"fit_location_membership": membership_hash},
                    feature_names=list(FEATURE_NAMES), center=q[2].tolist(), scale=scale.tolist(),
                    support_low=q[0].tolist(), support_high=q[4].tolist())


def transform(x: np.ndarray, preprocessing: dict, *, family: str):
    validate(preprocessing, "PREPROCESSING")
    if preprocessing["feature_names"] != list(FEATURE_NAMES) or family not in {"table", "smooth", "tree"}:
        raise ValueError("Predictor interface differs")
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid predictors")
    lo, hi = np.asarray(preprocessing["support_low"]), np.asarray(preprocessing["support_high"])
    clamped = (x < lo) | (x > hi)
    result = (np.clip(x,lo,hi)-preprocessing["center"])/preprocessing["scale"]
    if family == "table":
        result = result[:, :len(BASIC_NAMES)]
        clamped = clamped[:, :len(BASIC_NAMES)]
    return result, clamped
