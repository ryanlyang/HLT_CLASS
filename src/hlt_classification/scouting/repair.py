"""HLT-anchored P4_ONLY/v1 alpha repair with exact endpoint semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import numpy as np

from .inputs import ParticleInputs, build_hlt_inputs
from .matching import p4_kinematics, physical_p4_mask, wrapped_delta_phi
from .schema import HLT_FEATURE_SPECS, HLT_VECTOR_BRANCHES

ALPHA_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
REPAIR_FAMILY = "P4_ONLY/v1"
REPAIR_FAMILIES = (
    "P4_ONLY", "TRACK_ONLY", "P4_PLUS_TRACK", "DIRECTION_ONLY",
    "RESPONSE_ONLY", "WRONG_DIRECTION", "RANDOM_DIRECTION",
    "LOG_ANGULAR", "CONFIDENCE_WEIGHTED", "MATCH_SHUFFLED",
)
RECOMPUTED_CHANNELS = frozenset((7, 8, 9, 10, 19))
RETAINED_CHANNELS = frozenset(set(range(21)) - RECOMPUTED_CHANNELS)


def _rows(value: object) -> list[np.ndarray]:
    try:
        import awkward as ak
        if isinstance(value, ak.Array):
            return [np.asarray(row) for row in ak.to_list(value)]
    except ImportError:
        pass
    return [np.asarray(row) for row in value]  # type: ignore[arg-type]


def combined_offline_p4(
    charged: Mapping[str, object], neutral: Mapping[str, object], row: int,
) -> np.ndarray:
    def collection(arrays: Mapping[str, object], prefix: str) -> np.ndarray:
        columns = [_rows(arrays[f"{prefix}_{name}"])[row] for name in ("px", "py", "pz", "energy")]
        if len({len(item) for item in columns}) != 1:
            raise ValueError(f"{prefix} p4 collection lengths differ")
        return np.stack(columns, axis=1).astype(np.float32, copy=False)
    cpf = collection(charged, "cpfcandlt")
    npf = collection(neutral, "npfcand")
    return np.concatenate((cpf, npf), axis=0)


def build_alpha_repaired_inputs(
    hlt_arrays: Mapping[str, object], offline_p4_by_row: Sequence[np.ndarray],
    assignments: np.ndarray, *, alpha: float,
    repair_family: str = "P4_ONLY",
    confidence_weights: np.ndarray | None = None,
) -> ParticleInputs:
    if alpha not in ALPHA_GRID:
        raise ValueError(f"alpha must be one of {ALPHA_GRID}")
    if repair_family not in REPAIR_FAMILIES:
        raise ValueError("unknown repair family")
    if repair_family in {"TRACK_ONLY", "P4_PLUS_TRACK"}:
        raise PermissionError(
            "track repair is disabled until a locked branch-semantics compatibility audit exists"
        )
    canonical = build_hlt_inputs(hlt_arrays)
    if alpha == 0:
        return canonical
    mapping = np.asarray(assignments)
    rows = canonical.features.shape[0]
    if mapping.shape != (rows, canonical.features.shape[2]):
        raise ValueError("repair assignment shape differs")
    if len(offline_p4_by_row) != rows:
        raise ValueError("offline endpoint row count differs")
    confidence = None if confidence_weights is None else np.asarray(confidence_weights, np.float64)
    if repair_family == "CONFIDENCE_WEIGHTED":
        if confidence is None or confidence.shape != mapping.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence-weighted repair requires finite aligned probabilities")
    raw = {name: [row.copy() for row in _rows(value)] for name, value in hlt_arrays.items()}
    for row in range(rows):
        visible = min(int(canonical.raw_lengths[row]), canonical.features.shape[2])
        hlt_p4 = np.stack([raw[name][row][:visible] for name in HLT_VECTOR_BRANCHES], axis=1).astype(np.float64)
        offline = np.asarray(offline_p4_by_row[row], dtype=np.float64)
        if not physical_p4_mask(hlt_p4).all():
            raise ValueError(f"nonphysical HLT p4 endpoint in row {row}")
        offline_valid = physical_p4_mask(offline)
        repaired = hlt_p4.copy()
        accepted = mapping[row, :visible] >= 0
        for i in np.flatnonzero(accepted):
            j = int(mapping[row, i])
            if j >= len(offline) or not offline_valid[j]:
                raise ValueError(f"invalid offline repair endpoint in row {row}, token {i}")
            endpoint = offline[j].copy()
            if repair_family in {"DIRECTION_ONLY", "WRONG_DIRECTION", "RANDOM_DIRECTION", "RESPONSE_ONLY"}:
                h_pt, h_eta, h_phi, h_e = (item[0] for item in p4_kinematics(hlt_p4[i:i + 1]))
                o_pt, o_eta, o_phi, o_e = (item[0] for item in p4_kinematics(offline[j:j + 1]))
                if repair_family == "DIRECTION_ONLY":
                    magnitude = np.linalg.norm(hlt_p4[i, :3]); eta, phi, energy = o_eta, o_phi, h_e
                elif repair_family == "RESPONSE_ONLY":
                    magnitude = np.linalg.norm(offline[j, :3]); eta, phi, energy = h_eta, h_phi, o_e
                else:
                    magnitude = np.linalg.norm(offline[j, :3])
                    if repair_family == "WRONG_DIRECTION":
                        eta = h_eta - (o_eta - h_eta)
                        phi = h_phi - float(wrapped_delta_phi(o_phi, h_phi))
                    else:
                        dr = np.hypot(o_eta - h_eta, float(wrapped_delta_phi(o_phi, h_phi)))
                        theta = 2 * np.pi * (((row + 1) * 2654435761 + (i + 1) * 2246822519) % 2**32) / 2**32
                        eta = h_eta + dr * np.cos(theta); phi = h_phi + dr * np.sin(theta)
                    energy = o_e
                pt = magnitude / np.cosh(eta)
                endpoint = np.array(
                    [pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), energy],
                    dtype=np.float64,
                )
            effective_alpha = alpha * (confidence[row, i] if repair_family == "CONFIDENCE_WEIGHTED" else 1.0)
            if repair_family == "LOG_ANGULAR":
                h_pt, h_eta, h_phi, h_e = (item[0] for item in p4_kinematics(hlt_p4[i:i + 1]))
                o_pt, o_eta, o_phi, o_e = (item[0] for item in p4_kinematics(offline[j:j + 1]))
                pt = np.exp((1 - effective_alpha) * np.log(h_pt) + effective_alpha * np.log(o_pt))
                energy = np.exp((1 - effective_alpha) * np.log(h_e) + effective_alpha * np.log(o_e))
                eta = (1 - effective_alpha) * h_eta + effective_alpha * o_eta
                phi = h_phi + effective_alpha * float(wrapped_delta_phi(o_phi, h_phi))
                repaired[i] = np.array([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), energy])
            else:
                repaired[i] = (1.0 - effective_alpha) * hlt_p4[i] + effective_alpha * endpoint
        if not physical_p4_mask(repaired).all():
            raise ValueError(f"interpolated p4 became nonphysical in row {row}")
        hpt, heta, hphi, henergy = p4_kinematics(hlt_p4)
        rpt, reta, rphi, renergy = p4_kinematics(repaired)
        deltas = {
            "scoutpfcand_phirel": wrapped_delta_phi(rphi, hphi),
            "scoutpfcand_etarel": reta - heta,
            "scoutpfcand_abseta": np.abs(reta) - np.abs(heta),
            "scoutpfcand_pt_log": np.log(rpt / hpt),
            "scoutpfcand_e_log": np.log(renergy / henergy),
        }
        if any(not np.isfinite(value).all() for value in deltas.values()):
            raise ValueError(f"nonfinite repair delta in row {row}")
        for branch, delta in deltas.items():
            raw[branch][row][:visible] = raw[branch][row][:visible] + np.where(accepted, delta, 0)
        for channel, branch in enumerate(HLT_VECTOR_BRANCHES):
            raw[branch][row][:visible] = repaired[:, channel].astype(raw[branch][row].dtype, copy=False)
    result = build_hlt_inputs(raw)
    if not np.array_equal(result.mask, canonical.mask) or not np.array_equal(result.raw_lengths, canonical.raw_lengths):
        raise RuntimeError("alpha repair changed HLT token identity")
    for channel in RETAINED_CHANNELS:
        if not np.array_equal(result.features[:, channel], canonical.features[:, channel]):
            raise RuntimeError(f"repair changed retained channel {channel}")
    return result


__all__ = [
    "ALPHA_GRID", "RECOMPUTED_CHANNELS", "REPAIR_FAMILIES", "REPAIR_FAMILY", "RETAINED_CHANNELS",
    "build_alpha_repaired_inputs", "combined_offline_p4",
]
