"""Deterministic track-dominant HLT-v3 controlled degradation proxy.

This module preserves the registered
``fixed_hlt_v3_track_dominant_proxy/v1`` scientific semantics. It is a
controlled HLT-like proxy, not genuine detector-HLT reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import hashlib
import inspect
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

from .replicas import (
    DOMAIN_SEEDS,
    RANDOM_MULTIPLIERS,
    REALIZATION_POLICIES,
    event_rng_seed,
)
from .schema import RAW_TOKEN_DIM

HLT_V3_PROFILE_NAME = "fixed_hlt_v3_track_dominant_proxy"
HLT_V3_PROFILE_VERSION = "v1"
HLT_V3_PROFILE_ID = f"{HLT_V3_PROFILE_NAME}/{HLT_V3_PROFILE_VERSION}"
HLT_V3_PROFILE_CONTRACT = "hlt_classification_hlt_v3_profile_v1"
HLT_V3_PROFILE_SCHEMA_VERSION = 1
RAW_DIM = RAW_TOKEN_DIM
INVALID_TRACK_SENTINEL = 0.0

PID_NAMES = (
    "charged_hadron",
    "neutral_hadron",
    "photon",
    "electron",
    "muon",
    "unknown",
)
CHARGED_PID = frozenset({0, 3, 4})
NEUTRAL_PID = frozenset({1, 2})

SUBSTREAM_IDS = {
    "merge": 11,
    "efficiency_quality": 21,
    "efficiency_loss": 22,
    "kinematic_quality": 31,
    "kinematic_core": 32,
    "kinematic_tail": 33,
    "angular_core": 34,
    "reassignment": 35,
    "pid_confusion": 41,
    "track_loss": 51,
    "track_error_scale": 52,
    "track_core": 53,
    "track_tail": 54,
    "charge_flip": 61,
}

OPERATION_ORDER = (
    "validate_offline_raw_schema",
    "mild_constituent_threshold",
    "type_aware_neutral_local_merging",
    "constituent_efficiency_loss",
    "mild_kinematic_response_and_local_reassignment",
    "pid_confusion",
    "charge_pid_consistency",
    "track_measurement_loss",
    "surviving_track_response",
    "rare_charge_flips",
    "mass_preserving_energy_recomputation",
    "zero_invalid_and_padded_rows",
    "stable_descending_pt_sort",
    "diagnostics_and_hashes",
)


@dataclass(frozen=True)
class _V2BaseParameters:
    hlt_pt_threshold: float
    merge_radius: float
    merge_probability: float
    eff_plateau_barrel: float
    eff_plateau_endcap: float
    eff_turnon_pt_barrel: float
    eff_turnon_pt_endcap: float
    eff_width_pt_barrel: float
    eff_width_pt_endcap: float
    density_loss_scale: float
    jet_quality_sigma: float
    smear_scale: float
    tail_probability_base: float
    tail_probability_eta: float
    tail_probability_density: float
    reassign_scale: float


@dataclass(frozen=True)
class HltV3Parameters:
    profile_name: str = HLT_V3_PROFILE_NAME
    profile_version: str = HLT_V3_PROFILE_VERSION
    hlt_pt_threshold: float = 0.10
    merge_radius: float = 0.0015
    merge_probability: float = 0.15
    eff_plateau_barrel: float = 0.9995
    eff_plateau_endcap: float = 0.9980
    eff_turnon_pt_barrel: float = 0.10
    eff_turnon_pt_endcap: float = 0.20
    eff_width_pt_barrel: float = 0.05
    eff_width_pt_endcap: float = 0.07
    density_loss_scale: float = 0.005
    jet_quality_sigma: float = 0.010
    kinematic_smear_scale: float = 0.080
    kinematic_tail_base: float = 0.0005
    kinematic_tail_eta: float = 0.0005
    kinematic_tail_density: float = 0.0005
    local_reassign_scale: float = 0.050

    def __post_init__(self) -> None:
        if self.profile_name != HLT_V3_PROFILE_NAME:
            raise ValueError("profile_name differs from registered HLT-v3 v1")
        if self.profile_version != HLT_V3_PROFILE_VERSION:
            raise ValueError("profile_version differs from registered HLT-v3 v1")
        for field in fields(self):
            if field.name in {"profile_name", "profile_version"}:
                continue
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{field.name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"{field.name} must be nonnegative")
        for name in ("eff_plateau_barrel", "eff_plateau_endcap", "merge_probability"):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must not exceed one")

    def v2_base_parameters(self) -> _V2BaseParameters:
        return _V2BaseParameters(
            hlt_pt_threshold=self.hlt_pt_threshold,
            merge_radius=self.merge_radius,
            merge_probability=self.merge_probability,
            eff_plateau_barrel=self.eff_plateau_barrel,
            eff_plateau_endcap=self.eff_plateau_endcap,
            eff_turnon_pt_barrel=self.eff_turnon_pt_barrel,
            eff_turnon_pt_endcap=self.eff_turnon_pt_endcap,
            eff_width_pt_barrel=self.eff_width_pt_barrel,
            eff_width_pt_endcap=self.eff_width_pt_endcap,
            density_loss_scale=self.density_loss_scale,
            jet_quality_sigma=self.jet_quality_sigma,
            smear_scale=self.kinematic_smear_scale,
            tail_probability_base=self.kinematic_tail_base,
            tail_probability_eta=self.kinematic_tail_eta,
            tail_probability_density=self.kinematic_tail_density,
            reassign_scale=self.local_reassign_scale,
        )


@dataclass(frozen=True)
class DegradationProfile:
    profile_id: str
    strength: float
    threshold: bool
    merging: bool
    constituent_loss: bool
    kinematic_response: bool
    reassignment: bool
    pid_confusion: bool
    track_loss: bool
    track_response: bool
    charge_flip: bool
    legacy_profile: str | None = None


def _profile(
    profile_id: str,
    strength: float,
    *,
    kinematics: bool,
    constituent_missing: bool,
    track_missing: bool,
    track_response: bool,
    pid_charge: bool,
) -> DegradationProfile:
    return DegradationProfile(
        profile_id=profile_id,
        strength=strength,
        threshold=constituent_missing,
        merging=constituent_missing,
        constituent_loss=constituent_missing,
        kinematic_response=kinematics,
        reassignment=kinematics,
        pid_confusion=pid_charge,
        track_loss=track_missing,
        track_response=track_response,
        charge_flip=pid_charge,
    )


DEGRADATION_PROFILES = {
    "D_OFFLINE_IDENTITY": _profile(
        "D_OFFLINE_IDENTITY",
        0.0,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_KIN_ONLY": _profile(
        "D_KIN_ONLY",
        1.0,
        kinematics=True,
        constituent_missing=True,
        track_missing=False,
        track_response=False,
        pid_charge=False,
    ),
    "D_TRACK_ONLY": _profile(
        "D_TRACK_ONLY",
        1.0,
        kinematics=False,
        constituent_missing=False,
        track_missing=True,
        track_response=True,
        pid_charge=False,
    ),
    "D_MISSING_ONLY": _profile(
        "D_MISSING_ONLY",
        1.0,
        kinematics=False,
        constituent_missing=True,
        track_missing=True,
        track_response=False,
        pid_charge=False,
    ),
    "D_NOMINAL": _profile(
        "D_NOMINAL",
        1.0,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_MILD": _profile(
        "D_MILD",
        0.5,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_SEVERE": _profile(
        "D_SEVERE",
        1.5,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_LEGACY_V1": DegradationProfile(
        "D_LEGACY_V1", 0.6, False, False, False, False, False,
        False, False, False, False, "fixed_hlt_v1",
    ),
    "D_LEGACY_V2": DegradationProfile(
        "D_LEGACY_V2", 1.0, False, False, False, False, False,
        False, False, False, False, "fixed_hlt_v2_realistic",
    ),
}

TYPE_MULTIPLIERS = {
    "charged_hadron": (0.50, 0.45, 0.35, 0.00),
    "electron": (0.60, 0.55, 0.45, 0.00),
    "muon": (0.40, 0.35, 0.30, 0.00),
    "photon": (0.90, 0.85, 0.90, 1.00),
    "neutral_hadron": (1.35, 1.30, 1.25, 1.50),
    "unknown": (1.00, 1.00, 1.00, 1.00),
}

PID_TRANSITIONS = {
    0: ((3, 0.002), (4, 0.002)),
    3: ((0, 0.010), (4, 0.001)),
    4: ((0, 0.010), (3, 0.001)),
    1: ((2, 0.010),),
    2: ((1, 0.010),),
    5: (),
}


def degradation_profile(profile_id: str) -> DegradationProfile:
    try:
        profile = DEGRADATION_PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"unknown degradation profile {profile_id!r}") from error
    if profile.legacy_profile is not None:
        raise ValueError(
            f"{profile_id} is a comparison-only legacy profile, not an HLT-v3 mode"
        )
    return profile


def wrap_phi_np(phi: np.ndarray) -> np.ndarray:
    """Wrap angles to the half-open interval [-pi, pi)."""

    return (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi


def compute_local_density_np(
    eta: np.ndarray,
    phi: np.ndarray,
    valid_idx: np.ndarray | None = None,
    radius: float = 0.04,
) -> np.ndarray:
    """Count neighboring constituents in a fixed local delta-R cone."""

    eta_values = np.asarray(eta, dtype=np.float64)
    phi_values = np.asarray(phi, dtype=np.float64)
    if eta_values.shape != phi_values.shape or eta_values.ndim != 1:
        raise ValueError("eta and phi must be equal-length one-dimensional arrays")
    if valid_idx is None:
        valid_idx = np.arange(len(eta_values), dtype=np.int64)
    indices = np.asarray(valid_idx, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(eta_values)):
        raise IndexError("local-density valid index is out of range")
    density = np.zeros((len(eta_values),), dtype=np.float32)
    if len(indices) <= 1:
        return density
    selected_eta = eta_values[indices]
    selected_phi = phi_values[indices]
    for local_index, global_index in enumerate(indices):
        deta = selected_eta[local_index] - selected_eta
        dphi = wrap_phi_np(selected_phi[local_index] - selected_phi)
        distance = np.sqrt(deta * deta + dphi * dphi)
        density[global_index] = float(
            np.count_nonzero((distance < radius) & (distance > 0.0))
        )
    return density


def fixed_hlt_v2_efficiency_base_terms(
    *,
    pt: np.ndarray,
    eta: np.ndarray,
    density: np.ndarray,
    params: _V2BaseParameters,
    jet_quality: float,
) -> dict[str, np.ndarray]:
    """Exact deterministic v2 efficiency terms inherited by HLT-v3."""

    pt_values = np.maximum(np.asarray(pt), 1e-8)
    abs_eta = np.abs(np.asarray(eta))
    density_values = np.asarray(density)
    plateau = np.where(
        abs_eta < 1.5,
        float(params.eff_plateau_barrel),
        float(params.eff_plateau_endcap),
    )
    pt50 = np.where(
        abs_eta < 1.5,
        float(params.eff_turnon_pt_barrel),
        float(params.eff_turnon_pt_endcap),
    )
    width = np.where(
        abs_eta < 1.5,
        float(params.eff_width_pt_barrel),
        float(params.eff_width_pt_endcap),
    )
    positive_width = width > 0.0
    turn_on = np.ones_like(pt_values, dtype=np.float64)
    if np.any(positive_width):
        turn_on[positive_width] = 1.0 / (
            1.0
            + np.exp(
                -(pt_values[positive_width] - pt50[positive_width])
                / np.maximum(width[positive_width], 1e-6)
            )
        )
    if np.any(~positive_width):
        turn_on[~positive_width] = (
            pt_values[~positive_width] >= pt50[~positive_width]
        ).astype(np.float64)
    density_term = np.exp(
        -float(max(0.0, params.density_loss_scale)) * density_values
    )
    quality = np.clip(float(jet_quality), 0.90, 1.06)
    efficiency = np.clip(
        np.asarray(plateau * turn_on * density_term * quality, dtype=np.float64),
        0.0,
        1.0,
    )
    return {
        "plateau": np.asarray(plateau, dtype=np.float64),
        "turn_on": turn_on,
        "density_term": np.asarray(density_term, dtype=np.float64),
        "efficiency": efficiency,
        "loss_probability": 1.0 - efficiency,
    }


def fixed_hlt_v2_kinematic_base_terms(
    *,
    pt: np.ndarray,
    eta: np.ndarray,
    density: np.ndarray,
    params: _V2BaseParameters,
    jet_quality: float,
) -> dict[str, np.ndarray]:
    """Exact v2 core, tail, angular, and reassignment base amplitudes."""

    pt_values = np.maximum(np.asarray(pt), 1e-8)
    abs_eta = np.abs(np.asarray(eta))
    density_values = np.asarray(density)
    smear_scale = float(max(0.0, params.smear_scale))
    quality = float(jet_quality)
    sigma_p = np.sqrt(
        ((0.35 * smear_scale) / np.sqrt(pt_values)) ** 2
        + (0.012 * smear_scale) ** 2
        + ((0.08 * smear_scale) / pt_values) ** 2
    )
    sigma_p = sigma_p * (1.0 + 0.08 * abs_eta) * quality
    sigma_p = np.clip(sigma_p, 0.0, 0.25)
    tail_probability = np.clip(
        float(max(0.0, params.tail_probability_base))
        + float(max(0.0, params.tail_probability_eta)) * abs_eta
        + float(max(0.0, params.tail_probability_density)) * density_values,
        0.0,
        0.25,
    )
    sigma_angle = (
        0.0008 * smear_scale + (0.010 * smear_scale) / np.sqrt(pt_values)
    ) * (1.0 + 0.08 * abs_eta) * quality
    reassignment = np.clip(
        (0.01 + 0.006 * density_values)
        * float(max(0.0, params.reassign_scale)),
        0.0,
        0.08,
    )
    return {
        "sigma_p": np.asarray(sigma_p, dtype=np.float64),
        "tail_probability": np.asarray(tail_probability, dtype=np.float64),
        "tail_sigma": np.asarray(
            2.5 * sigma_p + 0.015 * min(1.0, smear_scale),
            dtype=np.float64,
        ),
        "tail_mean": np.full_like(
            sigma_p,
            1.0 - 0.02 * min(1.0, smear_scale),
            dtype=np.float64,
        ),
        "sigma_eta": np.asarray(sigma_angle, dtype=np.float64),
        "sigma_phi": np.asarray(sigma_angle, dtype=np.float64),
        "reassignment_probability": np.asarray(reassignment, dtype=np.float64),
    }


def _pid_categories(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim not in {2, 3} or values.shape[-1] != RAW_DIM:
        raise ValueError("PID category input must end in the 14-field raw schema")
    if valid.shape != values.shape[:-1]:
        raise ValueError("PID category mask shape differs")
    flags = np.asarray(values[..., 5:10], dtype=np.float64)
    if np.any(np.abs(flags - np.rint(flags)) > 1.0e-6):
        raise ValueError("PID flags must be binary within tolerance")
    binary = np.rint(flags).astype(np.int8)
    if np.any(valid & np.any((binary != 0) & (binary != 1), axis=-1)):
        raise ValueError("PID flags must be exact binary values")
    counts = np.sum(binary, axis=-1)
    if np.any(counts[valid] > 1):
        raise ValueError("multi-hot PID input is invalid")
    categories = np.full(valid.shape, 5, dtype=np.int8)
    one_hot = counts == 1
    categories[one_hot] = np.argmax(binary[one_hot], axis=-1).astype(np.int8)
    return categories


def _validate_raw_tokens(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if tokens.ndim != 2 or tokens.shape[1] != RAW_DIM:
        raise ValueError(f"single-jet tokens must have shape [N,{RAW_DIM}]")
    if tokens.dtype != np.float32:
        raise ValueError("raw HLT-v3 tokens must have float32 dtype")
    if mask.shape != (tokens.shape[0],):
        raise ValueError("single-jet mask shape mismatch")
    valid = np.asarray(mask, dtype=bool)
    if not bool(np.isfinite(tokens).all()):
        raise ValueError("raw-token values, including padding, must be finite")
    if np.any(tokens[~valid] != 0):
        raise ValueError("masked raw-token padding rows must be exactly zero")
    if np.any(tokens[valid, 0] < 0.0) or np.any(tokens[valid, 3] < 0.0):
        raise ValueError("valid pT and energy must be nonnegative")
    charge = tokens[valid, 4]
    if len(charge) and np.any(
        np.min(np.abs(charge[:, None] - np.array([-1, 0, 1])), axis=1) > 1e-6
    ):
        raise ValueError("charge must be one of -1, 0, +1")
    categories = _pid_categories(tokens, valid)
    neutral = np.isin(categories, list(NEUTRAL_PID))
    if np.any(np.abs(tokens[neutral & valid, 4]) > 1e-6):
        raise ValueError("neutral PID rows must have zero charge")
    return categories


def measurement_validity_states(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Encode 0=not-track-domain, 1=available, 2=missing."""

    values = np.asarray(tokens)
    valid = np.asarray(mask, dtype=bool)
    categories = _pid_categories(values, valid)
    states = np.zeros(valid.shape, dtype=np.int8)
    charged = valid & np.isin(categories, list(CHARGED_PID))
    available = (
        charged
        & np.all(np.isfinite(values[..., 10:14]), axis=-1)
        & (values[..., 11] > 0.0)
        & (values[..., 13] > 0.0)
    )
    states[charged] = 2
    states[available] = 1
    return states


def _substream_seed(base_seed: int, family: str) -> int:
    if family not in SUBSTREAM_IDS:
        raise ValueError(f"unknown HLT-v3 random substream {family!r}")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, Integral)
        or base_seed < 0
    ):
        raise ValueError("base_seed must be a non-negative integer")
    base_seed = int(base_seed)
    digest = hashlib.sha256()
    digest.update(b"retb_hlt_v3_substream_v1\0")
    digest.update(base_seed.to_bytes(8, "big", signed=False))
    digest.update(b"\0")
    digest.update(str(SUBSTREAM_IDS[family]).encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _rng(base_seed: int, family: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(_substream_seed(base_seed, family)))


def _four_vector(token: np.ndarray) -> np.ndarray:
    pt, eta, phi, energy = (float(token[index]) for index in range(4))
    return np.array(
        [pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), energy],
        dtype=np.float64,
    )


def _mass_from_token(token: np.ndarray) -> float:
    px, py, pz, energy = _four_vector(token)
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def merge_equal_neutral_tokens(
    first: np.ndarray,
    second: np.ndarray,
    *,
    category: int,
) -> tuple[np.ndarray, float]:
    """Merge equal neutral PID categories by exact four-vector addition."""

    if category not in NEUTRAL_PID:
        raise ValueError("only neutral-hadron or photon rows may merge")
    vector = _four_vector(first) + _four_vector(second)
    px, py, pz, energy = (float(value) for value in vector)
    pt = math.hypot(px, py)
    phi = math.atan2(py, px)
    eta = math.asinh(pz / max(pt, 1.0e-12)) if pt > 0.0 else 0.0
    if not all(math.isfinite(value) for value in (pt, eta, phi, energy)):
        raise FloatingPointError("merged four-vector is nonfinite")
    mass = math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))
    output = np.zeros((RAW_DIM,), dtype=np.float64)
    output[:4] = (pt, eta, phi, max(energy, 0.0))
    output[4] = 0.0
    output[5 + category] = 1.0
    output[10:14] = INVALID_TRACK_SENTINEL
    return output, mass


def _replica_multipliers(
    policy: str,
    replica_id: int,
) -> tuple[float, float, float, float]:
    if policy not in {"R_FIXED", "R_MULTI", "R_RANDOM"}:
        raise ValueError(f"unknown realization policy {policy!r}")
    if isinstance(replica_id, bool) or not isinstance(replica_id, Integral):
        raise ValueError("replica_id must be an integer in [0,3]")
    replica_id = int(replica_id)
    if replica_id not in range(4):
        raise ValueError("replica_id must be in [0,3]")
    if policy != "R_RANDOM":
        return (1.0, 1.0, 1.0, 1.0)
    row = RANDOM_MULTIPLIERS[str(replica_id)]
    return (
        float(row["kinematic"]),
        float(row["track_loss"]),
        float(row["track_core_noise"]),
        float(row["tail_probability"]),
    )


def _clip01(value: np.ndarray | float) -> np.ndarray:
    return np.clip(value, 0.0, 1.0)


def scale_mechanism_terms(
    base_terms: Mapping[str, np.ndarray | float],
    *,
    pid_category: int,
    strength: float,
    replica_multipliers: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Apply the locked type, strength, and replica multiplication table."""

    if (
        isinstance(pid_category, bool)
        or not isinstance(pid_category, Integral)
        or int(pid_category) not in range(len(PID_NAMES))
    ):
        raise ValueError("pid_category lies outside the six-domain schema")
    pid_category = int(pid_category)
    if not math.isfinite(float(strength)) or float(strength) < 0.0:
        raise ValueError("degradation strength must be finite and nonnegative")
    if len(replica_multipliers) != 4:
        raise ValueError("four replica-family multipliers are required")
    r_kin, _r_track_loss, _r_track_core, r_tail = (
        float(value) for value in replica_multipliers
    )
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (r_kin, _r_track_loss, _r_track_core, r_tail)
    ):
        raise ValueError("replica-family multipliers must be finite and nonnegative")
    a_loss, a_momentum, a_angle, a_reassign = TYPE_MULTIPLIERS[
        PID_NAMES[pid_category]
    ]
    strength_value = float(strength)

    def array(name: str) -> np.ndarray:
        if name not in base_terms:
            raise KeyError(f"v2 base terms lack {name!r}")
        value = np.asarray(base_terms[name], dtype=np.float64)
        if not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"v2 base term {name!r} is nonfinite")
        return value

    output = {
        "reassignment_delta_scale": np.asarray(
            a_reassign * strength_value * r_kin, dtype=np.float64
        ),
        "kinematic_tail_delta_scale": np.asarray(
            a_momentum * strength_value * r_kin, dtype=np.float64
        ),
    }
    if "loss_probability" in base_terms:
        output["loss_probability"] = _clip01(
            array("loss_probability") * a_loss * strength_value * r_kin
        )
    if "sigma_p" in base_terms:
        output["sigma_p"] = np.minimum(
            array("sigma_p") * a_momentum * strength_value * r_kin, 0.25
        )
    if "tail_probability" in base_terms:
        output["kinematic_tail_probability"] = _clip01(
            array("tail_probability") * a_momentum * strength_value * r_tail
        )
    if "sigma_eta" in base_terms:
        output["sigma_eta"] = np.minimum(
            array("sigma_eta") * a_angle * strength_value * r_kin, 0.25
        )
    if "sigma_phi" in base_terms:
        output["sigma_phi"] = np.minimum(
            array("sigma_phi") * a_angle * strength_value * r_kin, 0.25
        )
    if "reassignment_probability" in base_terms:
        output["reassignment_probability"] = _clip01(
            array("reassignment_probability")
            * a_reassign
            * strength_value
            * r_kin
        )
    return output


def track_loss_probability(
    *,
    pt: np.ndarray,
    eta: np.ndarray,
    density: np.ndarray,
    strength: float,
    replica_multiplier: float = 1.0,
) -> np.ndarray:
    pt_values = np.asarray(pt, dtype=np.float64)
    eta_values = np.asarray(eta, dtype=np.float64)
    density_values = np.asarray(density, dtype=np.float64)
    if not all(
        bool(np.isfinite(values).all())
        for values in (pt_values, eta_values, density_values)
    ):
        raise ValueError("track-loss inputs must be finite")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in (strength, replica_multiplier)
    ):
        raise ValueError("track-loss scales must be finite and nonnegative")
    sigmoid = 1.0 / (1.0 + np.exp(-(0.80 - pt_values) / 0.25))
    base = np.clip(
        0.030
        + 0.030 * (np.abs(eta_values) >= 1.5)
        + 0.080 * sigmoid
        + 0.020 * np.minimum(density_values / 8.0, 1.0),
        0.0,
        0.35,
    )
    return _clip01(base * float(strength) * float(replica_multiplier))


def track_tail_probability(
    *,
    eta: np.ndarray,
    density: np.ndarray,
    strength: float,
    replica_multiplier: float = 1.0,
) -> np.ndarray:
    eta_values = np.asarray(eta, dtype=np.float64)
    density_values = np.asarray(density, dtype=np.float64)
    if not bool(np.isfinite(eta_values).all()) or not bool(
        np.isfinite(density_values).all()
    ):
        raise ValueError("track-tail inputs must be finite")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in (strength, replica_multiplier)
    ):
        raise ValueError("track-tail scales must be finite and nonnegative")
    base = np.clip(
        0.010
        + 0.005 * (np.abs(eta_values) >= 1.5)
        + 0.002 * np.minimum(density_values, 5.0),
        0.0,
        0.08,
    )
    return _clip01(base * float(strength) * float(replica_multiplier))


def charge_flip_probability(
    *,
    pt: np.ndarray,
    eta: np.ndarray,
    strength: float,
) -> np.ndarray:
    pt_values = np.asarray(pt, dtype=np.float64)
    eta_values = np.asarray(eta, dtype=np.float64)
    if not bool(np.isfinite(pt_values).all()) or not bool(
        np.isfinite(eta_values).all()
    ):
        raise ValueError("charge-flip inputs must be finite")
    if not math.isfinite(float(strength)) or float(strength) < 0.0:
        raise ValueError("charge-flip strength must be finite and nonnegative")
    base = np.clip(
        0.002
        + 0.002 * (np.abs(eta_values) >= 1.5)
        + 0.001 * np.minimum(pt_values / 100.0, 1.0),
        0.0,
        0.01,
    )
    return _clip01(base * float(strength))


def _stable_pt_order(
    tokens: np.ndarray,
    canonical_indices: np.ndarray,
) -> np.ndarray:
    return np.lexsort((canonical_indices, -tokens[:, 0]))


def apply_hlt_v3_single_jet(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identity: str,
    logical_role: str,
    replica_id: int,
    realization_policy: str = "R_MULTI",
    profile_id: str = "D_NOMINAL",
    parameters: HltV3Parameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Degrade one jet using identity-addressed independent random streams."""

    parameters = parameters or HltV3Parameters()
    profile = degradation_profile(profile_id)
    input_tokens = np.asarray(tokens)
    raw_mask = np.asarray(mask)
    if raw_mask.dtype != np.bool_:
        raise ValueError("raw HLT-v3 mask must have boolean dtype")
    input_mask = raw_mask
    if not isinstance(canonical_identity, str) or not canonical_identity:
        raise ValueError("canonical identity must be a nonempty string")
    if logical_role not in DOMAIN_SEEDS:
        raise ValueError(f"unknown logical role {logical_role!r}")
    if realization_policy not in REALIZATION_POLICIES:
        raise ValueError(f"unknown realization policy {realization_policy!r}")
    replica_multipliers = _replica_multipliers(realization_policy, replica_id)
    categories = _validate_raw_tokens(input_tokens, input_mask)
    strength = float(profile.strength)

    # This short circuit is part of the registered scientific contract. Do not
    # even derive a seed or construct a Generator on the identity path.
    if strength == 0.0:
        copied_tokens = np.array(input_tokens, copy=True)
        copied_mask = np.array(input_mask, copy=True)
        states = measurement_validity_states(copied_tokens, copied_mask)
        return copied_tokens, copied_mask, states, {
            "profile_id": profile_id,
            "strength": 0.0,
            "identity_short_circuit": True,
            "rng_constructed": False,
            "n_offline": int(np.sum(input_mask)),
            "n_output": int(np.sum(input_mask)),
            "operation_order": list(OPERATION_ORDER),
        }

    base_seed = event_rng_seed(
        logical_role=logical_role,
        replica_id=replica_id,
        canonical_identity=canonical_identity,
    )
    r_kin, r_track_loss, r_track_core, r_tail = replica_multipliers
    valid_indices = np.flatnonzero(input_mask)
    rows = np.asarray(input_tokens[valid_indices], dtype=np.float64).copy()
    row_categories = categories[valid_indices].astype(np.int8, copy=True)
    canonical_indices = valid_indices.astype(np.int64, copy=True)
    source_masses = np.array(
        [_mass_from_token(row) for row in rows], dtype=np.float64
    )
    original_density = compute_local_density_np(
        rows[:, 1],
        rows[:, 2],
        np.arange(len(rows), dtype=np.int64),
        radius=0.04,
    ).astype(np.float64)
    diagnostics: dict[str, Any] = {
        "profile_id": profile_id,
        "strength": strength,
        "identity_short_circuit": False,
        "rng_constructed": True,
        "n_offline": int(len(rows)),
        "operation_order": list(OPERATION_ORDER),
        "mechanism_counts": {
            "threshold_drop": 0,
            "merge": 0,
            "constituent_loss": 0,
            "reassignment": 0,
            "pid_transition": 0,
            "track_loss": 0,
            "track_tail": 0,
            "charge_flip": 0,
        },
        "probability_sums": {},
        "type_input_counts": {
            name: int(np.sum(row_categories == index))
            for index, name in enumerate(PID_NAMES)
        },
        "replica_multipliers": {
            "kinematic": r_kin,
            "track_loss": r_track_loss,
            "track_core": r_track_core,
            "tail": r_tail,
        },
    }
    if len(rows) == 0:
        return (
            np.zeros_like(input_tokens),
            np.zeros_like(input_mask),
            np.zeros_like(input_mask, dtype=np.int8),
            {**diagnostics, "n_output": 0},
        )

    if profile.threshold:
        threshold = parameters.hlt_pt_threshold * strength * r_kin
        keep = rows[:, 0] >= threshold
        diagnostics["mechanism_counts"]["threshold_drop"] = int(np.sum(~keep))
        rows = rows[keep]
        row_categories = row_categories[keep]
        canonical_indices = canonical_indices[keep]
        source_masses = source_masses[keep]
        original_density = original_density[keep]

    if profile.merging and len(rows) > 1:
        merge_rng = _rng(base_seed, "merge")
        radius = parameters.merge_radius * strength * r_kin
        probability = float(
            _clip01(parameters.merge_probability * strength * r_kin)
        )
        removed: set[int] = set()
        for index in range(len(rows)):
            if index in removed or row_categories[index] not in NEUTRAL_PID:
                continue
            for other in range(index + 1, len(rows)):
                if (
                    other in removed
                    or row_categories[other] != row_categories[index]
                ):
                    continue
                deta = rows[index, 1] - rows[other, 1]
                dphi = float(
                    wrap_phi_np(
                        np.array([rows[index, 2] - rows[other, 2]])
                    )[0]
                )
                if math.hypot(deta, dphi) >= radius:
                    continue
                if merge_rng.random() >= probability:
                    continue
                rows[index], source_masses[index] = merge_equal_neutral_tokens(
                    rows[index],
                    rows[other],
                    category=int(row_categories[index]),
                )
                canonical_indices[index] = min(
                    canonical_indices[index], canonical_indices[other]
                )
                original_density[index] = max(
                    original_density[index], original_density[other]
                )
                removed.add(other)
                diagnostics["mechanism_counts"]["merge"] += 1
        if removed:
            keep = np.array(
                [index not in removed for index in range(len(rows))],
                dtype=bool,
            )
            rows = rows[keep]
            row_categories = row_categories[keep]
            canonical_indices = canonical_indices[keep]
            source_masses = source_masses[keep]
            original_density = original_density[keep]

    if profile.constituent_loss and len(rows):
        quality = float(
            np.clip(
                _rng(base_seed, "efficiency_quality").lognormal(
                    mean=0.0,
                    sigma=parameters.jet_quality_sigma,
                ),
                0.75,
                1.35,
            )
        )
        base = fixed_hlt_v2_efficiency_base_terms(
            pt=rows[:, 0],
            eta=rows[:, 1],
            density=original_density,
            params=parameters.v2_base_parameters(),
            jet_quality=quality,
        )
        loss_probability = np.empty((len(rows),), dtype=np.float64)
        for category in range(len(PID_NAMES)):
            selected = row_categories == category
            if np.any(selected):
                scaled = scale_mechanism_terms(
                    {
                        "loss_probability": base["loss_probability"][selected]
                    },
                    pid_category=category,
                    strength=strength,
                    replica_multipliers=(
                        r_kin,
                        r_track_loss,
                        r_track_core,
                        r_tail,
                    ),
                )
                loss_probability[selected] = scaled["loss_probability"]
        diagnostics["probability_sums"]["constituent_loss"] = float(
            np.sum(loss_probability)
        )
        keep = (
            _rng(base_seed, "efficiency_loss").random(len(rows))
            >= loss_probability
        )
        diagnostics["mechanism_counts"]["constituent_loss"] = int(np.sum(~keep))
        rows = rows[keep]
        row_categories = row_categories[keep]
        canonical_indices = canonical_indices[keep]
        source_masses = source_masses[keep]
        original_density = original_density[keep]

    if profile.kinematic_response and len(rows):
        quality = float(
            np.clip(
                _rng(base_seed, "kinematic_quality").lognormal(
                    mean=0.0,
                    sigma=parameters.jet_quality_sigma,
                ),
                0.75,
                1.35,
            )
        )
        base = fixed_hlt_v2_kinematic_base_terms(
            pt=rows[:, 0],
            eta=rows[:, 1],
            density=original_density,
            params=parameters.v2_base_parameters(),
            jet_quality=quality,
        )
        sigma_p = np.empty((len(rows),), dtype=np.float64)
        tail_probability = np.empty((len(rows),), dtype=np.float64)
        tail_delta_scale = np.empty((len(rows),), dtype=np.float64)
        sigma_eta = np.empty((len(rows),), dtype=np.float64)
        sigma_phi = np.empty((len(rows),), dtype=np.float64)
        reassignment_probability = np.empty((len(rows),), dtype=np.float64)
        reassignment_delta_scale = np.empty((len(rows),), dtype=np.float64)
        for category in range(len(PID_NAMES)):
            selected = row_categories == category
            if not np.any(selected):
                continue
            scaled = scale_mechanism_terms(
                {name: value[selected] for name, value in base.items()},
                pid_category=category,
                strength=strength,
                replica_multipliers=(
                    r_kin,
                    r_track_loss,
                    r_track_core,
                    r_tail,
                ),
            )
            sigma_p[selected] = scaled["sigma_p"]
            tail_probability[selected] = scaled[
                "kinematic_tail_probability"
            ]
            tail_delta_scale[selected] = scaled[
                "kinematic_tail_delta_scale"
            ]
            sigma_eta[selected] = scaled["sigma_eta"]
            sigma_phi[selected] = scaled["sigma_phi"]
            reassignment_probability[selected] = scaled[
                "reassignment_probability"
            ]
            reassignment_delta_scale[selected] = scaled[
                "reassignment_delta_scale"
            ]

        core_rng = _rng(base_seed, "kinematic_core")
        log_ratio = core_rng.normal(size=len(rows)) * sigma_p
        tail_rng = _rng(base_seed, "kinematic_tail")
        tail_mask = tail_rng.random(len(rows)) < tail_probability
        tail_delta = (
            (base["tail_mean"] - 1.0)
            + base["tail_sigma"] * tail_rng.normal(size=len(rows))
        ) * tail_delta_scale
        log_ratio[tail_mask] = tail_delta[tail_mask]
        rows[:, 0] *= np.clip(np.exp(log_ratio), 0.55, 1.45)

        angular_rng = _rng(base_seed, "angular_core")
        rows[:, 1] += angular_rng.normal(size=len(rows)) * sigma_eta
        rows[:, 1] = np.clip(rows[:, 1], -5.0, 5.0)
        rows[:, 2] = wrap_phi_np(
            rows[:, 2] + angular_rng.normal(size=len(rows)) * sigma_phi
        )

        if profile.reassignment and len(rows) > 1:
            reassignment_rng = _rng(base_seed, "reassignment")
            selected_rows = (
                reassignment_rng.random(len(rows))
                < reassignment_probability
            )
            for index in np.flatnonzero(selected_rows):
                if row_categories[index] in CHARGED_PID:
                    raise AssertionError("charged rows may not be locally reassigned")
                deta = rows[index, 1] - rows[:, 1]
                dphi = wrap_phi_np(rows[index, 2] - rows[:, 2])
                distances = np.hypot(deta, dphi)
                distances[index] = np.inf
                nearest = int(np.argmin(distances))
                if distances[nearest] > 0.08:
                    continue
                amplitude = min(1.0, reassignment_delta_scale[index])
                fraction = reassignment_rng.uniform(0.20, 0.65) * amplitude
                rows[index, 1] = (
                    (1.0 - fraction) * rows[index, 1]
                    + fraction * rows[nearest, 1]
                )
                rows[index, 2] = math.atan2(
                    (1.0 - fraction) * math.sin(rows[index, 2])
                    + fraction * math.sin(rows[nearest, 2]),
                    (1.0 - fraction) * math.cos(rows[index, 2])
                    + fraction * math.cos(rows[nearest, 2]),
                )
                diagnostics["mechanism_counts"]["reassignment"] += 1

    if profile.pid_confusion and len(rows):
        pid_rng = _rng(base_seed, "pid_confusion")
        draws = pid_rng.random(len(rows))
        for index, category in enumerate(row_categories.copy()):
            cumulative = 0.0
            for target, probability in PID_TRANSITIONS[int(category)]:
                cumulative += probability * strength
                if draws[index] < min(cumulative, 1.0):
                    row_categories[index] = target
                    diagnostics["mechanism_counts"]["pid_transition"] += 1
                    break
        rows[:, 5:10] = 0.0
        known = row_categories < 5
        known_indices = np.flatnonzero(known)
        rows[known_indices, 5 + row_categories[known]] = 1.0

    # PID owns charge/track-domain consistency. Neutral rows never carry a
    # charge or an available track measurement.
    neutral = np.isin(row_categories, list(NEUTRAL_PID))
    rows[neutral, 4] = 0.0
    rows[neutral, 10:14] = INVALID_TRACK_SENTINEL
    track_states = measurement_validity_states(
        rows.astype(np.float32),
        np.ones((len(rows),), dtype=bool),
    )

    if profile.track_loss and len(rows):
        eligible = track_states == 1
        loss_probability = track_loss_probability(
            pt=rows[:, 0],
            eta=rows[:, 1],
            density=original_density,
            strength=strength,
            replica_multiplier=r_track_loss,
        )
        diagnostics["probability_sums"]["track_loss"] = float(
            np.sum(loss_probability[eligible])
        )
        lost = (
            _rng(base_seed, "track_loss").random(len(rows))
            < loss_probability
        ) & eligible
        rows[lost, 10:14] = INVALID_TRACK_SENTINEL
        diagnostics["mechanism_counts"]["track_loss"] = int(np.sum(lost))
        track_states[lost] = 2

    if profile.track_response and len(rows):
        surviving = track_states == 1
        if np.any(surviving):
            original_d0_error = rows[:, 11].copy()
            original_dz_error = rows[:, 13].copy()
            error_rng = _rng(base_seed, "track_error_scale")
            error_z = error_rng.normal(size=(len(rows), 2))
            displacement = strength * r_track_core
            rows[surviving, 11] *= np.exp(
                displacement
                * (math.log(1.35) + 0.15 * error_z[surviving, 0])
            )
            rows[surviving, 13] *= np.exp(
                displacement
                * (math.log(1.30) + 0.15 * error_z[surviving, 1])
            )

            core_rng = _rng(base_seed, "track_core")
            z0 = core_rng.normal(size=len(rows))
            z1 = core_rng.normal(size=len(rows))
            correlated = 0.25 * z0 + math.sqrt(1.0 - 0.25**2) * z1
            track_tail_probability_values = track_tail_probability(
                eta=rows[:, 1],
                density=original_density,
                strength=strength,
                replica_multiplier=r_tail,
            )
            tail = (
                _rng(base_seed, "track_tail").random(len(rows))
                < track_tail_probability_values
            ) & surviving
            tail_scale = np.where(tail, 4.0, 1.0)
            rows[surviving, 10] += (
                0.75
                * original_d0_error[surviving]
                * z0[surviving]
                * displacement
                * tail_scale[surviving]
            )
            rows[surviving, 12] += (
                0.65
                * original_dz_error[surviving]
                * correlated[surviving]
                * displacement
                * tail_scale[surviving]
            )
            diagnostics["mechanism_counts"]["track_tail"] = int(np.sum(tail))

    if profile.charge_flip and len(rows):
        eligible = np.isin(row_categories, list(CHARGED_PID)) & np.isin(
            np.rint(rows[:, 4]).astype(np.int8),
            [-1, 1],
        )
        flip_probability = charge_flip_probability(
            pt=rows[:, 0],
            eta=rows[:, 1],
            strength=strength,
        )
        flipped = (
            _rng(base_seed, "charge_flip").random(len(rows))
            < flip_probability
        ) & eligible
        rows[flipped, 4] *= -1.0
        diagnostics["mechanism_counts"]["charge_flip"] = int(np.sum(flipped))

    # Preserve each surviving/merged constituent mass while recomputing energy
    # from the degraded momentum.
    momentum = rows[:, 0] * np.cosh(rows[:, 1])
    rows[:, 3] = np.sqrt(
        np.maximum(momentum * momentum + source_masses**2, 0.0)
    )
    if not bool(np.isfinite(rows).all()):
        raise FloatingPointError("HLT-v3 output contains nonfinite values")

    order = _stable_pt_order(rows, canonical_indices)
    rows = rows[order]
    canonical_indices = canonical_indices[order]
    take = min(len(rows), input_tokens.shape[0])
    output = np.zeros(input_tokens.shape, dtype=np.float32)
    output_mask = np.zeros(input_mask.shape, dtype=bool)
    output[:take] = rows[:take].astype(np.float32)
    output_mask[:take] = True
    states = measurement_validity_states(output, output_mask)
    output_categories = _pid_categories(output, output_mask)
    diagnostics["n_output"] = int(take)
    diagnostics["canonical_output_indices"] = canonical_indices[:take].tolist()
    diagnostics["type_output_counts"] = {
        name: int(np.sum(output_categories[output_mask] == index))
        for index, name in enumerate(PID_NAMES)
    }
    diagnostics["measurement_states"] = {
        "not_track_domain": int(np.sum(states[output_mask] == 0)),
        "available": int(np.sum(states[output_mask] == 1)),
        "missing": int(np.sum(states[output_mask] == 2)),
    }
    return output, output_mask, states, diagnostics


def build_hlt_v3_view(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identities: Sequence[str],
    logical_role: str,
    replica_id: int,
    realization_policy: str = "R_MULTI",
    profile_id: str = "D_NOMINAL",
    parameters: HltV3Parameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build a batch without introducing batch-, worker-, or shard-level RNG."""

    input_tokens = np.asarray(tokens)
    raw_mask = np.asarray(mask)
    if raw_mask.dtype != np.bool_:
        raise ValueError("raw HLT-v3 batch mask must have boolean dtype")
    input_mask = raw_mask
    if input_tokens.ndim != 3 or input_tokens.shape[-1] != RAW_DIM:
        raise ValueError(f"tokens must have shape [B,N,{RAW_DIM}]")
    if input_tokens.dtype != np.float32:
        raise ValueError("raw HLT-v3 batch tokens must have float32 dtype")
    if input_mask.shape != input_tokens.shape[:2]:
        raise ValueError("batch mask shape mismatch")
    if len(canonical_identities) != len(input_tokens):
        raise ValueError("canonical identity count differs from batch")
    if any(
        not isinstance(identity, str) or not identity
        for identity in canonical_identities
    ):
        raise ValueError("every canonical identity must be a nonempty string")
    outputs = np.empty_like(input_tokens)
    output_masks = np.empty_like(input_mask)
    states = np.empty_like(input_mask, dtype=np.int8)
    diagnostics: list[dict[str, Any]] = []
    for index, identity in enumerate(canonical_identities):
        output, output_mask, state, diagnostic = apply_hlt_v3_single_jet(
            input_tokens[index],
            input_mask[index],
            canonical_identity=identity,
            logical_role=logical_role,
            replica_id=replica_id,
            realization_policy=realization_policy,
            profile_id=profile_id,
            parameters=parameters,
        )
        outputs[index] = output
        output_masks[index] = output_mask
        states[index] = state
        diagnostics.append(diagnostic)
    return outputs, output_masks, states, diagnostics


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _function_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _profile_payload(
    *,
    raw_input_schema_sha256: str,
    hlt_replica_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "contract": HLT_V3_PROFILE_CONTRACT,
        "schema_version": HLT_V3_PROFILE_SCHEMA_VERSION,
        "profile_name": HLT_V3_PROFILE_NAME,
        "profile_version": HLT_V3_PROFILE_VERSION,
        "profile_id": HLT_V3_PROFILE_ID,
        "proxy_claim": "HLT_like_controlled_proxy_not_real_HLT",
        "nominal_strength": 1.0,
        "parameters": asdict(HltV3Parameters()),
        "type_multiplier_order": [
            "constituent_loss",
            "momentum",
            "angular",
            "local_reassignment",
        ],
        "type_multipliers": {
            name: list(values) for name, values in TYPE_MULTIPLIERS.items()
        },
        "pid_transitions": {
            PID_NAMES[source]: [
                {
                    "target": PID_NAMES[target],
                    "probability": probability,
                }
                for target, probability in transitions
            ]
            for source, transitions in PID_TRANSITIONS.items()
        },
        "invalid_track_sentinel": [0.0, 0.0, 0.0, 0.0],
        "measurement_states": [
            "not_track_domain",
            "available",
            "missing",
        ],
        "operation_order": list(OPERATION_ORDER),
        "substream_ids": copy.deepcopy(SUBSTREAM_IDS),
        "random_domain_separators": {
            "event": "retb_hlt_v3_rng_v1",
            "substream": "retb_hlt_v3_substream_v1",
            "replica_cycle": "retb_replica_cycle_v1",
            "retained_for_registered_v1_parity": True,
        },
        "degradation_profiles": {
            name: asdict(profile)
            for name, profile in DEGRADATION_PROFILES.items()
        },
        "v2_base_term_helpers": {
            "efficiency": {
                "qualified_name": (
                    "hlt_classification.data.hlt_v3."
                    "fixed_hlt_v2_efficiency_base_terms"
                ),
                "source_sha256": _function_sha256(
                    fixed_hlt_v2_efficiency_base_terms
                ),
            },
            "kinematic": {
                "qualified_name": (
                    "hlt_classification.data.hlt_v3."
                    "fixed_hlt_v2_kinematic_base_terms"
                ),
                "source_sha256": _function_sha256(
                    fixed_hlt_v2_kinematic_base_terms
                ),
            },
        },
        "raw_input_schema_sha256": _require_sha256(
            raw_input_schema_sha256,
            name="raw_input_schema_sha256",
        ),
        "hlt_replica_manifest_sha256": _require_sha256(
            hlt_replica_manifest_sha256,
            name="hlt_replica_manifest_sha256",
        ),
        "fake_duplicate_split_constituents": False,
        "strength_zero_rng_constructed": False,
        "derived_features_rebuilt_from_degraded_view_only": True,
    }


def build_hlt_v3_profile_contract(
    *,
    raw_input_schema_sha256: str,
    hlt_replica_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the self-authenticating registered-v1 profile contract."""

    payload = _profile_payload(
        raw_input_schema_sha256=raw_input_schema_sha256,
        hlt_replica_manifest_sha256=hlt_replica_manifest_sha256,
    )
    payload["content_hash"] = _canonical_sha256(payload)
    return payload


def validate_hlt_v3_profile_contract(payload: Mapping[str, Any]) -> str:
    """Reject hash, source-helper, parent, or semantic drift."""

    if payload.get("contract") != HLT_V3_PROFILE_CONTRACT:
        raise ValueError("HLT-v3 profile contract mismatch")
    if payload.get("schema_version") != HLT_V3_PROFILE_SCHEMA_VERSION:
        raise ValueError("HLT-v3 profile schema version mismatch")
    supplied_hash = _require_sha256(payload.get("content_hash"), name="content_hash")
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    if _canonical_sha256(unhashed) != supplied_hash:
        raise ValueError("HLT-v3 profile content hash mismatch")
    raw_parent = _require_sha256(
        payload.get("raw_input_schema_sha256"),
        name="raw_input_schema_sha256",
    )
    replica_parent = _require_sha256(
        payload.get("hlt_replica_manifest_sha256"),
        name="hlt_replica_manifest_sha256",
    )
    expected = build_hlt_v3_profile_contract(
        raw_input_schema_sha256=raw_parent,
        hlt_replica_manifest_sha256=replica_parent,
    )
    if dict(payload) != expected:
        raise ValueError("HLT-v3 profile differs from the locked v1 contract")
    return supplied_hash


__all__ = [
    "DEGRADATION_PROFILES",
    "HLT_V3_PROFILE_CONTRACT",
    "HLT_V3_PROFILE_ID",
    "HLT_V3_PROFILE_NAME",
    "HLT_V3_PROFILE_VERSION",
    "HltV3Parameters",
    "INVALID_TRACK_SENTINEL",
    "OPERATION_ORDER",
    "PID_NAMES",
    "SUBSTREAM_IDS",
    "TYPE_MULTIPLIERS",
    "apply_hlt_v3_single_jet",
    "build_hlt_v3_profile_contract",
    "build_hlt_v3_view",
    "charge_flip_probability",
    "degradation_profile",
    "fixed_hlt_v2_efficiency_base_terms",
    "fixed_hlt_v2_kinematic_base_terms",
    "measurement_validity_states",
    "merge_equal_neutral_tokens",
    "scale_mechanism_terms",
    "track_loss_probability",
    "track_tail_probability",
    "validate_hlt_v3_profile_contract",
]
