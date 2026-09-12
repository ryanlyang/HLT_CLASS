"""Shared 17-feature Particle Transformer interface, with exact HLT isolation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import artifact
from .reader import Particles

FEATURE_NAMES = (
    "pt_log", "e_log", "logptrel", "logerel", "deltaR", "charge",
    "isChargedHadron", "isNeutralHadron", "isPhoton", "isElectron", "isMuon",
    "d0", "d0err", "dz", "dzerr", "deta", "dphi",
)


def input_contract(*, capacity: int) -> dict:
    if type(capacity) is not int or capacity < 16:
        raise ValueError("Input capacity must be an integer >=16")
    return artifact(
        "INPUTS", feature_names=list(FEATURE_NAMES), capacity=capacity, minimum_padding=16,
        truncation="forbidden", raw_p4="stored_px_py_pz_energy_float32",
        axis="sum_visible_p4_own_view", eta_reflection="sign_jet_eta_zero_positive",
        normalization="part_inputs_analytic_17_v1", epsilon=1e-8,
        impact_parameter_units="stored_mm_provisional", error_transform="clip_0_1_no_division",
        categorical_fields="exclusive_one_hot_and_charge", missing_quality_fields="omitted",
        forbidden_inputs=["row_identity", "label", "native_index", "source", "matching", "hlt_matched"],
        preprocessing_fit_roles=[], class_count=11,
    )


def eta_phi(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p4, np.float64)
    pt = np.hypot(p[..., 0], p[..., 1])
    return np.arcsinh(p[..., 2] / np.maximum(pt, 1e-8)), np.arctan2(p[..., 1], p[..., 0])


def wrap_phi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class ModelInputs:
    features: np.ndarray
    vectors: np.ndarray
    mask: np.ndarray

    def model_inputs(self) -> dict:
        return dict(features=self.features, vectors=self.vectors, mask=self.mask)


def build_inputs(particles: Particles, *, capacity: int) -> ModelInputs:
    input_contract(capacity=capacity)
    if len(particles) > capacity:
        raise ValueError(f"Particle capacity overflow: {len(particles)} > {capacity}; truncation forbidden")
    raw = particles.values.astype(np.float64)
    p4, aux = raw[:, :4], raw[:, 4:]
    summed = p4.sum(axis=0)
    jet_pt, jet_e = max(float(np.hypot(*summed[:2])), 1e-8), max(float(summed[3]), 1e-8)
    jet_eta, jet_phi = eta_phi(summed)
    eta, phi = eta_phi(p4)
    deta = (eta - jet_eta) * (-1 if jet_eta < 0 else 1)
    dphi = wrap_phi(phi - jet_phi)
    pt, energy = np.maximum(np.hypot(p4[:, 0], p4[:, 1]), 1e-8), np.maximum(p4[:, 3], 1e-8)
    values = np.column_stack((
        np.clip((np.log(pt) - 1.7) * .7, -5, 5),
        np.clip((np.log(energy) - 2) * .7, -5, 5),
        np.clip((np.log(pt / jet_pt) + 4.7) * .7, -5, 5),
        np.clip((np.log(energy / jet_e) + 4.7) * .7, -5, 5),
        np.clip((np.hypot(deta, dphi) - .2) * 4, -5, 5),
        aux[:, :6], np.tanh(aux[:, 6]), np.clip(aux[:, 7], 0, 1),
        np.tanh(aux[:, 8]), np.clip(aux[:, 9], 0, 1), deta, dphi,
    )).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite transformed particle features")
    features = np.zeros((17, capacity), np.float32)
    vectors = np.zeros((4, capacity), np.float32)
    mask = np.zeros((1, capacity), np.bool_)
    features[:, :len(raw)] = values.T
    vectors[:, :len(raw)] = particles.p4.T
    mask[:, :len(raw)] = True
    return ModelInputs(features, vectors, mask)
