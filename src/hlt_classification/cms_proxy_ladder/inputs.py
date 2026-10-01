"""Validity-aware 17-feature adapter for common physical particles."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.jetclass2_delphes.inputs import FEATURE_NAMES

from .contracts import artifact


def input_contract(*, capacity: int = 512) -> dict:
    if type(capacity) is not int or capacity < 16:
        raise ValueError("Input capacity must be an integer >=16")
    return artifact(
        "INPUTS",
        feature_names=list(FEATURE_NAMES), capacity=capacity, minimum_padding=16,
        truncation="forbidden_fail_before_training", raw_p4="physical_float64_to_model_float32",
        axis="sum_visible_p4_own_view", eta_reflection="sign_jet_eta_zero_positive",
        normalization="part_inputs_analytic_17_v1", epsilon=1e-8,
        impact_parameter_units="mm", error_transform="clip_0_1_no_division",
        invalid_tracking="zero_without_invented_measurement_or_significance",
        category="five_known_one_hot_unknown_all_zero", class_count=11,
        forbidden_inputs=[
            "row_identity", "label", "particle_key", "source", "matching",
            "pairing_validity", "degradation_index",
        ],
    )


def _eta_phi(p4: np.ndarray):
    pt = np.hypot(p4[..., 0], p4[..., 1])
    return np.arcsinh(p4[..., 2] / np.maximum(pt, 1e-8)), np.arctan2(p4[..., 1], p4[..., 0])


def _wrap_phi(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class RaggedInputs:
    features: np.ndarray
    vectors: np.ndarray


def build_inputs(particles: Particles, *, capacity: int = 512) -> RaggedInputs:
    input_contract(capacity=capacity)
    if len(particles) > capacity:
        raise ValueError(f"Particle capacity overflow: {len(particles)} > {capacity}; truncation forbidden")
    p4 = particles.p4
    summed = p4.sum(axis=0)
    jet_pt = max(float(np.hypot(*summed[:2])), 1e-8)
    jet_energy = max(float(summed[3]), 1e-8)
    jet_eta, jet_phi = _eta_phi(summed)
    eta, phi = _eta_phi(p4)
    deta = (eta - jet_eta) * (-1 if jet_eta < 0 else 1)
    dphi = _wrap_phi(phi - jet_phi)
    pt = np.maximum(particles.pt, 1e-8)
    energy = np.maximum(p4[:, 3], 1e-8)
    one_hot = np.zeros((len(particles), 5), np.float64)
    known = particles.category < 5
    one_hot[np.flatnonzero(known), particles.category[known]] = 1.
    tracking = np.where(particles.valid, particles.tracking, 0.)
    values = np.column_stack((
        np.clip((np.log(pt) - 1.7) * .7, -5, 5),
        np.clip((np.log(energy) - 2.) * .7, -5, 5),
        np.clip((np.log(pt / jet_pt) + 4.7) * .7, -5, 5),
        np.clip((np.log(energy / jet_energy) + 4.7) * .7, -5, 5),
        np.clip((np.hypot(deta, dphi) - .2) * 4., -5, 5),
        particles.charge.astype(np.float64), one_hot,
        np.tanh(tracking[:, 0]), np.clip(tracking[:, 2], 0, 1),
        np.tanh(tracking[:, 1]), np.clip(tracking[:, 3], 0, 1),
        deta, dphi,
    )).astype(np.float32)
    if values.shape != (len(particles), 17) or not np.isfinite(values).all():
        raise ValueError("Invalid transformed CMS-proxy particle inputs")
    return RaggedInputs(values, np.asarray(p4, np.float32))


__all__ = ["RaggedInputs", "build_inputs", "input_contract"]
