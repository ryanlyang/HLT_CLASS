"""Canonical 17-feature Particle Transformer input construction.

Every jet axis and every derived kinematic quantity is reconstructed from the
supplied particle view.  No offline metadata or alternate view is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .schema import CLASS_LABELS, RAW_TOKEN_DIM

if TYPE_CHECKING:
    from .dataset import CacheBatch
    from .identity import JetIdentity
    from .root_reader import JetView


PART_INPUT_CONTRACT = "hlt_classification_part_inputs_v1"
PART_INPUT_SCHEMA_VERSION = 1
EPSILON = np.float32(1.0e-8)

POINT_NAMES = ("part_deta", "part_dphi")
FEATURE_NAMES = (
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
    "part_deta",
    "part_dphi",
)
VECTOR_NAMES = ("part_px", "part_py", "part_pz", "part_energy")
JET_FEATURE_NAMES = ("pt", "eta", "phi", "energy", "mass", "nparticles")


@dataclass(frozen=True)
class ParticleTransformerInputs:
    """Batched Weaver inputs with identity and label lineage."""

    points: np.ndarray
    features: np.ndarray
    lorentz_vectors: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    identities: tuple["JetIdentity | str", ...]
    jet_features: np.ndarray
    all_empty_rows_repaired: np.ndarray
    source_view: str

    def model_inputs(self) -> dict[str, np.ndarray]:
        return {
            "points": self.points,
            "features": self.features,
            "lorentz_vectors": self.lorentz_vectors,
            "mask": self.mask,
        }

    def contract_payload(self) -> dict[str, object]:
        return {
            "contract": PART_INPUT_CONTRACT,
            "schema_version": PART_INPUT_SCHEMA_VERSION,
            "source_view": self.source_view,
            "point_names": list(POINT_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "vector_names": list(VECTOR_NAMES),
            "jet_feature_names": list(JET_FEATURE_NAMES),
            "shapes": {
                "points": list(self.points.shape),
                "features": list(self.features.shape),
                "lorentz_vectors": list(self.lorentz_vectors.shape),
                "mask": list(self.mask.shape),
                "labels": list(self.labels.shape),
                "jet_features": list(self.jet_features.shape),
            },
            "all_empty_rows_repaired": np.flatnonzero(
                self.all_empty_rows_repaired
            ).tolist(),
        }


def wrap_phi(values: np.ndarray) -> np.ndarray:
    """Wrap angles to the half-open interval ``[-pi, pi)``."""

    return ((values + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _validate_raw_inputs(
    tokens: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    identities: tuple["JetIdentity | str", ...],
) -> tuple[int, int]:
    if not isinstance(tokens, np.ndarray) or tokens.dtype != np.float32:
        raise TypeError("tokens must be a float32 numpy array")
    if tokens.ndim != 3 or tokens.shape[2] != RAW_TOKEN_DIM:
        raise ValueError(
            f"tokens must have shape [batch, particles, {RAW_TOKEN_DIM}]"
        )
    batch_size, particles, _ = tokens.shape
    if particles < 1:
        raise ValueError("at least one particle slot is required")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("mask must be a bool numpy array")
    if mask.shape != (batch_size, particles):
        raise ValueError("mask must have shape [batch, particles]")
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int64:
        raise TypeError("labels must be an int64 numpy array")
    if labels.shape != (batch_size,):
        raise ValueError("labels must have shape [batch]")
    if np.any((labels < 0) | (labels >= len(CLASS_LABELS))):
        raise ValueError("labels contain a class outside the frozen class contract")
    if len(identities) != batch_size:
        raise ValueError("identity count must equal the batch size")
    if not np.isfinite(tokens).all():
        raise ValueError("tokens contain non-finite values")
    if np.any(tokens[~mask] != 0):
        raise ValueError("padded token values must be exactly zero")
    return batch_size, particles


def build_particle_transformer_inputs(
    tokens: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    identities: tuple["JetIdentity | str", ...],
    *,
    source_view: str,
) -> ParticleTransformerInputs:
    """Build the frozen standard Particle Transformer representation.

    Raw token order is the repository's 14-field schema:
    ``pt, eta, phi, energy, charge, five PID flags, d0, d0err, dz,
    dzerr``.
    """

    if not source_view or not source_view.strip():
        raise ValueError("source_view must be a non-empty explicit view identifier")
    batch_size, particles = _validate_raw_inputs(
        tokens, mask, labels, identities
    )

    work = tokens.copy()
    work_mask = mask.copy()
    empty = ~work_mask.any(axis=1)
    if np.any(empty):
        # Weaver cannot safely process an entirely padded sequence.  This
        # placeholder carries zero momentum and no semantic measurements.
        work[empty, 0, :] = 0.0
        work[empty, 0, 0] = EPSILON
        work[empty, 0, 3] = EPSILON
        work_mask[empty, 0] = True

    valid = work_mask.astype(np.float32)
    pt = work[:, :, 0]
    eta = work[:, :, 1]
    phi = work[:, :, 2]
    energy = work[:, :, 3]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    px *= valid
    py *= valid
    pz *= valid
    energy_masked = energy * valid

    jet_px = px.sum(axis=1)
    jet_py = py.sum(axis=1)
    jet_pz = pz.sum(axis=1)
    jet_energy = energy_masked.sum(axis=1)
    jet_pt = np.hypot(jet_px, jet_py).astype(np.float32)
    positive_axis = jet_pt > EPSILON
    jet_phi = np.zeros(batch_size, dtype=np.float32)
    jet_eta = np.zeros(batch_size, dtype=np.float32)
    jet_phi[positive_axis] = np.arctan2(
        jet_py[positive_axis], jet_px[positive_axis]
    )
    jet_eta[positive_axis] = np.arcsinh(
        jet_pz[positive_axis] / np.maximum(jet_pt[positive_axis], EPSILON)
    )
    jet_phi = wrap_phi(jet_phi)

    eta_sign = np.sign(jet_eta).astype(np.float32)
    eta_sign[eta_sign == 0] = 1.0
    deta = (eta - jet_eta[:, None]) * eta_sign[:, None]
    dphi = wrap_phi(phi - jet_phi[:, None])
    delta_r = np.hypot(deta, dphi).astype(np.float32)

    safe_pt = np.maximum(pt, EPSILON)
    safe_energy = np.maximum(energy, EPSILON)
    safe_jet_pt = np.maximum(jet_pt, EPSILON)
    safe_jet_energy = np.maximum(jet_energy, EPSILON)

    feature_columns = (
        np.clip((np.log(safe_pt) - 1.7) * 0.7, -5.0, 5.0),
        np.clip((np.log(safe_energy) - 2.0) * 0.7, -5.0, 5.0),
        np.clip(
            (np.log(safe_pt / safe_jet_pt[:, None]) + 4.7) * 0.7,
            -5.0,
            5.0,
        ),
        np.clip(
            (np.log(safe_energy / safe_jet_energy[:, None]) + 4.7) * 0.7,
            -5.0,
            5.0,
        ),
        np.clip((delta_r - 0.2) * 4.0, -5.0, 5.0),
        work[:, :, 4],
        work[:, :, 5],
        work[:, :, 6],
        work[:, :, 7],
        work[:, :, 8],
        work[:, :, 9],
        np.tanh(work[:, :, 10]),
        np.clip(work[:, :, 11], 0.0, 1.0),
        np.tanh(work[:, :, 12]),
        np.clip(work[:, :, 13], 0.0, 1.0),
        deta,
        dphi,
    )
    features = np.stack(feature_columns, axis=1).astype(np.float32)
    points = np.stack((deta, dphi), axis=1).astype(np.float32)
    vectors = np.stack((px, py, pz, energy_masked), axis=1).astype(np.float32)
    output_mask = work_mask[:, None, :]

    feature_valid = valid[:, None, :]
    points *= feature_valid
    features *= feature_valid
    vectors *= feature_valid

    momentum_squared = jet_px**2 + jet_py**2 + jet_pz**2
    jet_mass = np.sqrt(
        np.maximum(jet_energy**2 - momentum_squared, 0.0)
    ).astype(np.float32)
    jet_features = np.stack(
        (
            jet_pt,
            jet_eta,
            jet_phi,
            jet_energy.astype(np.float32),
            jet_mass,
            work_mask.sum(axis=1).astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)

    for name, array in (
        ("points", points),
        ("features", features),
        ("lorentz_vectors", vectors),
        ("jet_features", jet_features),
    ):
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise RuntimeError(f"constructed {name} violates the FP32 finite contract")

    return ParticleTransformerInputs(
        points=points,
        features=features,
        lorentz_vectors=vectors,
        mask=output_mask,
        labels=labels.copy(),
        identities=tuple(identities),
        jet_features=jet_features,
        all_empty_rows_repaired=empty.astype(np.bool_),
        source_view=source_view,
    )


def build_particle_transformer_inputs_from_view(
    view: "JetView",
    *,
    source_view: str,
) -> ParticleTransformerInputs:
    """Build inputs from one authenticated in-memory view."""

    return build_particle_transformer_inputs(
        view.tokens,
        view.mask,
        view.labels,
        view.identities,
        source_view=source_view,
    )


def build_particle_transformer_inputs_from_cache_batch(
    batch: "CacheBatch",
    *,
    source_view: str,
) -> ParticleTransformerInputs:
    """Build canonical inputs from a deployable bounded cache batch."""

    return build_particle_transformer_inputs(
        batch.tokens,
        batch.mask,
        batch.labels,
        tuple(batch.identity_keys),
        source_view=source_view,
    )
