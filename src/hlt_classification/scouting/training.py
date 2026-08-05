"""PMARD loss, weighting, seed, and exact-state training primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

KD_MIXTURES = MappingProxyType({
    "K0": (1.00, 0.00, 0.00),
    "K1": (0.25, 0.75, 0.00),
    "K2": (0.25, 0.60, 0.15),
    "K3": (0.25, 0.50, 0.25),
    "K4": (0.25, 0.00, 0.75),
    "K5": (0.10, 0.75, 0.15),
    "K6": (0.25, 0.60, 0.15),
})
TEMPERATURE_GRID = (1.0, 2.0, 4.0)
REPRESENTATION_ARMS = ("R0", "R1", "R2", "R3", "R4", "R5")
REPRESENTATION_COEFFICIENT = 0.10
SCREENING_SEED = 1337
CONFIRMATION_SEEDS = (11, 22, 33, 44, 55)
BOOTSTRAP_SEED = 8041


@dataclass(frozen=True)
class LossConfiguration:
    arm: str
    ce: float
    hlt_kd: float
    privileged_kd: float
    temperature: float

    @classmethod
    def for_arm(cls, arm: str, *, temperature: float) -> "LossConfiguration":
        if arm not in KD_MIXTURES or temperature not in TEMPERATURE_GRID:
            raise ValueError("unknown KD arm or unlocked temperature")
        return cls(arm, *KD_MIXTURES[arm], temperature)


def derive_seed(master_seed: int, domain: str) -> int:
    if not domain:
        raise ValueError("seed domain must be nonempty")
    digest = hashlib.sha256(f"pmard/v1/{master_seed}/{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def sqrt_inverse_class_weights(counts: Sequence[int]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (15,) or np.any(values <= 0):
        raise ValueError("PMARD class counts must contain 15 positive values")
    inverse = 1.0 / np.sqrt(values)
    kappa = values.sum() / np.sum(values * inverse)
    weights = kappa * inverse
    if not np.isclose(np.sum(values * weights) / values.sum(), 1.0, atol=1e-12):
        raise RuntimeError("class weights do not have unit population mean")
    return weights.astype(np.float32)


def pmard_loss(
    student_logits, labels, *, class_weights, configuration: LossConfiguration,
    hlt_teacher_logits=None, privileged_teacher_logits=None,
):
    """Compute exact per-row weighted CE and forward KL components."""
    import torch
    import torch.nn.functional as functional
    if student_logits.ndim != 2 or student_logits.shape[1] != 15:
        raise ValueError("student logits must be [batch,15]")
    labels = labels.long()
    weights = class_weights.to(student_logits)[labels]
    ce_rows = functional.cross_entropy(student_logits, labels, reduction="none")
    tau = configuration.temperature

    def kd_rows(teacher):
        if teacher is None:
            raise ValueError("required frozen teacher logits are absent")
        if teacher.shape != student_logits.shape or teacher.requires_grad:
            raise ValueError("teacher logits must be shape-matched and detached")
        target = functional.softmax(teacher / tau, dim=-1)
        return functional.kl_div(
            functional.log_softmax(student_logits / tau, dim=-1), target,
            reduction="none",
        ).sum(-1) * tau * tau

    zero = student_logits.sum(dim=-1) * 0
    hlt_rows = kd_rows(hlt_teacher_logits) if configuration.hlt_kd else zero
    privileged_rows = kd_rows(privileged_teacher_logits) if configuration.privileged_kd else zero
    components = {
        "ce": (weights * ce_rows).mean(),
        "hlt_kd": (weights * hlt_rows).mean(),
        "privileged_kd": (weights * privileged_rows).mean(),
    }
    total = (configuration.ce * components["ce"]
             + configuration.hlt_kd * components["hlt_kd"]
             + configuration.privileged_kd * components["privileged_kd"])
    if not torch.isfinite(total):
        raise FloatingPointError("PMARD loss is nonfinite")
    components["total"] = total
    return components


def normalized_representation_loss(student, teacher, *, mask=None, mode: str = "cosine_mse"):
    import torch
    import torch.nn.functional as functional
    if teacher.requires_grad or student.shape != teacher.shape:
        raise ValueError("representation target must be detached and shape-matched")
    left = functional.normalize(student, dim=-1)
    right = functional.normalize(teacher, dim=-1)
    rows = (1 - (left * right).sum(-1)) + (left - right).square().mean(-1)
    if mask is not None:
        valid = mask.bool()
        if valid.shape != rows.shape:
            raise ValueError("representation mask shape differs")
        if not valid.any():
            raise ValueError("representation mask has no valid elements")
        rows = rows[valid]
    result = rows.mean()
    if not torch.isfinite(result):
        raise FloatingPointError("representation loss is nonfinite")
    return result


def representation_kd_loss(student, teacher, *, mask=None):
    """Apply normalized loss to tensors, depth tuples, or pair-similarity matrices."""
    import torch
    if isinstance(student, tuple) or isinstance(teacher, tuple):
        if not isinstance(student, tuple) or not isinstance(teacher, tuple) or len(student) != len(teacher):
            raise ValueError("multi-depth representation structures differ")
        return torch.stack([
            normalized_representation_loss(left, right, mask=mask)
            for left, right in zip(student, teacher, strict=True)
        ]).mean()
    if student.ndim == 3 and student.shape[-1] == student.shape[-2] and mask is not None:
        if teacher.requires_grad or student.shape != teacher.shape:
            raise ValueError("pair representation target differs")
        pair_mask = mask[:, :, None] & mask[:, None, :]
        if not pair_mask.any(): raise ValueError("pair representation mask is empty")
        return (student - teacher).square()[pair_mask].mean()
    return normalized_representation_loss(student, teacher, mask=mask)


def freeze_teacher(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = [
    "BOOTSTRAP_SEED", "CONFIRMATION_SEEDS", "KD_MIXTURES", "LossConfiguration",
    "REPRESENTATION_ARMS", "SCREENING_SEED", "TEMPERATURE_GRID", "derive_seed",
    "freeze_teacher", "normalized_representation_loss", "pmard_loss",
    "representation_kd_loss", "sqrt_inverse_class_weights",
]
