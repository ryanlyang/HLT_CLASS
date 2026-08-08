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
REPRESENTATION_ARMS = ("R0", "R1", "R2", "R3", "R4_PAIR", "R4_GRAM", "R5")
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
    privileged_temperature: float | None = None

    def __post_init__(self) -> None:
        values = (self.ce, self.hlt_kd, self.privileged_kd)
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("PMARD loss coefficients must be finite and nonnegative")
        if not np.isclose(sum(values), 1.0, rtol=0, atol=1e-12):
            raise ValueError("PMARD loss coefficients must sum to one")
        temperatures = (
            self.temperature,
            self.temperature if self.privileged_temperature is None
            else self.privileged_temperature,
        )
        if self.arm.startswith("HCWDL_"):
            if any(not np.isfinite(value) or value <= 0 for value in temperatures):
                raise ValueError("HCWDL KD temperatures must be finite and positive")
        elif any(value not in TEMPERATURE_GRID for value in temperatures):
            raise ValueError("unknown PMARD KD temperature")

    @property
    def hlt_temperature(self) -> float:
        """The legacy/common temperature is the frozen HLT-teacher temperature."""

        return self.temperature

    @property
    def effective_privileged_temperature(self) -> float:
        return (
            self.temperature
            if self.privileged_temperature is None
            else self.privileged_temperature
        )

    @classmethod
    def for_arm(
        cls, arm: str, *, temperature: float,
        privileged_temperature: float | None = None,
    ) -> "LossConfiguration":
        if arm not in KD_MIXTURES or temperature not in TEMPERATURE_GRID:
            raise ValueError("unknown KD arm or unlocked temperature")
        return cls(
            arm, *KD_MIXTURES[arm], temperature,
            privileged_temperature=privileged_temperature,
        )

    @classmethod
    def for_mixture(
        cls, *, arm: str, ce: float, hlt_kd: float, privileged_kd: float,
        hlt_temperature: float, privileged_temperature: float,
    ) -> "LossConfiguration":
        if not arm or arm in KD_MIXTURES:
            raise ValueError("custom PMARD mixture requires a distinct nonempty arm")
        return cls(
            arm=arm, ce=ce, hlt_kd=hlt_kd,
            privileged_kd=privileged_kd, temperature=hlt_temperature,
            privileged_temperature=privileged_temperature,
        )


def derive_seed(master_seed: int, domain: str) -> int:
    if not domain:
        raise ValueError("seed domain must be nonempty")
    digest = hashlib.sha256(f"pmard/v1/{master_seed}/{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


MATCHER_FOLD_SEED = derive_seed(SCREENING_SEED, "matcher_folds")


def teacher_target_cache_enabled(
    *, max_rows_per_role: int | None, bounded_cache_miniature: bool,
) -> bool:
    """Use caches for production or for the one explicit bounded cache check."""

    if max_rows_per_role is not None and max_rows_per_role <= 0:
        raise ValueError("student role row bound must be positive")
    return max_rows_per_role is None or bool(bounded_cache_miniature)


def requires_privileged_training_views(
    *, representation_arm: str, representation_coefficient: float,
) -> bool:
    """Only positive-coefficient representation KD needs per-epoch aligned views."""

    if representation_arm not in REPRESENTATION_ARMS:
        raise ValueError("unknown representation arm")
    if representation_coefficient < 0:
        raise ValueError("representation coefficient must be nonnegative")
    return representation_arm != "R0" and representation_coefficient > 0


def generational_anchor_input_domain(
    *, alpha: float, native_offline: bool, has_initialization: bool,
) -> str:
    """Validate a companion's anchor path, including the HLT alpha-zero endpoint."""

    if native_offline:
        raise ValueError("generational anchor is unavailable to the native-offline oracle")
    if alpha == 0 and not has_initialization:
        raise ValueError("alpha-zero generational anchor requires HLT initialization")
    return "hlt_identity_endpoint" if alpha == 0 else "hlt_anchor_on_repaired_companion"


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
    student_fp32 = student_logits.float()
    labels = labels.long()
    weights = class_weights.to(device=student_fp32.device, dtype=torch.float32)[labels]
    ce_rows = functional.cross_entropy(student_fp32, labels, reduction="none")
    def kd_rows(teacher, *, tau):
        if teacher is None:
            raise ValueError("required frozen teacher logits are absent")
        if teacher.shape != student_logits.shape or teacher.requires_grad:
            raise ValueError("teacher logits must be shape-matched and detached")
        teacher_fp32 = teacher.float()
        target = functional.softmax(teacher_fp32 / tau, dim=-1)
        return functional.kl_div(
            functional.log_softmax(student_fp32 / tau, dim=-1), target,
            reduction="none",
        ).sum(-1) * tau * tau

    zero = student_fp32.sum(dim=-1) * 0
    hlt_rows = (
        kd_rows(hlt_teacher_logits, tau=configuration.hlt_temperature)
        if configuration.hlt_kd else zero
    )
    privileged_rows = (
        kd_rows(
            privileged_teacher_logits,
            tau=configuration.effective_privileged_temperature,
        )
        if configuration.privileged_kd else zero
    )
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
    left = functional.normalize(student.float(), dim=-1)
    right = functional.normalize(teacher.float(), dim=-1)
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
    if student.ndim in {3, 4} and student.shape[-1] == student.shape[-2] and mask is not None:
        if teacher.requires_grad or student.shape != teacher.shape:
            raise ValueError("pair representation target differs")
        pair_mask = mask[:, :, None] & mask[:, None, :]
        if student.ndim == 4:
            pair_mask = pair_mask[:, None].expand_as(student)
        if not pair_mask.any(): raise ValueError("pair representation mask is empty")
        return (student.float() - teacher.float()).square()[pair_mask].mean()
    return normalized_representation_loss(student, teacher, mask=mask)


def freeze_teacher(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = [
    "BOOTSTRAP_SEED", "CONFIRMATION_SEEDS", "KD_MIXTURES", "LossConfiguration",
    "MATCHER_FOLD_SEED", "REPRESENTATION_ARMS", "SCREENING_SEED", "TEMPERATURE_GRID", "derive_seed",
    "freeze_teacher", "generational_anchor_input_domain",
    "normalized_representation_loss", "pmard_loss",
    "representation_kd_loss", "requires_privileged_training_views",
    "sqrt_inverse_class_weights", "teacher_target_cache_enabled",
]
