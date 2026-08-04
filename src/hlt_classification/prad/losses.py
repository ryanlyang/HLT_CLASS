"""Normalized teacher and student losses for PRAD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RelationLossResult:
    total: torch.Tensor
    bottleneck: torch.Tensor
    bias: torch.Tensor
    valid_pair_count: int
    weight_sum: float


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def relation_distillation_loss(
    *,
    student_relation: torch.Tensor,
    student_bias: torch.Tensor,
    teacher_relation: torch.Tensor,
    teacher_bias: torch.Tensor,
    pair_mask: torch.Tensor,
    teacher_true_class_confidence: torch.Tensor,
    use_bottleneck: bool = True,
    use_bias: bool = True,
) -> RelationLossResult:
    """Compute independently normalized, reliability-weighted Smooth L1."""

    if not use_bottleneck and not use_bias:
        raise ValueError("at least one PRAD relation loss component is required")
    if student_relation.shape != teacher_relation.shape:
        raise ValueError("student and teacher relation shapes differ")
    if student_bias.shape != teacher_bias.shape:
        raise ValueError("student and teacher bias shapes differ")
    if student_relation.ndim != 4 or student_bias.ndim != 4:
        raise ValueError("PRAD relation and bias tensors must be rank four")
    batch, particles, other_particles, _ = student_relation.shape
    if particles != other_particles:
        raise ValueError("PRAD relation matrix must be square")
    if pair_mask.shape != (batch, particles, particles):
        raise ValueError("PRAD pair loss mask shape differs")
    if teacher_true_class_confidence.shape != (batch,):
        raise ValueError("teacher confidence must have shape [B]")
    if not torch.isfinite(teacher_true_class_confidence).all():
        raise FloatingPointError("teacher confidence is nonfinite")
    mask = pair_mask.to(dtype=torch.bool)
    count = int(mask.sum().detach().cpu())
    if count == 0:
        zero = _zero(student_relation) + _zero(student_bias)
        return RelationLossResult(zero, zero, zero, 0, 0.0)
    jet_weight = teacher_true_class_confidence.clamp(0.1, 1.0)[
        :, None, None
    ]
    pair_weight = (
        1.0 + 2.0 * teacher_bias.detach().abs().mean(dim=1)
    ).clamp(1.0, 5.0)
    weights = jet_weight * pair_weight
    effective = weights * mask.to(weights.dtype)
    denominator = effective.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    if use_bottleneck:
        raw_relation = F.smooth_l1_loss(
            student_relation,
            teacher_relation.detach(),
            reduction="none",
        ).mean(dim=-1)
        bottleneck = (raw_relation * effective).sum() / denominator
    else:
        bottleneck = _zero(student_relation)
    if use_bias:
        raw_bias = F.smooth_l1_loss(
            student_bias,
            teacher_bias.detach(),
            reduction="none",
        ).mean(dim=1)
        bias = (raw_bias * effective).sum() / denominator
    else:
        bias = _zero(student_bias)
    if use_bottleneck and use_bias:
        total = 0.5 * bottleneck + 0.5 * bias
    else:
        total = bottleneck + bias
    if not torch.isfinite(total):
        raise FloatingPointError("PRAD relation loss is nonfinite")
    return RelationLossResult(
        total=total,
        bottleneck=bottleneck,
        bias=bias,
        valid_pair_count=count,
        weight_sum=float(denominator.detach().cpu()),
    )


def semantic_pair_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    """Normalize class-balanced binary semantic losses by valid components."""

    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("semantic logits, targets, and mask shapes differ")
    if logits.ndim != 4:
        raise ValueError("semantic pair tensors must have shape [B,N,N,C]")
    if positive_weights.shape != (logits.shape[-1],):
        raise ValueError("semantic positive-weight shape differs")
    mask = valid_mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        return _zero(logits)
    raw = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        pos_weight=positive_weights.to(device=logits.device, dtype=logits.dtype),
        reduction="none",
    )
    result = raw[mask].mean()
    if not torch.isfinite(result):
        raise FloatingPointError("PRAD semantic loss is nonfinite")
    return result


def temperature_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher class-logit shapes differ")
    if temperature <= 0.0:
        raise ValueError("KD temperature must be positive")
    teacher_probability = F.softmax(
        teacher_logits.detach() / temperature, dim=-1
    )
    student_log_probability = F.log_softmax(
        student_logits / temperature, dim=-1
    )
    result = (
        temperature**2
        * F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="batchmean",
        )
    )
    if not torch.isfinite(result):
        raise FloatingPointError("PRAD KD loss is nonfinite")
    return result


__all__ = [
    "RelationLossResult",
    "relation_distillation_loss",
    "semantic_pair_loss",
    "temperature_kl_loss",
]
