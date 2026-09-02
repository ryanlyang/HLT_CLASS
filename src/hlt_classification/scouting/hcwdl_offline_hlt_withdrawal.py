"""Pure withdrawal schedules and losses shared by production and tests."""

from __future__ import annotations

import math
from typing import Any, Mapping


def validate_alpha_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    schedule = dict(value)
    kind = schedule.get("kind")
    if kind == "hold_cosine_zero_tail_v1":
        if set(schedule) != {
            "kind", "hold_through_pass", "decay_through_pass", "zero_from_pass",
        }:
            raise ValueError("cosine withdrawal schedule fields differ")
        hold = int(schedule["hold_through_pass"])
        decay = int(schedule["decay_through_pass"])
        zero = int(schedule["zero_from_pass"])
        if (hold, decay, zero) != (10, 60, 61):
            raise ValueError("cosine withdrawal schedule differs")
    elif kind == "step_to_zero_v1":
        if set(schedule) != {"kind", "hold_through_pass", "zero_from_pass"}:
            raise ValueError("step withdrawal schedule fields differ")
        if (
            int(schedule["hold_through_pass"]),
            int(schedule["zero_from_pass"]),
        ) != (60, 61):
            raise ValueError("step withdrawal schedule differs")
    else:
        raise ValueError("withdrawal schedule kind differs")
    return schedule


def alpha_for_effective_pass(
    schedule: Mapping[str, Any], *, effective_pass: float,
) -> float:
    schedule = validate_alpha_schedule(schedule)
    if not math.isfinite(effective_pass) or effective_pass <= 0:
        raise ValueError("withdrawal effective pass differs")
    hold = float(schedule["hold_through_pass"])
    if effective_pass <= hold:
        return 1.0
    if schedule["kind"] == "step_to_zero_v1":
        return 0.0
    decay = float(schedule["decay_through_pass"])
    if effective_pass >= float(schedule["zero_from_pass"]):
        return 0.0
    progress = min(1.0, max(0.0, (effective_pass - hold) / (decay - hold)))
    return .5 * (1.0 + math.cos(math.pi * progress))


def withdrawal_loss(
    output, labels, teacher_probabilities, *, temperature: float = 2.0,
):
    """Compute the registered dual-route loss in stable FP32."""

    import torch
    import torch.nn.functional as functional

    if temperature != 2.0:
        raise ValueError("withdrawal temperature differs")
    zero = output.zero.logits.float()
    privileged = output.privileged.logits.float()
    labels = labels.to(torch.long)
    teacher = teacher_probabilities.float()
    if teacher.shape != zero.shape or zero.shape != privileged.shape:
        raise ValueError("withdrawal logit/teacher shapes differ")

    def kd(logits):
        return functional.kl_div(
            functional.log_softmax(logits / temperature, dim=1),
            teacher, reduction="batchmean",
        ) * temperature * temperature

    ce_zero = functional.cross_entropy(zero, labels)
    kd_zero = kd(zero)
    if float(output.alpha) == 0.0:
        # At the exact endpoint both route coefficients collapse onto the one
        # deployable call; consistency terms are mathematically zero.
        ce_privileged = ce_zero
        kd_privileged = kd_zero
        logit = zero.new_zeros(())
        representation = zero.new_zeros(())
    else:
        ce_privileged = functional.cross_entropy(privileged, labels)
        kd_privileged = kd(privileged)
        with torch.no_grad():
            privileged_probability = functional.softmax(privileged, dim=1)
        logit = functional.kl_div(
            functional.log_softmax(zero, dim=1), privileged_probability,
            reduction="batchmean",
        )
        if (
            len(output.zero.hlt_states) != 4
            or len(output.privileged.hlt_states) != 4
            or not torch.equal(output.zero.hlt_mask, output.privileged.hlt_mask)
        ):
            raise ValueError("withdrawal representation surfaces differ")
        active = output.zero.hlt_mask[..., None]
        terms = []
        for left, right in zip(
            output.zero.hlt_states, output.privileged.hlt_states, strict=True,
        ):
            left = functional.layer_norm(left.float(), (left.shape[-1],))
            right = functional.layer_norm(right.float(), (right.shape[-1],))
            values = functional.smooth_l1_loss(
                left, right.detach(), reduction="none",
            )
            terms.append(
                (values * active).sum()
                / (active.sum().clamp_min(1) * values.shape[-1])
            )
        representation = torch.stack(terms).mean()
    total = (
        .25 * ce_zero + .30 * kd_zero
        + .15 * ce_privileged + .20 * kd_privileged
        + .05 * logit + .05 * representation
    )
    return {
        "total": total, "ce_zero": ce_zero, "kd_zero": kd_zero,
        "ce_privileged": ce_privileged, "kd_privileged": kd_privileged,
        "logit_consistency": logit,
        "representation_consistency": representation,
        "alpha": zero.new_tensor(float(output.alpha)),
    }


__all__ = [
    "alpha_for_effective_pass", "validate_alpha_schedule", "withdrawal_loss",
]
