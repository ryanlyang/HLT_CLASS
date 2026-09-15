"""Identity-safe, RAM-only mixtures of durable single-model probability banks."""
from __future__ import annotations

from fractions import Fraction

import numpy as np


def fraction(value: dict) -> Fraction:
    if set(value) != {"numerator", "denominator"}:
        raise ValueError("MT20 rational weight fields differ")
    result = Fraction(value["numerator"], value["denominator"])
    if result.numerator != value["numerator"] or result.denominator != value["denominator"]:
        raise ValueError("MT20 rational weight is not canonical")
    return result


def mix_probabilities(
    components: list[np.ndarray], teacher_registry: list[dict],
    *, kd_weight: Fraction = Fraction(4, 5),
) -> np.ndarray:
    """Accumulate normalized conditional teacher probabilities in float64.

    Registry weights are contributions to the complete objective and therefore
    sum to ``kd_weight``.  Dividing by that total produces the probability
    target used by a single forward-KL term with global KD coefficient 0.8.
    """
    if not components or len(components) != len(teacher_registry):
        raise ValueError("MT20 component/teacher registry coverage differs")
    weights = [fraction(row["loss_weight"]) for row in teacher_registry]
    if sum(weights, Fraction()) != kd_weight or any(value <= 0 for value in weights):
        raise ValueError("MT20 teacher loss contributions differ")
    shape = np.asarray(components[0]).shape
    if len(shape) != 2 or shape[1] != 11:
        raise ValueError("MT20 component shape differs")
    result = np.zeros(shape, dtype=np.float64)
    for values, loss_weight in zip(components, weights, strict=True):
        values = np.asarray(values)
        if (
            values.dtype != np.float32 or values.shape != shape
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(-1), 1., atol=2e-6, rtol=0)
        ):
            raise ValueError("MT20 probability component differs")
        result += values.astype(np.float64) * float(loss_weight / kd_weight)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("MT20 RAM mixture is invalid")
    result /= result.sum(axis=1, keepdims=True)
    output = np.ascontiguousarray(result, dtype=np.float32)
    if not np.allclose(output.sum(-1), 1., atol=2e-6, rtol=0):
        raise ValueError("MT20 RAM mixture normalization differs")
    return output


__all__ = ["fraction", "mix_probabilities"]
