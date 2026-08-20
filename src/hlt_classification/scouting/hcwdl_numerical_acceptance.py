"""Executable numerical acceptance for HCWDL's frozen finite kernels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final, Literal

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_json_bytes,
    canonical_sha256,
    with_content_hash,
    write_immutable_json,
)

from .hcwdl_representation_kernels import (
    SpectralKernelResources,
    analytic_finite_mmd_gradient,
    cached_finite_mmd,
    generate_spectral_resources,
    ideal_rbf_mmd,
    slow_pairwise_finite_mmd,
    weighted_feature_mean,
)


NUMERICAL_ACCEPTANCE_CONTRACT: Final = "HCWDL_REPRESENTATION_NUMERICAL_ACCEPTANCE/v1"
NUMERICAL_FIXTURE_CONTRACT: Final = "HCWDL_REP_NUMERICAL_FIXTURE/v1"
TOKEN_VALUE_SEED: Final = 991
RELATION_VALUE_SEED: Final = 992
TOKEN_GRADIENT_SEED: Final = 993
RELATION_GRADIENT_SEED: Final = 994
ROTATION_SEED: Final = 995
VALUE_FIXTURES: Final = 512
GRADIENT_FIXTURES: Final = 64
ROTATION_FIXTURES: Final = 64


def _seed_payload(purpose: str, seed: int) -> dict[str, object]:
    return {"contract": NUMERICAL_FIXTURE_CONTRACT, "purpose": purpose, "seed": seed}


def _normalized_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise FloatingPointError("numerical token fixture contains a zero/nonfinite row")
    return value / norm


def _token_fixture(generator: np.random.Generator, *, maximum_count: int = 32):
    student_count = int(generator.integers(2, maximum_count + 1))
    teacher_count = int(generator.integers(2, maximum_count + 1))
    center = generator.standard_normal(128)
    center /= np.linalg.norm(center)
    student_scale = math.exp(generator.uniform(math.log(.02), math.log(.8)))
    teacher_scale = math.exp(generator.uniform(math.log(.02), math.log(.8)))
    student = _normalized_rows(
        center[None, :] + student_scale * generator.standard_normal((student_count, 128))
    )
    teacher = _normalized_rows(
        center[None, :] + teacher_scale * generator.standard_normal((teacher_count, 128))
    )
    student_weight = generator.uniform(0.0, 1.0, size=student_count)
    teacher_weight = generator.uniform(0.0, 1.0, size=teacher_count)
    student_weight /= student_weight.sum(dtype=np.float64)
    teacher_weight /= teacher_weight.sum(dtype=np.float64)
    return student, teacher, student_weight, teacher_weight


def _relation_fixture(generator: np.random.Generator, *, maximum_count: int = 496):
    student_count = int(generator.integers(2, maximum_count + 1))
    teacher_count = int(generator.integers(2, maximum_count + 1))
    center = generator.uniform(-.8, .8)
    student_scale = math.exp(generator.uniform(math.log(.01), math.log(.5)))
    teacher_scale = math.exp(generator.uniform(math.log(.01), math.log(.5)))
    student = np.clip(
        center + student_scale * generator.standard_normal(student_count), -1.0, 1.0,
    )[:, None]
    teacher = np.clip(
        center + teacher_scale * generator.standard_normal(teacher_count), -1.0, 1.0,
    )[:, None]
    student_weight = generator.uniform(0.0, 1.0, size=student_count)
    teacher_weight = generator.uniform(0.0, 1.0, size=teacher_count)
    student_weight /= student_weight.sum(dtype=np.float64)
    teacher_weight /= teacher_weight.sum(dtype=np.float64)
    return student, teacher, student_weight, teacher_weight


def _features_fp64(values: np.ndarray, resources: SpectralKernelResources) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if resources.kind == "relation":
        value = value.reshape(-1, 1)
    scale = math.sqrt(2.0 / resources.total_features)
    return np.concatenate([
        scale * np.cos(
            value @ block.omega.astype(np.float64).T
            + block.phase.astype(np.float64)[None, :]
        )
        for block in resources.blocks
    ], axis=1)


def _finite_values_fp64(student, teacher, student_weight, teacher_weight, resources):
    student_features = _features_fp64(student, resources)
    teacher_features = _features_fp64(teacher, resources)
    student_mean = np.einsum("i,if->f", student_weight, student_features, optimize=False)
    teacher_mean = np.einsum("i,if->f", teacher_weight, teacher_features, optimize=False)
    cached = float(np.square(student_mean - teacher_mean).sum(dtype=np.float64))
    # This is the exact pairwise finite-kernel contraction, evaluated without
    # materializing a potentially 496x496 matrix.  It deliberately uses a
    # distinct contraction/reduction path from the cached difference above.
    ss = float(np.einsum(
        "i,j,if,jf->", student_weight, student_weight,
        student_features, student_features, optimize=True,
    ))
    tt = float(np.einsum(
        "i,j,if,jf->", teacher_weight, teacher_weight,
        teacher_features, teacher_features, optimize=True,
    ))
    st = float(np.einsum(
        "i,j,if,jf->", student_weight, teacher_weight,
        student_features, teacher_features, optimize=True,
    ))
    return cached, ss + tt - 2.0 * st


def _finite_value_diagnostics(
    *, kind: Literal["token", "relation"], seed: int, fixtures: int,
) -> dict[str, object]:
    generator = np.random.Generator(np.random.PCG64(seed))
    resources = generate_spectral_resources(kind)
    fp64_disagreement: list[float] = []
    fp32_disagreement: list[float] = []
    finite_values: list[float] = []
    ideal_values: list[float] = []
    for _ in range(fixtures):
        row = (
            _token_fixture(generator)
            if kind == "token" else _relation_fixture(generator)
        )
        student, teacher, student_weight, teacher_weight = row
        cached64, pairwise64 = _finite_values_fp64(*row, resources)
        fp64_disagreement.append(abs(cached64 - pairwise64))
        target32 = weighted_feature_mean(
            teacher.astype(np.float32), teacher_weight.astype(np.float32), resources,
        ).detach()
        cached32 = cached_finite_mmd(
            student.astype(np.float32), student_weight.astype(np.float32),
            target32, resources,
        )
        pairwise32 = slow_pairwise_finite_mmd(
            student.astype(np.float32), student_weight.astype(np.float32),
            teacher.astype(np.float32), teacher_weight.astype(np.float32), resources,
        )
        fp32_disagreement.append(abs(float(cached32) - float(pairwise32)))
        finite_values.append(cached64)
        ideal_values.append(ideal_rbf_mmd(
            student, student_weight, teacher, teacher_weight, kind=kind,
        ))
    finite_array = np.asarray(finite_values, np.float64)
    ideal_array = np.asarray(ideal_values, np.float64)
    ideal_error = np.abs(finite_array - ideal_array)
    correlation = float(np.corrcoef(finite_array, ideal_array)[0, 1])
    thresholds = (
        {"mae": .015, "p95": .040, "correlation_minimum": .990}
        if kind == "token" else
        {"mae": .030, "p95": .100, "correlation_minimum": .985}
    )
    summary = {
        "fp64_pairwise_cached_absolute_errors": fp64_disagreement,
        "fp32_pairwise_cached_absolute_errors": fp32_disagreement,
        "finite_values": finite_values,
        "ideal_rbf_values": ideal_values,
        "ideal_absolute_errors": ideal_error.tolist(),
        "maximum_fp64_pairwise_cached_error": float(max(fp64_disagreement, default=0)),
        "maximum_fp32_pairwise_cached_error": float(max(fp32_disagreement, default=0)),
        "ideal_value_mae": float(ideal_error.mean()),
        "ideal_value_p95": float(np.quantile(ideal_error, .95)),
        "ideal_value_correlation": correlation,
        "thresholds": thresholds,
    }
    summary["passed"] = bool(
        summary["maximum_fp64_pairwise_cached_error"] <= 1.0e-10
        and summary["maximum_fp32_pairwise_cached_error"] <= 1.0e-5
        and summary["ideal_value_mae"] <= thresholds["mae"]
        and summary["ideal_value_p95"] <= thresholds["p95"]
        and correlation >= thresholds["correlation_minimum"]
    )
    return summary


def _gradient_diagnostics(
    *, kind: Literal["token", "relation"], seed: int, fixtures: int,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional

    generator = np.random.Generator(np.random.PCG64(seed))
    resources = generate_spectral_resources(kind)
    rows: list[dict[str, float | bool]] = []
    for _ in range(fixtures):
        row = (
            _token_fixture(generator, maximum_count=16)
            if kind == "token" else _relation_fixture(generator, maximum_count=16)
        )
        student, teacher, student_weight, teacher_weight = row
        leaf = torch.tensor(student, dtype=torch.float64, requires_grad=True)
        if kind == "token":
            runtime_student = functional.normalize(leaf.float(), dim=-1, eps=1.0e-12)
            runtime_teacher = functional.normalize(
                torch.tensor(teacher, dtype=torch.float32), dim=-1, eps=1.0e-12,
            )
        else:
            runtime_student = leaf.float()
            runtime_teacher = torch.tensor(teacher, dtype=torch.float32)
        target = weighted_feature_mean(
            runtime_teacher, teacher_weight.astype(np.float32), resources,
        ).detach()
        value = cached_finite_mmd(
            runtime_student, student_weight.astype(np.float32), target, resources,
        )
        value.backward()
        automatic = leaf.grad.detach().cpu().numpy()
        analytic = analytic_finite_mmd_gradient(
            student, student_weight, teacher, teacher_weight, resources,
            normalize_inputs=kind == "token",
        )
        analytic_flat = analytic.reshape(-1)
        automatic_flat = automatic.reshape(-1)
        analytic_norm = float(np.linalg.norm(analytic_flat))
        automatic_norm = float(np.linalg.norm(automatic_flat))
        if analytic_norm >= 1.0e-8 and automatic_norm > 0:
            cosine = float(
                np.dot(analytic_flat, automatic_flat) / (analytic_norm * automatic_norm)
            )
            ratio = automatic_norm / analytic_norm
            passed = (
                cosine >= .99999 and .999 <= ratio <= 1.001
                and np.max(np.abs(analytic_flat - automatic_flat)) <= 1.0e-5
            )
        else:
            cosine = 1.0 if automatic_norm == 0 else 0.0
            ratio = 1.0 if analytic_norm == automatic_norm == 0 else math.inf
            passed = automatic_norm <= 1.0e-6
        rows.append({
            "analytic_gradient_norm": analytic_norm,
            "autograd_gradient_norm": automatic_norm,
            "flattened_cosine": cosine,
            "norm_ratio_autograd_over_analytic": ratio,
            "maximum_absolute_entry_error": float(
                np.max(np.abs(analytic_flat - automatic_flat))
            ),
            "passed": bool(passed),
        })
    return {"fixtures": rows, "passed": all(bool(row["passed"]) for row in rows)}


def _rotation_diagnostics(*, fixtures: int) -> dict[str, object]:
    generator = np.random.Generator(np.random.PCG64(ROTATION_SEED))
    matrix = generator.standard_normal((128, 128))
    q, r = np.linalg.qr(matrix)
    diagonal_sign = np.sign(np.diag(r)); diagonal_sign[diagonal_sign == 0] = 1
    q = q * diagonal_sign[None, :]
    resources = generate_spectral_resources("token")
    finite_changes: list[float] = []
    ideal_changes: list[float] = []
    transformed_gradient_cosines: list[float] = []
    transformed_gradient_norm_ratios: list[float] = []
    for _ in range(fixtures):
        student, teacher, student_weight, teacher_weight = _token_fixture(generator)
        original, _ = _finite_values_fp64(
            student, teacher, student_weight, teacher_weight, resources,
        )
        rotated_student = student @ q
        rotated_teacher = teacher @ q
        rotated, _ = _finite_values_fp64(
            rotated_student, rotated_teacher, student_weight, teacher_weight, resources,
        )
        finite_changes.append(abs(rotated - original))
        ideal_original = ideal_rbf_mmd(
            student, student_weight, teacher, teacher_weight, kind="token",
        )
        ideal_rotated = ideal_rbf_mmd(
            rotated_student, student_weight, rotated_teacher, teacher_weight,
            kind="token",
        )
        ideal_changes.append(abs(ideal_rotated - ideal_original))
        gradient = analytic_finite_mmd_gradient(
            student, student_weight, teacher, teacher_weight, resources,
            normalize_inputs=True,
        )
        rotated_gradient = analytic_finite_mmd_gradient(
            rotated_student, student_weight, rotated_teacher, teacher_weight,
            resources, normalize_inputs=True,
        ) @ q.T
        left = gradient.reshape(-1); right = rotated_gradient.reshape(-1)
        left_norm = np.linalg.norm(left); right_norm = np.linalg.norm(right)
        transformed_gradient_cosines.append(float(np.dot(left, right) / (left_norm * right_norm)))
        transformed_gradient_norm_ratios.append(float(right_norm / left_norm))
    mean_change = float(np.mean(finite_changes))
    p95_change = float(np.quantile(finite_changes, .95))
    return {
        "haar_matrix_sha256": canonical_sha256({
            "dtype": q.dtype.str,
            "shape": list(q.shape),
            "bytes_sha256": __import__("hashlib").sha256(q.tobytes(order="C")).hexdigest(),
        }),
        "finite_kernel_absolute_changes": finite_changes,
        "ideal_rbf_absolute_changes": ideal_changes,
        "transformed_gradient_cosines_report_only": transformed_gradient_cosines,
        "transformed_gradient_norm_ratios_report_only": transformed_gradient_norm_ratios,
        "finite_change_mean": mean_change,
        "finite_change_p95": p95_change,
        "maximum_ideal_invariance_error": float(max(ideal_changes, default=0)),
        "passed": bool(
            mean_change <= .020 and p95_change <= .065
            and max(ideal_changes, default=0) <= 1.0e-12
        ),
    }


def _build_acceptance(
    *, value_fixtures: int, gradient_fixtures: int, rotation_fixtures: int,
    scientific_authorization: bool,
) -> dict[str, object]:
    token_resources = generate_spectral_resources("token")
    relation_resources = generate_spectral_resources("relation")
    diagnostics = {
        "token_values": _finite_value_diagnostics(
            kind="token", seed=TOKEN_VALUE_SEED, fixtures=value_fixtures,
        ),
        "relation_values": _finite_value_diagnostics(
            kind="relation", seed=RELATION_VALUE_SEED, fixtures=value_fixtures,
        ),
        "token_gradients": _gradient_diagnostics(
            kind="token", seed=TOKEN_GRADIENT_SEED, fixtures=gradient_fixtures,
        ),
        "relation_gradients": _gradient_diagnostics(
            kind="relation", seed=RELATION_GRADIENT_SEED, fixtures=gradient_fixtures,
        ),
        "joint_rotation": _rotation_diagnostics(fixtures=rotation_fixtures),
    }
    passed = all(bool(value["passed"]) for value in diagnostics.values())
    payload = with_content_hash({
        "contract": NUMERICAL_ACCEPTANCE_CONTRACT,
        "schema_version": 1,
        "scientific_authorization": scientific_authorization,
        "passed": bool(passed),
        "fixture_counts": {
            "value": value_fixtures,
            "gradient": gradient_fixtures,
            "rotation": rotation_fixtures,
        },
        "fixture_streams": [
            {
                "payload": _seed_payload(purpose, seed),
                "payload_canonical_utf8": canonical_json_bytes(
                    _seed_payload(purpose, seed),
                ).decode("ascii"),
                "payload_sha256": canonical_sha256(_seed_payload(purpose, seed)),
            }
            for purpose, seed in (
                ("token_value", TOKEN_VALUE_SEED),
                ("relation_value", RELATION_VALUE_SEED),
                ("token_gradient", TOKEN_GRADIENT_SEED),
                ("relation_gradient", RELATION_GRADIENT_SEED),
                ("joint_rotation", ROTATION_SEED),
            )
        ],
        "resource_hashes": {
            "token": token_resources.content_hash,
            "relation": relation_resources.content_hash,
        },
        "resource_array_logical_hashes": {
            "token": [block.logical_hashes for block in token_resources.blocks],
            "relation": [block.logical_hashes for block in relation_resources.blocks],
        },
        "diagnostics": diagnostics,
        "claims": {
            "training_estimand": "frozen_finite_spectral_kernel",
            "ideal_rbf_value_quality": "acceptance",
            "ideal_rbf_gradient_fidelity": "report_only_not_claimed",
            "rotation_gradient_invariance": "report_only_not_claimed",
        },
    })
    return payload


def build_numerical_acceptance(*, output: str | Path | None = None) -> dict[str, object]:
    """Run and optionally publish the complete frozen scientific acceptance."""

    result = _build_acceptance(
        value_fixtures=VALUE_FIXTURES,
        gradient_fixtures=GRADIENT_FIXTURES,
        rotation_fixtures=ROTATION_FIXTURES,
        scientific_authorization=True,
    )
    if not result["passed"]:
        raise RuntimeError("HCWDL frozen numerical acceptance failed")
    if output is not None:
        write_immutable_json(output, result)
    return result


def build_numerical_acceptance_preview(
    *, value_fixtures: int = 4, gradient_fixtures: int = 4,
    rotation_fixtures: int = 4,
) -> dict[str, object]:
    """Bounded local probe that is explicitly unable to authorize a campaign."""

    if min(value_fixtures, gradient_fixtures, rotation_fixtures) <= 0:
        raise ValueError("numerical preview fixture counts must be positive")
    return _build_acceptance(
        value_fixtures=value_fixtures,
        gradient_fixtures=gradient_fixtures,
        rotation_fixtures=rotation_fixtures,
        scientific_authorization=False,
    )


__all__ = [
    "GRADIENT_FIXTURES", "NUMERICAL_ACCEPTANCE_CONTRACT",
    "NUMERICAL_FIXTURE_CONTRACT", "RELATION_GRADIENT_SEED",
    "RELATION_VALUE_SEED", "ROTATION_FIXTURES", "ROTATION_SEED",
    "TOKEN_GRADIENT_SEED", "TOKEN_VALUE_SEED", "VALUE_FIXTURES",
    "build_numerical_acceptance", "build_numerical_acceptance_preview",
]
