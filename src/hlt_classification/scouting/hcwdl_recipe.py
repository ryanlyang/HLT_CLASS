"""Immutable, evidence-bound HCWDL optimization recipe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash


RECIPE_CONTRACT: Final = "HCWDL_RECIPE/v3"
RECIPE_SCHEMA_VERSION: Final = 3
PRIMARY_RECIPE_PROFILE: Final = "primary_ladder"
PRIMARY_DUAL_TEACHER_DECISION: Final = {
    "peak_learning_rate": 3e-4,
    "ce": 0.25,
    "predecessor_kd": 0.40,
    "privileged_kd": 0.35,
    "privileged_temperature": 2.0,
}
PRIMARY_RECIPE_DECISION: Final = {
    "batching": {
        "microbatch_size": 256,
        "gradient_accumulation": 1,
        "effective_batch_size": 256,
    },
    "optimizer": {
        "name": "AdamW",
        "peak_learning_rates": {
            "cold_root": 3e-4,
            "cold_child": 3e-4,
            "warm_child": 3e-4,
        },
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "gradient_clipping": {"enabled": False},
    },
    "schedule": {
        "name": "warmup_cosine",
        "warmup_fraction": 0.05,
        "minimum_lr_fraction": 0.05,
    },
    "coefficient_schedule": "constant",
    "single_teacher_coefficients": {"ce": 0.25, "teacher_kd": 0.75},
    "dual_teacher_coefficients": {
        "ce": 0.25,
        "predecessor_kd": 0.40,
        "privileged_kd": 0.35,
    },
    "controls": {
        "predecessor_only_coefficients": {"ce": 0.25, "predecessor_kd": 0.75},
        "include_label_only_warm_continuation": True,
    },
    "single_privileged_temperature": 2.0,
    "predecessor_temperature": 1.0,
    "privileged_temperature": 2.0,
    "dual_teacher_peak_learning_rate": 3e-4,
    "amp_dtype": "bfloat16",
}
CLASS_WEIGHT_POLICY: Final = "sqrt_inverse_frequency_unit_population_mean_v1"
FORBIDDEN_PLACEHOLDERS: Final = frozenset(("", "tbd", "todo", "placeholder", "unknown", "auto", "default"))


def _positive(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _weights(value: object, names: Sequence[str], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{label} coefficient names differ")
    result = {name: float(value[name]) for name in names}
    if any(not math.isfinite(item) or item < 0 for item in result.values()):
        raise ValueError(f"{label} coefficients must be finite and nonnegative")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{label} coefficients must sum to one")
    return result


def _reject_placeholders(value: object, path: str = "recipe") -> None:
    if isinstance(value, str) and value.strip().lower() in FORBIDDEN_PLACEHOLDERS:
        raise ValueError(f"unresolved placeholder at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_placeholders(item, f"{path}[{index}]")


def validate_recipe(
    value: Mapping[str, Any], *, require_authorized: bool = True,
    expected_profile: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RECIPE_CONTRACT,
        expected_schema_version=RECIPE_SCHEMA_VERSION,
    )
    _reject_placeholders(value)
    if require_authorized and value.get("authorized_for_execution") is not True:
        raise PermissionError("HCWDL recipe is an example or has not been authorized")
    profile = value.get("recipe_profile")
    if profile not in {PRIMARY_RECIPE_PROFILE, "registered_ablation", "local_test"}:
        raise ValueError("HCWDL recipe profile differs")
    if expected_profile is not None and profile != expected_profile:
        raise PermissionError("HCWDL recipe profile is not authorized for this campaign")
    if require_authorized and profile == "local_test":
        raise PermissionError("HCWDL local-test recipe cannot authorize execution")
    if value.get("repair_family") != "HIGHCOV_SHELL_EXACT/v1":
        raise ValueError("HCWDL recipe repair family differs")
    if value.get("training_passes") != 60 or value.get("validation_every_passes") != 1:
        raise ValueError("HCWDL training/validation budget differs")
    batching = value.get("batching")
    if not isinstance(batching, Mapping):
        raise ValueError("HCWDL batching recipe differs")
    for name in ("microbatch_size", "gradient_accumulation", "effective_batch_size"):
        if not isinstance(batching.get(name), int) or int(batching[name]) <= 0:
            raise ValueError(f"HCWDL {name} must be a positive integer")
    if batching["microbatch_size"] * batching["gradient_accumulation"] != batching["effective_batch_size"]:
        raise ValueError("HCWDL effective batch does not equal microbatch times accumulation")
    optimizer = value.get("optimizer")
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer) != {
            "name", "peak_learning_rates", "betas", "epsilon", "weight_decay",
            "gradient_clipping",
        }
        or optimizer.get("name") != "AdamW"
    ):
        raise ValueError("HCWDL optimizer differs")
    learning_rates = optimizer.get("peak_learning_rates")
    if not isinstance(learning_rates, Mapping) or set(learning_rates) != {"cold_root", "cold_child", "warm_child"}:
        raise ValueError("HCWDL learning-rate roles differ")
    for name, item in learning_rates.items():
        _positive(item, f"peak learning rate {name}")
    betas = optimizer.get("betas")
    if (
        not isinstance(betas, Sequence) or isinstance(betas, (str, bytes))
        or len(betas) != 2
        or any(not math.isfinite(float(item)) or not 0 <= float(item) < 1 for item in betas)
    ):
        raise ValueError("HCWDL AdamW betas differ")
    if [float(item) for item in betas] != [0.9, 0.999]:
        raise ValueError("HCWDL runtime supports only the bound AdamW betas")
    clipping = optimizer.get("gradient_clipping")
    if not isinstance(clipping, Mapping) or not isinstance(clipping.get("enabled"), bool):
        raise ValueError("HCWDL gradient-clipping policy differs")
    if clipping["enabled"]:
        raise ValueError("HCWDL runtime does not enable gradient clipping")
    elif set(clipping) != {"enabled"}:
        raise ValueError("disabled HCWDL gradient clipping has unexpected fields")
    dual_peak_learning_rate = _positive(
        value.get("dual_teacher_peak_learning_rate"),
        "dual-teacher peak learning rate",
    )
    _positive(optimizer.get("epsilon"), "optimizer epsilon")
    weight_decay = float(optimizer.get("weight_decay"))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("optimizer weight decay differs")
    schedule = value.get("schedule")
    if not isinstance(schedule, Mapping) or schedule.get("name") != "warmup_cosine":
        raise ValueError("HCWDL schedule differs")
    warmup = float(schedule.get("warmup_fraction"))
    minimum = float(schedule.get("minimum_lr_fraction"))
    if not 0 <= warmup < 1 or not 0 < minimum <= 1:
        raise ValueError("HCWDL schedule fractions differ")
    if value.get("coefficient_schedule") != "constant":
        raise ValueError("HCWDL coefficient schedule differs")
    single = _weights(value.get("single_teacher_coefficients"), ("ce", "teacher_kd"), "single-teacher")
    dual = _weights(value.get("dual_teacher_coefficients"), ("ce", "predecessor_kd", "privileged_kd"), "dual-teacher")
    controls = value.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError("HCWDL control recipe differs")
    _weights(
        controls.get("predecessor_only_coefficients"), ("ce", "predecessor_kd"),
        "predecessor-only control",
    )
    if not isinstance(controls.get("include_label_only_warm_continuation"), bool):
        raise ValueError("HCWDL warm-continuation control decision differs")
    _positive(value.get("single_privileged_temperature"), "single privileged temperature")
    _positive(value.get("predecessor_temperature"), "predecessor temperature")
    _positive(value.get("privileged_temperature"), "privileged temperature")
    class_weighting = value.get("class_weighting")
    if not isinstance(class_weighting, Mapping) or set(class_weighting) != {
        "policy", "train_class_counts", "train_row_selection_sha256",
    }:
        raise ValueError("HCWDL class-weight lineage differs")
    if class_weighting.get("policy") != CLASS_WEIGHT_POLICY:
        raise ValueError("HCWDL class-weight policy differs")
    counts_raw = class_weighting.get("train_class_counts")
    if (
        not isinstance(counts_raw, Sequence) or isinstance(counts_raw, (str, bytes))
        or len(counts_raw) != 15
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in counts_raw)
    ):
        raise ValueError("HCWDL train class counts differ")
    require_sha256(
        class_weighting.get("train_row_selection_sha256"),
        name="HCWDL train row-selection SHA-256",
    )
    counts = np.asarray(counts_raw, np.float64)
    inverse = 1.0 / np.sqrt(counts)
    expected_weights = (counts.sum() / np.sum(counts * inverse) * inverse).astype(np.float32)
    class_weights = np.asarray(value.get("class_weights"), np.float32)
    if class_weights.shape != (15,) or not np.isfinite(class_weights).all() or np.any(class_weights <= 0):
        raise ValueError("HCWDL class weights must contain 15 finite positive values")
    if not np.array_equal(class_weights, expected_weights):
        raise ValueError("HCWDL class weights differ from authenticated train counts")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("HCWDL recipe requires evidence parents")
    for name, parent in evidence.items():
        require_sha256(parent, name=f"recipe evidence {name}")
    if single["teacher_kd"] == 0 or dual["predecessor_kd"] == 0 or dual["privileged_kd"] == 0:
        raise ValueError("primary HCWDL recipe cannot remove a declared teacher")
    if profile == PRIMARY_RECIPE_PROFILE:
        if value.get("purpose") != "hcwdl_primary_ladder":
            raise ValueError("primary HCWDL recipe purpose differs")
        expected = PRIMARY_DUAL_TEACHER_DECISION
        actual = {
            "peak_learning_rate": dual_peak_learning_rate,
            **dual,
            "privileged_temperature": float(value["privileged_temperature"]),
        }
        if actual != expected:
            raise ValueError("primary HCWDL dual-teacher decision differs")
        primary_actual = {
            name: value[name] for name in (
                "batching", "optimizer", "schedule", "coefficient_schedule",
                "single_teacher_coefficients", "dual_teacher_coefficients", "controls",
                "single_privileged_temperature", "predecessor_temperature",
                "privileged_temperature", "dual_teacher_peak_learning_rate", "amp_dtype",
            )
        }
        if primary_actual != PRIMARY_RECIPE_DECISION:
            raise ValueError("primary HCWDL complete recipe decision differs")
    return digest


def validate_recipe_class_weight_lineage(
    recipe: Mapping[str, Any], row_selection: Mapping[str, Any],
) -> None:
    """Bind recipe weights to the exact authenticated train selection."""

    validate_recipe(recipe, require_authorized=bool(recipe.get("authorized_for_execution")))
    from .selective_assignment import ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION

    selection_hash = validate_content_hash(
        row_selection, expected_contract=ROW_SELECTION_CONTRACT,
        expected_schema_version=ROW_SELECTION_VERSION,
    )
    train = row_selection.get("roles", {}).get("train")
    if not isinstance(train, Mapping):
        raise ValueError("HCWDL row selection lacks the train role")
    weighting = recipe["class_weighting"]
    if (
        weighting["train_row_selection_sha256"] != selection_hash
        or weighting["train_class_counts"] != train.get("class_counts")
    ):
        raise ValueError("HCWDL recipe class weights have different row-selection lineage")


def build_recipe(payload: Mapping[str, Any], *, authorized: bool) -> dict[str, Any]:
    result = with_content_hash({
        "contract": RECIPE_CONTRACT,
        "schema_version": RECIPE_SCHEMA_VERSION,
        "authorized_for_execution": bool(authorized),
        **dict(payload),
    })
    validate_recipe(result, require_authorized=authorized)
    return result


def example_recipe() -> dict[str, Any]:
    """Complete test-only values that are cryptographically marked unauthorized."""

    return build_recipe({
        "recipe_profile": "local_test",
        "purpose": "local_test_only_not_a_pilot_recipe",
        "repair_family": "HIGHCOV_SHELL_EXACT/v1",
        "training_passes": 60,
        "validation_every_passes": 1,
        "batching": dict(PRIMARY_RECIPE_DECISION["batching"]),
        "optimizer": {
            "name": "AdamW", "peak_learning_rates": {
                "cold_root": 3e-4, "cold_child": 3e-4, "warm_child": 3e-4,
            }, "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.01,
            "gradient_clipping": {"enabled": False},
        },
        "dual_teacher_peak_learning_rate": 3e-4,
        "schedule": {"name": "warmup_cosine", "warmup_fraction": 0.05, "minimum_lr_fraction": 0.05},
        "coefficient_schedule": "constant",
        "single_teacher_coefficients": {"ce": 0.25, "teacher_kd": 0.75},
        "dual_teacher_coefficients": {"ce": 0.25, "predecessor_kd": 0.40, "privileged_kd": 0.35},
        "controls": {
            "predecessor_only_coefficients": {"ce": 0.25, "predecessor_kd": 0.75},
            "include_label_only_warm_continuation": True,
        },
        "single_privileged_temperature": 2.0,
        "predecessor_temperature": 1.0,
        "privileged_temperature": 2.0,
        "class_weighting": {
            "policy": CLASS_WEIGHT_POLICY,
            "train_class_counts": [1] * 15,
            "train_row_selection_sha256": "1" * 64,
        },
        "class_weights": [1.0] * 15,
        "amp_dtype": "bfloat16",
        "evidence": {"local_fixture": "0" * 64},
    }, authorized=False)


__all__ = [
    "CLASS_WEIGHT_POLICY", "PRIMARY_DUAL_TEACHER_DECISION", "PRIMARY_RECIPE_DECISION",
    "PRIMARY_RECIPE_PROFILE",
    "RECIPE_CONTRACT", "RECIPE_SCHEMA_VERSION", "build_recipe",
    "example_recipe", "validate_recipe", "validate_recipe_class_weight_lineage",
]
