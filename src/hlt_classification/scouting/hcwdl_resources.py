"""Measured HCWDL resource profile and prelaunch storage estimator contracts."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Final

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash


RESOURCE_PROFILE_CONTRACT: Final = "HCWDL_RESOURCE_PROFILE/v1"
STORAGE_ESTIMATE_CONTRACT: Final = "HCWDL_STORAGE_ESTIMATE/v1"
RESOURCE_CLASSES: Final = ("cpu_small", "cpu_assignment", "gpu_root", "gpu_single", "gpu_dual")


def estimate_storage(
    *, visible_tokens_by_role: Mapping[str, int], selected_checkpoint_bytes: int,
    rolling_checkpoint_bytes: int, concurrent_training_jobs: int,
    headroom_fraction: float = 0.25,
) -> dict[str, Any]:
    if set(visible_tokens_by_role) != {"train", "validation", "final_test"}:
        raise ValueError("HCWDL storage estimate roles differ")
    if any(isinstance(value, bool) or int(value) < 0 for value in visible_tokens_by_role.values()):
        raise ValueError("HCWDL storage token totals differ")
    if min(selected_checkpoint_bytes, rolling_checkpoint_bytes, concurrent_training_jobs) <= 0:
        raise ValueError("HCWDL checkpoint/concurrency estimate differs")
    if not math.isfinite(headroom_fraction) or not 0 < headroom_fraction < 1:
        raise ValueError("HCWDL storage headroom differs")
    # int16 index + uint16 confidence, plus conservative row/metadata overhead.
    assignment_bytes = 4 * sum(map(int, visible_tokens_by_role.values()))
    checkpoint_bytes = concurrent_training_jobs * (selected_checkpoint_bytes + rolling_checkpoint_bytes)
    subtotal = assignment_bytes + checkpoint_bytes
    return with_content_hash({
        "contract": STORAGE_ESTIMATE_CONTRACT, "schema_version": 1,
        "visible_tokens_by_role": dict(sorted(visible_tokens_by_role.items())),
        "assignment_bytes": assignment_bytes, "checkpoint_bytes": checkpoint_bytes,
        "subtotal_bytes": subtotal,
        "headroom_fraction_hex": float(headroom_fraction).hex(),
        "required_bytes_with_headroom": math.ceil(subtotal * (1 + headroom_fraction)),
        "durable_repaired_dataset_bytes": 0,
    })


def build_resource_profile(
    *, requests: Mapping[str, Mapping[str, Any]], miniature_report_sha256: str,
    storage_estimate_sha256: str, measurement_report_sha256: str,
    safety_factor: float,
) -> dict[str, Any]:
    if set(requests) != set(RESOURCE_CLASSES):
        raise ValueError("HCWDL resource classes differ")
    normalized = {}
    for name in RESOURCE_CLASSES:
        row = requests[name]
        if set(row) != {"cpus", "memory", "walltime", "gpu"}:
            raise ValueError(f"HCWDL resource fields differ for {name}")
        if not isinstance(row["cpus"], int) or row["cpus"] <= 0:
            raise ValueError("HCWDL CPU request differs")
        if not isinstance(row["memory"], str) or not row["memory"]:
            raise ValueError("HCWDL memory request differs")
        if not isinstance(row["walltime"], str) or not row["walltime"]:
            raise ValueError("HCWDL walltime request differs")
        if name.startswith("gpu_") and row["gpu"] != "gpu:gh200:1":
            raise ValueError("HCWDL measured GPU class differs")
        if name.startswith("cpu_") and row["gpu"] is not None:
            raise ValueError("HCWDL CPU class unexpectedly requests a GPU")
        normalized[name] = dict(row)
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("HCWDL resource safety factor differs")
    return with_content_hash({
        "contract": RESOURCE_PROFILE_CONTRACT, "schema_version": 1,
        "requests": normalized,
        "miniature_report_sha256": require_sha256(miniature_report_sha256, name="miniature SHA-256"),
        "storage_estimate_sha256": require_sha256(storage_estimate_sha256, name="storage SHA-256"),
        "measurement_report_sha256": require_sha256(measurement_report_sha256, name="measurement SHA-256"),
        "safety_factor_hex": float(safety_factor).hex(),
        "measured_on_tigris": True, "authorized": True,
    })


def validate_resource_profile(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=RESOURCE_PROFILE_CONTRACT, expected_schema_version=1,
    )
    if value.get("measured_on_tigris") is not True or value.get("authorized") is not True:
        raise PermissionError("HCWDL resource profile is not measured and authorized")
    rebuilt = build_resource_profile(
        requests=value["requests"], miniature_report_sha256=value["miniature_report_sha256"],
        storage_estimate_sha256=value["storage_estimate_sha256"],
        measurement_report_sha256=value["measurement_report_sha256"],
        safety_factor=float.fromhex(value["safety_factor_hex"]),
    )
    if rebuilt["content_hash"] != digest:
        raise ValueError("HCWDL resource profile semantics differ")
    return digest


__all__ = [
    "RESOURCE_CLASSES", "RESOURCE_PROFILE_CONTRACT", "STORAGE_ESTIMATE_CONTRACT",
    "build_resource_profile", "estimate_storage", "validate_resource_profile",
]
