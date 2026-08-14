"""Versioned contracts shared by the HCWDL structural-feature homotopy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)


COUPLING_CONFIG_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_COUPLING_CONFIG/v1"
SCALE_CALIBRATION_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_SCALE_CALIBRATION/v1"
SWITCH_CALIBRATION_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_SWITCH_CALIBRATION/v1"
BASE_SHARD_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_BASE_SHARD/v1"
BASE_MANIFEST_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_BASE_MANIFEST/v1"
SWITCH_SIDECAR_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_SWITCH_SIDECAR/v1"
COUPLING_MANIFEST_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_COUPLING_MANIFEST/v1"
COUPLING_AUDIT_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_COUPLING_AUDIT/v1"
COUPLING_LOCK_CONTRACT: Final = "HCWDL_RESIDUAL_SHELL_COUPLING_LOCK/v1"

TOFF_TARGET_SHARD_CONTRACT: Final = "HCWDL_TOFF_TARGET_SHARD/v1"
TOFF_TARGET_MANIFEST_CONTRACT: Final = "HCWDL_TOFF_TARGET_MANIFEST/v1"
TOFF_TARGET_LOCK_CONTRACT: Final = "HCWDL_TOFF_TARGET_LOCK/v1"

COORDINATE_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_COORDINATE/v2"
NODE_SPEC_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_NODE_SPEC/v2"
GRAPH_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_GRAPH/v2"
RECIPE_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_RECIPE/v2"
ENDPOINT_LOCK_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_ENDPOINT_EQUALITY_LOCK/v1"
GRAPH_RECIPE_LOCK_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_GRAPH_RECIPE_LOCK/v2"
PILOT_SPEC_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v2"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_COMMAND_PLAN/v2"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_TRAINING_REPORT/v1"
NODE_RUNTIME_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_NODE_RUNTIME/v1"
AGGREGATE_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_AGGREGATE/v2"
RESOURCE_PROFILE_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_RESOURCE_PROFILE/v1"
OPERATIONAL_WAIVER_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_OPERATIONAL_EVIDENCE_WAIVER/v1"
)
CACHE_RESOURCE_MEASUREMENT_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_CACHE_RESOURCE_MEASUREMENT/v1"
)
CACHE_MINIATURE_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_CACHE_MINIATURE/v1"
TARGET_RESOURCE_MEASUREMENT_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_TARGET_RESOURCE_MEASUREMENT/v1"
)
SMOKE_RESOURCE_MEASUREMENT_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_SMOKE_RESOURCE_MEASUREMENT/v1"
)
RESUME_EVIDENCE_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_RESUME_EVIDENCE/v1"
)
WEAVER_PARITY_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_WEAVER_PARITY/v1"
CAMPAIGN_COMPLETION_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_CAMPAIGN_COMPLETE/v2"
SMOKE_SELECTION_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_SMOKE_SELECTION/v1"
STUDY_SELECTION_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_SELECTION/v1"

RECOVERY_SPEC_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_RECOVERY_SPEC/v2"
RECOVERY_COMMAND_PLAN_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_RECOVERY_COMMAND_PLAN/v2"
)
RESOURCE_RECOVERY_SPEC_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_RESOURCE_RECOVERY_SPEC/v2"
)
RESOURCE_RECOVERY_COMMAND_PLAN_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_RESOURCE_RECOVERY_COMMAND_PLAN/v2"
)
EXECUTION_RESOURCE_RECOVERY_SPEC_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_EXECUTION_RESOURCE_RECOVERY_SPEC/v1"
)
EXECUTION_RESOURCE_RECOVERY_COMMAND_PLAN_CONTRACT: Final = (
    "HCWDL_STRUCTURAL_FEATURE_EXECUTION_RESOURCE_RECOVERY_COMMAND_PLAN/v1"
)

COUPLING_CONFIG_VERSION: Final = 1
ROLE_COUNTS: Final = {"train": 300_000, "validation": 100_000, "final_test": 0}
SMOKE_ROLE_COUNTS: Final = {"train": 4096, "validation": 4096, "final_test": 0}
REPLICATE_SEED: Final = 1337
P0_CHARGED_MAX: Final = 90
P0_NEUTRAL_MAX: Final = 60
MAX_TOKENS: Final = 200
CLASS_COUNT: Final = 15
SOURCE_SENTINEL: Final = -1
TARGET_SLOT_SENTINEL: Final = 65535
TARGET_NATIVE_SENTINEL: Final = -1
EDIT_SUBSTITUTION: Final = 0
EDIT_REMOVAL: Final = 1
EDIT_INSERTION: Final = 2
TARGET_HLT_DUSTBIN: Final = 0
TARGET_ASSIGNED_OFFLINE: Final = 1
COUPLING_HASH_DOMAIN: Final = "HCWDL_RESIDUAL_SHELL_SWITCH/v1"
AUTHORIZATION_PHRASE: Final = "AUTHORIZE HCWDL UJ VALIDATION CAMPAIGN EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL UJ VALIDATION CAMPAIGN EXACT SPEC"
RECOVERY_AUTHORIZATION_PHRASE: Final = "AUTHORIZE HCWDL UJ FAILED CLOSURE RECOVERY"
RECOVERY_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL UJ FAILED CLOSURE RECOVERY"
RESOURCE_RECOVERY_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE HCWDL UJ RESOURCE ONLY RECOVERY"
)
RESOURCE_RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL UJ RESOURCE ONLY RECOVERY"
)
EXECUTION_RESOURCE_RECOVERY_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE HCWDL UJ EXECUTION AND RESOURCE RECOVERY"
)
EXECUTION_RESOURCE_RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL UJ EXECUTION AND RESOURCE RECOVERY"
)

_SHA = re.compile(r"^[0-9a-f]{64}$")


def artifact_reference(path: str | Path) -> dict[str, str]:
    """Create a path+content-hash reference to a validated JSON artifact."""

    resolved = Path(path).resolve()
    payload = load_json(resolved)
    contract = payload.get("contract")
    version = payload.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"unversioned HCWDL-UJ artifact: {resolved}")
    digest = validate_content_hash(
        payload, expected_contract=contract, expected_schema_version=version,
    )
    return {"path": str(resolved), "content_hash": digest}


def load_artifact_reference(
    reference: Mapping[str, object], *, expected_contract: str | None = None,
    name: str = "artifact",
) -> dict[str, Any]:
    if set(reference) != {"path", "content_hash"}:
        raise ValueError(f"HCWDL-UJ {name} reference differs")
    payload = load_json(Path(str(reference["path"])))
    contract = payload.get("contract")
    version = payload.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"HCWDL-UJ {name} is unversioned")
    if expected_contract is not None and contract != expected_contract:
        raise ValueError(f"HCWDL-UJ {name} contract differs")
    digest = validate_content_hash(
        payload, expected_contract=contract, expected_schema_version=version,
    )
    if digest != reference["content_hash"]:
        raise ValueError(f"HCWDL-UJ {name} content hash differs")
    return payload


def build_coupling_config(*, projection_sha256: str, shell_exact_sha256: str) -> dict[str, Any]:
    """Return the complete immutable v1 scientific coupling configuration."""

    payload = {
        "contract": COUPLING_CONFIG_CONTRACT,
        "schema_version": 1,
        "p0_bounds": {"charged": P0_CHARGED_MAX, "neutral": P0_NEUTRAL_MAX},
        "max_tokens": MAX_TOKENS,
        "physical_p4_tolerance": 1.0e-5,
        "epsilon_p4": 1.0e-6,
        "cost_weights": {
            "kinematics": 0.30,
            "identity": 0.20,
            "validity": 0.15,
            "track": 0.20,
            "field": 0.15,
        },
        "scale_histograms": {
            "bins": 65536,
            "delta_r_upper": 5.0,
            "log_response_upper": 8.0,
            "field_delta_upper": 64.0,
            "quantile_numerator": 90,
            "quantile_denominator": 100,
        },
        "scale_floors": {
            "delta_r": 0.02,
            "log_pt": 0.25,
            "log_energy": 0.25,
            "field": {
                "0": 0.20, "7": 0.05, "8": 0.05, "9": 0.25,
                "10": 0.25, "11": 0.25, "12": 0.25, "13": 0.25,
                "14": 0.25, "15": 0.25, "16": 0.25, "17": 0.25,
                "18": 0.25, "19": 0.25, "20": 1.00,
            },
        },
        "field_groups": {
            "quality": [0], "relative_kinematics": [7, 8, 9],
            "scale_kinematics": [10, 19], "track_fit": [11],
            "track_dz": [12, 18], "track_dxy": [13, 14],
            "track_btag": [15, 16, 17], "lost_inner_hits": [20],
        },
        "cost_quantum": 1_000_000,
        "mass_quantum": 1_000_000,
        "mass_weights": {"constant": 1.0, "pt_share": 4.0,
                         "energy_share": 2.0, "disruption": 2.0},
        "switch_bins": 4096,
        "hash_domain": COUPLING_HASH_DOMAIN,
        "solver_objective": (
            "maximum_cardinality_then_minimum_integer_cost_then_"
            "lexicographically_smallest_sorted_edge_set_v1"
        ),
        "carrier_order": "active_hlt_slots_then_native_index_tail_v1",
        "projection_sha256": require_sha256(
            projection_sha256, name="endpoint projection SHA-256",
        ),
        "shell_exact_sha256": require_sha256(
            shell_exact_sha256, name="Shell Exact SHA-256",
        ),
        "labels_forbidden": True,
        "labels_read": False,
        "final_test_accessed": False,
    }
    return with_content_hash(payload)


def validate_coupling_config(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=COUPLING_CONFIG_CONTRACT,
        expected_schema_version=1,
    )
    reference = build_coupling_config(
        projection_sha256=str(value.get("projection_sha256")),
        shell_exact_sha256=str(value.get("shell_exact_sha256")),
    )
    if value != reference:
        raise ValueError("HCWDL-UJ coupling configuration differs from v1")
    return digest


def coordinate_payload() -> dict[str, Any]:
    def row(node: str, s_num: int, denominator: int, f_num: int) -> dict[str, Any]:
        s = s_num / denominator
        f = f_num / denominator
        return {
            "node": node,
            "structural": {"numerator": s_num, "denominator": denominator,
                           "decimal": f"{s:.2f}", "float_hex": float(s).hex()},
            "feature": {"numerator": f_num, "denominator": denominator,
                        "decimal": f"{f:.2f}", "float_hex": float(f).hex()},
            "alpha_hex": float(1.0 - f).hex(),
            "structural_u16": (
                2 * s_num * 65535 + denominator
            ) // (2 * denominator),
        }

    rows = []
    rows.extend(row(f"U{index * 20:03d}", index, 5, 0) for index in range(1, 6))
    rows.extend(row(f"D{100-index * 20}F", 5, 5, index) for index in range(1, 6))
    rows.extend(row(f"J{index * 10:03d}", index, 10, index) for index in range(1, 11))
    return with_content_hash({
        "contract": COORDINATE_CONTRACT,
        "schema_version": 1,
        "rows": rows,
        "endpoint_overrides": {
            "s0f0": "P0_multiset", "s1f0": "D100_byte_exact",
            "s1f1": "HLT_byte_exact",
        },
    })


def validate_coordinate(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=COORDINATE_CONTRACT, expected_schema_version=1,
    )
    if value != coordinate_payload():
        raise ValueError("HCWDL-UJ coordinate table differs")
    return digest


def validate_exact_contract(
    value: Mapping[str, Any], *, contract: str, required_hashes: Sequence[str] = (),
    required_true: Sequence[str] = (),
) -> str:
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )
    for name in required_hashes:
        require_sha256(value.get(name), name=name)
    for name in required_true:
        if value.get(name) is not True:
            raise ValueError(f"HCWDL-UJ required invariant {name} differs")
    if value.get("final_test_accessed", False) is not False:
        raise PermissionError("HCWDL-UJ validation-only artifact accessed final test")
    return digest


def content_payload(contract: str, **fields: Any) -> dict[str, Any]:
    """Build an authenticated schema-v1 payload for an exact contract identity."""

    return with_content_hash({"contract": contract, "schema_version": 1, **fields})


def validate_commit(value: object, *, name: str = "source commit") -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return text


__all__ = [name for name in globals() if name.endswith("_CONTRACT") or name in {
    "AGGREGATE_CONTRACT", "AUTHORIZATION_PHRASE", "CLASS_COUNT",
    "COUPLING_HASH_DOMAIN", "EDIT_INSERTION", "EDIT_REMOVAL",
    "EDIT_SUBSTITUTION", "MAX_TOKENS", "P0_CHARGED_MAX", "P0_NEUTRAL_MAX",
    "REPLICATE_SEED", "ROLE_COUNTS", "SMOKE_ROLE_COUNTS", "SOURCE_SENTINEL",
    "SUBMISSION_PHRASE", "TARGET_ASSIGNED_OFFLINE", "TARGET_HLT_DUSTBIN",
    "TARGET_NATIVE_SENTINEL", "TARGET_SLOT_SENTINEL", "artifact_reference",
    "build_coupling_config", "content_payload", "coordinate_payload",
    "load_artifact_reference", "validate_commit", "validate_content_hash",
    "validate_coordinate", "validate_coupling_config", "validate_exact_contract",
}]
