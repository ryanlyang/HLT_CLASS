"""Versioned HCWDL-RKD artifact identities and strict JSON envelopes.

This module is deliberately additive.  The logit-only HCWDL contracts keep
their existing meanings; representation artifacts must use one of the
identifiers frozen here and cannot be passed off as ``HCWDL_ARTIFACT/v1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from numbers import Integral
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)


SCHEMA_VERSION: Final = 1

# This is a hash-domain tag, not a reusable-artifact contract.  It therefore
# deliberately does not end in ``_CONTRACT`` and is not admitted by the
# versioned-artifact registry below.
LOGICAL_ARRAY_HASH_DOMAIN: Final = "HCWDL_REPRESENTATION_LOGICAL_ARRAY/v1"

PARENT_IMPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_PARENT_IMPORT/v3"
PARENT_LOSS_ATTESTATION_CONTRACT: Final = "HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v3"
ARCHITECTURE_ATTESTATION_CONTRACT: Final = "HCWDL_REPRESENTATION_ARCHITECTURE_ATTESTATION/v2"
ASCENT_GRAPH_CONTRACT: Final = "HCWDL_REPRESENTATION_DENSE_DESCENT_GRAPH/v1"
REPRESENTATION_RECIPE_CONTRACT: Final = "HCWDL_REPRESENTATION_RECIPE/v3"
KERNEL_RESOURCES_CONTRACT: Final = "HCWDL_REPRESENTATION_KERNEL_RESOURCES/v1"
TAP_CONTRACT: Final = "HCWDL_REPRESENTATION_TAP/v1"
SURFACE_PARITY_CONTRACT: Final = "HCWDL_REPRESENTATION_SURFACE_PARITY/v2"
TARGET_FORWARD_SPEC_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_FORWARD_SPEC/v2"
TARGET_EXECUTION_ATTESTATION_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_EXECUTION_ATTESTATION/v2"
TARGET_LOGICAL_BANK_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_LOGICAL_BANK/v2"
TARGET_CONSUMER_REGISTRY_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_CONSUMER_REGISTRY/v2"
TARGET_BUILD_INTENT_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_BUILD_INTENT/v2"
TARGET_GENERATION_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_GENERATION/v2"
TARGET_SHARD_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_SHARD/v2"
TARGET_MANIFEST_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_MANIFEST/v2"
TARGET_CLEANUP_AUTHORIZATION_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v1"
TARGET_CLEANUP_COMPLETION_CONTRACT: Final = "HCWDL_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v1"
TARGET_RECOVERY_PLAN_CONTRACT: Final = "HCWDL_REPRESENTATION_RECOVERY_PLAN/v1"
GRADIENT_CALIBRATION_CONTRACT: Final = "HCWDL_REPRESENTATION_GRADIENT_CALIBRATION/v1"
CALIBRATION_SELECTION_CONTRACT: Final = "HCWDL_REP_GRAD_CAL/v1"
GRADIENT_CALIBRATION_MANIFEST_CONTRACT: Final = "HCWDL_REPRESENTATION_GRADIENT_CALIBRATION_MANIFEST/v1"
DIAGNOSTIC_BATCH_CONTRACT: Final = "HCWDL_REPRESENTATION_DIAGNOSTIC_BATCH/v1"
NUMERICAL_ACCEPTANCE_CONTRACT: Final = "HCWDL_REPRESENTATION_NUMERICAL_ACCEPTANCE/v1"
SMOKE_PROBE_CONTRACT: Final = "HCWDL_REPRESENTATION_SMOKE_PROBE/v1"
PAIRED_BOOTSTRAP_CONTRACT: Final = "HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1"
CONTROL_REGISTRY_CONTRACT: Final = "HCWDL_REPRESENTATION_CONTROL_REGISTRY/v2"
ZERO_COEFFICIENT_ACCEPTANCE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_ZERO_COEFFICIENT_ACCEPTANCE/v1"
)
SHUFFLE_MAP_CONTRACT: Final = "HCWDL_REPRESENTATION_SHUFFLE_MAP/v1"
RESUME_STATE_CONTRACT: Final = "HCWDL_REPRESENTATION_RESUME_STATE/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_TRAINING_REPORT/v2"
SELECTED_TRAINING_CHECKPOINT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_SELECTED_TRAINING_CHECKPOINT/v2"
)
FINAL_TRAINING_CHECKPOINT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FINAL_TRAINING_CHECKPOINT/v2"
)
CHECKPOINT_SELECTION_CONTRACT: Final = "HCWDL_REPRESENTATION_CHECKPOINT_SELECTION/v1"
DEPLOYABLE_EXTRACTION_CONTRACT: Final = "HCWDL_REPRESENTATION_MODEL_EXTRACTION/v2"
SCREEN_AGGREGATE_CONTRACT: Final = "HCWDL_REPRESENTATION_SCREEN_AGGREGATE/v2"
CONFIRMATION_REGISTRY_CONTRACT: Final = "HCWDL_REPRESENTATION_CONFIRMATION_REGISTRY/v2"
CONFIRMATION_AGGREGATE_CONTRACT: Final = "HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v2"
CONFIRMATION_RUN_CONTRACT: Final = "HCWDL_REPRESENTATION_CONFIRMATION_RUN/v2"
VALIDATION_ONLY_AGGREGATE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_VALIDATION_ONLY_AGGREGATE/v1"
)
FINAL_DISPOSITION_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_DISPOSITION/v1"
PARENT_FINAL_STATE_CONTRACT: Final = "HCWDL_REPRESENTATION_PARENT_FINAL_STATE/v1"
SHARED_BINARY_ENVELOPE_CONTRACT: Final = "HCWDL_SHARED_IMMUTABLE_BINARY_ENVELOPE/v1"
SHARED_FINAL_POPULATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_POPULATION/v1"
SHARED_FINAL_POPULATION_DISJOINTNESS_CONTRACT: Final = "HCWDL_SHARED_FINAL_POPULATION_DISJOINTNESS/v1"
SHARED_FINAL_EXPOSURE_LEDGER_CONTRACT: Final = "HCWDL_SHARED_FINAL_EXPOSURE_LEDGER/v1"
SHARED_LEGACY_FINAL_EXPOSURE_CONTRACT: Final = (
    "HCWDL_SHARED_LEGACY_FINAL_EXPOSURE/v1"
)
SHARED_FINAL_POPULATION_REGISTRATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_POPULATION_REGISTRATION/v1"
SHARED_FINAL_RESERVATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_RESERVATION/v1"
SHARED_FINAL_LEGACY_CANCELLATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_LEGACY_CANCELLATION/v1"
FINAL_ASSIGNMENT_SPEC_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FINAL_ASSIGNMENT_SPEC/v1"
)
PRETRAINING_FINALIST_POLICY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_PRETRAINING_FINALIST_POLICY/v1"
)
FINALIST_LOCK_CONTRACT: Final = "HCWDL_REPRESENTATION_FINALIST_LOCK/v2"
SHARED_FINAL_TASK_REGISTRY_CONTRACT: Final = "HCWDL_SHARED_FINAL_TASK_REGISTRY/v1"
SHARED_FINAL_EXECUTION_CLAIM_CONTRACT: Final = "HCWDL_SHARED_FINAL_EXECUTION_CLAIM/v1"
SHARED_FINAL_ROLE_CAPABILITY_CONTRACT: Final = "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1"
SHARED_FINAL_RECOVERY_PLAN_CONTRACT: Final = "HCWDL_SHARED_FINAL_RECOVERY_PLAN/v1"
SHARED_FINAL_ROW_SELECTION_CONTRACT: Final = "HCWDL_SHARED_FINAL_ROW_SELECTION/v1"
SHARED_FINAL_LABEL_ESCROW_CONTRACT: Final = "HCWDL_SHARED_FINAL_LABEL_ESCROW/v1"
SHARED_FINAL_ASSIGNMENT_SHARD_CONTRACT: Final = (
    "HCWDL_SHARED_FINAL_ASSIGNMENT_SHARD/v1"
)
SHARED_FINAL_BRANCH_ACCESS_CONTRACT: Final = (
    "HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1"
)
SHARED_FINAL_ASSIGNMENT_AUDIT_CONTRACT: Final = "HCWDL_SHARED_FINAL_ASSIGNMENT_AUDIT/v1"
SHARED_FINAL_DATA_ATTESTATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1"
EXECUTION_LOCK_CONTRACT: Final = "HCWDL_REPRESENTATION_EXECUTION_LOCK/v2"
FINAL_PREDICTION_SPEC_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v2"
FINAL_EVALUATION_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_EVALUATION/v2"
PREDICTION_SHARD_CONTRACT: Final = "HCWDL_REPRESENTATION_PREDICTION_SHARD/v2"
PREDICTION_MANIFEST_CONTRACT: Final = "HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v2"
METRIC_JOIN_CONTRACT: Final = "HCWDL_REPRESENTATION_METRIC_JOIN/v2"
FINAL_AGGREGATE_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_AGGREGATE/v2"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_REPRESENTATION_CAMPAIGN_SPEC/v2"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_REPRESENTATION_COMMAND_PLAN/v2"
RUNTIME_BINDING_CONTRACT: Final = "HCWDL_REPRESENTATION_RUNTIME_BINDING/v1"
RUNTIME_PREREQUISITES_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_RUNTIME_PREREQUISITES/v1"
)
RUNTIME_DRY_RUN_AUDIT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_RUNTIME_DRY_RUN_AUDIT/v1"
)
LIVE_WORKER_RUNTIME_DOMAIN: Final = (
    "HCWDL_REPRESENTATION_LIVE_WORKER_RUNTIME/v1"
)
ROW_RUNTIME_SIGNATURE_DOMAIN: Final = (
    "HCWDL_REPRESENTATION_ROW_RUNTIME_SIGNATURE/v1"
)
WORKER_RUNTIME_MEASUREMENT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_WORKER_RUNTIME_MEASUREMENT/v1"
)
EXECUTABLE_CANDIDATE_AUDIT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_EXECUTABLE_CANDIDATE_AUDIT/v1"
)
ACCEPTANCE_BOOTSTRAP_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_ACCEPTANCE_BOOTSTRAP/v1"
)
NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_AUTHORITY/v1"
)
NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_INPUTS/v1"
)
NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY/v1"
)
NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT/v1"
)
ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_ACCEPTANCE_REAL_BATCH_FULL_LOSS/v1"
)
NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_RESULT/v1"
)
SUBMISSION_EVENT_CONTRACT: Final = "HCWDL_REPRESENTATION_SUBMISSION_EVENT/v1"
SUBMISSION_LEDGER_CONTRACT: Final = "HCWDL_REPRESENTATION_SUBMISSION_LEDGER/v1"
RECOVERY_SUBMISSION_LEDGER_CONTRACT: Final = "HCWDL_REPRESENTATION_RECOVERY_SUBMISSION_LEDGER/v1"
MONITOR_REPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_MONITOR_REPORT/v1"
RESOURCE_PROFILE_CONTRACT: Final = "HCWDL_REPRESENTATION_RESOURCE_PROFILE/v1"
STORAGE_ESTIMATE_CONTRACT: Final = "HCWDL_REPRESENTATION_STORAGE_ESTIMATE/v1"
FIXED_SIZE_INVENTORY_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_FIXED_SIZE_INVENTORY/v1"
)
SCHEDULER_EVIDENCE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_SCHEDULER_EVIDENCE/v1"
)
MINIATURE_EVIDENCE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_MINIATURE_EVIDENCE/v1"
)
TIGRIS_EVIDENCE_BUNDLE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_TIGRIS_EVIDENCE_BUNDLE/v2"
)
TIGRIS_ACTION_PROOF_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_TIGRIS_ACTION_PROOF/v1"
)
TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_TWO_UPDATE_ACCEPTANCE_PROOF/v1"
)
USR1_DELIVERY_RECEIPT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_USR1_DELIVERY_RECEIPT/v1"
)
USR1_EXACT_RESUME_PROOF_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_USR1_EXACT_RESUME_PROOF/v2"
)
VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_VALIDATION_PROXY_BRANCH_ACCESS/v1"
)
VALIDATION_PROXY_PROOF_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_VALIDATION_PROXY_PROOF/v2"
)
NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE/v1"
)
PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_PRODUCTION_WORKER_SMOKE_PROOF/v1"
)
LOCAL_SMOKE_REPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_LOCAL_SMOKE_REPORT/v1"
CACHE_MINIATURE_CONTRACT: Final = "HCWDL_REPRESENTATION_CACHE_MINIATURE/v1"
CACHE_MINIATURE_BANK_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_CACHE_MINIATURE_BANK/v1"
)
TIGRIS_ACCEPTANCE_CONTRACT: Final = "HCWDL_REPRESENTATION_TIGRIS_ACCEPTANCE/v2"
SUBMISSION_AUTHORIZATION_CONTRACT: Final = "HCWDL_REPRESENTATION_SUBMISSION_AUTHORIZATION/v1"

# Long-form aliases keep import sites self-describing and avoid confusing the
# additive representation graph/recipe with their logit-only parent names.
REPRESENTATION_ASCENT_GRAPH_CONTRACT: Final = ASCENT_GRAPH_CONTRACT
REPRESENTATION_DESCENT_GRAPH_CONTRACT: Final = ASCENT_GRAPH_CONTRACT
REPRESENTATION_CONTROL_REGISTRY_CONTRACT: Final = CONTROL_REGISTRY_CONTRACT
REPRESENTATION_RESUME_STATE_CONTRACT: Final = RESUME_STATE_CONTRACT


CONTRACTS: Final = frozenset(
    value for name, value in tuple(globals().items())
    if name.endswith("_CONTRACT") and isinstance(value, str)
)
CONTRACT_SCHEMA_VERSIONS: Final = {
    contract: (
        3 if contract == PARENT_LOSS_ATTESTATION_CONTRACT else
        2
        if contract in {
            PARENT_IMPORT_CONTRACT,
            ARCHITECTURE_ATTESTATION_CONTRACT,
            SURFACE_PARITY_CONTRACT,
        }
        else 1
    )
    for contract in CONTRACTS
}
CUSTOM_ENVELOPE_CONTRACTS: Final = frozenset({
    ARCHITECTURE_ATTESTATION_CONTRACT,
    PARENT_IMPORT_CONTRACT,
    PARENT_LOSS_ATTESTATION_CONTRACT,
    SURFACE_PARITY_CONTRACT,
})


def contract_schema_version(contract: str) -> int:
    """Return the exact JSON schema version for one registered contract."""

    if contract not in CONTRACT_SCHEMA_VERSIONS:
        raise ValueError(f"unknown HCWDL-RKD contract {contract!r}")
    return CONTRACT_SCHEMA_VERSIONS[contract]


def logical_array_sha256_from_byte_hash(
    *,
    name: str,
    dtype: str,
    shape: Sequence[int],
    c_order_byte_sha256: str,
    byte_length: int,
) -> str:
    """Return the canonical scientific identity of one logical C-order array.

    A serialized payload's byte SHA-256 is intentionally distinct from its
    logical array SHA-256.  The latter is a canonical-JSON digest over explicit
    type/shape/order metadata plus the raw-byte digest, as required by the
    campaign's single scientific hashing rule.  The helper also supports
    streaming producers, which can hash raw chunks first and wrap the finished
    byte digest without materializing the full array.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("HCWDL-RKD logical-array name is empty")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("HCWDL-RKD logical-array dtype is empty")
    normalized_shape: list[int] = []
    for raw_dimension in shape:
        if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, Integral):
            raise TypeError("HCWDL-RKD logical-array shape differs")
        dimension = int(raw_dimension)
        if dimension < 0:
            raise ValueError("HCWDL-RKD logical-array shape differs")
        normalized_shape.append(dimension)
    if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 0:
        raise ValueError("HCWDL-RKD logical-array byte length differs")
    digest = require_sha256(
        c_order_byte_sha256,
        name="HCWDL-RKD logical-array C-order byte SHA-256",
    )
    return canonical_sha256({
        "byte_length": int(byte_length),
        "c_order_byte_sha256": digest,
        "contract": LOGICAL_ARRAY_HASH_DOMAIN,
        "dtype": dtype,
        "name": name,
        "shape": normalized_shape,
        "storage_order": "C",
    })


def logical_array_sha256(name: str, value: Any) -> str:
    """Hash an in-memory NumPy-compatible array under the HCWDL logical rule."""

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("object arrays are forbidden from HCWDL-RKD logical hashes")
    raw = array.tobytes(order="C")
    return logical_array_sha256_from_byte_hash(
        name=name,
        dtype=array.dtype.str,
        shape=[int(dimension) for dimension in array.shape],
        c_order_byte_sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
    )


def validate_parent_hashes(parents: Mapping[str, Any], *, allow_empty: bool = False) -> dict[str, str]:
    """Return a sorted, strictly typed immutable-parent registry."""

    if not isinstance(parents, Mapping) or (not parents and not allow_empty):
        raise ValueError("HCWDL-RKD immutable parent registry is empty or invalid")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in parents.items():
        name = str(raw_name)
        if not name or name != raw_name or name in normalized:
            raise ValueError("HCWDL-RKD immutable parent name differs")
        normalized[name] = require_sha256(raw_digest, name=f"HCWDL-RKD parent {name}")
    return dict(sorted(normalized.items()))


def build_versioned_artifact(
    contract: str,
    *,
    parents: Mapping[str, Any],
    payload: Mapping[str, Any],
    allow_empty_parents: bool = False,
) -> dict[str, Any]:
    """Build the common strict JSON envelope used by RKD contracts."""

    if contract not in CONTRACTS:
        raise ValueError(f"unknown HCWDL-RKD contract {contract!r}")
    if contract in CUSTOM_ENVELOPE_CONTRACTS:
        raise ValueError(
            f"HCWDL-RKD contract {contract!r} requires its typed artifact builder"
        )
    if not isinstance(payload, Mapping):
        raise TypeError("HCWDL-RKD artifact payload must be a mapping")
    return with_content_hash({
        "contract": contract,
        "schema_version": contract_schema_version(contract),
        "parents": validate_parent_hashes(parents, allow_empty=allow_empty_parents),
        "payload": dict(payload),
    })


def validate_versioned_artifact(
    value: Mapping[str, Any],
    *,
    expected_contract: str,
    expected_parents: Mapping[str, Any] | None = None,
    required_payload_keys: Sequence[str] = (),
    allow_empty_parents: bool = False,
) -> str:
    """Authenticate a JSON artifact and optionally bind all expected parents."""

    if expected_contract not in CONTRACTS:
        raise ValueError(f"unknown HCWDL-RKD contract {expected_contract!r}")
    if expected_contract in CUSTOM_ENVELOPE_CONTRACTS:
        raise ValueError(
            f"HCWDL-RKD contract {expected_contract!r} requires its typed validator"
        )
    digest = validate_content_hash(
        value,
        expected_contract=expected_contract,
        expected_schema_version=contract_schema_version(expected_contract),
    )
    parents = validate_parent_hashes(
        value.get("parents", {}), allow_empty=allow_empty_parents,
    )
    if expected_parents is not None and parents != validate_parent_hashes(
        expected_parents, allow_empty=allow_empty_parents,
    ):
        raise ValueError("HCWDL-RKD artifact parent lineage differs")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("HCWDL-RKD artifact payload differs")
    missing = set(required_payload_keys) - set(payload)
    if missing:
        raise ValueError(f"HCWDL-RKD artifact payload is missing {sorted(missing)}")
    return digest


def derive_envelope_id(
    *,
    contract: str,
    producer_task_id: str,
    schema: Mapping[str, Any],
    immutable_parent_hashes: Mapping[str, Any],
    registered_output_row: Mapping[str, Any],
) -> str:
    """Derive the plan-frozen identity of a binary artifact envelope."""

    if contract not in CONTRACTS or not producer_task_id:
        raise ValueError("HCWDL-RKD envelope contract or producer task differs")
    parents = validate_parent_hashes(immutable_parent_hashes)
    if not isinstance(schema, Mapping) or not isinstance(registered_output_row, Mapping):
        raise TypeError("HCWDL-RKD envelope schema/output row must be mappings")
    return canonical_sha256({
        "contract": contract,
        "producer_task_id": producer_task_id,
        "schema": dict(schema),
        "immutable_parent_hashes": parents,
        "registered_output_row": dict(registered_output_row),
    })


def derive_envelope_owner_id(
    *,
    envelope_id: str,
    campaign_or_recovery_owner: Mapping[str, Any],
) -> str:
    """Derive an owner identity without incorporating produced bytes."""

    envelope = require_sha256(envelope_id, name="HCWDL-RKD envelope ID")
    if not isinstance(campaign_or_recovery_owner, Mapping) or not campaign_or_recovery_owner:
        raise ValueError("HCWDL-RKD envelope owner payload is empty")
    return canonical_sha256({
        "envelope_id": envelope,
        "campaign_or_recovery_owner": dict(campaign_or_recovery_owner),
    })


__all__ = sorted(
    [name for name in globals() if name.endswith("_CONTRACT")]
    + [
        "CONTRACTS", "CONTRACT_SCHEMA_VERSIONS", "CUSTOM_ENVELOPE_CONTRACTS",
        "SCHEMA_VERSION", "build_versioned_artifact", "contract_schema_version",
        "derive_envelope_id", "derive_envelope_owner_id",
        "logical_array_sha256", "logical_array_sha256_from_byte_hash",
        "validate_parent_hashes", "validate_versioned_artifact",
    ]
)
