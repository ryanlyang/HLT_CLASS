from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from hlt_classification.scouting.hcwdl_representation_contracts import (
    CALIBRATION_SELECTION_CONTRACT,
    CONTRACTS,
    LOGICAL_ARRAY_HASH_DOMAIN,
    PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT,
    RUNTIME_BINDING_CONTRACT,
    SHARED_LEGACY_FINAL_EXPOSURE_CONTRACT,
    SUBMISSION_EVENT_CONTRACT,
    TIGRIS_ACTION_PROOF_CONTRACT,
    USR1_EXACT_RESUME_PROOF_CONTRACT,
    VALIDATION_PROXY_PROOF_CONTRACT,
    logical_array_sha256,
    logical_array_sha256_from_byte_hash,
)


# This is the literal Section 21 registry.  Keeping the expected spellings in
# the test prevents a similarly named parent/PMARD contract from being accepted
# accidentally and makes an in-place semantic expansion fail loudly.
EXPECTED_CONTRACTS = {
    "HCWDL_REPRESENTATION_PARENT_IMPORT/v1",
    "HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v1",
    "HCWDL_REPRESENTATION_ARCHITECTURE_ATTESTATION/v1",
    "HCWDL_REPRESENTATION_ASCENT_GRAPH/v1",
    "HCWDL_REPRESENTATION_RECIPE/v1",
    "HCWDL_REPRESENTATION_KERNEL_RESOURCES/v1",
    "HCWDL_REPRESENTATION_TAP/v1",
    "HCWDL_REPRESENTATION_SURFACE_PARITY/v1",
    "HCWDL_REPRESENTATION_TARGET_FORWARD_SPEC/v1",
    "HCWDL_REPRESENTATION_TARGET_EXECUTION_ATTESTATION/v1",
    "HCWDL_REPRESENTATION_TARGET_LOGICAL_BANK/v1",
    "HCWDL_REPRESENTATION_TARGET_CONSUMER_REGISTRY/v1",
    "HCWDL_REPRESENTATION_TARGET_BUILD_INTENT/v1",
    "HCWDL_REPRESENTATION_TARGET_GENERATION/v1",
    "HCWDL_REPRESENTATION_TARGET_SHARD/v1",
    "HCWDL_REPRESENTATION_TARGET_MANIFEST/v1",
    "HCWDL_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v1",
    "HCWDL_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v1",
    "HCWDL_REPRESENTATION_RECOVERY_PLAN/v1",
    "HCWDL_REPRESENTATION_GRADIENT_CALIBRATION/v1",
    "HCWDL_REP_GRAD_CAL/v1",
    "HCWDL_REPRESENTATION_GRADIENT_CALIBRATION_MANIFEST/v1",
    "HCWDL_REPRESENTATION_DIAGNOSTIC_BATCH/v1",
    "HCWDL_REPRESENTATION_NUMERICAL_ACCEPTANCE/v1",
    "HCWDL_REPRESENTATION_SMOKE_PROBE/v1",
    "HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1",
    "HCWDL_REPRESENTATION_CONTROL_REGISTRY/v1",
    "HCWDL_REPRESENTATION_ZERO_COEFFICIENT_ACCEPTANCE/v1",
    "HCWDL_REPRESENTATION_SHUFFLE_MAP/v1",
    "HCWDL_REPRESENTATION_RESUME_STATE/v1",
    "HCWDL_REPRESENTATION_TRAINING_REPORT/v1",
    "HCWDL_REPRESENTATION_SELECTED_TRAINING_CHECKPOINT/v1",
    "HCWDL_REPRESENTATION_FINAL_TRAINING_CHECKPOINT/v1",
    "HCWDL_REPRESENTATION_CHECKPOINT_SELECTION/v1",
    "HCWDL_REPRESENTATION_DEPLOYABLE_EXTRACTION/v1",
    "HCWDL_REPRESENTATION_SCREEN_AGGREGATE/v1",
    "HCWDL_REPRESENTATION_CONFIRMATION_REGISTRY/v1",
    "HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v1",
    "HCWDL_REPRESENTATION_CONFIRMATION_RUN/v1",
    "HCWDL_REPRESENTATION_VALIDATION_ONLY_AGGREGATE/v1",
    "HCWDL_REPRESENTATION_FINAL_DISPOSITION/v1",
    "HCWDL_REPRESENTATION_PARENT_FINAL_STATE/v1",
    "HCWDL_SHARED_IMMUTABLE_BINARY_ENVELOPE/v1",
    "HCWDL_SHARED_FINAL_POPULATION/v1",
    "HCWDL_SHARED_FINAL_POPULATION_DISJOINTNESS/v1",
    "HCWDL_SHARED_FINAL_EXPOSURE_LEDGER/v1",
    "HCWDL_SHARED_LEGACY_FINAL_EXPOSURE/v1",
    "HCWDL_SHARED_FINAL_POPULATION_REGISTRATION/v1",
    "HCWDL_SHARED_FINAL_RESERVATION/v1",
    "HCWDL_SHARED_FINAL_LEGACY_CANCELLATION/v1",
    "HCWDL_REPRESENTATION_FINAL_ASSIGNMENT_SPEC/v1",
    "HCWDL_REPRESENTATION_PRETRAINING_FINALIST_POLICY/v1",
    "HCWDL_REPRESENTATION_FINALIST_LOCK/v1",
    "HCWDL_SHARED_FINAL_TASK_REGISTRY/v1",
    "HCWDL_SHARED_FINAL_EXECUTION_CLAIM/v1",
    "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1",
    "HCWDL_SHARED_FINAL_RECOVERY_PLAN/v1",
    "HCWDL_SHARED_FINAL_ROW_SELECTION/v1",
    "HCWDL_SHARED_FINAL_LABEL_ESCROW/v1",
    "HCWDL_SHARED_FINAL_ASSIGNMENT_SHARD/v1",
    "HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1",
    "HCWDL_SHARED_FINAL_ASSIGNMENT_AUDIT/v1",
    "HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1",
    "HCWDL_REPRESENTATION_EXECUTION_LOCK/v1",
    "HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v1",
    "HCWDL_REPRESENTATION_FINAL_EVALUATION/v1",
    "HCWDL_REPRESENTATION_PREDICTION_SHARD/v1",
    "HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v1",
    "HCWDL_REPRESENTATION_METRIC_JOIN/v1",
    "HCWDL_REPRESENTATION_FINAL_AGGREGATE/v1",
    "HCWDL_REPRESENTATION_CAMPAIGN_SPEC/v1",
    "HCWDL_REPRESENTATION_COMMAND_PLAN/v1",
    "HCWDL_REPRESENTATION_RUNTIME_BINDING/v1",
    "HCWDL_REPRESENTATION_RUNTIME_PREREQUISITES/v1",
    "HCWDL_REPRESENTATION_RUNTIME_DRY_RUN_AUDIT/v1",
    "HCWDL_REPRESENTATION_WORKER_RUNTIME_MEASUREMENT/v1",
    "HCWDL_REPRESENTATION_EXECUTABLE_CANDIDATE_AUDIT/v1",
    "HCWDL_REPRESENTATION_SUBMISSION_EVENT/v1",
    "HCWDL_REPRESENTATION_SUBMISSION_LEDGER/v1",
    "HCWDL_REPRESENTATION_RECOVERY_SUBMISSION_LEDGER/v1",
    "HCWDL_REPRESENTATION_MONITOR_REPORT/v1",
    "HCWDL_REPRESENTATION_RESOURCE_PROFILE/v1",
    "HCWDL_REPRESENTATION_STORAGE_ESTIMATE/v1",
    "HCWDL_REPRESENTATION_FIXED_SIZE_INVENTORY/v1",
    "HCWDL_REPRESENTATION_SCHEDULER_EVIDENCE/v1",
    "HCWDL_REPRESENTATION_MINIATURE_EVIDENCE/v1",
    "HCWDL_REPRESENTATION_TIGRIS_EVIDENCE_BUNDLE/v1",
    "HCWDL_REPRESENTATION_TIGRIS_ACTION_PROOF/v1",
    "HCWDL_REPRESENTATION_USR1_EXACT_RESUME_PROOF/v1",
    "HCWDL_REPRESENTATION_VALIDATION_PROXY_PROOF/v1",
    "HCWDL_REPRESENTATION_PRODUCTION_WORKER_SMOKE_PROOF/v1",
    "HCWDL_REPRESENTATION_LOCAL_SMOKE_REPORT/v1",
    "HCWDL_REPRESENTATION_CACHE_MINIATURE/v1",
    "HCWDL_REPRESENTATION_CACHE_MINIATURE_BANK/v1",
    "HCWDL_REPRESENTATION_TIGRIS_ACCEPTANCE/v1",
    "HCWDL_REPRESENTATION_SUBMISSION_AUTHORIZATION/v1",
    "HCWDL_REPRESENTATION_ACCEPTANCE_BOOTSTRAP/v1",
}


def test_contract_registry_is_exactly_the_plan_frozen_section_21_registry() -> None:
    assert CONTRACTS == EXPECTED_CONTRACTS
    assert not any("PMARD" in contract for contract in CONTRACTS)
    assert "HCWDL_GRAPH/v1" not in CONTRACTS
    assert "HCWDL_RECIPE/v3" not in CONTRACTS


def test_durable_module_publications_use_central_contract_identities() -> None:
    from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
        PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT as evidence_worker_smoke,
        TIGRIS_ACTION_PROOF_CONTRACT as evidence_action,
        USR1_EXACT_RESUME_PROOF_CONTRACT as evidence_usr1,
        VALIDATION_PROXY_PROOF_CONTRACT as evidence_validation_proxy,
    )
    from hlt_classification.scouting.hcwdl_representation_calibration import (
        CALIBRATION_SELECTION_CONTRACT as calibration_selection,
    )
    from hlt_classification.scouting.hcwdl_representation_campaign import (
        SUBMISSION_EVENT_CONTRACT as campaign_submission_event,
    )
    from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
        RUNTIME_BINDING_CONTRACT as runtime_binding,
    )
    from hlt_classification.scouting.hcwdl_shared_final import (
        LEGACY_FINAL_EXPOSURE_CONTRACT as legacy_final_exposure,
    )

    assert calibration_selection == CALIBRATION_SELECTION_CONTRACT
    assert runtime_binding == RUNTIME_BINDING_CONTRACT
    assert legacy_final_exposure == SHARED_LEGACY_FINAL_EXPOSURE_CONTRACT
    assert campaign_submission_event == SUBMISSION_EVENT_CONTRACT
    assert evidence_action == TIGRIS_ACTION_PROOF_CONTRACT
    assert evidence_usr1 == USR1_EXACT_RESUME_PROOF_CONTRACT
    assert evidence_validation_proxy == VALIDATION_PROXY_PROOF_CONTRACT
    assert evidence_worker_smoke == PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT


def test_logical_array_hash_uses_the_plan_canonical_json_rule() -> None:
    value = np.asarray([[1.25, -3.5], [7.0, 0.0]], dtype="<f4")
    raw = value.tobytes(order="C")
    payload = {
        "byte_length": len(raw),
        "c_order_byte_sha256": hashlib.sha256(raw).hexdigest(),
        "contract": LOGICAL_ARRAY_HASH_DOMAIN,
        "dtype": value.dtype.str,
        "name": "fixture",
        "shape": [2, 2],
        "storage_order": "C",
    }
    expected = hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    assert logical_array_sha256("fixture", value) == expected
    assert logical_array_sha256_from_byte_hash(
        name="fixture",
        dtype=value.dtype.str,
        shape=value.shape,
        c_order_byte_sha256=payload["c_order_byte_sha256"],
        byte_length=len(raw),
    ) == expected


def test_logical_array_hash_is_type_shape_and_name_explicit() -> None:
    value = np.arange(8, dtype=np.uint8)
    baseline = logical_array_sha256("fixture", value)
    assert logical_array_sha256("other", value) != baseline
    assert logical_array_sha256("fixture", value.reshape(2, 4)) != baseline
    assert logical_array_sha256("fixture", value.astype(np.int8)) != baseline
    with pytest.raises(TypeError, match="object arrays"):
        logical_array_sha256("fixture", np.asarray([object()], dtype=object))
