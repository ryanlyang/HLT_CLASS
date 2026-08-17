"""Versioned identities for the HCWDL factorized homotopy RKD supplement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)


PREFIX: Final = "HCWDL_HOMOTOPY_REPRESENTATION"
SCHEMA_VERSION: Final = 2
PARENT_IMPORT_CONTRACT: Final = f"{PREFIX}_PARENT_IMPORT/v2"
INTEGRATION_ATTESTATION_CONTRACT: Final = f"{PREFIX}_INTEGRATION_ATTESTATION/v2"
NODE_SPEC_CONTRACT: Final = f"{PREFIX}_NODE_SPEC/v2"
GRAPH_CONTRACT: Final = f"{PREFIX}_GRAPH/v2"
RECIPE_CONTRACT: Final = f"{PREFIX}_RECIPE/v2"
GRAPH_RECIPE_LOCK_CONTRACT: Final = f"{PREFIX}_GRAPH_RECIPE_LOCK/v2"
TARGET_SPEC_CONTRACT: Final = f"{PREFIX}_TARGET_SPEC/v2"
TARGET_GENERATION_CONTRACT: Final = f"{PREFIX}_TARGET_GENERATION/v2"
TARGET_SHARD_CONTRACT: Final = f"{PREFIX}_TARGET_SHARD/v2"
TARGET_MANIFEST_CONTRACT: Final = f"{PREFIX}_TARGET_MANIFEST/v2"
TARGET_CLEANUP_AUTHORIZATION_CONTRACT: Final = f"{PREFIX}_TARGET_CLEANUP_AUTHORIZATION/v2"
TARGET_CLEANUP_COMPLETION_CONTRACT: Final = f"{PREFIX}_TARGET_CLEANUP_COMPLETION/v2"
CALIBRATION_CONTRACT: Final = f"{PREFIX}_CALIBRATION/v2"
RESUME_STATE_CONTRACT: Final = f"{PREFIX}_RESUME_STATE/v2"
TRAINING_REPORT_CONTRACT: Final = f"{PREFIX}_TRAINING_REPORT/v2"
SELECTED_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_SELECTED_CHECKPOINT/v2"
DEPLOYABLE_EXTRACTION_CONTRACT: Final = f"{PREFIX}_DEPLOYABLE_EXTRACTION/v2"
AGGREGATE_CONTRACT: Final = f"{PREFIX}_AGGREGATE/v2"
CAMPAIGN_SPEC_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_SPEC/v2"
COMMAND_PLAN_CONTRACT: Final = f"{PREFIX}_COMMAND_PLAN/v2"
RUNTIME_BINDING_CONTRACT: Final = f"{PREFIX}_RUNTIME_BINDING/v2"
TASK_ATTESTATION_CONTRACT: Final = f"{PREFIX}_TASK_ATTESTATION/v2"
SUBMISSION_LEDGER_CONTRACT: Final = f"{PREFIX}_SUBMISSION_LEDGER/v2"
MONITOR_REPORT_CONTRACT: Final = f"{PREFIX}_MONITOR_REPORT/v2"
SOURCE_RECOVERY_CONTRACT: Final = f"{PREFIX}_SOURCE_RECOVERY/v2"
RESOURCE_RECOVERY_CONTRACT: Final = f"{PREFIX}_RESOURCE_RECOVERY/v2"
CAMPAIGN_COMPLETE_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_COMPLETE/v2"
RESUME_RETIREMENT_AUTHORIZATION_CONTRACT: Final = (
    f"{PREFIX}_RESUME_RETIREMENT_AUTHORIZATION/v1"
)
RESUME_RETIREMENT_COMPLETION_CONTRACT: Final = (
    f"{PREFIX}_RESUME_RETIREMENT_COMPLETION/v1"
)
RECIPE_COMPATIBILITY_CONTRACT: Final = f"{PREFIX}_RECIPE_COMPATIBILITY/v1"
PREREQUISITE_BUNDLE_CONTRACT: Final = f"{PREFIX}_PREREQUISITE_BUNDLE/v1"

AUTHORIZATION_PHRASE: Final = "AUTHORIZE HCWDL U RKD VALIDATION CAMPAIGN EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL U RKD VALIDATION CAMPAIGN EXACT SPEC"
SOURCE_RECOVERY_PHRASE: Final = "AUTHORIZE HCWDL U RKD FAILED CLOSURE RECOVERY"
RESOURCE_RECOVERY_PHRASE: Final = "AUTHORIZE HCWDL U RKD RESOURCE ONLY RECOVERY"
RESUME_RETIREMENT_PHRASE: Final = (
    "DELETE HCWDL U RKD COMPLETED RESUME GENERATIONS"
)

FIT_COUNT: Final = 22
TARGET_BANK_COUNT: Final = 21
REPLICATE_SEED: Final = 1337
ROLE_COUNTS: Final = {"train": 300_000, "validation": 100_000, "final_test": 0}
SMOKE_ROLE_COUNTS: Final = {"train": 4096, "validation": 4096, "final_test": 0}


def build_artifact(contract: str, *, parents: Mapping[str, object], **payload: Any) -> dict[str, Any]:
    """Publish one strict v2 envelope with normalized immutable parents."""

    normalized = {
        str(name): require_sha256(value, name=f"{contract} parent {name}")
        for name, value in sorted(parents.items())
    }
    return with_content_hash({
        "contract": contract,
        "schema_version": SCHEMA_VERSION,
        "parents": normalized,
        **payload,
        "final_test_accessed": False,
    })


def validate_artifact(
    value: Mapping[str, object], *, contract: str,
    required_parents: Sequence[str] = (), required_fields: Sequence[str] = (),
) -> str:
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=SCHEMA_VERSION,
    )
    parents = value.get("parents")
    if not isinstance(parents, Mapping):
        raise ValueError(f"{contract} lacks immutable parents")
    for name in required_parents:
        require_sha256(parents.get(name), name=f"{contract} parent {name}")
    missing = set(required_fields) - set(value)
    if missing:
        raise ValueError(f"{contract} lacks required fields {sorted(missing)}")
    if value.get("final_test_accessed") is not False:
        raise PermissionError("HCWDL-U-RKD artifacts may not access final test")
    return digest


__all__ = [name for name in globals() if name.endswith("_CONTRACT") or name in {
    "AUTHORIZATION_PHRASE", "FIT_COUNT", "REPLICATE_SEED", "RESOURCE_RECOVERY_PHRASE",
    "RESUME_RETIREMENT_PHRASE",
    "ROLE_COUNTS", "SCHEMA_VERSION", "SMOKE_ROLE_COUNTS", "SOURCE_RECOVERY_PHRASE", "SUBMISSION_PHRASE",
    "TARGET_BANK_COUNT", "build_artifact", "validate_artifact",
}]
