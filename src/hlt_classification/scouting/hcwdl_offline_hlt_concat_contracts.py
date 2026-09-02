"""Versioned contracts for the isolated tagged concatenation pilot."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash, with_content_hash,
)


PREFIX: Final = "HCWDL_OFFLINE_HLT_TAGGED_CONCAT_PILOT"
SCHEMA_VERSION: Final = 1
GRAPH_CONTRACT: Final = f"{PREFIX}_GRAPH/v1"
RECIPE_CONTRACT: Final = f"{PREFIX}_RECIPE/v1"
SOURCE_LOCK_CONTRACT: Final = f"{PREFIX}_SOURCE_LOCK/v1"
CAPACITY_AUDIT_CONTRACT: Final = f"{PREFIX}_CAPACITY_AUDIT/v1"
EXECUTION_ACCEPTANCE_CONTRACT: Final = f"{PREFIX}_EXECUTION_ACCEPTANCE/v1"
SPEC_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_SPEC/v1"
PLAN_CONTRACT: Final = f"{PREFIX}_COMMAND_PLAN/v1"
NODE_CONTRACT: Final = f"{PREFIX}_NODE/v1"
TRAINING_REPORT_CONTRACT: Final = f"{PREFIX}_TRAINING_REPORT/v1"
SELECTED_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_SELECTED_CHECKPOINT/v1"
FINAL_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_FINAL_CHECKPOINT/v1"
AGGREGATE_CONTRACT: Final = f"{PREFIX}_AGGREGATE/v1"
COMPLETE_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_COMPLETE/v1"
MONITOR_CONTRACT: Final = f"{PREFIX}_MONITOR/v1"
RECOVERY_SPEC_CONTRACT: Final = f"{PREFIX}_RECOVERY_SPEC/v1"

CONTRACTS: Final = tuple(
    value for name, value in list(globals().items())
    if name.endswith("_CONTRACT") and isinstance(value, str)
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("tagged concatenation artifact contract differs")
    return with_content_hash({
        "contract": contract, "schema_version": SCHEMA_VERSION, **dict(payload),
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("tagged concatenation artifact contract differs")
    return validate_content_hash(
        value, expected_contract=contract,
        expected_schema_version=SCHEMA_VERSION,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "PREFIX", "SCHEMA_VERSION", "artifact", "validate_artifact",
]
