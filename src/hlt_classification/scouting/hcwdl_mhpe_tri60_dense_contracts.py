"""Versioned artifacts for the isolated TRI60 dense-extension campaign."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)


PREFIX: Final = "HCWDL_MHPE_TRI60_DENSE_EXTENSION"
GRAPH_CONTRACT: Final = f"{PREFIX}_GRAPH/v1"
NODE_CONTRACT: Final = f"{PREFIX}_NODE_SPEC/v1"
SOURCE_LOCK_CONTRACT: Final = f"{PREFIX}_SOURCE_LOCK/v1"
SOURCE_GATE_CONTRACT: Final = f"{PREFIX}_SOURCE_GATE/v1"
SPEC_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_SPEC/v1"
PLAN_CONTRACT: Final = f"{PREFIX}_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = f"{PREFIX}_TRAINING_REPORT/v1"
SELECTED_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_SELECTED_CHECKPOINT/v1"
FINAL_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_FINAL_CHECKPOINT/v1"
PROBABILITY_SHARD_CONTRACT: Final = f"{PREFIX}_PROBABILITY_SHARD/v1"
PROBABILITY_MANIFEST_CONTRACT: Final = f"{PREFIX}_PROBABILITY_MANIFEST/v1"
PROBABILITY_LOCK_CONTRACT: Final = f"{PREFIX}_PROBABILITY_LOCK/v1"
STAGE_REPORT_CONTRACT: Final = f"{PREFIX}_STAGE_REPORT/v1"
AGGREGATE_CONTRACT: Final = f"{PREFIX}_AGGREGATE/v1"
FINALIST_LOCK_CONTRACT: Final = f"{PREFIX}_FINALIST_LOCK/v1"
COMPLETE_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_COMPLETE/v1"
RECOVERY_SPEC_CONTRACT: Final = f"{PREFIX}_RECOVERY_SPEC/v1"
MONITOR_CONTRACT: Final = f"{PREFIX}_MONITOR/v1"
SCHEMA_VERSION: Final = 1

CONTRACTS: Final = (
    GRAPH_CONTRACT, NODE_CONTRACT, SOURCE_LOCK_CONTRACT, SOURCE_GATE_CONTRACT,
    SPEC_CONTRACT, PLAN_CONTRACT, TRAINING_REPORT_CONTRACT,
    SELECTED_CHECKPOINT_CONTRACT, FINAL_CHECKPOINT_CONTRACT,
    PROBABILITY_SHARD_CONTRACT, PROBABILITY_MANIFEST_CONTRACT,
    PROBABILITY_LOCK_CONTRACT, STAGE_REPORT_CONTRACT, AGGREGATE_CONTRACT,
    FINALIST_LOCK_CONTRACT, COMPLETE_CONTRACT, RECOVERY_SPEC_CONTRACT,
    MONITOR_CONTRACT,
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 dense-extension contract")
    return with_content_hash({
        **dict(payload), "contract": contract, "schema_version": SCHEMA_VERSION,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 dense-extension contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=SCHEMA_VERSION,
    )


def hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not values:
        raise ValueError("TRI60 dense-extension parent registry is empty")
    return {
        str(name): require_sha256(value, name=str(name))
        for name, value in sorted(values.items())
    }


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "SCHEMA_VERSION", "artifact", "hashes", "validate_artifact",
]
