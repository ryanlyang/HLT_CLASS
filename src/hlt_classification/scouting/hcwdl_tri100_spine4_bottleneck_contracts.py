"""Versioned artifacts for the bottleneck-pairing TRI100 four-spine control."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash,
    with_content_hash,
)


PREFIX: Final = "HCWDL_TRI100_FOUR_SPINE_FULLCARD_BOTTLENECK"
GRAPH_CONTRACT: Final = f"{PREFIX}_GRAPH/v3"
RECIPE_CONTRACT: Final = f"{PREFIX}_RECIPE/v3"
SOURCE_LOCK_CONTRACT: Final = f"{PREFIX}_SOURCE_LOCK/v3"
EXECUTION_ACCEPTANCE_CONTRACT: Final = f"{PREFIX}_EXECUTION_ACCEPTANCE/v3"
SPEC_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_SPEC/v3"
PLAN_CONTRACT: Final = f"{PREFIX}_COMMAND_PLAN/v3"
TRAINING_REPORT_CONTRACT: Final = f"{PREFIX}_TRAINING_REPORT/v3"
SELECTED_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_SELECTED_CHECKPOINT/v3"
FINAL_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_FINAL_CHECKPOINT/v3"
PROBABILITY_SHARD_CONTRACT: Final = f"{PREFIX}_PROBABILITY_SHARD/v3"
PROBABILITY_MANIFEST_CONTRACT: Final = f"{PREFIX}_PROBABILITY_MANIFEST/v3"
PROBABILITY_LOCK_CONTRACT: Final = f"{PREFIX}_PROBABILITY_LOCK/v3"
STAGE_REPORT_CONTRACT: Final = f"{PREFIX}_STAGE_REPORT/v3"
AGGREGATE_CONTRACT: Final = f"{PREFIX}_AGGREGATE/v3"
COMPLETE_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_COMPLETE/v3"
RECOVERY_SPEC_CONTRACT: Final = f"{PREFIX}_RECOVERY_SPEC/v3"
MONITOR_CONTRACT: Final = f"{PREFIX}_MONITOR/v3"
SCHEMA_VERSION: Final = 1

CONTRACTS: Final = (
    GRAPH_CONTRACT, RECIPE_CONTRACT, SOURCE_LOCK_CONTRACT,
    EXECUTION_ACCEPTANCE_CONTRACT, SPEC_CONTRACT, PLAN_CONTRACT,
    TRAINING_REPORT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT, PROBABILITY_SHARD_CONTRACT,
    PROBABILITY_MANIFEST_CONTRACT, PROBABILITY_LOCK_CONTRACT,
    STAGE_REPORT_CONTRACT, AGGREGATE_CONTRACT,
    COMPLETE_CONTRACT, RECOVERY_SPEC_CONTRACT, MONITOR_CONTRACT,
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown bottleneck four-spine contract")
    return with_content_hash({
        **dict(payload), "contract": contract, "schema_version": SCHEMA_VERSION,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown bottleneck four-spine contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=SCHEMA_VERSION,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "SCHEMA_VERSION", "artifact", "validate_artifact",
]
