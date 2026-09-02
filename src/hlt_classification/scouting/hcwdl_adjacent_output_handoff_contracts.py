"""Versioned artifacts for the adjacent-view output-handoff campaign."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash, with_content_hash,
)


PREFIX: Final = "HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF"
SCHEMA_VERSION: Final = 1

_NAMES: Final = (
    "GRAPH", "RECIPE", "SOURCE_LOCK", "POPULATION_LOCK",
    "VALIDATION_PARTITION", "CONTROL_LOCK", "SEED_LOCK",
    "EXECUTION_ACCEPTANCE", "CAMPAIGN_SPEC", "COMMAND_PLAN",
    "TRAINING_REPORT", "SELECTED_CHECKPOINT", "FINAL_CHECKPOINT",
    "PROBABILITY_SHARD", "PROBABILITY_MANIFEST", "PROBABILITY_LOCK",
    "TEMPERATURE_CALIBRATION", "MIXTURE_CURVE", "BOOTSTRAP_REPORT",
    "SELECTED_MIXTURE", "ENSEMBLE_REPORT", "STAGE_REPORT",
    "AGGREGATE", "CAMPAIGN_COMPLETE", "TASK_ATTESTATION",
    "SUBMISSION_LEDGER", "MONITOR", "RECOVERY_SPEC",
    "RECOVERY_COMMAND_PLAN",
)
CONTRACTS: Final = tuple(f"{PREFIX}_{name}/v1" for name in _NAMES)

(
    GRAPH_CONTRACT, RECIPE_CONTRACT, SOURCE_LOCK_CONTRACT,
    POPULATION_LOCK_CONTRACT, VALIDATION_PARTITION_CONTRACT,
    CONTROL_LOCK_CONTRACT, SEED_LOCK_CONTRACT,
    EXECUTION_ACCEPTANCE_CONTRACT, SPEC_CONTRACT, PLAN_CONTRACT,
    TRAINING_REPORT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT, PROBABILITY_SHARD_CONTRACT,
    PROBABILITY_MANIFEST_CONTRACT, PROBABILITY_LOCK_CONTRACT,
    TEMPERATURE_CALIBRATION_CONTRACT, MIXTURE_CURVE_CONTRACT,
    BOOTSTRAP_REPORT_CONTRACT, SELECTED_MIXTURE_CONTRACT,
    ENSEMBLE_REPORT_CONTRACT, STAGE_REPORT_CONTRACT, AGGREGATE_CONTRACT,
    COMPLETE_CONTRACT, TASK_ATTESTATION_CONTRACT,
    SUBMISSION_LEDGER_CONTRACT, MONITOR_CONTRACT, RECOVERY_SPEC_CONTRACT,
    RECOVERY_PLAN_CONTRACT,
) = CONTRACTS


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown adjacent output-handoff contract")
    return with_content_hash({
        **dict(payload), "contract": contract, "schema_version": SCHEMA_VERSION,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown adjacent output-handoff contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=SCHEMA_VERSION,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "PREFIX", "SCHEMA_VERSION", "artifact", "validate_artifact",
]
