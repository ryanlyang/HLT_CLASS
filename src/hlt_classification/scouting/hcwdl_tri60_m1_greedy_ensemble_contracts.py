"""Contracts for the validation-only TRI60 M1 greedy ensemble diagnostic."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash, with_content_hash,
)


PREFIX: Final = "HCWDL_TRI60_M1_GREEDY_ENSEMBLE"
SOURCE_LOCK_CONTRACT: Final = f"{PREFIX}_SOURCE_LOCK/v1"
SPEC_CONTRACT: Final = f"{PREFIX}_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = f"{PREFIX}_COMMAND_PLAN/v1"
SHARD_REPORT_CONTRACT: Final = f"{PREFIX}_PREDICTION_SHARD/v1"
RESULT_REPORT_CONTRACT: Final = f"{PREFIX}_RESULT/v2"
CAMPAIGN_COMPLETE_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_COMPLETE/v1"
REDUCER_RECOVERY_SPEC_CONTRACT: Final = f"{PREFIX}_REDUCER_RECOVERY_SPEC/v1"

CONTRACTS: Final = (
    SOURCE_LOCK_CONTRACT, SPEC_CONTRACT, COMMAND_PLAN_CONTRACT,
    SHARD_REPORT_CONTRACT, RESULT_REPORT_CONTRACT,
    CAMPAIGN_COMPLETE_CONTRACT, REDUCER_RECOVERY_SPEC_CONTRACT,
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 M1 greedy-ensemble contract")
    return with_content_hash({
        **dict(payload), "contract": contract,
        "schema_version": 2 if contract == RESULT_REPORT_CONTRACT else 1,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 M1 greedy-ensemble contract")
    return validate_content_hash(
        value, expected_contract=contract,
        expected_schema_version=2 if contract == RESULT_REPORT_CONTRACT else 1,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "PREFIX", "artifact", "validate_artifact",
]
