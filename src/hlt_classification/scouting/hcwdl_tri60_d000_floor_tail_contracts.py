"""Versioned artifacts for the matched D000 floor-tail confirmation."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash, with_content_hash,
)


PREFIX: Final = "HCWDL_TRI60_D000_FLOOR_TAIL_CONFIRMATION"
REFERENCE_LOCK_CONTRACT: Final = f"{PREFIX}_REFERENCE_LOCK/v1"
GRAPH_CONTRACT: Final = f"{PREFIX}_GRAPH/v1"
NODE_CONTRACT: Final = f"{PREFIX}_NODE/v1"
SPEC_CONTRACT: Final = f"{PREFIX}_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = f"{PREFIX}_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = f"{PREFIX}_TRAINING_REPORT/v1"
SELECTED_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_SELECTED_CHECKPOINT/v1"
FINAL_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_FINAL_CHECKPOINT/v1"
CAMPAIGN_COMPLETE_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_COMPLETE/v1"

CONTRACTS: Final = (
    REFERENCE_LOCK_CONTRACT, GRAPH_CONTRACT, NODE_CONTRACT, SPEC_CONTRACT,
    COMMAND_PLAN_CONTRACT, TRAINING_REPORT_CONTRACT,
    SELECTED_CHECKPOINT_CONTRACT, FINAL_CHECKPOINT_CONTRACT,
    CAMPAIGN_COMPLETE_CONTRACT,
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown D000 floor-tail confirmation contract")
    return with_content_hash({
        **dict(payload), "contract": contract, "schema_version": 1,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown D000 floor-tail confirmation contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "PREFIX", "artifact", "validate_artifact",
]
