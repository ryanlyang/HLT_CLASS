"""Contracts for the standalone TRI60 D000-from-D033E 180-pass study."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash, with_content_hash,
)


GRAPH_CONTRACT: Final = "HCWDL_TRI60_D000_D033E_LONG180_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_TRI60_D000_D033E_LONG180_NODE/v1"
SOURCE_LOCK_CONTRACT: Final = "HCWDL_TRI60_D000_D033E_LONG180_SOURCE_LOCK/v1"
SPEC_CONTRACT: Final = "HCWDL_TRI60_D000_D033E_LONG180_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_TRI60_D000_D033E_LONG180_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = (
    "HCWDL_TRI60_D000_D033E_LONG180_TRAINING_REPORT/v1"
)
SELECTED_CHECKPOINT_CONTRACT: Final = (
    "HCWDL_TRI60_D000_D033E_LONG180_SELECTED_CHECKPOINT/v1"
)
FINAL_CHECKPOINT_CONTRACT: Final = (
    "HCWDL_TRI60_D000_D033E_LONG180_FINAL_CHECKPOINT/v1"
)
COMPARISON_CONTRACT: Final = "HCWDL_TRI60_D000_D033E_LONG180_COMPARISON/v1"

CONTRACTS: Final = (
    GRAPH_CONTRACT, NODE_CONTRACT, SOURCE_LOCK_CONTRACT, SPEC_CONTRACT,
    COMMAND_PLAN_CONTRACT, TRAINING_REPORT_CONTRACT,
    SELECTED_CHECKPOINT_CONTRACT, FINAL_CHECKPOINT_CONTRACT,
    COMPARISON_CONTRACT,
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 D000 long180 contract")
    return with_content_hash({
        **dict(payload), "contract": contract, "schema_version": 1,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 D000 long180 contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "artifact", "validate_artifact",
]
