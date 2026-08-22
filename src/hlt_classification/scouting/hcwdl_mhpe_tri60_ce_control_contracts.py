"""Contracts for the additive full-data 60-pass exact-HLT CE control."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash, with_content_hash,
)


GRAPH_CONTRACT: Final = "HCWDL_MHPE_TRI60_CE60_CONTROL_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_MHPE_TRI60_CE60_CONTROL_NODE/v1"
SPEC_CONTRACT: Final = "HCWDL_MHPE_TRI60_CE60_CONTROL_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_TRI60_CE60_CONTROL_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MHPE_TRI60_CE60_CONTROL_TRAINING_REPORT/v1"
SELECTED_CHECKPOINT_CONTRACT: Final = (
    "HCWDL_MHPE_TRI60_CE60_CONTROL_SELECTED_CHECKPOINT/v1"
)
FINAL_CHECKPOINT_CONTRACT: Final = (
    "HCWDL_MHPE_TRI60_CE60_CONTROL_FINAL_CHECKPOINT/v1"
)
CONTRACTS: Final = (
    GRAPH_CONTRACT, NODE_CONTRACT, SPEC_CONTRACT, COMMAND_PLAN_CONTRACT,
    TRAINING_REPORT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT,
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown HCWDL-MHPE CE60 control contract")
    return with_content_hash({
        **dict(payload), "contract": contract, "schema_version": 1,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown HCWDL-MHPE CE60 control contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "artifact", "validate_artifact",
]
