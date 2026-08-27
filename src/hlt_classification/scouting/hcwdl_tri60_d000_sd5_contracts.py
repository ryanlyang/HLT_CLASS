"""Contracts for the TRI60 matched-seed D000 ablation."""

from __future__ import annotations

from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash, with_content_hash,
)


PREFIX: Final = "HCWDL_TRI60_D000_SD5"
GRAPH_CONTRACT: Final = f"{PREFIX}_GRAPH/v1"
NODE_CONTRACT: Final = f"{PREFIX}_NODE/v1"
SPEC_CONTRACT: Final = f"{PREFIX}_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = f"{PREFIX}_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = f"{PREFIX}_TRAINING_REPORT/v1"
SELECTED_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_SELECTED_CHECKPOINT/v1"
FINAL_CHECKPOINT_CONTRACT: Final = f"{PREFIX}_FINAL_CHECKPOINT/v1"
ENSEMBLE_REPORT_CONTRACT: Final = f"{PREFIX}_ENSEMBLE_REPORT/v1"
AGGREGATE_CONTRACT: Final = f"{PREFIX}_AGGREGATE/v1"
CAMPAIGN_COMPLETE_CONTRACT: Final = f"{PREFIX}_CAMPAIGN_COMPLETE/v1"

CONTRACTS: Final = (
    GRAPH_CONTRACT, NODE_CONTRACT, SPEC_CONTRACT, COMMAND_PLAN_CONTRACT,
    TRAINING_REPORT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT, ENSEMBLE_REPORT_CONTRACT,
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
)


def artifact(payload: Mapping[str, Any], *, contract: str) -> dict[str, Any]:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 D000 SD5 contract")
    return with_content_hash({
        **dict(payload), "contract": contract, "schema_version": 1,
    })


def validate_artifact(value: Mapping[str, Any], *, contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError("unknown TRI60 D000 SD5 contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "CONTRACTS", "PREFIX", "artifact", "validate_artifact",
]
