"""Versioned contracts for the JetClass2 salience Strategy-B campaign."""
from __future__ import annotations

from typing import Final, Mapping

from .contracts import artifact as _artifact, validate as _validate


PREFIX: Final = "SALIENCE_LEARNED_HANDOFF_"
KINDS: Final = (
    "GRAPH", "NODE_SPEC", "RECIPE", "SOURCE_LOCK", "POPULATION_LOCK",
    "SEED_LOCK", "VALIDATION_PARTITION", "STORAGE_AUDIT",
    "EXECUTION_ACCEPTANCE", "CAMPAIGN_SPEC", "COMMAND_PLAN",
    "TRAINING_REPORT", "CHECKPOINT", "EXTRACTION", "PROBABILITY_BANK",
    "DIAGNOSTIC", "AGGREGATE", "CAMPAIGN_COMPLETE", "TASK_REPORT",
    "SUBMISSION_LEDGER", "MONITOR", "RECOVERY",
    "AUTOLAUNCH_SPEC", "AUTOLAUNCH_PLAN", "AUTOLAUNCH_RECEIPT",
)


def artifact(kind: str, *, version: int = 1, **fields) -> dict:
    if kind not in KINDS:
        raise ValueError(f"Unknown salience learned-handoff artifact: {kind}")
    return _artifact(PREFIX + kind, version=version, **fields)


def validate(value: Mapping, kind: str, *, version: int = 1) -> str:
    if kind not in KINDS:
        raise ValueError(f"Unknown salience learned-handoff artifact: {kind}")
    return _validate(value, PREFIX + kind, version=version)


__all__ = ["KINDS", "PREFIX", "artifact", "validate"]
