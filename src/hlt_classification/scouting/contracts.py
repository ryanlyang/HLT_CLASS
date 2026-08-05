"""Versioned PMARD artifact envelopes and data-role capability enforcement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from hlt_classification.data.cache_contracts import validate_content_hash, with_content_hash

PMARD_CAMPAIGN_NAME: Final = "particle_matched_alpha_repair_distillation"
PMARD_ARTIFACT_CONTRACT: Final = "hlt_classification_pmard_artifact_v1"
PMARD_CONTRACT_VERSION: Final = 1
DATA_ROLES: Final = ("train", "validation", "final_test")
LOCK_ORDER: Final = (
    "data", "matcher_design", "matcher_result", "full_endpoint", "training",
    "screen_confirmation", "finalist", "execution",
)


def require_role_access(
    role: str,
    *,
    branch_read: bool = False,
    fit: bool = False,
    select: bool = False,
    completed_locks: Sequence[str] = (),
) -> None:
    if role not in DATA_ROLES:
        raise ValueError(f"unknown PMARD role {role!r}")
    if fit and role != "train":
        raise PermissionError("matcher/statistic fitting is train-only")
    if select and role != "validation":
        raise PermissionError("model selection is validation-only")
    if role == "final_test" and branch_read:
        required = {"finalist", "execution"}
        if not required.issubset(completed_locks):
            raise PermissionError("final-test branches are sealed until finalist and execution locks")


def artifact_envelope(
    *, kind: str, payload: Mapping[str, Any], parents: Mapping[str, str]
) -> dict[str, Any]:
    if not kind or not isinstance(kind, str):
        raise ValueError("artifact kind must be nonempty")
    return with_content_hash({
        "contract": PMARD_ARTIFACT_CONTRACT,
        "schema_version": PMARD_CONTRACT_VERSION,
        "campaign": PMARD_CAMPAIGN_NAME,
        "kind": kind,
        "parents": dict(sorted(parents.items())),
        "payload": dict(payload),
    })


def validate_artifact(
    artifact: Mapping[str, Any], *, expected_kind: str, expected_parents: Mapping[str, str]
) -> str:
    digest = validate_content_hash(
        artifact, expected_contract=PMARD_ARTIFACT_CONTRACT,
        expected_schema_version=PMARD_CONTRACT_VERSION,
    )
    if artifact.get("campaign") != PMARD_CAMPAIGN_NAME:
        raise ValueError("artifact campaign differs")
    if artifact.get("kind") != expected_kind:
        raise ValueError("artifact kind differs")
    if artifact.get("parents") != dict(sorted(expected_parents.items())):
        raise ValueError("artifact parent lineage differs")
    return digest


__all__ = [
    "DATA_ROLES", "LOCK_ORDER", "PMARD_ARTIFACT_CONTRACT", "PMARD_CAMPAIGN_NAME",
    "artifact_envelope", "require_role_access", "validate_artifact",
]
