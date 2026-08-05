"""Hash-chained PMARD freeze sequence and one-time final execution claim."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_json_bytes, require_sha256,
    validate_content_hash, with_content_hash,
)
from .contracts import LOCK_ORDER, PMARD_CAMPAIGN_NAME

PMARD_LOCK_CONTRACT = "hlt_classification_pmard_lock_v1"
PMARD_EXECUTION_CLAIM_CONTRACT = "hlt_classification_pmard_execution_claim_v1"


def create_lock(
    level: str, *, payload: Mapping[str, Any], campaign_spec_sha256: str,
    parent_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if level not in LOCK_ORDER:
        raise ValueError("unknown PMARD lock level")
    index = LOCK_ORDER.index(level)
    if index == 0 and parent_lock is not None:
        raise ValueError("data lock cannot have a predecessor")
    if index > 0:
        if parent_lock is None:
            raise ValueError(f"{level} lock requires its predecessor")
        validate_lock(parent_lock, expected_level=LOCK_ORDER[index - 1])
        parent_hash = parent_lock["content_hash"]
    else:
        parent_hash = None
    return with_content_hash({
        "contract": PMARD_LOCK_CONTRACT, "schema_version": 1,
        "campaign": PMARD_CAMPAIGN_NAME, "level": level,
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign_spec_sha256"),
        "parent_lock_sha256": parent_hash, "payload": dict(payload),
    })


def validate_lock(lock: Mapping[str, Any], *, expected_level: str) -> str:
    digest = validate_content_hash(lock, expected_contract=PMARD_LOCK_CONTRACT)
    if lock.get("campaign") != PMARD_CAMPAIGN_NAME or lock.get("level") != expected_level:
        raise ValueError("PMARD lock identity differs")
    require_sha256(lock.get("campaign_spec_sha256"), name="campaign_spec_sha256")
    if expected_level != "data":
        require_sha256(lock.get("parent_lock_sha256"), name="parent_lock_sha256")
    return digest


def claim_final_execution(
    path: str | Path, *, execution_lock: Mapping[str, Any], final_test_manifest_sha256: str,
) -> dict[str, Any]:
    execution_hash = validate_lock(execution_lock, expected_level="execution")
    claim = with_content_hash({
        "contract": PMARD_EXECUTION_CLAIM_CONTRACT, "schema_version": 1,
        "campaign": PMARD_CAMPAIGN_NAME, "execution_lock_sha256": execution_hash,
        "final_test_manifest_sha256": require_sha256(
            final_test_manifest_sha256, name="final_test_manifest_sha256"
        ),
        "state": "claimed_once",
    })
    destination = Path(path)
    data = canonical_json_bytes(claim) + b"\n"
    if destination.exists():
        raise FileExistsError("final-test execution was already claimed")
    atomic_publish_bytes(destination, data)
    return claim


__all__ = ["claim_final_execution", "create_lock", "validate_lock"]
