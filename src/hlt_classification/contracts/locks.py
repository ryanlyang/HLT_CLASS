"""Finalist and final-test execution locks shared by future campaigns."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

FINALIST_LOCK_CONTRACT = "hlt_classification_finalist_lock_v1"
FINAL_TEST_EXECUTION_LOCK_CONTRACT = (
    "hlt_classification_final_test_execution_lock_v1"
)
LOCK_SCHEMA_VERSION = 1


def build_finalist_lock(
    *,
    campaign_spec_sha256: str,
    finalists: Sequence[Mapping[str, Any]],
    selection_artifacts: Mapping[str, str],
    source_snapshot_sha256: str,
) -> dict[str, Any]:
    rows = []
    graph_ids: set[str] = set()
    for finalist in finalists:
        graph_id = str(finalist["graph_id"])
        if not graph_id or graph_id in graph_ids:
            raise ValueError("finalist graph ids must be nonempty and unique")
        graph_ids.add(graph_id)
        rows.append(
            {
                "graph_id": graph_id,
                "checkpoint_sha256": require_sha256(
                    finalist["checkpoint_sha256"],
                    name="checkpoint_sha256",
                ),
                "training_report_sha256": require_sha256(
                    finalist["training_report_sha256"],
                    name="training_report_sha256",
                ),
            }
        )
    if not rows:
        raise ValueError("finalist lock requires at least one graph")
    artifacts = {
        str(name): require_sha256(value, name=f"selection_artifacts[{name}]")
        for name, value in sorted(selection_artifacts.items())
    }
    if not artifacts:
        raise ValueError("finalist lock requires selection artifact lineage")
    return with_content_hash(
        {
            "contract": FINALIST_LOCK_CONTRACT,
            "schema_version": LOCK_SCHEMA_VERSION,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256,
                name="campaign_spec_sha256",
            ),
            "source_snapshot_sha256": require_sha256(
                source_snapshot_sha256,
                name="source_snapshot_sha256",
            ),
            "finalists": rows,
            "selection_artifacts": artifacts,
        }
    )


def validate_finalist_lock(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=FINALIST_LOCK_CONTRACT,
    )
    expected = build_finalist_lock(
        campaign_spec_sha256=payload["campaign_spec_sha256"],
        finalists=payload["finalists"],
        selection_artifacts=payload["selection_artifacts"],
        source_snapshot_sha256=payload["source_snapshot_sha256"],
    )
    if dict(payload) != expected:
        raise ValueError("finalist lock semantics differ")
    return digest


def build_final_test_execution_lock(
    *,
    campaign_spec_sha256: str,
    finalist_lock_sha256: str,
    final_test_cache_manifest_sha256: str,
    source_snapshot_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": FINAL_TEST_EXECUTION_LOCK_CONTRACT,
            "schema_version": LOCK_SCHEMA_VERSION,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256,
                name="campaign_spec_sha256",
            ),
            "finalist_lock_sha256": require_sha256(
                finalist_lock_sha256,
                name="finalist_lock_sha256",
            ),
            "final_test_cache_manifest_sha256": require_sha256(
                final_test_cache_manifest_sha256,
                name="final_test_cache_manifest_sha256",
            ),
            "source_snapshot_sha256": require_sha256(
                source_snapshot_sha256,
                name="source_snapshot_sha256",
            ),
            "authorization": "one_locked_final_test_execution",
        }
    )


def validate_final_test_execution_lock(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    expected = build_final_test_execution_lock(
        campaign_spec_sha256=payload["campaign_spec_sha256"],
        finalist_lock_sha256=payload["finalist_lock_sha256"],
        final_test_cache_manifest_sha256=payload[
            "final_test_cache_manifest_sha256"
        ],
        source_snapshot_sha256=payload["source_snapshot_sha256"],
    )
    if dict(payload) != expected:
        raise ValueError("final-test execution lock semantics differ")
    return digest


def authorize_final_test_inference(
    *,
    finalist_lock: Mapping[str, Any],
    execution_lock: Mapping[str, Any],
    checkpoint_sha256: str,
    final_test_cache_manifest_sha256: str,
    source_snapshot_sha256: str,
    campaign_spec_sha256: str,
) -> str:
    finalist_hash = validate_finalist_lock(finalist_lock)
    execution_hash = validate_final_test_execution_lock(execution_lock)
    checkpoint_hash = require_sha256(
        checkpoint_sha256,
        name="checkpoint_sha256",
    )
    if checkpoint_hash not in {
        row["checkpoint_sha256"] for row in finalist_lock["finalists"]
    }:
        raise PermissionError("checkpoint is not present in finalist lock")
    campaign_hash = require_sha256(
        campaign_spec_sha256,
        name="campaign_spec_sha256",
    )
    if finalist_lock.get("campaign_spec_sha256") != campaign_hash:
        raise PermissionError("finalist campaign specification differs")
    if execution_lock.get("campaign_spec_sha256") != campaign_hash:
        raise PermissionError("execution campaign specification differs")
    expected = {
        "finalist_lock_sha256": finalist_hash,
        "final_test_cache_manifest_sha256": require_sha256(
            final_test_cache_manifest_sha256,
            name="final_test_cache_manifest_sha256",
        ),
        "source_snapshot_sha256": require_sha256(
            source_snapshot_sha256,
            name="source_snapshot_sha256",
        ),
    }
    for key, value in expected.items():
        if execution_lock.get(key) != value:
            raise PermissionError(f"final-test execution lock differs for {key}")
    if (
        finalist_lock["source_snapshot_sha256"]
        != expected["source_snapshot_sha256"]
    ):
        raise PermissionError("finalist source snapshot differs")
    return execution_hash


__all__ = [
    "FINALIST_LOCK_CONTRACT",
    "FINAL_TEST_EXECUTION_LOCK_CONTRACT",
    "authorize_final_test_inference",
    "build_final_test_execution_lock",
    "build_finalist_lock",
    "validate_final_test_execution_lock",
    "validate_finalist_lock",
]
