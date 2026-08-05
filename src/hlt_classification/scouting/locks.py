"""Hash-chained PMARD freeze sequence and one-time final execution claim."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_json_bytes, require_sha256,
    validate_content_hash, with_content_hash,
)
from .contracts import LOCK_ORDER, PMARD_CAMPAIGN_NAME
from .coverage import validate_full_role_coverage_report
from .matcher_validation import MATCHER_VALIDATION_CONTRACT
from .repair import FULL_REPAIR_FAMILY, SELECTIVE_FULL_REPAIR_FAMILY
from .selective_assignment import validate_assignment_manifest, validate_row_selection
from .training import MATCHER_FOLD_SEED

PMARD_LOCK_CONTRACT = "hlt_classification_pmard_lock_v4"
PMARD_LOCK_VERSION = 4
PMARD_EXECUTION_CLAIM_CONTRACT = "hlt_classification_pmard_execution_claim_v1"


def create_lock(
    level: str, *, payload: Mapping[str, Any], campaign_spec_sha256: str,
    parent_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    campaign_hash = require_sha256(campaign_spec_sha256, name="campaign_spec_sha256")
    if level not in LOCK_ORDER:
        raise ValueError("unknown PMARD lock level")
    index = LOCK_ORDER.index(level)
    if index == 0 and parent_lock is not None:
        raise ValueError("data lock cannot have a predecessor")
    if index > 0:
        if parent_lock is None:
            raise ValueError(f"{level} lock requires its predecessor")
        validate_lock(parent_lock, expected_level=LOCK_ORDER[index - 1])
        if parent_lock.get("campaign_spec_sha256") != campaign_hash:
            raise ValueError("PMARD lock predecessor belongs to a different campaign spec")
        parent_hash = parent_lock["content_hash"]
    else:
        parent_hash = None
    return with_content_hash({
        "contract": PMARD_LOCK_CONTRACT, "schema_version": PMARD_LOCK_VERSION,
        "campaign": PMARD_CAMPAIGN_NAME, "level": level,
        "campaign_spec_sha256": campaign_hash,
        "parent_lock_sha256": parent_hash, "payload": dict(payload),
    })


def validate_lock(lock: Mapping[str, Any], *, expected_level: str) -> str:
    digest = validate_content_hash(
        lock, expected_contract=PMARD_LOCK_CONTRACT,
        expected_schema_version=PMARD_LOCK_VERSION,
    )
    if lock.get("campaign") != PMARD_CAMPAIGN_NAME or lock.get("level") != expected_level:
        raise ValueError("PMARD lock identity differs")
    require_sha256(lock.get("campaign_spec_sha256"), name="campaign_spec_sha256")
    if expected_level != "data":
        require_sha256(lock.get("parent_lock_sha256"), name="parent_lock_sha256")
    return digest


def create_full_endpoint_authorization(
    *, matcher_result_lock: Mapping[str, Any], full_validation: Mapping[str, Any],
    full_role_coverage: Mapping[str, Any],
    campaign_spec_sha256: str,
) -> dict[str, Any]:
    """Authorize full repair only for complete, all-category matcher coverage."""

    matcher_result_sha256 = validate_lock(matcher_result_lock, expected_level="matcher_result")
    validation_sha256 = validate_content_hash(
        full_validation, expected_contract=MATCHER_VALIDATION_CONTRACT,
        expected_schema_version=2,
    )
    payload = matcher_result_lock.get("payload", {})
    split_sha256 = require_sha256(
        payload.get("split_manifest_sha256"), name="split_manifest_sha256",
    )
    fold_matcher_hashes = payload.get("fold_matcher_report_sha256")
    if not isinstance(fold_matcher_hashes, list) or len(fold_matcher_hashes) != 5:
        raise ValueError("matcher-result lock does not name five fold matcher reports")
    fold_matcher_hashes = [
        require_sha256(value, name=f"fold_matcher_report_sha256[{index}]")
        for index, value in enumerate(fold_matcher_hashes)
    ]
    if payload.get("validation_report_sha256") != validation_sha256:
        raise ValueError("full matcher validation lineage differs from matcher-result lock")
    categories = payload.get("category_eligibility")
    all_categories = {str(index): True for index in range(5)}
    selected_variant = payload.get("selected_variant")
    threshold = float(payload.get("threshold", -1.0))
    if selected_variant not in {f"M{index}" for index in range(6)} or not 0 <= threshold <= 1:
        raise ValueError("matcher-result execution settings are invalid")
    if payload.get("matcher_fold_seed") != MATCHER_FOLD_SEED:
        raise ValueError("matcher-result fold assignment seed differs")
    full_matcher_sha256 = require_sha256(
        payload.get("full_matcher_report_sha256"), name="full_matcher_report_sha256",
    )
    candidates = payload.get("matching_only_selection", ())
    selected = next(
        (row for row in candidates if row.get("variant") == selected_variant), None
    )
    full_variant = full_validation.get("variants", {}).get(selected_variant, {})
    validation_parents = full_validation.get("parents", {})
    coverage_sha256 = validate_full_role_coverage_report(full_role_coverage)
    coverage_parents = full_role_coverage.get("parents", {})
    failures = []
    if categories != all_categories:
        failures.append("all five particle categories are not authorized")
    if payload.get("meets_initial_precision_target") is not True:
        failures.append("matching precision selector did not pass")
    if selected is None:
        failures.append("selected cross-fit matcher candidate is absent")
    if full_variant.get("passes_initial_99pct_lcb") is not True:
        failures.append("selected full matcher did not pass its own precision bound")
    if full_validation.get("threshold") != threshold:
        failures.append("full matcher validation threshold differs")
    if validation_parents.get("split_manifest_sha256") != split_sha256:
        failures.append("full matcher validation split lineage differs")
    if validation_parents.get("matcher_report_sha256") != full_matcher_sha256:
        failures.append("full matcher validation model lineage differs")
    expected_coverage_parents = {
        "split_manifest_sha256": split_sha256,
        "matcher_result_lock_sha256": matcher_result_sha256,
        "full_matcher_report_sha256": full_matcher_sha256,
        **{
            f"matcher_fold_{fold}_report_sha256": fold_matcher_hashes[fold]
            for fold in range(5)
        },
    }
    if coverage_parents != dict(sorted(expected_coverage_parents.items())):
        failures.append("full-role coverage matcher lineage differs")
    if full_role_coverage.get("selected_variant") != selected_variant:
        failures.append("full-role coverage matcher variant differs")
    if full_role_coverage.get("threshold") != threshold:
        failures.append("full-role coverage threshold differs")
    if full_role_coverage.get("matcher_fold_seed") != MATCHER_FOLD_SEED:
        failures.append("full-role coverage fold assignment seed differs")
    if failures:
        raise PermissionError("full endpoint cannot be authorized: " + "; ".join(failures))
    return create_lock(
        "full_endpoint", campaign_spec_sha256=campaign_spec_sha256,
        parent_lock=matcher_result_lock, payload={
            "authorized": True, "repair_family": FULL_REPAIR_FAMILY,
            "eligible_categories": [0, 1, 2, 3, 4],
            "complete_assignment_fraction": 1.0,
            "selected_variant": selected_variant, "threshold": threshold,
            "threshold_hex": threshold.hex(), "split_manifest_sha256": split_sha256,
            "matcher_fold_seed": MATCHER_FOLD_SEED,
            "fold_matcher_report_sha256": fold_matcher_hashes,
            "matcher_result_lock_sha256": matcher_result_sha256,
            "full_matcher_report_sha256": full_matcher_sha256,
            "validation_report_sha256": validation_sha256,
            "full_role_coverage_sha256": coverage_sha256,
            "coverage_scope": "all_mapped_train_and_validation_rows_v1",
            "full_matcher_passes_initial_99pct_lcb": True,
        },
    )


def validate_full_endpoint_authorization(
    lock: Mapping[str, Any], *, matcher_report_sha256: str,
    matcher_variant: str, matcher_threshold: float, split_manifest_sha256: str,
    fold_matcher_report_sha256: Sequence[str],
) -> str:
    matcher_report_sha256 = require_sha256(
        matcher_report_sha256, name="matcher_report_sha256",
    )
    split_manifest_sha256 = require_sha256(
        split_manifest_sha256, name="split_manifest_sha256",
    )
    if len(fold_matcher_report_sha256) != 5:
        raise ValueError("full endpoint execution requires five fold matcher hashes")
    fold_matcher_report_sha256 = [
        require_sha256(value, name=f"fold_matcher_report_sha256[{index}]")
        for index, value in enumerate(fold_matcher_report_sha256)
    ]
    digest = validate_lock(lock, expected_level="full_endpoint")
    payload = lock.get("payload", {})
    expected = {
        "authorized": True, "repair_family": FULL_REPAIR_FAMILY,
        "eligible_categories": [0, 1, 2, 3, 4],
        "complete_assignment_fraction": 1.0,
        "selected_variant": matcher_variant, "threshold": float(matcher_threshold),
        "threshold_hex": float(matcher_threshold).hex(),
        "matcher_fold_seed": MATCHER_FOLD_SEED,
        "split_manifest_sha256": split_manifest_sha256,
        "fold_matcher_report_sha256": list(fold_matcher_report_sha256),
        "coverage_scope": "all_mapped_train_and_validation_rows_v1",
        "full_matcher_passes_initial_99pct_lcb": True,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise PermissionError("full endpoint authorization payload is incomplete")
    for name in (
        "matcher_result_lock_sha256", "full_matcher_report_sha256",
        "validation_report_sha256", "full_role_coverage_sha256",
    ):
        require_sha256(payload.get(name), name=name)
    if payload.get("full_matcher_report_sha256") != matcher_report_sha256:
        raise ValueError("full endpoint authorization names a different matcher report")
    return digest


def create_selective_assignment_authorization(
    *, matcher_result_lock: Mapping[str, Any], assignment_manifest: Mapping[str, Any],
    row_selection: Mapping[str, Any], campaign_spec_sha256: str,
) -> dict[str, Any]:
    """Bind selective repair to the canonical fitted matcher and durable cache."""

    matcher_lock_hash = validate_lock(matcher_result_lock, expected_level="matcher_result")
    payload = matcher_result_lock.get("payload", {})
    split_hash = require_sha256(payload.get("split_manifest_sha256"), name="split_manifest_sha256")
    selection_hash = validate_row_selection(row_selection, split_manifest_sha256=split_hash)
    assignment_hash = validate_assignment_manifest(
        assignment_manifest, split_manifest_sha256=split_hash,
        selection_manifest_sha256=selection_hash,
    )
    expected = {
        "selected_variant": assignment_manifest.get("variant"),
        "threshold": assignment_manifest.get("threshold"),
        "matcher_artifact_sha256": assignment_manifest.get("matcher_artifact_sha256"),
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("matcher lock and selective assignment cache differ")
    if set(assignment_manifest.get("roles", {})) != {"train", "validation"}:
        raise ValueError("training assignment cache must cover exactly train and validation")
    return create_lock(
        "full_endpoint", campaign_spec_sha256=campaign_spec_sha256,
        parent_lock=matcher_result_lock, payload={
            "authorized": True, "repair_family": SELECTIVE_FULL_REPAIR_FAMILY,
            "eligible_categories": [0, 1, 2, 3, 4],
            "unmatched_policy": "retain_exact_hlt_token_v1",
            "selected_variant": expected["selected_variant"],
            "threshold": expected["threshold"],
            "matcher_artifact_sha256": expected["matcher_artifact_sha256"],
            "split_manifest_sha256": split_hash,
            "row_selection_sha256": selection_hash,
            "assignment_manifest_sha256": assignment_hash,
            "matcher_result_lock_sha256": matcher_lock_hash,
        },
    )


def validate_selective_assignment_authorization(
    lock: Mapping[str, Any], *, assignment_manifest: Mapping[str, Any],
    row_selection: Mapping[str, Any], split_manifest_sha256: str,
) -> str:
    digest = validate_lock(lock, expected_level="full_endpoint")
    split_hash = require_sha256(split_manifest_sha256, name="split_manifest_sha256")
    selection_hash = validate_row_selection(row_selection, split_manifest_sha256=split_hash)
    assignment_hash = validate_assignment_manifest(
        assignment_manifest, split_manifest_sha256=split_hash,
        selection_manifest_sha256=selection_hash,
    )
    payload = lock.get("payload", {})
    expected = {
        "authorized": True, "repair_family": SELECTIVE_FULL_REPAIR_FAMILY,
        "eligible_categories": [0, 1, 2, 3, 4],
        "unmatched_policy": "retain_exact_hlt_token_v1",
        "selected_variant": assignment_manifest.get("variant"),
        "threshold": assignment_manifest.get("threshold"),
        "matcher_artifact_sha256": assignment_manifest.get("matcher_artifact_sha256"),
        "split_manifest_sha256": split_hash,
        "row_selection_sha256": selection_hash,
        "assignment_manifest_sha256": assignment_hash,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise PermissionError("selective assignment authorization payload differs")
    require_sha256(payload.get("matcher_result_lock_sha256"), name="matcher_result_lock_sha256")
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


__all__ = [
    "claim_final_execution", "create_full_endpoint_authorization", "create_lock",
    "create_selective_assignment_authorization", "validate_full_endpoint_authorization",
    "validate_selective_assignment_authorization", "validate_lock",
]
