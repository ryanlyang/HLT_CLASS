"""Fail-closed HCWDL authorization locks and the one-time test claim."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_json_bytes, load_json, require_sha256,
    validate_content_hash, with_content_hash,
)

from .hcwdl_ladder import GRAPH_SHA256
from .highcov_cache import validate_assignment_manifest


LOCK_CONTRACT: Final = "HCWDL_LOCK/v1"
EXECUTION_CLAIM_CONTRACT: Final = "HCWDL_EXECUTION_CLAIM/v1"
LOCK_ORDER: Final = (
    "assignment", "recipe", "shell_endpoint_qualification",
    "confirmation_registry", "finalist", "execution",
)


def create_lock(
    level: str, *, campaign_spec_sha256: str, payload: Mapping[str, Any],
    parent_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if level not in LOCK_ORDER:
        raise ValueError("unknown HCWDL lock level")
    campaign_hash = require_sha256(campaign_spec_sha256, name="campaign spec SHA-256")
    index = LOCK_ORDER.index(level)
    if index == 0:
        if parent_lock is not None:
            raise ValueError("HCWDL assignment lock has no lock parent")
        parent_hash = None
    else:
        if parent_lock is None:
            raise ValueError(f"HCWDL {level} lock requires its predecessor")
        parent_hash = validate_lock(parent_lock, expected_level=LOCK_ORDER[index - 1])
        if parent_lock.get("campaign_spec_sha256") != campaign_hash:
            raise ValueError("HCWDL lock parent belongs to another campaign")
    return with_content_hash({
        "contract": LOCK_CONTRACT, "schema_version": 1,
        "campaign": "HCWDL", "level": level,
        "campaign_spec_sha256": campaign_hash,
        "parent_lock_sha256": parent_hash,
        "graph_sha256": GRAPH_SHA256, "payload": dict(payload),
    })


def validate_lock(value: Mapping[str, Any], *, expected_level: str) -> str:
    digest = validate_content_hash(value, expected_contract=LOCK_CONTRACT, expected_schema_version=1)
    if expected_level not in LOCK_ORDER or value.get("campaign") != "HCWDL":
        raise ValueError("HCWDL lock identity differs")
    if value.get("level") != expected_level or value.get("graph_sha256") != GRAPH_SHA256:
        raise ValueError("HCWDL lock level or graph differs")
    require_sha256(value.get("campaign_spec_sha256"), name="campaign spec SHA-256")
    if expected_level != "assignment":
        require_sha256(value.get("parent_lock_sha256"), name="parent lock SHA-256")
    return digest


def create_assignment_lock(
    *, campaign_spec_sha256: str, train_manifest_path: str | Path,
    validation_manifest_path: str | Path, expected_train_rows: int,
    expected_validation_rows: int, expected_parents: Mapping[str, str],
    train_recomputation_sha256: str, validation_recomputation_sha256: str,
    matcher_resources_sha256: str,
) -> dict[str, Any]:
    train = validate_assignment_manifest(
        train_manifest_path, expected_role="train", expected_mapped_jets=expected_train_rows,
        expected_parents=expected_parents, require_sub10pct_dustbins=True,
    )
    validation = validate_assignment_manifest(
        validation_manifest_path, expected_role="validation",
        expected_mapped_jets=expected_validation_rows, expected_parents=expected_parents,
        require_sub10pct_dustbins=True,
    )
    return create_lock(
        "assignment", campaign_spec_sha256=campaign_spec_sha256, payload={
            "authorized": True,
            "scope": "complete_train_and_validation_roles_v1",
            "train_manifest_sha256": train["content_hash"],
            "validation_manifest_sha256": validation["content_hash"],
            "train_recomputation_sha256": require_sha256(
                train_recomputation_sha256, name="train recomputation SHA-256",
            ),
            "validation_recomputation_sha256": require_sha256(
                validation_recomputation_sha256, name="validation recomputation SHA-256",
            ),
            "matcher_resources_sha256": require_sha256(
                matcher_resources_sha256, name="matcher resources SHA-256",
            ),
            "train_dustbin_fraction_hex": float(train["dustbin_fraction"]).hex(),
            "validation_dustbin_fraction_hex": float(validation["dustbin_fraction"]).hex(),
            "strict_dustbin_fraction_upper_bound": 0.10,
        },
    )


def create_recipe_lock(
    *, campaign_spec_sha256: str, assignment_lock: Mapping[str, Any],
    recipe_sha256: str, evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return create_lock(
        "recipe", campaign_spec_sha256=campaign_spec_sha256,
        parent_lock=assignment_lock, payload={
            "authorized": True,
            "recipe_sha256": require_sha256(recipe_sha256, name="recipe SHA-256"),
            "evidence": {
                name: require_sha256(value, name=f"recipe evidence {name}")
                for name, value in sorted(evidence_hashes.items())
            },
        },
    )


def create_shell_endpoint_qualification_lock(
    *, campaign_spec_sha256: str, recipe_lock: Mapping[str, Any],
    qualification_report_sha256: str, assignment_manifest_sha256: str,
    endpoint_invariants_passed: bool,
) -> dict[str, Any]:
    if not endpoint_invariants_passed:
        raise PermissionError("Shell Exact endpoint invariants did not pass")
    return create_lock(
        "shell_endpoint_qualification", campaign_spec_sha256=campaign_spec_sha256,
        parent_lock=recipe_lock, payload={
            "authorized": True, "repair_family": "HIGHCOV_SHELL_EXACT/v1",
            "selection_role": "fixed_primary_not_validation_selected",
            "qualification_report_sha256": require_sha256(
                qualification_report_sha256, name="qualification report SHA-256",
            ),
            "assignment_manifest_sha256": require_sha256(
                assignment_manifest_sha256, name="assignment manifest SHA-256",
            ),
            "endpoint_invariants_passed": True,
        },
    )


def create_confirmation_registry_lock(
    *, campaign_spec_sha256: str, qualification_lock: Mapping[str, Any],
    screen_aggregate_sha256: str, registry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not registry or any("node_id" not in row or "seed" not in row for row in registry):
        raise ValueError("HCWDL confirmation registry differs")
    identities = [(str(row["node_id"]), int(row["seed"])) for row in registry]
    if len(identities) != len(set(identities)):
        raise ValueError("HCWDL confirmation registry repeats a node/seed")
    return create_lock(
        "confirmation_registry", campaign_spec_sha256=campaign_spec_sha256,
        parent_lock=qualification_lock, payload={
            "screen_aggregate_sha256": require_sha256(
                screen_aggregate_sha256, name="screen aggregate SHA-256",
            ),
            "registry": [dict(row) for row in registry],
            "poor_scientific_performance_does_not_remove_rows": True,
        },
    )


def create_finalist_lock(
    *, campaign_spec_sha256: str, confirmation_lock: Mapping[str, Any],
    confirmation_report_sha256: str, finalists: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not finalists:
        raise ValueError("HCWDL finalist set is empty")
    required = {"node_id", "checkpoint_sha256", "report_sha256"}
    for row in finalists:
        if not required.issubset(row):
            raise ValueError("HCWDL finalist record is incomplete")
        require_sha256(row["checkpoint_sha256"], name="finalist checkpoint SHA-256")
        require_sha256(row["report_sha256"], name="finalist report SHA-256")
    return create_lock(
        "finalist", campaign_spec_sha256=campaign_spec_sha256,
        parent_lock=confirmation_lock, payload={
            "confirmation_report_sha256": require_sha256(
                confirmation_report_sha256, name="confirmation report SHA-256",
            ),
            "finalists": [dict(row) for row in finalists],
            "selection_used_validation_only": True,
        },
    )


def create_execution_lock(
    *, campaign_spec_sha256: str, finalist_lock: Mapping[str, Any],
    split_manifest_sha256: str, final_test_selection_rule_sha256: str,
    matcher_resources_sha256: str, recipe_sha256: str, source_commit: str,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("HCWDL execution source commit differs")
    return create_lock(
        "execution", campaign_spec_sha256=campaign_spec_sha256,
        parent_lock=finalist_lock, payload={
            "authorized": True,
            "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split SHA-256"),
            "final_test_selection_rule_sha256": require_sha256(
                final_test_selection_rule_sha256, name="final-test selection-rule SHA-256",
            ),
            "matcher_resources_sha256": require_sha256(
                matcher_resources_sha256, name="matcher resources SHA-256",
            ),
            "recipe_sha256": require_sha256(recipe_sha256, name="recipe SHA-256"),
            "source_commit": source_commit, "repair_family": "HIGHCOV_SHELL_EXACT/v1",
            "graph_sha256": GRAPH_SHA256,
        },
    )


def claim_final_execution(
    path: str | Path, *, execution_lock: Mapping[str, Any],
    test_assignment_manifest_sha256: str,
) -> dict[str, Any]:
    execution_hash = validate_lock(execution_lock, expected_level="execution")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("HCWDL final-test execution was already claimed")
    claim = with_content_hash({
        "contract": EXECUTION_CLAIM_CONTRACT, "schema_version": 1,
        "execution_lock_sha256": execution_hash,
        "test_assignment_manifest_sha256": require_sha256(
            test_assignment_manifest_sha256, name="test assignment manifest SHA-256",
        ),
        "state": "claimed_once",
    })
    atomic_publish_bytes(destination, canonical_json_bytes(claim) + b"\n")
    return claim


def validate_final_execution_claim(
    value: Mapping[str, Any], *, execution_lock: Mapping[str, Any],
    test_assignment_manifest_sha256: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=EXECUTION_CLAIM_CONTRACT,
        expected_schema_version=1,
    )
    expected = {
        "execution_lock_sha256": validate_lock(
            execution_lock, expected_level="execution",
        ),
        "test_assignment_manifest_sha256": require_sha256(
            test_assignment_manifest_sha256,
            name="test assignment manifest SHA-256",
        ),
        "state": "claimed_once",
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise PermissionError("HCWDL final-test execution claim lineage differs")
    return digest


def recover_or_claim_final_execution(
    path: str | Path, *, execution_lock: Mapping[str, Any],
    test_assignment_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Create the one-time claim, or authenticate the exact prior claim.

    Reuse authorizes only an interrupted execution of the already frozen
    finalist registry.  A different lock or test assignment always fails.
    """

    destination = Path(path)
    if destination.exists():
        claim = load_json(destination)
        validate_final_execution_claim(
            claim, execution_lock=execution_lock,
            test_assignment_manifest_sha256=test_assignment_manifest_sha256,
        )
        return claim, "reused_existing_exact_claim"
    claim = claim_final_execution(
        destination, execution_lock=execution_lock,
        test_assignment_manifest_sha256=test_assignment_manifest_sha256,
    )
    return claim, "created_new_claim"


__all__ = [
    "EXECUTION_CLAIM_CONTRACT", "LOCK_CONTRACT", "LOCK_ORDER",
    "claim_final_execution", "create_assignment_lock", "create_confirmation_registry_lock",
    "create_execution_lock", "create_finalist_lock", "create_lock", "create_recipe_lock",
    "create_shell_endpoint_qualification_lock", "recover_or_claim_final_execution",
    "validate_final_execution_claim", "validate_lock",
]
