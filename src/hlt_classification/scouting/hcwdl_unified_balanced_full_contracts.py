"""Versioned contracts for the all-mapped HCWDL-UB three-arm scale-up."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_unified_balanced_full_graph import (
    ARM_IDS, ARM_WEIGHTS, META_GRAPH_SHA256, META_REGISTRY, arm_registry,
)


FOUNDATION_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_FOUNDATION_SPEC/v1"
FOUNDATION_COMMAND_PLAN_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_FOUNDATION_COMMAND_PLAN/v1"
FOUNDATION_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_FOUNDATION_LOCK/v1"
ARM_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_ARM_SPEC/v1"
ARM_COMMAND_PLAN_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_ARM_COMMAND_PLAN/v1"
GRAPH_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_GRAPH/v1"
RECIPE_OVERLAY_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_RECIPE_OVERLAY/v1"
ARM_RECIPE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_ARM_RECIPE/v1"
SWEEP_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_SWEEP/v1"
ENDPOINT_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_ENDPOINT_LOCK/v1"
ASSIGNMENT_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_ASSIGNMENT_LOCK/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_TRAINING_REPORT/v1"
ARM_AGGREGATE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_ARM_AGGREGATE/v1"
ARM_COMPLETION_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_ARM_COMPLETE/v1"
MONITOR_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_MONITOR/v1"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_RECOVERY_SPEC/v1"
MAPPED_IDENTITY_REPAIR_EVIDENCE_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_MAPPED_IDENTITY_REPAIR_EVIDENCE/v1"
)
MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_MAPPED_IDENTITY_RECOVERY_SPEC/v1"
)
RESOURCE_PROFILE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_RESOURCE_PROFILE/v1"
AUTOLAUNCH_RECEIPT_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_AUTOLAUNCH_RECEIPT/v1"
)
CAMPAIGN_SUBMISSION_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_CAMPAIGN_SUBMISSION/v1"
)
AUTOLAUNCH_EVENT_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_AUTOLAUNCH_EVENT/v1"
)


def _contract(value: Mapping[str, Any], contract: str) -> str:
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )


def _hashes(values: Mapping[str, str]) -> dict[str, str]:
    return {
        str(name): require_sha256(value, name=str(name))
        for name, value in sorted(values.items())
    }


def graph_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "meta_graph_sha256": META_GRAPH_SHA256,
        "fit_count": 38, "shared_fit_count": 2,
        "arm_fit_counts": {arm: 12 for arm in ARM_IDS},
        "arm_order": list(ARM_IDS),
        "factorized_only": True, "joint_nodes": [],
        "nodes": [META_REGISTRY[key].payload() for key in sorted(META_REGISTRY)],
        "final_test_accessed": False,
    })


def validate_graph(value: Mapping[str, Any]) -> str:
    digest = _contract(value, GRAPH_CONTRACT)
    if (
        value.get("meta_graph_sha256") != META_GRAPH_SHA256
        or value.get("fit_count") != 38
        or value.get("shared_fit_count") != 2
        or value.get("arm_fit_counts") != {arm: 12 for arm in ARM_IDS}
        or value.get("arm_order") != list(ARM_IDS)
        or value.get("factorized_only") is not True
        or value.get("joint_nodes") != []
        or [row.get("canonical_id") for row in value.get("nodes", [])]
        != sorted(META_REGISTRY)
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULL3 graph differs")
    return digest


def recipe_overlay_payload(*, base_recipe_sha256: str) -> dict[str, Any]:
    return with_content_hash({
        "contract": RECIPE_OVERLAY_CONTRACT, "schema_version": 1,
        "base_recipe_sha256": require_sha256(base_recipe_sha256, name="base recipe"),
        "training_passes": 20, "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "homotopy_temperature": 2.0,
        "m1": {"ce": .25, "parent_kd": .75, "grandparent_kd": 0.0,
               "temperature": 1.0},
        "performance_early_stopping": False,
    })


def arm_recipe_payload(
    *, arm_id: str, overlay_sha256: str,
) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB-FULL3 arm")
    ce, parent, grandparent = ARM_WEIGHTS[arm_id]
    return with_content_hash({
        "contract": ARM_RECIPE_CONTRACT, "schema_version": 1,
        "arm_id": arm_id,
        "overlay_sha256": require_sha256(overlay_sha256, name="recipe overlay"),
        "ce": ce, "parent_kd": parent, "grandparent_kd": grandparent,
        "first_edge_parent_kd": parent + grandparent,
        "first_edge_grandparent_kd": 0.0,
        "node_count": 12, "m1_fixed": True,
    })


def _role_counts(value: Mapping[str, Any]) -> dict[str, int]:
    result = {role: int(value[role]) for role in ("train", "validation", "final_test")}
    if any(count <= 0 for count in result.values()):
        raise ValueError("HCWDL-UB-FULL3 role counts must be positive")
    return result


def foundation_spec_payload(
    *, source_commit: str, project_dir: str | Path,
    campaign_root: str | Path, parent_homotopy_spec: str | Path,
    data_root: str | Path, role_counts: Mapping[str, int],
    parents: Mapping[str, str], artifact_paths: Mapping[str, str | Path],
    resources: Mapping[str, Mapping[str, Any]],
    semantic_source_sha256: Mapping[str, str], replicate_seed: int,
    live_submission_authorized: bool, authorization_phrase: str | None,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("HCWDL-UB-FULL3 source commit differs")
    counts = _role_counts(role_counts)
    required_paths = {
        "parent_homotopy_spec", "split_manifest", "parent_recipe",
        "parent_shell_lock", "parent_assignment_lock", "parent_matcher_resources",
        "matcher_resources",
        "selection_manifest", "recipe", "train_assignment_manifest",
        "validation_assignment_manifest", "train_base_manifest",
        "validation_base_manifest", "legacy_train_manifest",
        "legacy_validation_manifest",
    }
    if set(artifact_paths) != required_paths:
        raise ValueError("HCWDL-UB-FULL3 foundation artifact registry differs")
    if live_submission_authorized and authorization_phrase != (
        "AUTHORIZE HCWDL UB FULL3 ALL MAPPED FOUNDATION EXACT SPEC"
    ):
        raise PermissionError("HCWDL-UB-FULL3 foundation phrase differs")
    return with_content_hash({
        "contract": FOUNDATION_SPEC_CONTRACT, "schema_version": 1,
        "mode": "all_mapped_full3", "source_commit": source_commit,
        "project_dir": str(Path(project_dir).resolve()),
        "campaign_root": str(Path(campaign_root).resolve()),
        "parent_homotopy_spec": str(Path(parent_homotopy_spec).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "role_counts": counts,
        "ordinary_access_role_counts": {
            "train": counts["train"], "validation": counts["validation"],
            "final_test": 0,
        },
        "population_policy": "all_authenticated_mapped_rows_v1",
        "replicate_seed": int(replicate_seed),
        "parents": _hashes(parents),
        "artifact_paths": {
            key: str(Path(path).resolve()) for key, path in sorted(artifact_paths.items())
        },
        "resources": {key: dict(row) for key, row in sorted(resources.items())},
        "semantic_source_sha256": _hashes(semantic_source_sha256),
        "shared_nodes": ["shared/U000", "shared/M0paired"],
        "live_submission_authorized": bool(live_submission_authorized),
        "authorization_phrase": authorization_phrase if live_submission_authorized else None,
        "final_test_accessed": False,
    })


def assignment_lock_payload(
    *, foundation_spec_sha256: str, role_rows: Mapping[str, int],
    parents: Mapping[str, str], manifests: Mapping[str, str],
    recomputation_audits: Mapping[str, str], dustbin_fractions: Mapping[str, float],
) -> dict[str, Any]:
    roles = {"train", "validation"}
    if set(role_rows) != roles or set(manifests) != roles or set(recomputation_audits) != roles:
        raise ValueError("HCWDL-UB-FULL3 assignment-lock role registry differs")
    if set(dustbin_fractions) != roles:
        raise ValueError("HCWDL-UB-FULL3 assignment dustbin registry differs")
    rows = {role: int(role_rows[role]) for role in sorted(roles)}
    if any(value <= 0 for value in rows.values()):
        raise ValueError("HCWDL-UB-FULL3 assignment rows differ")
    dustbins = {role: float(dustbin_fractions[role]) for role in sorted(roles)}
    if any(not math.isfinite(value) or not 0 <= value < .10 for value in dustbins.values()):
        raise ValueError("HCWDL-UB-FULL3 assignment dustbin bound differs")
    return with_content_hash({
        "contract": ASSIGNMENT_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": require_sha256(
            foundation_spec_sha256, name="foundation spec",
        ),
        "role_rows": rows, "parents": _hashes(parents),
        "assignment_manifest_sha256": _hashes(manifests),
        "recomputation_audit_sha256": _hashes(recomputation_audits),
        "dustbin_fraction_hex": {
            role: dustbins[role].hex() for role in sorted(roles)
        },
        "strict_dustbin_fraction_upper_bound": .10,
        "complete_train_validation_coverage": True,
        "final_test_accessed": False,
    })


def validate_assignment_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ASSIGNMENT_LOCK_CONTRACT)
    if (
        set(value.get("role_rows", {})) != {"train", "validation"}
        or any(int(item) <= 0 for item in value.get("role_rows", {}).values())
        or value.get("strict_dustbin_fraction_upper_bound") != .10
        or value.get("complete_train_validation_coverage") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB-FULL3 assignment lock is incomplete")
    require_sha256(value.get("foundation_spec_sha256"), name="foundation spec")
    _hashes(value.get("parents", {}))
    _hashes(value.get("assignment_manifest_sha256", {}))
    _hashes(value.get("recomputation_audit_sha256", {}))
    for role, text in value.get("dustbin_fraction_hex", {}).items():
        parsed = float.fromhex(str(text))
        if role not in {"train", "validation"} or not 0 <= parsed < .10:
            raise ValueError("HCWDL-UB-FULL3 assignment dustbin evidence differs")
    return digest


def validate_foundation_spec(value: Mapping[str, Any]) -> str:
    digest = _contract(value, FOUNDATION_SPEC_CONTRACT)
    counts = _role_counts(value.get("role_counts", {}))
    if (
        value.get("mode") != "all_mapped_full3"
        or value.get("population_policy") != "all_authenticated_mapped_rows_v1"
        or value.get("ordinary_access_role_counts") != {
            "train": counts["train"], "validation": counts["validation"],
            "final_test": 0,
        }
        or value.get("shared_nodes") != ["shared/U000", "shared/M0paired"]
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULL3 foundation semantics differ")
    _hashes(value.get("parents", {}))
    _hashes(value.get("semantic_source_sha256", {}))
    if set(value.get("artifact_paths", {})) != {
        "parent_homotopy_spec", "split_manifest", "parent_recipe",
        "parent_shell_lock", "parent_assignment_lock", "parent_matcher_resources",
        "matcher_resources",
        "selection_manifest", "recipe", "train_assignment_manifest",
        "validation_assignment_manifest", "train_base_manifest",
        "validation_base_manifest", "legacy_train_manifest",
        "legacy_validation_manifest",
    }:
        raise ValueError("HCWDL-UB-FULL3 foundation artifact paths differ")
    return digest


def endpoint_lock_payload(
    *, foundation_spec_sha256: str, role_rows: Mapping[str, int],
    parents: Mapping[str, str], resource_measurement_sha256: str,
) -> dict[str, Any]:
    rows = {role: int(role_rows[role]) for role in ("train", "validation")}
    if any(value <= 0 for value in rows.values()):
        raise ValueError("HCWDL-UB-FULL3 endpoint rows differ")
    return with_content_hash({
        "contract": ENDPOINT_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": require_sha256(
            foundation_spec_sha256, name="foundation spec",
        ),
        "role_rows": rows, "parents": _hashes(parents),
        "resource_measurement_sha256": require_sha256(
            resource_measurement_sha256, name="resource measurement",
        ),
        "endpoint_checks": {
            "u000_exact_p0": True, "u100_exact_d100": True,
            "d0_exact_hlt": True, "no_durable_views": True,
            "prepared_endpoints_reused": True,
        },
        "labels_used_for_homotopy_construction": False,
        "final_test_accessed": False,
    })


def validate_endpoint_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ENDPOINT_LOCK_CONTRACT)
    if (
        set(value.get("role_rows", {})) != {"train", "validation"}
        or any(int(item) <= 0 for item in value.get("role_rows", {}).values())
        or not all(value.get("endpoint_checks", {}).values())
        or value.get("labels_used_for_homotopy_construction") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB-FULL3 endpoint lock differs")
    _hashes(value.get("parents", {}))
    require_sha256(value.get("resource_measurement_sha256"), name="resource measurement")
    return digest


def foundation_lock_payload(
    *, foundation_spec_sha256: str, role_counts: Mapping[str, int],
    parents: Mapping[str, str], u000_report_sha256: str,
    m0paired_report_sha256: str, u000_checkpoint_sha256: str,
    m0paired_checkpoint_sha256: str, u000_target_manifest_sha256: str,
    recipe_sha256: str,
) -> dict[str, Any]:
    return with_content_hash({
        "contract": FOUNDATION_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": require_sha256(
            foundation_spec_sha256, name="foundation spec",
        ),
        "role_counts": _role_counts(role_counts), "parents": _hashes(parents),
        "u000_report_sha256": require_sha256(u000_report_sha256, name="U000 report"),
        "m0paired_report_sha256": require_sha256(m0paired_report_sha256, name="M0 report"),
        "u000_checkpoint_sha256": require_sha256(u000_checkpoint_sha256, name="U000 checkpoint"),
        "m0paired_checkpoint_sha256": require_sha256(m0paired_checkpoint_sha256, name="M0 checkpoint"),
        "u000_target_manifest_sha256": require_sha256(
            u000_target_manifest_sha256, name="U000 target manifest",
        ),
        "recipe_sha256": require_sha256(recipe_sha256, name="full recipe"),
        "complete": True, "final_test_accessed": False,
    })


def validate_foundation_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, FOUNDATION_LOCK_CONTRACT)
    _role_counts(value.get("role_counts", {}))
    if value.get("complete") is not True or value.get("final_test_accessed") is not False:
        raise PermissionError("HCWDL-UB-FULL3 foundation lock is incomplete")
    for name in (
        "foundation_spec_sha256", "u000_report_sha256", "m0paired_report_sha256",
        "u000_checkpoint_sha256", "m0paired_checkpoint_sha256",
        "u000_target_manifest_sha256", "recipe_sha256",
    ):
        require_sha256(value.get(name), name=name)
    _hashes(value.get("parents", {}))
    return digest


def arm_spec_payload(
    *, arm_id: str, source_commit: str, project_dir: str | Path,
    campaign_root: str | Path, foundation_lock_path: str | Path,
    foundation_lock_sha256: str, graph_sha256: str,
    arm_recipe_sha256: str, resources: Mapping[str, Any],
    live_submission_authorized: bool, authorization_phrase: str | None,
) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB-FULL3 arm")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("HCWDL-UB-FULL3 arm source commit differs")
    if live_submission_authorized and authorization_phrase != (
        "AUTHORIZE HCWDL UB FULL3 THREE ARMS EXACT SPECS"
    ):
        raise PermissionError("HCWDL-UB-FULL3 arm phrase differs")
    return with_content_hash({
        "contract": ARM_SPEC_CONTRACT, "schema_version": 1,
        "arm_id": arm_id, "source_commit": source_commit,
        "project_dir": str(Path(project_dir).resolve()),
        "campaign_root": str(Path(campaign_root).resolve()),
        "foundation_lock_path": str(Path(foundation_lock_path).resolve()),
        "foundation_lock_sha256": require_sha256(
            foundation_lock_sha256, name="foundation lock",
        ),
        "graph_sha256": require_sha256(graph_sha256, name="graph"),
        "arm_recipe_sha256": require_sha256(arm_recipe_sha256, name="arm recipe"),
        "node_ids": list(arm_registry(arm_id)), "node_count": 12,
        "resources": dict(resources),
        "live_submission_authorized": bool(live_submission_authorized),
        "authorization_phrase": authorization_phrase if live_submission_authorized else None,
        "final_test_accessed": False,
    })


def validate_arm_spec(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_SPEC_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if (
        arm_id not in ARM_IDS
        or value.get("node_ids") != list(arm_registry(arm_id))
        or value.get("node_count") != 12
        or len(str(value.get("source_commit"))) != 40
        or any(c not in "0123456789abcdef" for c in str(value.get("source_commit")))
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULL3 arm specification differs")
    for name in ("foundation_lock_sha256", "graph_sha256", "arm_recipe_sha256"):
        require_sha256(value.get(name), name=name)
    return digest


def sweep_payload(*, foundation_lock_sha256: str, arm_specs: Mapping[str, str]) -> dict[str, Any]:
    if tuple(arm_specs) != ARM_IDS:
        raise ValueError("HCWDL-UB-FULL3 sweep arm order differs")
    return with_content_hash({
        "contract": SWEEP_CONTRACT, "schema_version": 1,
        "foundation_lock_sha256": require_sha256(
            foundation_lock_sha256, name="foundation lock",
        ),
        "arm_order": list(ARM_IDS), "arm_spec_sha256": _hashes(arm_specs),
        "expected_total_fit_count": 38, "read_only": True,
        "final_test_accessed": False,
    })


def aggregate_payload(
    *, arm_id: str, arm_spec_sha256: str,
    rows: Sequence[Mapping[str, Any]], shared: Mapping[str, Any],
    gpu_hours: float,
) -> dict[str, Any]:
    if [row.get("node_id") for row in rows] != list(arm_registry(arm_id)):
        raise ValueError("HCWDL-UB-FULL3 aggregate rows differ")
    if set(shared) != {"U000", "M0paired"}:
        raise ValueError("HCWDL-UB-FULL3 shared controls differ")
    if not math.isfinite(gpu_hours) or gpu_hours < 0:
        raise ValueError("HCWDL-UB-FULL3 GPU hours differ")
    return with_content_hash({
        "contract": ARM_AGGREGATE_CONTRACT, "schema_version": 1,
        "arm_id": arm_id,
        "arm_spec_sha256": require_sha256(arm_spec_sha256, name="arm spec"),
        "rows": [dict(row) for row in rows], "shared_controls": dict(shared),
        "gpu_hours": float(gpu_hours), "primary_metric": "macro_ovr_auc",
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    })


def validate_aggregate(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_AGGREGATE_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if (
        arm_id not in ARM_IDS
        or [row.get("node_id") for row in value.get("rows", [])]
        != list(arm_registry(arm_id))
        or set(value.get("shared_controls", {})) != {"U000", "M0paired"}
        or not math.isfinite(float(value.get("gpu_hours", -1)))
        or float(value.get("gpu_hours", -1)) < 0
        or value.get("primary_metric") != "macro_ovr_auc"
        or value.get("scientific_result_does_not_control_completion") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULL3 aggregate differs")
    require_sha256(value.get("arm_spec_sha256"), name="arm spec")
    return digest


def completion_payload(
    *, arm_id: str, arm_spec_sha256: str, aggregate_sha256: str,
    reports: Mapping[str, str], gpu_hours: float,
) -> dict[str, Any]:
    if set(reports) != set(arm_registry(arm_id)):
        raise ValueError("HCWDL-UB-FULL3 completion rows differ")
    return with_content_hash({
        "contract": ARM_COMPLETION_CONTRACT, "schema_version": 1,
        "arm_id": arm_id,
        "arm_spec_sha256": require_sha256(arm_spec_sha256, name="arm spec"),
        "aggregate_sha256": require_sha256(aggregate_sha256, name="aggregate"),
        "completed_node_report_sha256": _hashes(reports),
        "completed_node_count": 12, "gpu_hours": float(gpu_hours),
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    })


def validate_completion(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_COMPLETION_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if (
        arm_id not in ARM_IDS
        or set(value.get("completed_node_report_sha256", {}))
        != set(arm_registry(arm_id))
        or value.get("completed_node_count") != 12
        or not math.isfinite(float(value.get("gpu_hours", -1)))
        or float(value.get("gpu_hours", -1)) < 0
        or value.get("scientific_result_does_not_control_completion") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULL3 completion differs")
    require_sha256(value.get("arm_spec_sha256"), name="arm spec")
    require_sha256(value.get("aggregate_sha256"), name="aggregate")
    _hashes(value.get("completed_node_report_sha256", {}))
    return digest


def validate_autolaunch_receipt(value: Mapping[str, Any]) -> str:
    digest = _contract(value, AUTOLAUNCH_RECEIPT_CONTRACT)
    if (
        value.get("arm_order") != list(ARM_IDS)
        or set(value.get("arm_spec_sha256", {})) != set(ARM_IDS)
        or set(value.get("submission_ledger_sha256", {})) != set(ARM_IDS)
        or len(str(value.get("source_commit"))) != 40
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULL3 autolaunch receipt differs")
    require_sha256(value.get("foundation_lock_sha256"), name="foundation lock")
    require_sha256(value.get("recipe_sweep_sha256"), name="recipe sweep")
    _hashes(value.get("arm_spec_sha256", {}))
    _hashes(value.get("submission_ledger_sha256", {}))
    return digest


def validate_campaign_submission(value: Mapping[str, Any]) -> str:
    digest = _contract(value, CAMPAIGN_SUBMISSION_CONTRACT)
    dry_run = value.get("dry_run")
    if (
        dry_run not in {True, False}
        or value.get("arm_order") != list(ARM_IDS)
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULL3 campaign submission differs")
    require_sha256(value.get("foundation_spec_sha256"), name="foundation spec")
    require_sha256(
        value.get("foundation_submission_ledger_sha256"),
        name="foundation submission ledger",
    )
    if dry_run:
        if value.get("autolaunch_deferred_until_foundation_lock") is not True:
            raise ValueError("HCWDL-UB-FULL3 dry-run autolaunch boundary differs")
    else:
        require_sha256(value.get("autolaunch_event_sha256"), name="autolaunch event")
        if not str(value.get("foundation_lock_job_id", "")) or not str(
            value.get("autolaunch_job_id", "")
        ):
            raise ValueError("HCWDL-UB-FULL3 live submission IDs are absent")
    return digest


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "aggregate_payload", "arm_recipe_payload", "arm_spec_payload",
    "assignment_lock_payload",
    "completion_payload", "endpoint_lock_payload", "foundation_lock_payload",
    "foundation_spec_payload", "graph_payload", "recipe_overlay_payload",
    "sweep_payload", "validate_arm_spec", "validate_assignment_lock",
    "validate_aggregate", "validate_completion", "validate_endpoint_lock",
    "validate_foundation_lock", "validate_foundation_spec", "validate_graph",
    "validate_autolaunch_receipt", "validate_campaign_submission",
]
