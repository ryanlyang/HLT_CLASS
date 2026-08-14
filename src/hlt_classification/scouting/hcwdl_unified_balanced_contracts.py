"""Versioned persistent contracts for HCWDL unified-root balanced homotopy."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_unified_balanced import (
    BALANCED_ORDER_DOMAIN, BALANCED_PHASE_DOMAIN,
)
from .hcwdl_unified_balanced_graph import (
    ARM_IDS, ARM_WEIGHTS, META_GRAPH_SHA256, META_REGISTRY, arm_registry,
)


BALANCED_SWITCH_CONFIG_CONTRACT: Final = "HCWDL_BALANCED_STRUCTURAL_SWITCH_CONFIG/v1"
BALANCED_SWITCH_SIDECAR_CONTRACT: Final = "HCWDL_BALANCED_STRUCTURAL_SWITCH_SIDECAR/v1"
BALANCED_SWITCH_MANIFEST_CONTRACT: Final = "HCWDL_BALANCED_STRUCTURAL_SWITCH_MANIFEST/v1"
UNIFORM_REPAIR_CONTRACT: Final = "HCWDL_UNIFORM_SHELL_EXACT/v1"
COORDINATE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_COORDINATE/v1"
NODE_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_NODE_SPEC/v1"
GRAPH_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_GRAPH/v1"
RECIPE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_RECIPE/v1"
RECIPE_ARM_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_RECIPE_ARM/v1"
RECIPE_SWEEP_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_RECIPE_SWEEP/v1"
RECIPE_SWEEP_AGGREGATE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_RECIPE_SWEEP_AGGREGATE/v1"
ENDPOINT_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_ENDPOINT_LOCK/v1"
FOUNDATION_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FOUNDATION_SPEC/v1"
FOUNDATION_COMMAND_PLAN_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FOUNDATION_COMMAND_PLAN/v1"
FOUNDATION_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FOUNDATION_LOCK/v1"
ARM_CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_ARM_CAMPAIGN_SPEC/v1"
ARM_COMMAND_PLAN_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_ARM_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_TRAINING_REPORT/v1"
ARM_AGGREGATE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_ARM_AGGREGATE/v1"
ARM_COMPLETION_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_ARM_CAMPAIGN_COMPLETE/v1"
FINALIST_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FINALIST_LOCK/v1"
EXECUTION_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_EXECUTION_LOCK/v1"
FINAL_EVALUATION_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FINAL_EVALUATION/v1"
CAMPAIGN_COMPLETION_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_CAMPAIGN_COMPLETE/v1"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_RECOVERY_SPEC/v1"
RESOURCE_RECOVERY_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_RESOURCE_RECOVERY_SPEC/v1"
OPERATIONAL_WAIVER_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_OPERATIONAL_EVIDENCE_WAIVER/v1"
TARGET_SHARD_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_TARGET_SHARD/v1"
TARGET_MANIFEST_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_TARGET_MANIFEST/v1"
TARGET_LOCK_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_TARGET_LOCK/v1"

ROLE_COUNTS: Final = {"train": 300_000, "validation": 100_000, "final_test": 100_000}
ORDINARY_ROLE_COUNTS: Final = {"train": 300_000, "validation": 100_000, "final_test": 0}
WAIVER_VERIFICATION_CLAIMS: Final = {
    "python_compilation", "focused_tests", "complete_repository_suite",
    "synthetic_end_to_end", "cli_help", "contract_versions",
    "markdown_links", "diff_check",
    "installed_weaver_parity_carried_by_hash",
    "prepared_endpoint_baseline_bound_by_guide",
}


def _hashes(values: Mapping[str, str]) -> dict[str, str]:
    return {
        str(name): require_sha256(value, name=str(name))
        for name, value in sorted(values.items())
    }


def _contract(value: Mapping[str, Any], contract: str) -> str:
    return validate_content_hash(value, expected_contract=contract, expected_schema_version=1)


def balanced_switch_config_payload(*, base_coupling_lock_sha256: str) -> dict[str, Any]:
    return with_content_hash({
        "contract": BALANCED_SWITCH_CONFIG_CONTRACT, "schema_version": 1,
        "base_coupling_lock_sha256": require_sha256(
            base_coupling_lock_sha256, name="base coupling lock",
        ),
        "mass_source": "authenticated_residual_base_mass_q_v1",
        "stratum_fields": [
            "edit_kind", "source_category_six_state", "target_category_six_state",
            "source_target_charged_applicability", "validity_change_mask_u8",
        ],
        "order_hash_domain": BALANCED_ORDER_DOMAIN,
        "phase_hash_domain": BALANCED_PHASE_DOMAIN,
        "placement": "per_jet_per_stratum_mass_arc_rotated_midpoint_v1",
        "phase_bits": 64, "coordinate_dtype": "uint16_little_endian",
        "rounding": "exact_rational_round_half_up_v1",
        "endpoint_overrides": {"zero": "apply_none", "one": "apply_all"},
        "labels_read": False, "final_test_accessed": False,
    })


def validate_balanced_switch_config(value: Mapping[str, Any]) -> str:
    digest = _contract(value, BALANCED_SWITCH_CONFIG_CONTRACT)
    require_sha256(value.get("base_coupling_lock_sha256"), name="base coupling lock")
    if (
        value.get("order_hash_domain") != BALANCED_ORDER_DOMAIN
        or value.get("phase_hash_domain") != BALANCED_PHASE_DOMAIN
        or value.get("phase_bits") != 64
        or value.get("coordinate_dtype") != "uint16_little_endian"
        or value.get("labels_read") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("balanced switch configuration semantics differ")
    return digest


def uniform_repair_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": UNIFORM_REPAIR_CONTRACT, "schema_version": 1,
        "continuous_strength": "one_exact_rational_alpha_for_every_matched_slot",
        "discrete_hash_input": [
            "repair_contract", "campaign_discrete_seed", "canonical_jet_identity",
            "immutable_target_hlt_slot", "semantic_group",
        ],
        "discrete_rule": "hash_u64_times_denominator_lt_numerator_times_2pow64",
        "confidence_is_coordinate_input": False,
        "identity_atomic": True, "validity_group_atomic": True,
        "endpoint_zero": "canonical_hlt", "endpoint_one": "exact_d100",
    })


def coordinate_payload() -> dict[str, Any]:
    coordinates = {}
    for key, node in sorted(META_REGISTRY.items()):
        coordinate = node.coordinate.payload()
        coordinates.setdefault(node.node_id, coordinate)
        if coordinates[node.node_id] != coordinate:
            raise RuntimeError("same-name HCWDL-UB coordinates differ across arms")
    return with_content_hash({
        "contract": COORDINATE_CONTRACT, "schema_version": 1,
        "default_factorized_step_percent": 20,
        "default_joint_step_percent": 10,
        "coordinates": dict(sorted(coordinates.items())),
        "grid_independent_switches": True,
    })


def graph_payload() -> dict[str, Any]:
    nodes = []
    for canonical_id, node in sorted(META_REGISTRY.items()):
        payload = node.payload(); payload.update({
            "contract": NODE_SPEC_CONTRACT, "schema_version": 1,
        })
        nodes.append(payload)
    return with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "meta_graph_sha256": META_GRAPH_SHA256,
        "fit_count": len(nodes), "shared_fit_count": 2,
        "arm_fit_counts": {arm: len(arm_registry(arm)) for arm in ARM_IDS},
        "nodes": nodes,
        "no_cross_arm_teachers": True,
    })


def validate_graph(value: Mapping[str, Any]) -> str:
    digest = _contract(value, GRAPH_CONTRACT)
    if (
        value.get("meta_graph_sha256") != META_GRAPH_SHA256
        or value.get("fit_count") != 151
        or value.get("shared_fit_count") != 2
        or value.get("arm_fit_counts") != {
            arm: len(arm_registry(arm)) for arm in ARM_IDS
        }
        or value.get("no_cross_arm_teachers") is not True
    ):
        raise ValueError("HCWDL-UB graph registry differs")
    rows = value.get("nodes")
    if not isinstance(rows, list) or [row.get("canonical_id") for row in rows] != sorted(META_REGISTRY):
        raise ValueError("HCWDL-UB graph node order differs")
    return digest


def recipe_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": RECIPE_CONTRACT, "schema_version": 1,
        "arms": {
            arm: {
                "ce": weights[0], "parent_kd": weights[1],
                "grandparent_kd": weights[2], "homotopy_temperature": 2.0,
            }
            for arm, weights in ARM_WEIGHTS.items()
        },
        "first_edge_rule": "grandparent_allocation_transfers_to_parent",
        "m1": {"ce": .25, "parent_kd": .75, "grandparent_kd": 0.0,
               "temperature": 1.0},
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "training_passes": 60, "validation_every_pass": True,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
    })


def recipe_arm_payload(*, arm_id: str, recipe_sha256: str) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB recipe arm")
    weights = ARM_WEIGHTS[arm_id]
    return with_content_hash({
        "contract": RECIPE_ARM_CONTRACT, "schema_version": 1,
        "arm_id": arm_id, "recipe_sha256": require_sha256(recipe_sha256, name="recipe"),
        "ce": weights[0], "parent_kd": weights[1], "grandparent_kd": weights[2],
        "node_count": len(arm_registry(arm_id)),
        "first_edge_parent_kd": weights[1] + weights[2],
        "first_edge_grandparent_kd": 0.0,
        "m1_fixed": True,
    })


def foundation_spec_payload(
    *, source_commit: str, project_dir: str | Path, campaign_root: str | Path,
    parent_campaign_spec_path: str | Path, parents: Mapping[str, str],
    artifact_paths: Mapping[str, str | Path], data_root: str | Path,
    replicate_seed: int, resources: Mapping[str, Mapping[str, Any]],
    operational_waiver_sha256: str,
    semantic_source_sha256: Mapping[str, str],
    contextual_controls: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("foundation source commit must be a full lowercase Git SHA")
    required_paths = {
        "parent_homotopy_spec", "parent_campaign_spec", "split_manifest",
        "selection_manifest", "train_assignment_manifest",
        "validation_assignment_manifest", "train_base_manifest",
        "validation_base_manifest", "legacy_train_manifest",
        "legacy_validation_manifest", "assignment_lock", "recipe",
        "base_coupling_lock", "operational_waiver",
        "factorial_spec", "factorial_aggregate", "factorial_completion",
    }
    if set(artifact_paths) != required_paths:
        raise ValueError("foundation artifact path registry differs")
    expected_context = {"M0", "TOFF", "H_U", "H_S", "O_U", "O_S"}
    if set(contextual_controls) != expected_context:
        raise ValueError("foundation contextual control registry differs")
    context = {}
    for name, row in sorted(contextual_controls.items()):
        if set(row) != {"report_path", "report_sha256", "checkpoint_sha256"}:
            raise ValueError(f"foundation contextual {name} record differs")
        context[name] = {
            "report_path": str(Path(row["report_path"]).resolve()),
            "report_sha256": require_sha256(row["report_sha256"], name=f"{name} report"),
            "checkpoint_sha256": require_sha256(row["checkpoint_sha256"], name=f"{name} checkpoint"),
        }
    return with_content_hash({
        "contract": FOUNDATION_SPEC_CONTRACT, "schema_version": 1,
        "mode": "pilot_300k", "source_commit": source_commit,
        "project_dir": str(Path(project_dir).resolve()),
        "campaign_root": str(Path(campaign_root).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "replicate_seed": int(replicate_seed),
        "parent_campaign_spec_path": str(Path(parent_campaign_spec_path).resolve()),
        "role_counts": ROLE_COUNTS, "ordinary_access_role_counts": ORDINARY_ROLE_COUNTS,
        "parents": _hashes(parents),
        "artifact_paths": {
            name: str(Path(path).resolve()) for name, path in sorted(artifact_paths.items())
        },
        "shared_nodes": ["shared/U000", "shared/M0paired"],
        "resources": {name: dict(value) for name, value in sorted(resources.items())},
        "semantic_source_sha256": _hashes(semantic_source_sha256),
        "contextual_controls": context,
        "operational_waiver_sha256": require_sha256(
            operational_waiver_sha256, name="operational waiver",
        ),
        "child_training_writable": False,
        "final_test_accessed": False,
    })


def validate_foundation_spec(value: Mapping[str, Any]) -> str:
    digest = _contract(value, FOUNDATION_SPEC_CONTRACT)
    if (
        value.get("mode") != "pilot_300k" or value.get("role_counts") != ROLE_COUNTS
        or value.get("ordinary_access_role_counts") != ORDINARY_ROLE_COUNTS
        or value.get("shared_nodes") != ["shared/U000", "shared/M0paired"]
        or value.get("child_training_writable") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB foundation specification semantics differ")
    source_commit = str(value.get("source_commit"))
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("HCWDL-UB foundation source commit differs")
    _hashes(value.get("parents", {}))
    if not value.get("semantic_source_sha256"):
        raise ValueError("HCWDL-UB foundation semantic source registry is empty")
    _hashes(value.get("semantic_source_sha256", {}))
    if int(value.get("replicate_seed", -1)) < 0:
        raise ValueError("HCWDL-UB foundation replicate seed differs")
    require_sha256(value.get("operational_waiver_sha256"), name="operational waiver")
    if not isinstance(value.get("resources"), Mapping) or not value["resources"]:
        raise ValueError("HCWDL-UB foundation resources differ")
    if set(value.get("contextual_controls", {})) != {"M0", "TOFF", "H_U", "H_S", "O_U", "O_S"}:
        raise ValueError("HCWDL-UB foundation contextual registry differs")
    for name, row in value["contextual_controls"].items():
        require_sha256(row.get("report_sha256"), name=f"{name} report")
        require_sha256(row.get("checkpoint_sha256"), name=f"{name} checkpoint")
    return digest


def endpoint_lock_payload(
    *, foundation_spec_sha256: str, parents: Mapping[str, str],
    role_rows: Mapping[str, int], endpoint_checks: Mapping[str, bool],
    resource_measurement_sha256: str,
) -> dict[str, Any]:
    if dict(role_rows) != {"train": 300_000, "validation": 100_000}:
        raise ValueError("HCWDL-UB endpoint-lock row coverage differs")
    required = {"u000_exact_p0", "u100_exact_d100", "j100_exact_hlt",
                "d0_exact_hlt", "no_durable_views", "prepared_endpoints_reused"}
    if set(endpoint_checks) != required or not all(endpoint_checks.values()):
        raise ValueError("HCWDL-UB endpoint checks are incomplete")
    return with_content_hash({
        "contract": ENDPOINT_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": require_sha256(
            foundation_spec_sha256, name="foundation spec",
        ),
        "parents": _hashes(parents), "role_rows": dict(role_rows),
        "endpoint_checks": dict(endpoint_checks),
        "resource_measurement_sha256": require_sha256(
            resource_measurement_sha256, name="resource measurement",
        ),
        "labels_used": False, "final_test_accessed": False,
    })


def validate_endpoint_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ENDPOINT_LOCK_CONTRACT)
    if (
        value.get("role_rows") != {"train": 300_000, "validation": 100_000}
        or not all(value.get("endpoint_checks", {}).values())
        or value.get("labels_used") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB endpoint lock is incomplete")
    require_sha256(value.get("foundation_spec_sha256"), name="foundation spec")
    require_sha256(value.get("resource_measurement_sha256"), name="resource measurement")
    _hashes(value.get("parents", {}))
    return digest


def operational_waiver_payload(
    *, source_commit: str, parent_completion_sha256: str,
    prior_smoke_completion_sha256: str, performance_guide_sha256: str,
    parent_weaver_parity_sha256: str, readiness_evidence_sha256: str,
    semantic_source_sha256: Mapping[str, str],
    resources: Mapping[str, Mapping[str, Any]], authorization_phrase: str,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("HCWDL-UB waiver source commit differs")
    if authorization_phrase != "AUTHORIZE HCWDL UB 300K NO NEW SMOKE EXACT EVIDENCE":
        raise PermissionError("HCWDL-UB operational waiver phrase differs")
    return with_content_hash({
        "contract": OPERATIONAL_WAIVER_CONTRACT, "schema_version": 1,
        "source_commit": source_commit,
        "parent_completion_sha256": require_sha256(
            parent_completion_sha256, name="parent completion",
        ),
        "prior_smoke_completion_sha256": require_sha256(
            prior_smoke_completion_sha256, name="prior smoke completion",
        ),
        "performance_guide_sha256": require_sha256(
            performance_guide_sha256, name="performance guide",
        ),
        "parent_weaver_parity_sha256": require_sha256(
            parent_weaver_parity_sha256, name="parent Weaver parity",
        ),
        "readiness_evidence_sha256": require_sha256(
            readiness_evidence_sha256, name="readiness evidence",
        ),
        "semantic_source_sha256": _hashes(semantic_source_sha256),
        "verification_claims": {
            name: True for name in sorted(WAIVER_VERIFICATION_CLAIMS)
        },
        "resources": {name: dict(row) for name, row in sorted(resources.items())},
        "no_new_smoke": True, "authorization_phrase": authorization_phrase,
        "final_test_accessed": False,
    })


def validate_operational_waiver(value: Mapping[str, Any]) -> str:
    digest = _contract(value, OPERATIONAL_WAIVER_CONTRACT)
    if (
        value.get("no_new_smoke") is not True
        or value.get("authorization_phrase")
        != "AUTHORIZE HCWDL UB 300K NO NEW SMOKE EXACT EVIDENCE"
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB operational waiver differs")
    for name in ("parent_completion_sha256", "prior_smoke_completion_sha256",
                 "performance_guide_sha256", "parent_weaver_parity_sha256",
                 "readiness_evidence_sha256"):
        require_sha256(value.get(name), name=name)
    claims = value.get("verification_claims", {})
    if (
        not value.get("semantic_source_sha256")
        or set(claims) != WAIVER_VERIFICATION_CLAIMS
        or not all(item is True for item in claims.values())
    ):
        raise PermissionError("HCWDL-UB operational verification evidence differs")
    _hashes(value["semantic_source_sha256"])
    return digest


def aggregate_payload(
    *, arm_id: str, arm_spec_sha256: str, rows: Sequence[Mapping[str, Any]],
    imported: Mapping[str, str], contextual_controls: Mapping[str, Mapping[str, Any]],
    shared_controls: Mapping[str, Mapping[str, Any]], gpu_hours: float,
) -> dict[str, Any]:
    registry = arm_registry(arm_id)
    if [str(row.get("node_id")) for row in rows] != list(registry):
        raise ValueError("HCWDL-UB aggregate row registry differs")
    if set(contextual_controls) != {"M0", "TOFF", "H_U", "H_S", "O_U", "O_S"}:
        raise ValueError("HCWDL-UB aggregate contextual registry differs")
    if set(shared_controls) != {"U000", "M0paired"}:
        raise ValueError("HCWDL-UB aggregate shared registry differs")
    if not math.isfinite(float(gpu_hours)) or float(gpu_hours) < 0:
        raise ValueError("HCWDL-UB aggregate GPU hours differ")
    return with_content_hash({
        "contract": ARM_AGGREGATE_CONTRACT, "schema_version": 1,
        "arm_id": arm_id,
        "arm_spec_sha256": require_sha256(arm_spec_sha256, name="arm spec"),
        "rows": [dict(row) for row in rows], "imported": _hashes(imported),
        "contextual_controls": {
            name: dict(row) for name, row in sorted(contextual_controls.items())
        },
        "shared_controls": {
            name: dict(row) for name, row in sorted(shared_controls.items())
        },
        "gpu_hours": float(gpu_hours), "primary_metric": "macro_ovr_auc",
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    })


def validate_arm_aggregate(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_AGGREGATE_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if (
        arm_id not in ARM_IDS
        or [row.get("node_id") for row in value.get("rows", [])]
        != list(arm_registry(arm_id))
        or value.get("final_test_accessed") is not False
        or value.get("scientific_result_does_not_control_completion") is not True
        or set(value.get("contextual_controls", {}))
        != {"M0", "TOFF", "H_U", "H_S", "O_U", "O_S"}
        or set(value.get("shared_controls", {})) != {"U000", "M0paired"}
        or set(value.get("imported", {})) != {"foundation_lock", "U000", "M0paired"}
        or not math.isfinite(float(value.get("gpu_hours", float("nan"))))
        or float(value.get("gpu_hours", -1)) < 0
    ):
        raise ValueError("HCWDL-UB arm aggregate differs")
    _hashes(value["imported"])
    registry = arm_registry(arm_id)
    metric_names = (
        "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
    )
    recovery_names = {
        "recovery_m0paired_to_u000", "contextual_recovery_m0_to_toff",
    }
    for row, node in zip(value["rows"], registry.values(), strict=True):
        if (
            row.get("canonical_id") != node.canonical_id
            or row.get("parent_id") != node.parent_id
            or row.get("grandparent_id") != node.grandparent_id
            or row.get("coordinate") != node.coordinate.payload()
            or row.get("weights") != {
                "ce": node.ce_weight, "parent_kd": node.parent_kd_weight,
                "grandparent_kd": node.grandparent_kd_weight,
            }
            or not math.isfinite(float(row.get("idealized_u000_ancestry", float("nan"))))
            or not recovery_names <= set(row)
        ):
            raise ValueError(f"HCWDL-UB aggregate node lineage differs: {node.node_id}")
        metrics = row.get("metrics", {})
        if any(not math.isfinite(float(metrics.get(name, float("nan")))) for name in metric_names):
            raise FloatingPointError(f"HCWDL-UB aggregate metrics are nonfinite: {node.node_id}")
        for recovery_name in recovery_names:
            recovery = row[recovery_name]
            if set(recovery) != set(metric_names) or any(
                item is not None and not math.isfinite(float(item))
                for item in recovery.values()
            ):
                raise ValueError(f"HCWDL-UB aggregate recovery differs: {node.node_id}")
        for name in ("report_sha256", "checkpoint_sha256", "runtime_sha256"):
            require_sha256(row.get(name), name=f"{node.node_id} {name}")
    for group in (value["contextual_controls"], value["shared_controls"]):
        for name, record in group.items():
            require_sha256(record.get("report_sha256"), name=f"{name} report")
            require_sha256(record.get("checkpoint_sha256"), name=f"{name} checkpoint")
            if any(
                not math.isfinite(float(record.get("metrics", {}).get(metric, float("nan"))))
                for metric in metric_names
            ):
                raise FloatingPointError(f"HCWDL-UB control metrics are nonfinite: {name}")
    return digest


def validate_arm_completion(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_COMPLETION_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if arm_id not in ARM_IDS:
        raise ValueError("HCWDL-UB arm completion identity differs")
    registry = arm_registry(arm_id)
    if (
        set(value.get("completed_node_report_sha256", {})) != set(registry)
        or value.get("completed_node_count") != len(registry)
        or value.get("scientific_result_does_not_control_completion") is not True
        or value.get("final_test_accessed") is not False
        or not math.isfinite(float(value.get("gpu_hours", float("nan"))))
        or float(value.get("gpu_hours", -1)) < 0
    ):
        raise ValueError("HCWDL-UB arm completion differs")
    require_sha256(value.get("arm_spec_sha256"), name="arm spec")
    require_sha256(value.get("aggregate_sha256"), name="arm aggregate")
    _hashes(value["completed_node_report_sha256"])
    return digest


def sweep_aggregate_payload(
    *, recipe_sweep_sha256: str, arm_completions: Mapping[str, str],
    arm_aggregates: Mapping[str, str], rankings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(sorted(arm_completions)) != tuple(sorted(ARM_IDS)) or set(arm_aggregates) != set(ARM_IDS):
        raise ValueError("HCWDL-UB sweep aggregate arm registry differs")
    if len(rankings) != len(ARM_IDS) or {row.get("arm_id") for row in rankings} != set(ARM_IDS):
        raise ValueError("HCWDL-UB recipe ranking registry differs")
    metric_names = (
        "d0f_macro_ovr_auc", "j100_macro_ovr_auc", "m1f_macro_ovr_auc",
        "d0f_cross_entropy", "gpu_hours",
    )
    if any(
        not all(math.isfinite(float(row.get(name, float("nan")))) for name in metric_names)
        for row in rankings
    ):
        raise FloatingPointError("HCWDL-UB recipe ranking metrics are nonfinite")
    if [row.get("arm_id") for row in rankings] != [row["arm_id"] for row in sorted(
        rankings, key=lambda row: (
            -float(row["d0f_macro_ovr_auc"]), -float(row["j100_macro_ovr_auc"]),
            -float(row["m1f_macro_ovr_auc"]), float(row["d0f_cross_entropy"]),
            str(row["arm_id"]),
        ),
    )]:
        raise ValueError("HCWDL-UB recipe ranking differs")
    return with_content_hash({
        "contract": RECIPE_SWEEP_AGGREGATE_CONTRACT, "schema_version": 1,
        "recipe_sweep_sha256": require_sha256(recipe_sweep_sha256, name="recipe sweep"),
        "arm_completion_sha256": _hashes(arm_completions),
        "arm_aggregate_sha256": _hashes(arm_aggregates),
        "rankings": [dict(row) for row in rankings],
        "validation_only": True, "final_test_accessed": False,
    })


def validate_sweep_aggregate(value: Mapping[str, Any]) -> str:
    digest = _contract(value, RECIPE_SWEEP_AGGREGATE_CONTRACT)
    if (
        set(value.get("arm_completion_sha256", {})) != set(ARM_IDS)
        or set(value.get("arm_aggregate_sha256", {})) != set(ARM_IDS)
        or value.get("validation_only") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB sweep aggregate differs")
    expected = sweep_aggregate_payload(
        recipe_sweep_sha256=value["recipe_sweep_sha256"],
        arm_completions=value["arm_completion_sha256"],
        arm_aggregates=value["arm_aggregate_sha256"],
        rankings=value.get("rankings", ()),
    )
    if expected != value:
        raise ValueError("HCWDL-UB sweep aggregate semantics differ")
    return digest


def finalist_lock_payload(
    *, sweep_aggregate_sha256: str, foundation_lock_sha256: str,
    selected_arms: Sequence[str], finalists: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if len(selected_arms) != 2 or len(set(selected_arms)) != 2 or any(arm not in ARM_IDS for arm in selected_arms):
        raise ValueError("HCWDL-UB finalist arm set differs")
    expected = [f"{arm}/{node}" for arm in selected_arms for node in ("D0F", "J100", "M1F", "M1J")]
    if [row.get("canonical_id") for row in finalists] != expected:
        raise ValueError("HCWDL-UB finalist node set differs")
    for row in finalists:
        require_sha256(row.get("report_sha256"), name="finalist report")
        require_sha256(row.get("checkpoint_sha256"), name="finalist checkpoint")
    return with_content_hash({
        "contract": FINALIST_LOCK_CONTRACT, "schema_version": 1,
        "sweep_aggregate_sha256": require_sha256(
            sweep_aggregate_sha256, name="sweep aggregate",
        ),
        "foundation_lock_sha256": require_sha256(
            foundation_lock_sha256, name="foundation lock",
        ),
        "selected_arms": list(selected_arms), "finalists": [dict(row) for row in finalists],
        "selection_used_validation_only": True, "final_test_accessed": False,
    })


def validate_finalist_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, FINALIST_LOCK_CONTRACT)
    if (
        value.get("selection_used_validation_only") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB finalist selection is not validation-only")
    expected = finalist_lock_payload(
        sweep_aggregate_sha256=value["sweep_aggregate_sha256"],
        foundation_lock_sha256=value["foundation_lock_sha256"],
        selected_arms=value.get("selected_arms", ()),
        finalists=value.get("finalists", ()),
    )
    if expected != value:
        raise ValueError("HCWDL-UB finalist lock semantics differ")
    return digest


def execution_lock_payload(
    *, finalist_lock_sha256: str, source_commit: str,
    split_manifest_sha256: str, selection_manifest_sha256: str,
    authorization_phrase: str,
) -> dict[str, Any]:
    if authorization_phrase != "AUTHORIZE HCWDL UB SEALED FINAL TEST EXACT FINALISTS":
        raise PermissionError("HCWDL-UB final-test authorization phrase differs")
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("HCWDL-UB final-test source differs")
    return with_content_hash({
        "contract": EXECUTION_LOCK_CONTRACT, "schema_version": 1,
        "finalist_lock_sha256": require_sha256(finalist_lock_sha256, name="finalist lock"),
        "source_commit": source_commit,
        "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split manifest"),
        "selection_manifest_sha256": require_sha256(selection_manifest_sha256, name="selection manifest"),
        "authorization_phrase": authorization_phrase, "authorized": True,
        "single_execution_claim_required": True,
        "final_test_accessed": False,
    })


def validate_execution_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, EXECUTION_LOCK_CONTRACT)
    if (
        value.get("authorized") is not True
        or value.get("single_execution_claim_required") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB final execution lock is incomplete")
    expected = execution_lock_payload(
        finalist_lock_sha256=value["finalist_lock_sha256"],
        source_commit=value["source_commit"],
        split_manifest_sha256=value["split_manifest_sha256"],
        selection_manifest_sha256=value["selection_manifest_sha256"],
        authorization_phrase=value["authorization_phrase"],
    )
    if expected != value:
        raise ValueError("HCWDL-UB final execution lock semantics differ")
    return digest


def foundation_lock_payload(
    *, foundation_spec_sha256: str, parents: Mapping[str, str],
    u000_report_sha256: str, m0paired_report_sha256: str,
    u000_checkpoint_sha256: str, m0paired_checkpoint_sha256: str,
    u000_target_manifest_sha256: str,
) -> dict[str, Any]:
    return with_content_hash({
        "contract": FOUNDATION_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": require_sha256(
            foundation_spec_sha256, name="foundation spec",
        ),
        "parents": _hashes(parents),
        "u000_report_sha256": require_sha256(
            u000_report_sha256, name="U000 report",
        ),
        "m0paired_report_sha256": require_sha256(
            m0paired_report_sha256, name="M0paired report",
        ),
        "u000_checkpoint_sha256": require_sha256(
            u000_checkpoint_sha256, name="U000 checkpoint",
        ),
        "m0paired_checkpoint_sha256": require_sha256(
            m0paired_checkpoint_sha256, name="M0paired checkpoint",
        ),
        "u000_target_manifest_sha256": require_sha256(
            u000_target_manifest_sha256, name="U000 target manifest",
        ),
        "complete": True, "child_training_writable": False,
        "final_test_accessed": False,
    })


def validate_foundation_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, FOUNDATION_LOCK_CONTRACT)
    if (
        value.get("complete") is not True
        or value.get("child_training_writable") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB foundation lock is incomplete")
    for name in (
        "foundation_spec_sha256", "u000_report_sha256",
        "m0paired_report_sha256", "u000_checkpoint_sha256",
        "m0paired_checkpoint_sha256", "u000_target_manifest_sha256",
    ):
        require_sha256(value.get(name), name=name)
    _hashes(value.get("parents", {}))
    return digest


def arm_spec_payload(
    *, arm_id: str, source_commit: str, project_dir: str | Path,
    campaign_root: str | Path, foundation_lock_path: str | Path,
    foundation_lock_sha256: str, graph_sha256: str, recipe_arm_sha256: str,
    resources: Mapping[str, Any], operational_waiver_sha256: str,
) -> dict[str, Any]:
    if (
        arm_id not in ARM_IDS
        or len(source_commit) != 40
        or any(ch not in "0123456789abcdef" for ch in source_commit)
    ):
        raise ValueError("HCWDL-UB arm identity/source differs")
    return with_content_hash({
        "contract": ARM_CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "arm_id": arm_id, "source_commit": source_commit,
        "project_dir": str(Path(project_dir).resolve()),
        "campaign_root": str(Path(campaign_root).resolve()),
        "foundation_lock_path": str(Path(foundation_lock_path).resolve()),
        "foundation_lock_sha256": require_sha256(
            foundation_lock_sha256, name="foundation lock",
        ),
        "graph_sha256": require_sha256(graph_sha256, name="graph"),
        "recipe_arm_sha256": require_sha256(recipe_arm_sha256, name="recipe arm"),
        "operational_waiver_sha256": require_sha256(
            operational_waiver_sha256, name="operational waiver",
        ),
        "node_ids": list(arm_registry(arm_id)),
        "node_count": len(arm_registry(arm_id)),
        "resources": dict(resources), "final_test_accessed": False,
    })


def validate_arm_spec(
    value: Mapping[str, Any], *, foundation_lock_sha256: str | None = None,
) -> str:
    digest = _contract(value, ARM_CAMPAIGN_SPEC_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if arm_id not in ARM_IDS:
        raise ValueError("HCWDL-UB arm specification identity differs")
    source_commit = str(value.get("source_commit"))
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("HCWDL-UB arm source commit differs")
    if value.get("node_ids") != list(arm_registry(arm_id)) or value.get("node_count") != len(arm_registry(arm_id)):
        raise ValueError("HCWDL-UB arm specification registry differs")
    lock_hash = require_sha256(value.get("foundation_lock_sha256"), name="foundation lock")
    if foundation_lock_sha256 is not None and lock_hash != foundation_lock_sha256:
        raise ValueError("HCWDL-UB arm foundation differs")
    if value.get("final_test_accessed") is not False:
        raise PermissionError("HCWDL-UB arm specification accessed final test")
    return digest


def recipe_sweep_payload(
    *, foundation_lock_sha256: str, arm_specs: Mapping[str, str],
) -> dict[str, Any]:
    if tuple(sorted(arm_specs)) != tuple(sorted(ARM_IDS)):
        raise ValueError("HCWDL-UB recipe sweep must bind exactly six arms")
    return with_content_hash({
        "contract": RECIPE_SWEEP_CONTRACT, "schema_version": 1,
        "foundation_lock_sha256": require_sha256(
            foundation_lock_sha256, name="foundation lock",
        ),
        "arm_spec_sha256": _hashes(arm_specs),
        "arm_order": list(ARM_IDS), "training_campaign": False,
        "directory_discovery": False, "read_only": True,
        "expected_total_fit_count": 151,
    })


def validate_recipe_sweep(value: Mapping[str, Any]) -> str:
    digest = _contract(value, RECIPE_SWEEP_CONTRACT)
    if (
        value.get("arm_order") != list(ARM_IDS)
        or set(value.get("arm_spec_sha256", {})) != set(ARM_IDS)
        or value.get("training_campaign") is not False
        or value.get("directory_discovery") is not False
        or value.get("read_only") is not True
        or value.get("expected_total_fit_count") != 151
    ):
        raise ValueError("HCWDL-UB recipe sweep registry differs")
    _hashes(value["arm_spec_sha256"])
    return digest


def completion_payload(
    *, arm_id: str, arm_spec_sha256: str, aggregate_sha256: str,
    completed_node_reports: Mapping[str, str], gpu_hours: float,
) -> dict[str, Any]:
    registry = arm_registry(arm_id)
    if set(completed_node_reports) != set(registry):
        raise ValueError("HCWDL-UB arm completion node coverage differs")
    if gpu_hours < 0:
        raise ValueError("HCWDL-UB GPU hours must be nonnegative")
    return with_content_hash({
        "contract": ARM_COMPLETION_CONTRACT, "schema_version": 1,
        "arm_id": arm_id,
        "arm_spec_sha256": require_sha256(arm_spec_sha256, name="arm spec"),
        "aggregate_sha256": require_sha256(aggregate_sha256, name="aggregate"),
        "completed_node_report_sha256": _hashes(completed_node_reports),
        "completed_node_count": len(registry), "gpu_hours": float(gpu_hours),
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    })


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "ARM_IDS", "ORDINARY_ROLE_COUNTS", "ROLE_COUNTS", "arm_spec_payload",
    "aggregate_payload", "balanced_switch_config_payload", "completion_payload",
    "coordinate_payload", "endpoint_lock_payload",
    "execution_lock_payload", "finalist_lock_payload",
    "validate_execution_lock", "validate_finalist_lock",
    "foundation_lock_payload", "foundation_spec_payload", "graph_payload",
    "operational_waiver_payload",
    "recipe_arm_payload", "recipe_payload", "recipe_sweep_payload",
    "sweep_aggregate_payload",
    "uniform_repair_payload", "validate_arm_spec",
    "validate_arm_aggregate", "validate_arm_completion", "validate_endpoint_lock",
    "validate_balanced_switch_config", "validate_foundation_lock",
    "validate_foundation_spec", "validate_graph", "validate_operational_waiver",
    "validate_recipe_sweep", "validate_sweep_aggregate",
]
