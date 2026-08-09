"""One canonical publication-route template for every HCWDL-RKD contract.

``validation_surface`` is an auditable description of the owning validation
boundary.  It is deliberately not a dynamic function registry: consumers
import and call their typed validators directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

from .hcwdl_representation_contracts import (
    ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT,
    ARCHITECTURE_ATTESTATION_CONTRACT,
    CALIBRATION_SELECTION_CONTRACT,
    CONTRACTS,
    NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY_CONTRACT,
    NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
    NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT,
    NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
    NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT_CONTRACT,
    NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT,
    PARENT_IMPORT_CONTRACT,
    PARENT_LOSS_ATTESTATION_CONTRACT,
    REPRESENTATION_RECIPE_CONTRACT,
    SURFACE_PARITY_CONTRACT,
    TIGRIS_ACCEPTANCE_CONTRACT,
    TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
    TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT,
    USR1_DELIVERY_RECEIPT_CONTRACT,
    USR1_EXACT_RESUME_PROOF_CONTRACT,
    VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT,
    VALIDATION_PROXY_PROOF_CONTRACT,
)


CAMPAIGN_SPEC_PARENT_VALUES: Final = frozenset({".", "planning"})
GRADIENT_CALIBRATION_PHASE_VALUES: Final = frozenset({"jet_set", "relation"})


@dataclass(frozen=True)
class ArtifactRoute:
    path_template: str
    producer_kind: str
    validation_surface: str


def _c(name: str) -> str:
    if name.startswith("SHARED_"):
        return "HCWDL_" + name + "/v1"
    return "HCWDL_REPRESENTATION_" + name + "/v1"


ROUTES: Final = {
    PARENT_IMPORT_CONTRACT: ArtifactRoute("import/parent_import.json", "parent_import", "validate_parent_import"),
    PARENT_LOSS_ATTESTATION_CONTRACT: ArtifactRoute("import/parent_loss_attestation.json", "parent_loss_attestation", "validate_parent_loss_attestation"),
    ARCHITECTURE_ATTESTATION_CONTRACT: ArtifactRoute("import/architecture_attestation.json", "architecture_attestation", "validate_architecture_attestation"),
    _c("ASCENT_GRAPH"): ArtifactRoute("graph/ascent_graph.json", "graph_freeze", "validate_ascent_graph_artifact"),
    REPRESENTATION_RECIPE_CONTRACT: ArtifactRoute("recipes/representation_recipe.json", "representation_recipe", "validate_representation_recipe"),
    _c("KERNEL_RESOURCES"): ArtifactRoute("recipes/kernel_resources/committed/${envelope_id}", "kernel_resources", "validate_kernel_resources"),
    _c("TAP"): ArtifactRoute("architecture/tap.json", "tap_schema", "validate_tap_artifact"),
    SURFACE_PARITY_CONTRACT: ArtifactRoute("architecture/surface_parity.json", "surface_parity", "validate_surface_parity"),
    _c("TARGET_FORWARD_SPEC"): ArtifactRoute("targets/${bank}/generations/${generation_id}/target_forward_spec.json", "target_build", "validate_target_forward_spec"),
    _c("TARGET_EXECUTION_ATTESTATION"): ArtifactRoute("targets/${bank}/generations/${generation_id}/target_execution_attestation.json", "target_build", "validate_target_generation"),
    _c("TARGET_LOGICAL_BANK"): ArtifactRoute("targets/${bank}/logical_bank.json", "target_build", "validate_logical_target_bank"),
    _c("TARGET_CONSUMER_REGISTRY"): ArtifactRoute("targets/${bank}/generations/${generation_id}/consumer_registry.json", "target_build", "validate_target_consumer_registry"),
    _c("TARGET_BUILD_INTENT"): ArtifactRoute("targets/${bank}/generations/${generation_id}/build_intent.json", "target_build", "validate_target_generation"),
    _c("TARGET_GENERATION"): ArtifactRoute("targets/${bank}/generations/${generation_id}/generation.json", "target_build", "validate_target_generation"),
    _c("TARGET_SHARD"): ArtifactRoute("targets/${bank}/generations/${generation_id}/shards/${source}.json", "target_build", "validate_target_generation"),
    _c("TARGET_MANIFEST"): ArtifactRoute("targets/${bank}/generations/${generation_id}/manifest.json", "target_build", "validate_target_generation"),
    _c("TARGET_CLEANUP_AUTHORIZATION"): ArtifactRoute("cleanup/${bank}/${generation_id}/authorization.json", "target_cleanup", "validate_cleanup_authorization"),
    _c("TARGET_CLEANUP_COMPLETION"): ArtifactRoute("cleanup/${bank}/${generation_id}/completion.json", "target_cleanup", "validate_cleanup_completion"),
    _c("RECOVERY_PLAN"): ArtifactRoute("recovery/${scope}/${recovery_id}.json", "recovery", "validate_recovery_plan"),
    _c("GRADIENT_CALIBRATION"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/calibration/${phase}.json", "train_node", "gradient_calibration_artifact"),
    CALIBRATION_SELECTION_CONTRACT: ArtifactRoute("training/${scope}/${node_id}/${execution_id}/calibration/selection.json", "train_node", "validate_calibration_selection_artifact"),
    _c("GRADIENT_CALIBRATION_MANIFEST"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/calibration/manifest.json", "train_node", "gradient_calibration_manifest"),
    _c("DIAGNOSTIC_BATCH"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/calibration/diagnostic_batch.json", "train_node", "diagnostic_batch_artifact"),
    _c("NUMERICAL_ACCEPTANCE"): ArtifactRoute("acceptance/numerical.json", "numerical_acceptance", "validate_numerical_acceptance"),
    _c("SMOKE_PROBE"): ArtifactRoute("acceptance/smoke_probe.json", "smoke_probe", "validate_smoke_probe"),
    _c("PAIRED_BOOTSTRAP"): ArtifactRoute("reports/paired_bootstrap/${comparison_id}/committed/${envelope_id}", "metric_join", "validate_binary_envelope"),
    _c("CONTROL_REGISTRY"): ArtifactRoute("controls/registry.json", "control_registry", "validate_control_registry_artifact"),
    _c("ZERO_COEFFICIENT_ACCEPTANCE"): ArtifactRoute("controls/zero_coefficient/acceptance.json", "zero_coefficient_acceptance", "validate_zero_coefficient_acceptance"),
    _c("SHUFFLE_MAP"): ArtifactRoute("controls/shuffled_representation/committed/${envelope_id}", "shuffle_map", "validate_binary_envelope"),
    _c("RESUME_STATE"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/resume/commit_${sequence}.json", "train_node", "load_highest_valid_resume"),
    _c("TRAINING_REPORT"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/training_report.json", "train_node", "validate_training_report"),
    _c("CHECKPOINT_SELECTION"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/checkpoint_selection.json", "train_node", "validate_checkpoint_selection"),
    _c("DEPLOYABLE_EXTRACTION"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/deployable_extraction.json", "train_node", "validate_deployable_extraction"),
    _c("SELECTED_TRAINING_CHECKPOINT"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/checkpoints/selected/committed/${envelope_id}", "train_node", "validate_binary_envelope"),
    _c("FINAL_TRAINING_CHECKPOINT"): ArtifactRoute("training/${scope}/${node_id}/${execution_id}/checkpoints/final/committed/${envelope_id}", "train_node", "validate_binary_envelope"),
    _c("SCREEN_AGGREGATE"): ArtifactRoute("reports/screen_aggregate.json", "screen_aggregate", "validate_screen_aggregate"),
    _c("CONFIRMATION_REGISTRY"): ArtifactRoute("confirmation/registry.json", "confirmation_registry", "validate_confirmation_registry"),
    _c("CONFIRMATION_AGGREGATE"): ArtifactRoute("confirmation/aggregate.json", "confirmation_aggregate", "validate_confirmation_aggregate"),
    _c("CONFIRMATION_RUN"): ArtifactRoute("confirmation/runs/${execution_id}.json", "confirmation", "build_confirmation_run"),
    _c("VALIDATION_ONLY_AGGREGATE"): ArtifactRoute("reports/validation_only_aggregate.json", "validation_only_aggregate", "build_validation_only_aggregate"),
    _c("FINAL_DISPOSITION"): ArtifactRoute("import/final_disposition.json", "reservation", "validate_final_disposition"),
    _c("PARENT_FINAL_STATE"): ArtifactRoute("import/parent_final_state.json", "reservation", "validate_parent_final_state"),
    _c("SHARED_IMMUTABLE_BINARY_ENVELOPE"): ArtifactRoute("${artifact_root}/committed/${envelope_id}/commit.json", "binary_owner", "validate_binary_envelope"),
    _c("SHARED_FINAL_POPULATION"): ArtifactRoute("${population_namespace}/population.json", "reservation", "validate_final_population"),
    _c("SHARED_FINAL_POPULATION_DISJOINTNESS"): ArtifactRoute("${population_namespace}/population_disjointness.json", "reservation", "validate_population_disjointness"),
    _c("SHARED_FINAL_EXPOSURE_LEDGER"): ArtifactRoute("${claims_root}/exposure_ledger/generations/${sequence}_${content_hash}.json", "reservation", "validate_exposure_ledger"),
    _c("SHARED_LEGACY_FINAL_EXPOSURE"): ArtifactRoute("${claims_root}/exposure_ledger/legacy_final_exposure.json", "legacy_final_evaluation", "legacy_final_exposure_content_hash"),
    _c("SHARED_FINAL_POPULATION_REGISTRATION"): ArtifactRoute("${population_namespace}/population_registration.json", "reservation", "validate_population_registration"),
    _c("SHARED_FINAL_RESERVATION"): ArtifactRoute("${population_namespace}/reservation.json", "reservation", "validate_final_reservation"),
    _c("SHARED_FINAL_LEGACY_CANCELLATION"): ArtifactRoute("${population_namespace}/legacy_cancellation.json", "reservation", "validate_legacy_cancellation"),
    _c("FINAL_ASSIGNMENT_SPEC"): ArtifactRoute("final/assignment/specification.json", "reservation", "validate_final_assignment_spec"),
    _c("PRETRAINING_FINALIST_POLICY"): ArtifactRoute("final/pretraining_finalist_policy.json", "reservation", "validate_pretraining_finalist_policy_commitment"),
    _c("FINALIST_LOCK"): ArtifactRoute("locks/05_finalists.json", "finalist_lock", "validate_finalist_lock"),
    _c("SHARED_FINAL_TASK_REGISTRY"): ArtifactRoute("final/task_registry.json", "shared_final_claim", "validate_final_task_registry"),
    _c("SHARED_FINAL_EXECUTION_CLAIM"): ArtifactRoute("${population_namespace}/execution_claim.json", "shared_final_claim", "validate_final_execution_claim"),
    _c("SHARED_FINAL_ROLE_CAPABILITY"): ArtifactRoute("final/capabilities/${task_id}.json", "branch_opening_task", "validate_role_capability"),
    _c("SHARED_FINAL_RECOVERY_PLAN"): ArtifactRoute("final/recovery/${recovery_id}.json", "shared_final_recovery", "validate_shared_final_recovery"),
    _c("SHARED_FINAL_ROW_SELECTION"): ArtifactRoute("final/selection/row_selection.json", "final_selection", "validate_final_row_selection"),
    _c("SHARED_FINAL_LABEL_ESCROW"): ArtifactRoute("final/selection/label_escrow/committed/${envelope_id}", "final_selection", "validate_binary_envelope"),
    _c("SHARED_FINAL_ASSIGNMENT_SHARD"): ArtifactRoute("final/assignment/shards/${source_partition}/committed/${envelope_id}", "assignment_shard", "validate_binary_envelope"),
    _c("SHARED_FINAL_BRANCH_ACCESS"): ArtifactRoute("final/${branch_owner}/branch_access.json", "branch_opening_task", "validate_branch_access"),
    _c("SHARED_FINAL_ASSIGNMENT_AUDIT"): ArtifactRoute("final/assignment/audit.json", "assignment_finalize", "validate_assignment_audit"),
    _c("SHARED_FINAL_DATA_ATTESTATION"): ArtifactRoute("locks/06_final_data_attestation.json", "data_attestation", "validate_final_data_attestation"),
    _c("EXECUTION_LOCK"): ArtifactRoute("locks/07_execution.json", "execution_lock", "validate_execution_lock"),
    _c("FINAL_PREDICTION_SPEC"): ArtifactRoute("final/prediction_spec.json", "execution_lock", "validate_prediction_spec"),
    _c("FINAL_EVALUATION"): ArtifactRoute("final/evaluations/${finalist_id}.json", "metric_join", "validate_final_evaluation"),
    _c("PREDICTION_SHARD"): ArtifactRoute("final/predictions/${finalist_id}/shards/${source}/committed/${envelope_id}", "prediction_shard", "validate_binary_envelope"),
    _c("PREDICTION_MANIFEST"): ArtifactRoute("final/predictions/${finalist_id}/manifest.json", "prediction_finalize", "validate_prediction_manifest"),
    _c("METRIC_JOIN"): ArtifactRoute("final/metric_join.json", "metric_join", "validate_metric_join"),
    _c("FINAL_AGGREGATE"): ArtifactRoute("reports/final_aggregate.json", "final_aggregate", "validate_final_aggregate"),
    _c("CAMPAIGN_SPEC"): ArtifactRoute("${campaign_spec_parent}/campaign_spec.json", "campaign_create", "validate_campaign_spec"),
    _c("COMMAND_PLAN"): ArtifactRoute("command_plan.json", "campaign_dry_run", "validate_command_plan"),
    _c("RUNTIME_BINDING"): ArtifactRoute("runtime/runtime_binding.json", "runtime_binding", "validate_runtime_binding"),
    _c("RUNTIME_PREREQUISITES"): ArtifactRoute("runtime/runtime_prerequisites.json", "runtime_prerequisites", "validate_runtime_prerequisites"),
    _c("RUNTIME_DRY_RUN_AUDIT"): ArtifactRoute("runtime/dry_run_audit.json", "campaign_dry_run", "validate_runtime_dry_run_audit"),
    _c("WORKER_RUNTIME_MEASUREMENT"): ArtifactRoute("runtime/measurements/${resource_class}.json", "runtime_measurement", "validate_worker_runtime_measurement"),
    _c("EXECUTABLE_CANDIDATE_AUDIT"): ArtifactRoute("acceptance/executable_candidate_audit.json", "campaign_dry_run", "validate_executable_candidate_audit"),
    _c("ACCEPTANCE_BOOTSTRAP"): ArtifactRoute("acceptance/bootstrap/spec.json", "acceptance_bootstrap", "validate_acceptance_bootstrap"),
    NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT: ArtifactRoute("acceptance/nonfinal/action_inputs.json", "nonfinal_acceptance_authority", "validate_nonfinal_acceptance_action_inputs"),
    NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY_CONTRACT: ArtifactRoute("acceptance/nonfinal/assemblies/${action_id}.json", "nonfinal_acceptance_authority", "validate_nonfinal_acceptance_action_assembly"),
    NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT: ArtifactRoute("acceptance/nonfinal/results/${action_id}.json", "nonfinal_acceptance_action", "validate_nonfinal_acceptance_action_result"),
    NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT: ArtifactRoute("acceptance/nonfinal/authority.json", "nonfinal_acceptance_authority", "validate_nonfinal_acceptance_authority"),
    NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT_CONTRACT: ArtifactRoute("acceptance/nonfinal/evidence/${action_id}/execution_receipt.json", "nonfinal_acceptance_action", "validate_nonfinal_acceptance_execution_receipt"),
    ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT: ArtifactRoute("acceptance/nonfinal/workspaces/${action_id}/acceptance_real_batch_full_loss.json", "nonfinal_acceptance_action", "validate_acceptance_real_batch_full_loss_record"),
    _c("SUBMISSION_EVENT"): ArtifactRoute("submission_events/${sequence}_${content_hash}.json", "campaign_submit", "submission_event_content_hash"),
    _c("SUBMISSION_LEDGER"): ArtifactRoute("submission_ledger.json", "campaign_submit", "validate_submission_ledger"),
    _c("RECOVERY_SUBMISSION_LEDGER"): ArtifactRoute("recovery_submission_ledgers/${sequence}_${content_hash}.json", "campaign_resume", "validate_recovery_ledger"),
    _c("MONITOR_REPORT"): ArtifactRoute("monitoring/reports/${sequence}_${content_hash}.json", "campaign_monitor", "validate_monitor_report"),
    _c("RESOURCE_PROFILE"): ArtifactRoute("resources/measured_profile.json", "resource_measurement", "validate_measured_profile"),
    _c("STORAGE_ESTIMATE"): ArtifactRoute("resources/storage_estimate.json", "resource_measurement", "validate_storage_estimate"),
    _c("FIXED_SIZE_INVENTORY"): ArtifactRoute("resources/fixed_size_inventory.json", "resource_measurement", "validate_fixed_size_inventory"),
    _c("SCHEDULER_EVIDENCE"): ArtifactRoute("acceptance/evidence/scheduler/${evidence_id}.json", "acceptance_evidence", "validate_scheduler_evidence"),
    _c("MINIATURE_EVIDENCE"): ArtifactRoute("acceptance/evidence/miniature/${evidence_id}.json", "acceptance_evidence", "validate_miniature_evidence"),
    TIGRIS_EVIDENCE_BUNDLE_CONTRACT: ArtifactRoute("acceptance/tigris_evidence_bundle.json", "tigris_acceptance", "validate_tigris_evidence_bundle"),
    _c("TIGRIS_ACTION_PROOF"): ArtifactRoute("acceptance/proofs/${evidence_kind}.json", "acceptance_evidence", "validate_tigris_action_proof"),
    TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT: ArtifactRoute("acceptance/nonfinal/proofs/two_update.json", "nonfinal_acceptance", "validate_two_update_acceptance_proof"),
    USR1_DELIVERY_RECEIPT_CONTRACT: ArtifactRoute("acceptance/nonfinal/usr1/interrupt/receipt.json", "nonfinal_acceptance", "validate_usr1_delivery_receipt"),
    USR1_EXACT_RESUME_PROOF_CONTRACT: ArtifactRoute("acceptance/proofs/usr1_exact_resume_result.json", "acceptance_evidence", "validate_usr1_exact_resume_proof"),
    VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT: ArtifactRoute("acceptance/nonfinal/validation_proxy/access/${stage}.json", "validation_proxy", "validate_validation_proxy_branch_access"),
    VALIDATION_PROXY_PROOF_CONTRACT: ArtifactRoute("acceptance/nonfinal/validation_proxy/result.json", "validation_proxy", "validate_validation_proxy_proof"),
    NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT: ArtifactRoute("acceptance/nonfinal/evidence/${action_id}/scheduler.json", "nonfinal_acceptance_evidence", "validate_nonfinal_acceptance_scheduler_evidence"),
    _c("PRODUCTION_WORKER_SMOKE_PROOF"): ArtifactRoute("acceptance/proofs/production_worker_smoke_result.json", "acceptance_evidence", "validate_production_worker_smoke_proof"),
    _c("LOCAL_SMOKE_REPORT"): ArtifactRoute("acceptance/local_smoke_report.json", "local_smoke", "validate_local_smoke"),
    _c("CACHE_MINIATURE"): ArtifactRoute("acceptance/cache_miniature.json", "cache_miniature", "validate_cache_miniature"),
    _c("CACHE_MINIATURE_BANK"): ArtifactRoute("acceptance/cache_miniature_${bank}.json", "cache_miniature_bank", "validate_cache_miniature_bank"),
    TIGRIS_ACCEPTANCE_CONTRACT: ArtifactRoute("acceptance/tigris_acceptance.json", "tigris_acceptance", "validate_tigris_acceptance"),
    _c("SUBMISSION_AUTHORIZATION"): ArtifactRoute("locks/00_submission_authorization.json", "submission_authorization", "validate_submission_authorization"),
}


def validate_routes() -> None:
    if set(ROUTES) != set(CONTRACTS):
        missing = sorted(set(CONTRACTS) - set(ROUTES))
        extra = sorted(set(ROUTES) - set(CONTRACTS))
        raise RuntimeError(f"HCWDL-RKD artifact route registry differs: missing={missing}, extra={extra}")
    paths = [route.path_template for route in ROUTES.values()]
    # Only the generic recovery-plan scope intentionally shares a directory;
    # every contract itself still owns one unique path template.
    if len(paths) != len(set(paths)):
        collisions = sorted(path for path in set(paths) if paths.count(path) > 1)
        raise RuntimeError(f"HCWDL-RKD artifact routes collide: {collisions}")
    for contract, route in ROUTES.items():
        if (
            re.fullmatch(r".+/v[1-9][0-9]*", contract) is None
            or not route.producer_kind
            or not route.validation_surface
        ):
            raise RuntimeError("HCWDL-RKD artifact route is incomplete")


validate_routes()


__all__ = [
    "ArtifactRoute", "CAMPAIGN_SPEC_PARENT_VALUES",
    "GRADIENT_CALIBRATION_PHASE_VALUES", "ROUTES", "validate_routes",
]
