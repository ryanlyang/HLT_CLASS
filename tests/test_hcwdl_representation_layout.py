from hlt_classification.scouting.hcwdl_representation_contracts import CONTRACTS
from hlt_classification.scouting.hcwdl_representation_layout import (
    CAMPAIGN_SPEC_PARENT_VALUES,
    GRADIENT_CALIBRATION_PHASE_VALUES,
    ROUTES,
    validate_routes,
)


def test_every_contract_has_one_canonical_route() -> None:
    validate_routes()
    assert set(ROUTES) == set(CONTRACTS)
    assert len(ROUTES) == len(CONTRACTS)
    assert len({row.path_template for row in ROUTES.values()}) == len(ROUTES)
    assert ROUTES["HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v2"].path_template == (
        "confirmation/aggregate.json"
    )
    assert ROUTES["HCWDL_REPRESENTATION_CONFIRMATION_RUN/v2"].path_template == (
        "confirmation/runs/${execution_id}.json"
    )
    assert ROUTES["HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v2"].producer_kind == (
        "execution_lock"
    )
    assert ROUTES["HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1"].producer_kind == (
        "metric_join"
    )
    assert ROUTES["HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1"].producer_kind == (
        "branch_opening_task"
    )
    assert ROUTES["HCWDL_REPRESENTATION_CAMPAIGN_SPEC/v3"].path_template == (
        "${campaign_spec_parent}/campaign_spec.json"
    )
    assert CAMPAIGN_SPEC_PARENT_VALUES == {".", "planning"}
    assert ROUTES["HCWDL_REPRESENTATION_COMMAND_PLAN/v3"].path_template == (
        "command_plan.json"
    )
    assert ROUTES["HCWDL_REPRESENTATION_RUNTIME_BINDING/v1"].path_template == (
        "runtime/runtime_binding.json"
    )
    assert ROUTES["HCWDL_REP_GRAD_CAL/v1"].path_template == (
        "training/${scope}/${node_id}/${execution_id}/calibration/selection.json"
    )
    assert GRADIENT_CALIBRATION_PHASE_VALUES == {"jet_set", "relation"}
    assert ROUTES["HCWDL_SHARED_LEGACY_FINAL_EXPOSURE/v1"].path_template == (
        "${claims_root}/exposure_ledger/legacy_final_exposure.json"
    )
    assert ROUTES["HCWDL_REPRESENTATION_SUBMISSION_EVENT/v1"].path_template == (
        "submission_events/${sequence}_${content_hash}.json"
    )
    assert ROUTES["HCWDL_REPRESENTATION_TIGRIS_ACTION_PROOF/v1"].path_template == (
        "acceptance/proofs/${evidence_kind}.json"
    )
    assert ROUTES[
        "HCWDL_REPRESENTATION_TIGRIS_EVIDENCE_BUNDLE/v2"
    ].path_template == "acceptance/tigris_evidence_bundle.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_TIGRIS_ACCEPTANCE/v2"
    ].path_template == "acceptance/tigris_acceptance.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_DENSE_STORAGE_TEMPLATE/v1"
    ].path_template == "resources/dense_storage_template.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_DENSE_STORAGE_ESTIMATE/v1"
    ].path_template == "resources/dense_storage_estimate.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_DENSE_COMPATIBLE_RESOURCE_PROFILE/v1"
    ].path_template == "resources/measured_dense_profile_compatible.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_AUTHORIZATION/v1"
    ].path_template == "review/dense_resource_probe_collector_recovery_authorization.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_LEDGER/v1"
    ].path_template == "review/dense_resource_probe_collector_recovery_ledger.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_INPUTS/v1"
    ].path_template == "acceptance/nonfinal/action_inputs.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY/v1"
    ].path_template == "acceptance/nonfinal/assemblies/${action_id}.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_RESULT/v1"
    ].path_template == "acceptance/nonfinal/results/${action_id}.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_AUTHORITY/v1"
    ].path_template == "acceptance/nonfinal/authority.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT/v1"
    ].path_template == "acceptance/nonfinal/evidence/${action_id}/execution_receipt.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_ACCEPTANCE_REAL_BATCH_FULL_LOSS/v1"
    ].path_template == (
        "acceptance/nonfinal/workspaces/${action_id}/acceptance_real_batch_full_loss.json"
    )
    assert ROUTES[
        "HCWDL_REPRESENTATION_TWO_UPDATE_ACCEPTANCE_PROOF/v1"
    ].path_template == "acceptance/nonfinal/proofs/two_update.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_USR1_DELIVERY_RECEIPT/v1"
    ].path_template == "acceptance/nonfinal/usr1/interrupt/receipt.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_USR1_EXACT_RESUME_PROOF/v2"
    ].path_template == "acceptance/proofs/usr1_exact_resume_result.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_VALIDATION_PROXY_BRANCH_ACCESS/v1"
    ].path_template == "acceptance/nonfinal/validation_proxy/access/${stage}.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_VALIDATION_PROXY_PROOF/v2"
    ].path_template == "acceptance/nonfinal/validation_proxy/result.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE/v1"
    ].path_template == "acceptance/nonfinal/evidence/${action_id}/scheduler.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_PRODUCTION_WORKER_SMOKE_PROOF/v1"
    ].path_template == "acceptance/proofs/production_worker_smoke_result.json"
    assert all(route.validation_surface for route in ROUTES.values())
