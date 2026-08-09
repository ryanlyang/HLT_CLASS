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
    assert ROUTES["HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v1"].path_template == (
        "confirmation/aggregate.json"
    )
    assert ROUTES["HCWDL_REPRESENTATION_CONFIRMATION_RUN/v1"].path_template == (
        "confirmation/runs/${execution_id}.json"
    )
    assert ROUTES["HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v1"].producer_kind == (
        "execution_lock"
    )
    assert ROUTES["HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1"].producer_kind == (
        "metric_join"
    )
    assert ROUTES["HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1"].producer_kind == (
        "branch_opening_task"
    )
    assert ROUTES["HCWDL_REPRESENTATION_CAMPAIGN_SPEC/v1"].path_template == (
        "${campaign_spec_parent}/campaign_spec.json"
    )
    assert CAMPAIGN_SPEC_PARENT_VALUES == {".", "planning"}
    assert ROUTES["HCWDL_REPRESENTATION_COMMAND_PLAN/v1"].path_template == (
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
        "HCWDL_REPRESENTATION_USR1_EXACT_RESUME_PROOF/v1"
    ].path_template == "acceptance/proofs/usr1_exact_resume_result.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_VALIDATION_PROXY_PROOF/v1"
    ].path_template == "acceptance/proofs/final_role_validation_proxy_result.json"
    assert ROUTES[
        "HCWDL_REPRESENTATION_PRODUCTION_WORKER_SMOKE_PROOF/v1"
    ].path_template == "acceptance/proofs/production_worker_smoke_result.json"
    assert all(route.validation_surface for route in ROUTES.values())
