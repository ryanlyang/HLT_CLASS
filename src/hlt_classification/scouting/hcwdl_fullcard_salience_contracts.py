"""Versioned contracts for full-cardinality salience-weighted pairing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    validate_content_hash,
    with_content_hash,
)


MATCHER_REGISTRY_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_MATCHER_REGISTRY/v1"
MATCHER_SPEC_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_MATCHER_SPEC/v1"
ASSIGNMENT_SHARD_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_ASSIGNMENT_SHARD/v1"
ASSIGNMENT_MANIFEST_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_ASSIGNMENT_MANIFEST/v1"
ASSIGNMENT_AUDIT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_ASSIGNMENT_AUDIT/v1"
ASSIGNMENT_LOCK_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_ASSIGNMENT_LOCK/v1"
DIAGNOSTIC_REPORT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_DIAGNOSTIC_REPORT/v1"
MATCHER_ACCEPTANCE_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_MATCHER_ACCEPTANCE/v1"
COUPLING_LOCK_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_COUPLING_LOCK/v1"
U000_EQUIVALENCE_LOCK_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_U000_EQUIVALENCE_LOCK/v1"
SCREEN_SPLIT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_SPLIT/v1"
SCREEN_CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_CAMPAIGN_SPEC/v1"
SCREEN_COMMAND_PLAN_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_COMMAND_PLAN/v1"
SCREEN_GRAPH_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_GRAPH/v1"
SCREEN_RECIPE_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_RECIPE/v1"
SCREEN_AUTHENTICATION_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_AUTHENTICATION/v1"
SCREEN_EXECUTION_ACCEPTANCE_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_EXECUTION_ACCEPTANCE/v1"
SCREEN_TRAINING_REPORT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_TRAINING_REPORT/v1"
SCREEN_SELECTED_CHECKPOINT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_SELECTED_CHECKPOINT/v1"
SCREEN_FINAL_CHECKPOINT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_FINAL_CHECKPOINT/v1"
SCREEN_FIT_REPORT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_FIT_REPORT/v1"
SCREEN_COMPLETE_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_COMPLETE/v1"
SCREEN_REPORT_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN_REPORT/v1"
SELECTION_LOCK_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_SELECTION_LOCK/v1"
FOUNDATION_SPEC_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_FOUNDATION_SPEC/v1"
FOUNDATION_LOCK_CONTRACT: Final = "HCWDL_FULLCARD_SALIENCE_FOUNDATION_LOCK/v1"
SCHEMA_VERSION: Final = 1

DR_QUANTUM: Final = 1.0e-7
ABS_LOG_PT_RESPONSE_QUANTUM: Final = 1.0e-7
PT_SHARE_SCALE: Final = 1_000_000
SALIENCE_FLOOR: Final = 10_000
PAIR_DR_CAP: Final = 0.8
ROUNDING_MODE: Final = "IEEE-754_roundTiesToEven_v1"
PHI_WRAP: Final = "half_open_minus_pi_plus_pi_v1"
SOLVER: Final = "exact_mixed_radix_integer_hungarian_v1"

SALIENCE_PT_LINEAR: Final = "SALIENCE_PT_LINEAR"
SALIENCE_PT_QUADRATIC: Final = "SALIENCE_PT_QUADRATIC"
SALIENCE_PT_QUADRATIC_CORE25: Final = "SALIENCE_PT_QUADRATIC_CORE25"
CANDIDATES: Final = (
    SALIENCE_PT_LINEAR,
    SALIENCE_PT_QUADRATIC,
    SALIENCE_PT_QUADRATIC_CORE25,
)


def matcher_registry() -> dict[str, Any]:
    """Return the closed candidate family used by the validation screen."""

    return with_content_hash({
        "contract": MATCHER_REGISTRY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_order": list(CANDIDATES),
        "candidate_count": len(CANDIDATES),
        "post_metric_candidate_addition_forbidden": True,
        "bottleneck_is_contextual_control_not_candidate": True,
        "final_test_accessed": False,
    })


def validate_matcher_registry(value: Mapping[str, Any]) -> str:
    expected = matcher_registry()
    digest = validate_content_hash(
        value,
        expected_contract=MATCHER_REGISTRY_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    if dict(value) != expected:
        raise ValueError("full-cardinality salience matcher registry differs")
    return digest


def matcher_spec(candidate: str) -> dict[str, Any]:
    """Return one immutable candidate matcher specification."""

    if candidate not in CANDIDATES:
        raise ValueError("unknown full-cardinality salience candidate")
    salience = {
        SALIENCE_PT_LINEAR: "floor_plus_linear_endpoint_pt_share",
        SALIENCE_PT_QUADRATIC: "floor_plus_quadratic_endpoint_pt_share",
        SALIENCE_PT_QUADRATIC_CORE25: (
            "floor_plus_quadratic_endpoint_pt_share_plus_at_most_25pct_own_axis_core_bonus"
        ),
    }[candidate]
    return with_content_hash({
        "contract": MATCHER_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate,
        "claim_boundary": "forced_full_cardinality_pairing_control_not_truth",
        "cardinality": "min(valid_hlt_particles,valid_offline_particles)",
        "feasible_edges": "complete_bipartite_over_valid_particles",
        "durable_orientation": "hlt_to_native_offline_index_or_minus_one",
        "salience": salience,
        "salience_inputs": ["endpoint_scalar_pt_share", "own_endpoint_jet_axis_delta_r"],
        "forbidden_salience_inputs": [
            "jet_label", "classifier_gradient", "attention", "impact_parameter",
        ],
        "pt_share_scale": PT_SHARE_SCALE,
        "salience_floor": SALIENCE_FLOOR,
        "pair_dr_cap": PAIR_DR_CAP,
        "primary_objective": (
            "maximum_total_pair_salience_times_bounded_angular_closeness"
        ),
        "dr_quantum": DR_QUANTUM,
        "dr_dtype": "float64",
        "dr_rounding": ROUNDING_MODE,
        "phi_wrap": PHI_WRAP,
        "secondary_objectives": [
            "minimum_total_uncapped_canonical_qdr",
            "maximum_total_selected_larger_endpoint_salience",
            "minimum_total_canonical_abs_log_pt_response",
            "minimum_raw_particle_category_mismatch_count",
            "minimum_valid_charge_mismatch_count",
            "lexicographic_min_native_offline_index_by_hlt_index_minus_one_last",
        ],
        "abs_log_pt_response_quantum": ABS_LOG_PT_RESPONSE_QUANTUM,
        "abs_log_pt_response_dtype": "float64",
        "abs_log_pt_response_rounding": ROUNDING_MODE,
        "valid_charge_values": [-1, 0, 1],
        "solver": SOLVER,
        "correspondence_confidence": "absent_and_forbidden",
        "final_test_accessed": False,
    })


def validate_matcher_spec(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=MATCHER_SPEC_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    candidate = str(value.get("candidate"))
    if dict(value) != matcher_spec(candidate):
        raise ValueError("full-cardinality salience matcher semantics differ")
    return digest


__all__ = [
    "ABS_LOG_PT_RESPONSE_QUANTUM", "ASSIGNMENT_AUDIT_CONTRACT",
    "ASSIGNMENT_LOCK_CONTRACT", "ASSIGNMENT_MANIFEST_CONTRACT",
    "ASSIGNMENT_SHARD_CONTRACT", "CANDIDATES", "COUPLING_LOCK_CONTRACT",
    "DIAGNOSTIC_REPORT_CONTRACT",
    "DR_QUANTUM", "FOUNDATION_LOCK_CONTRACT", "FOUNDATION_SPEC_CONTRACT",
    "MATCHER_ACCEPTANCE_CONTRACT", "MATCHER_REGISTRY_CONTRACT",
    "MATCHER_SPEC_CONTRACT", "PAIR_DR_CAP",
    "PHI_WRAP", "PT_SHARE_SCALE", "ROUNDING_MODE", "SALIENCE_FLOOR",
    "SALIENCE_PT_LINEAR", "SALIENCE_PT_QUADRATIC",
    "SALIENCE_PT_QUADRATIC_CORE25", "SCHEMA_VERSION", "SCREEN_REPORT_CONTRACT",
    "SCREEN_CAMPAIGN_SPEC_CONTRACT", "SCREEN_COMMAND_PLAN_CONTRACT",
    "SCREEN_AUTHENTICATION_CONTRACT", "SCREEN_EXECUTION_ACCEPTANCE_CONTRACT",
    "SCREEN_COMPLETE_CONTRACT", "SCREEN_FINAL_CHECKPOINT_CONTRACT",
    "SCREEN_FIT_REPORT_CONTRACT", "SCREEN_SELECTED_CHECKPOINT_CONTRACT",
    "SCREEN_GRAPH_CONTRACT", "SCREEN_RECIPE_CONTRACT",
    "SCREEN_SPLIT_CONTRACT", "SCREEN_TRAINING_REPORT_CONTRACT",
    "SELECTION_LOCK_CONTRACT", "SOLVER",
    "matcher_registry", "matcher_spec", "validate_matcher_registry",
    "U000_EQUIVALENCE_LOCK_CONTRACT", "validate_matcher_spec",
]
