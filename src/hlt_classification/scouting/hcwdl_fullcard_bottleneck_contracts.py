"""Contracts for the forced full-cardinality bottleneck pairing control."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hlt_classification.data.cache_contracts import (
    validate_content_hash,
    with_content_hash,
)


MATCHER_SPEC_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_MATCHER_SPEC/v1"
ASSIGNMENT_SHARD_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_ASSIGNMENT_SHARD/v1"
ASSIGNMENT_MANIFEST_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_ASSIGNMENT_MANIFEST/v1"
ASSIGNMENT_AUDIT_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_ASSIGNMENT_AUDIT/v1"
ASSIGNMENT_LOCK_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_ASSIGNMENT_LOCK/v1"
DIAGNOSTIC_REPORT_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_DIAGNOSTIC_REPORT/v1"
MATCHER_ACCEPTANCE_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_MATCHER_ACCEPTANCE/v1"
COUPLING_LOCK_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_COUPLING_LOCK/v1"
FOUNDATION_SPEC_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_FOUNDATION_SPEC/v1"
FOUNDATION_LOCK_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_FOUNDATION_LOCK/v1"
U000_EQUIVALENCE_LOCK_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_U000_EQUIVALENCE_LOCK/v1"
SOURCE_LOCK_CONTRACT = "HCWDL_TRI100_SPINE4_BOTTLENECK_SOURCE_LOCK/v1"
CAMPAIGN_SPEC_CONTRACT = "HCWDL_TRI100_SPINE4_BOTTLENECK_CAMPAIGN_SPEC/v1"
RECOVERY_SPEC_CONTRACT = "HCWDL_TRI100_SPINE4_BOTTLENECK_RECOVERY_SPEC/v1"
SCHEMA_VERSION = 1

DR_QUANTUM = 1.0e-7
ABS_LOG_PT_RESPONSE_QUANTUM = 1.0e-7
ROUNDING_MODE = "IEEE-754_roundTiesToEven_v1"
PHI_WRAP = "half_open_minus_pi_plus_pi_v1"
SOLVER = "exact_bottleneck_pruned_mixed_radix_integer_hungarian_v1"


def matcher_spec() -> dict[str, Any]:
    """Return the immutable scientific specification of the pairing rule."""

    return with_content_hash({
        "contract": MATCHER_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": "forced_full_cardinality_pairing_control_not_truth",
        "cardinality": "min(valid_hlt_particles,valid_offline_particles)",
        "feasible_edges": "complete_bipartite_over_valid_particles",
        "durable_orientation": "hlt_to_native_offline_index_or_minus_one",
        "primary_objective": "lexicographic_min_sorted_descending_qdr_vector",
        "dr_quantum": DR_QUANTUM,
        "dr_dtype": "float64",
        "dr_rounding": ROUNDING_MODE,
        "phi_wrap": PHI_WRAP,
        "secondary_objectives": [
            "lexicographic_min_sorted_descending_canonical_abs_log_pt_response",
            "minimum_raw_particle_category_mismatch_count",
            "minimum_valid_charge_mismatch_count",
            "lexicographic_min_native_offline_index_by_hlt_index_minus_one_last",
        ],
        "abs_log_pt_response_quantum": ABS_LOG_PT_RESPONSE_QUANTUM,
        "abs_log_pt_response_dtype": "float64",
        "abs_log_pt_response_rounding": ROUNDING_MODE,
        "category_mismatch": "raw_integer_inequality_including_unclassified",
        "valid_charge_values": [-1, 0, 1],
        "charge_mismatch": "inequality_only_when_both_charges_are_valid",
        "solver": SOLVER,
        "correspondence_confidence": "absent_and_forbidden",
    })


def validate_matcher_spec(value: Mapping[str, Any]) -> str:
    expected = matcher_spec()
    digest = validate_content_hash(
        value,
        expected_contract=MATCHER_SPEC_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    if dict(value) != expected:
        raise ValueError("full-cardinality bottleneck matcher semantics differ")
    return digest


__all__ = [
    "ABS_LOG_PT_RESPONSE_QUANTUM", "ASSIGNMENT_AUDIT_CONTRACT",
    "ASSIGNMENT_LOCK_CONTRACT", "ASSIGNMENT_MANIFEST_CONTRACT",
    "ASSIGNMENT_SHARD_CONTRACT", "CAMPAIGN_SPEC_CONTRACT",
    "COUPLING_LOCK_CONTRACT", "DIAGNOSTIC_REPORT_CONTRACT", "DR_QUANTUM",
    "FOUNDATION_LOCK_CONTRACT", "MATCHER_ACCEPTANCE_CONTRACT",
    "FOUNDATION_SPEC_CONTRACT", "MATCHER_SPEC_CONTRACT", "PHI_WRAP",
    "RECOVERY_SPEC_CONTRACT", "ROUNDING_MODE", "SCHEMA_VERSION", "SOLVER",
    "SOURCE_LOCK_CONTRACT", "U000_EQUIVALENCE_LOCK_CONTRACT", "matcher_spec",
    "validate_matcher_spec",
]
