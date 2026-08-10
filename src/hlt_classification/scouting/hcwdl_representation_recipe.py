"""Immutable HCWDL-RKD overlay recipe and exact scientific-value validation.

The overlay authorizes only representation supervision.  It binds, but never
copies or overrides, the optimization policy in an authenticated
``HCWDL_RECIPE/v4`` teacher recipe.  Version 4 freezes the dense four-track descent
without importing the obsolete base-HCWDL ladder or finalist chain:
Offline -> D100 -> D95 -> ... -> D0 -> M1 for RSET/RREL, cold/warm.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    canonical_sha256, require_sha256, validate_content_hash,
)
from hlt_classification.provenance import (
    capture_source_snapshot,
    validate_source_snapshot_payload,
)

from .hcwdl_representation_contracts import (
    REPRESENTATION_RECIPE_CONTRACT,
    build_versioned_artifact,
    validate_parent_hashes,
    validate_versioned_artifact,
)
from .hcwdl_representation_graph import (
    ASCENT_GRAPH_SHA256,
    CONTROL_REGISTRY_SHA256,
    NODE_REGISTRY,
    RREL_STRATEGY,
    RSET_STRATEGY,
)


RECIPE_CONTRACT: Final = REPRESENTATION_RECIPE_CONTRACT
# RKD JSON artifacts share the common strict envelope schema v1; the scientific
# semantic version is carried by RECIPE_CONTRACT.
RECIPE_SCHEMA_VERSION: Final = 1
RECIPE_PROFILE: Final = "registered_ablation"
PARENT_RECIPE_CONTRACT: Final = "HCWDL_RECIPE/v4"

REQUIRED_PARENT_KEYS: Final = frozenset({
    "assignment_manifest",
    "dense_teacher_import",
    "historical_parent_graph",
    "kernel_resources",
    "parent_recipe",
    "producer_source",
    "representation_ascent_graph",
    "representation_control_registry",
    "row_selection",
    "source_manifest",
    "split_manifest",
})
REQUIRED_EVIDENCE_KEYS: Final = frozenset({
    "analytic_gradient",
    "diagnostic_reference",
    "finite_kernel",
    "zero_coefficient_parity",
})
KERNEL_RESOURCE_NAMES: Final = (
    "token_rbf_sigma_0p10",
    "token_rbf_sigma_0p25",
    "token_rbf_sigma_0p50",
    "token_rbf_sigma_1p00",
    "relation_rbf_sigma_0p05",
    "relation_rbf_sigma_0p10",
    "relation_rbf_sigma_0p20",
    "relation_rbf_sigma_0p40",
)


def _scientific_values() -> dict[str, Any]:
    """Return a fresh copy of every plan-frozen representation value."""

    return {
        "strategies": {
            RSET_STRATEGY: {
                "short_id": "RSET",
                "components_at_full_strength": {
                    "jet": 0.40, "set": 0.60, "relation": 0.0,
                },
            },
            RREL_STRATEGY: {
                "short_id": "RREL",
                "components_at_full_strength": {
                    "jet": 0.30, "set": 0.45, "relation": 0.25,
                },
            },
        },
        "representation_coefficient": 0.10,
        "orthogonality_coefficient": 1e-3,
        "ramps": {
            "coordinate": "one_based_pass_fraction_from_zero_based_optimizer_update",
            "updates_per_pass": "ceil(train_rows/256)",
            "jet_set": {
                "zero_through_pass": 2.0,
                "linear_full_at_pass": 6.0,
            },
            "relation": {
                "zero_through_pass": 4.0,
                "linear_full_at_pass": 8.0,
            },
            "rrel_common_weight": "r_js-0.25*r_rel",
            "base_coefficients_constant": True,
        },
        "taps": {
            "particle": {
                "name": "particle_block_2",
                "shape": ["batch", "tokens", 128],
                "weaver_block_index": 1,
            },
            "jet": {
                "name": "jet_penultimate",
                "shape": ["batch", 128],
                "meaning": "exact_vector_entering_final_15_class_linear_map",
            },
            "toff": {
                "charged_particle": "charged_particle_block_2",
                "neutral_particle": "neutral_particle_block_2",
                "pooled": "offline_jet_penultimate",
                "separate_charged_neutral_bases": True,
                "cross_family_relations": False,
            },
        },
        "token_set": {
            "projection": {
                "shape": [128, 128],
                "bias": False,
                "initialization": "identity",
                "reset_every_node": True,
            },
            "normalization": "fp32_l2",
            "weighting": {
                "uniform_fraction": 0.5,
                "softened_pt_fraction": 0.5,
                "softened_pt": "sqrt(sqrt(px**2+py**2))",
                "normalize_each_component_over_visible_tokens": True,
            },
            "kernel": {
                "kind": "fixed_multiscale_spectral_moment",
                "bandwidths": [0.10, 0.25, 0.50, 1.00],
                "features_per_bandwidth": 256,
                "total_features": 1024,
                "frequency_distribution": "Normal(0,sigma^-2 I)",
                "phase_distribution": "Uniform(0,2*pi)",
                "runtime_dtype": "float32",
            },
        },
        "relations": {
            "population": {
                "maximum_tokens": 32,
                "order": "descending_pt_then_ascending_canonical_token_id",
                "unordered_off_diagonal_pairs": True,
                "pair_weight": "product_of_token_weights_normalized_per_stratum",
                "student_teacher_selection_independent": True,
            },
            "strata": [
                {"name": "local", "lower_inclusive": 0.0, "upper_exclusive": 0.05},
                {"name": "medium", "lower_inclusive": 0.05, "upper_exclusive": 0.20},
                {"name": "wide", "lower_inclusive": 0.20, "upper_exclusive": None},
            ],
            "stratum_arithmetic": "float64",
            "latent_statistic": "cosine_similarity_of_projected_normalized_token_states",
            "kernel": {
                "kind": "fixed_multiscale_spectral_moment",
                "bandwidths": [0.05, 0.10, 0.20, 0.40],
                "features_per_bandwidth": 64,
                "features_per_stratum": 256,
                "runtime_dtype": "float32",
            },
            "reduce_active_strata_equally": True,
        },
        "family_policy": {
            "ordinary_families": ["all"],
            "toff_families": ["charged", "neutral"],
            "jointly_nonempty_families_equal_weight": True,
            "single_jointly_nonempty_family_weight": 1.0,
            "missing_either_side_ineligible": True,
            "hlt_family_source": "pre_transform_particle_identity",
            "unknown_nonzero_identity_fails": True,
        },
        "calibration": {
            "role": "train_only",
            "selection_rows": 4096,
            "selection": "smallest_canonical_sha256",
            "natural_class_population": True,
            "batch_size": 256,
            "expected_batches": 16,
            "jet_set_barrier_after_pass": 2,
            "relation_barrier_after_pass": 4,
            "minimum_supported_batches": 12,
            "scale": "median_batch_base_gradient_rms/median_batch_component_gradient_rms",
            "host_median_dtype": "float64",
            "active_scale_bounds_inclusive": [1e-4, 1e4],
            "weak_support_status": "inactive_valid_support",
            "weak_support_scale": 0.0,
            "no_coefficient_reallocation": True,
            "parameter_prefixes": [
                "mod.embed.", "mod.pair_embed.", "mod.blocks.0.", "mod.blocks.1.",
            ],
            "single_student_forward_per_batch": True,
            "no_optimizer_or_scheduler_step": True,
            "snapshot_and_restore_runtime_state": True,
        },
        "precision": {
            "student_forward": "parent_bfloat16_policy",
            "student_loss_boundary": "recursive_float32",
            "teacher_forward": "float32",
            "teacher_autocast": False,
            "teacher_tf32": False,
            "teacher_reduced_precision_reduction": False,
        },
        "kernel_generation": {
            "master_seed": 20260808,
            "canonical_payload_contract": "HCWDL_REP_RFF/v1",
            "digest_to_seed": "first_8_sha256_bytes_big_endian_unsigned",
            "bit_generator": "numpy.random.PCG64",
            "draw_dtype": "float64",
            "published_dtype": "float32_c_contiguous",
            "runtime_requires_exact_logical_hashes": True,
        },
        "seeds": {
            "screening": 1337,
            "confirmation": [11, 22, 33, 44, 55],
            "within_class_shuffle": 20260809,
            "rng_domains": {
                "kernel": "hcwdl_rkd/kernel_resources",
                "screen": "hcwdl_rkd/screen_training",
                "confirmation": "hcwdl_rkd/confirmation_training",
                "projection": "hcwdl_rkd/representation_projection",
                "shuffle": "hcwdl_rkd/within_class_shuffle",
                "calibration": "hcwdl_rkd/gradient_calibration_selection",
                "diagnostic": "hcwdl_rkd/representation_diagnostic",
            },
        },
        "training": {
            "passes": 60,
            "validation_every_passes": 1,
            "performance_early_stopping": False,
            "selection_order": [
                "highest_macro_ovr_auc",
                "lowest_cross_entropy",
                "highest_macro_mean_log_qcd_rejection_at_50pct_signal",
                "earliest_optimizer_update",
                "lexicographically_smallest_checkpoint_identity",
            ],
            "student_domains": sorted({
                node.student_domain for node in NODE_REGISTRY.values()
            }),
            "intermediate_repaired_views_are_training_only": True,
            "deployable_node_ids": sorted(
                node.node_id for node in NODE_REGISTRY.values()
                if node.deployable
            ),
            "primary_result_node_suffix": "M1",
            "terminal_after_m1": True,
            "training_heads_reset_each_node": True,
            "warm_start_loads_immediate_predecessor_model_state_only": True,
            "warm_start_resets_optimizer_and_scheduler": True,
            "cold_start_is_fresh_at_every_rung": True,
            "deployable_excludes_representation_heads": True,
            "representation_row_weight_source": "parent_recipe_exact_15_ones",
            "class_weighted_representation_row_reduction": False,
            "representation_row_reduction": "mean(per_jet_loss)",
            "parent_base_loss_inherited_without_override": True,
        },
        "target_forward": {
            "role": "train_only",
            "canonical_rows_per_source_batch": 256,
            "batches_never_cross_source_partition": True,
            "evaluation_mode": True,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "one_teacher_surface_forward_per_canonical_batch": True,
            "exact_float32_logical_reconstruction": True,
        },
        "target_lifecycle": {
            "logical_banks": [
                "TOFF",
                *sorted(
                    node_id for node_id, node in NODE_REGISTRY.items()
                    if node.stage != "terminal_m1"
                ),
            ],
            "campaign_teacher_commitments_are_topologically_derived": True,
            "superset_compact_targets": True,
            "just_in_time": True,
            "cleanup_after_all_authenticated_consumers": True,
            "mutable_latest_pointer": False,
            "no_raw_particle_or_match_arrays": True,
            "maximum_committed_scientific_banks_under_serial_dag": 1,
        },
        "controls": {
            "registered_count": 0,
            "retired_m5_ascent_controls_reused": False,
        },
        "forbidden_parent_overrides": [
            "ce_coefficients", "kd_coefficients", "temperatures", "batch_size",
            "optimizer", "learning_rate", "class_weight_rule", "duration",
            "validation_cadence", "checkpoint_selector",
        ],
    }


FROZEN_SCIENTIFIC_VALUES_SHA256: Final = canonical_sha256(_scientific_values())


def frozen_scientific_values() -> dict[str, Any]:
    """Return a defensive copy of the immutable overlay values."""

    return deepcopy(_scientific_values())


def derive_recipe_producer_source_sha256(repository: str | Path) -> str:
    """Derive the recipe producer identity from one clean Git checkout.

    The producer source is never a caller-entered digest.  The same source
    snapshot identity is measured again by every production worker and bound
    into the runtime facts before any scientific input is opened.
    """

    snapshot = capture_source_snapshot(repository, require_clean=True)
    validate_source_snapshot_payload(snapshot)
    if snapshot.get("worktree_clean") is not True:
        raise ValueError("representation recipe producer checkout is not clean")
    return require_sha256(
        snapshot["source_snapshot_sha256"],
        name="representation recipe producer source snapshot",
    )


def _hash_registry(
    value: Mapping[str, Any], *, expected_keys: frozenset[str] | tuple[str, ...], label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise ValueError(f"HCWDL-RKD {label} registry keys differ")
    return {
        name: require_sha256(value[name], name=f"HCWDL-RKD {label} {name}")
        for name in sorted(value)
    }


def build_representation_recipe(
    *,
    parents: Mapping[str, Any],
    kernel_array_logical_hashes: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact immutable representation overlay."""

    normalized_parents = validate_parent_hashes(parents)
    if set(normalized_parents) != REQUIRED_PARENT_KEYS:
        raise ValueError("HCWDL-RKD recipe parent lineage keys differ")
    kernel_hashes = _hash_registry(
        kernel_array_logical_hashes,
        expected_keys=KERNEL_RESOURCE_NAMES,
        label="kernel logical hash",
    )
    evidence_hashes = _hash_registry(
        evidence, expected_keys=REQUIRED_EVIDENCE_KEYS, label="acceptance evidence",
    )
    payload = {
        "recipe_profile": RECIPE_PROFILE,
        "purpose": "hcwdl_matching_free_representation_kd_four_dense_descents",
        "parent_recipe_contract": PARENT_RECIPE_CONTRACT,
        "ascent_graph_sha256": ASCENT_GRAPH_SHA256,
        "control_registry_sha256": CONTROL_REGISTRY_SHA256,
        "primary_node_ids": sorted(NODE_REGISTRY),
        "scientific_values": _scientific_values(),
        "scientific_values_sha256": FROZEN_SCIENTIFIC_VALUES_SHA256,
        "kernel_array_logical_hashes": kernel_hashes,
        "acceptance_evidence": evidence_hashes,
    }
    artifact = build_versioned_artifact(
        RECIPE_CONTRACT, parents=normalized_parents, payload=payload,
    )
    validate_representation_recipe(artifact, expected_parents=normalized_parents)
    return artifact


def derive_representation_recipe_evidence(
    *, numerical_acceptance: Mapping[str, Any],
    zero_coefficient_measurements: Mapping[str, Any],
) -> dict[str, str]:
    """Derive the four recipe evidence hashes from reproducible preflight work."""

    numerical_hash = validate_content_hash(
        numerical_acceptance,
        expected_contract="HCWDL_REPRESENTATION_NUMERICAL_ACCEPTANCE/v1",
        expected_schema_version=1,
    )
    if (
        numerical_acceptance.get("passed") is not True
        or numerical_acceptance.get("scientific_authorization") is not True
    ):
        raise ValueError("representation numerical evidence is nonauthorizing")
    from .hcwdl_representation_campaign_artifacts import (
        validate_zero_coefficient_measurements,
    )

    zero_hash = validate_zero_coefficient_measurements(
        zero_coefficient_measurements,
    )
    return {
        "analytic_gradient": numerical_hash,
        "diagnostic_reference": numerical_hash,
        "finite_kernel": numerical_hash,
        "zero_coefficient_parity": zero_hash,
    }


def validate_representation_recipe(
    value: Mapping[str, Any], *, expected_parents: Mapping[str, Any] | None = None,
) -> str:
    """Fail closed on any altered scientific value, graph, or lineage."""

    digest = validate_versioned_artifact(
        value,
        expected_contract=RECIPE_CONTRACT,
        expected_parents=expected_parents,
        required_payload_keys=(
            "recipe_profile", "purpose", "parent_recipe_contract",
            "ascent_graph_sha256", "control_registry_sha256", "primary_node_ids",
            "scientific_values", "scientific_values_sha256",
            "kernel_array_logical_hashes", "acceptance_evidence",
        ),
    )
    parents = validate_parent_hashes(value["parents"])
    if set(parents) != REQUIRED_PARENT_KEYS:
        raise ValueError("HCWDL-RKD recipe parent lineage keys differ")
    payload = value["payload"]
    expected_payload_keys = {
        "recipe_profile", "purpose", "parent_recipe_contract",
        "ascent_graph_sha256", "control_registry_sha256", "primary_node_ids",
        "scientific_values", "scientific_values_sha256",
        "kernel_array_logical_hashes", "acceptance_evidence",
    }
    if set(payload) != expected_payload_keys:
        raise ValueError("HCWDL-RKD recipe payload fields differ")
    if (
        payload["recipe_profile"] != RECIPE_PROFILE
        or payload["purpose"]
        != "hcwdl_matching_free_representation_kd_four_dense_descents"
        or payload["parent_recipe_contract"] != PARENT_RECIPE_CONTRACT
        or payload["ascent_graph_sha256"] != ASCENT_GRAPH_SHA256
        or payload["control_registry_sha256"] != CONTROL_REGISTRY_SHA256
        or payload["primary_node_ids"] != sorted(NODE_REGISTRY)
    ):
        raise ValueError("HCWDL-RKD recipe identity or graph binding differs")
    if (
        payload["scientific_values"] != _scientific_values()
        or payload["scientific_values_sha256"] != FROZEN_SCIENTIFIC_VALUES_SHA256
        or canonical_sha256(payload["scientific_values"])
        != FROZEN_SCIENTIFIC_VALUES_SHA256
    ):
        raise ValueError("HCWDL-RKD frozen scientific values differ")
    _hash_registry(
        payload["kernel_array_logical_hashes"],
        expected_keys=KERNEL_RESOURCE_NAMES,
        label="kernel logical hash",
    )
    _hash_registry(
        payload["acceptance_evidence"],
        expected_keys=REQUIRED_EVIDENCE_KEYS,
        label="acceptance evidence",
    )
    if payload["kernel_array_logical_hashes"] != dict(
        sorted(payload["kernel_array_logical_hashes"].items())
    ) or payload["acceptance_evidence"] != dict(
        sorted(payload["acceptance_evidence"].items())
    ):
        raise ValueError("HCWDL-RKD recipe hash registries are not canonical")
    return digest


def example_representation_recipe() -> dict[str, Any]:
    """Complete deterministic local fixture; it is not submission authority."""

    parents = {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(sorted(REQUIRED_PARENT_KEYS))
    }
    # Keep every fixture digest exactly 64 hexadecimal characters.
    parents = {name: digest[:64] for name, digest in parents.items()}
    kernel_hashes = {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(KERNEL_RESOURCE_NAMES)
    }
    kernel_hashes = {name: digest[:64] for name, digest in kernel_hashes.items()}
    evidence = {
        name: f"{index + 10:x}" * 64
        for index, name in enumerate(sorted(REQUIRED_EVIDENCE_KEYS))
    }
    evidence = {name: digest[:64] for name, digest in evidence.items()}
    return build_representation_recipe(
        parents=parents,
        kernel_array_logical_hashes=kernel_hashes,
        evidence=evidence,
    )


__all__ = [
    "FROZEN_SCIENTIFIC_VALUES_SHA256",
    "KERNEL_RESOURCE_NAMES",
    "RECIPE_CONTRACT",
    "RECIPE_PROFILE",
    "REQUIRED_EVIDENCE_KEYS",
    "REQUIRED_PARENT_KEYS",
    "build_representation_recipe",
    "derive_recipe_producer_source_sha256",
    "derive_representation_recipe_evidence",
    "example_representation_recipe",
    "frozen_scientific_values",
    "validate_representation_recipe",
]
