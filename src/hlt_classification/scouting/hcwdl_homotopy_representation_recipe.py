"""Immutable v1 overlay joining HCWDL recipe v4, RKD recipe v5, and U/D graph."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash

from .hcwdl_homotopy_representation_contracts import (
    FIT_COUNT, RECIPE_CONTRACT, SCHEMA_VERSION, TARGET_BANK_COUNT,
)
from .hcwdl_homotopy_representation_graph import (
    GRAPH_SHA256, NODE_REGISTRY, resolved_base_loss,
)
from .hcwdl_recipe import validate_recipe as validate_base_recipe
from .hcwdl_representation_graph import RREL_STRATEGY, RSET_STRATEGY
from .hcwdl_representation_recipe import validate_representation_recipe


def build_recipe(
    *, base_recipe: Mapping[str, Any], representation_recipe: Mapping[str, Any],
    parent_graph_recipe_lock_sha256: str, integration_attestation_sha256: str,
) -> dict[str, Any]:
    base_hash = validate_base_recipe(base_recipe, require_authorized=True)
    representation_hash = validate_representation_recipe(representation_recipe)
    if base_recipe.get("contract") != "HCWDL_RECIPE/v4":
        raise ValueError("HCWDL-U-RKD requires HCWDL_RECIPE/v4")
    if representation_recipe.get("contract") != "HCWDL_REPRESENTATION_RECIPE/v5":
        raise ValueError("HCWDL-U-RKD requires corrected representation recipe v5")
    if not np.array_equal(
        np.asarray(base_recipe.get("class_weights"), dtype=np.float32),
        np.ones(15, dtype=np.float32),
    ):
        raise ValueError("HCWDL-U-RKD requires the unweighted base recipe")
    # HCWDL_REPRESENTATION_RECIPE/v5 is a versioned artifact.  Its frozen
    # scientific values belong to the authenticated payload, not the artifact
    # envelope.  Reading the envelope directly made campaign creation fail
    # only after all prerequisite artifacts had been published.
    values = representation_recipe["payload"]["scientific_values"]
    if (
        values["relations"].get("latent_state_source") != "raw_particle_block_2"
        or values["relations"].get("latent_projection_applied") is not False
    ):
        raise ValueError("HCWDL-U-RKD RREL recipe is not raw-state v5")
    rows = []
    for node in NODE_REGISTRY.values():
        rows.append({
            "node_id": node.node_id,
            "strategy": node.strategy,
            "transition_index": node.transition_index,
            "student_domain": node.student_domain,
            "teacher": asdict(node.teacher),
            "target_bank_identity": node.target_bank_identity,
            "seed_alias": node.seed_alias,
            "base_loss": asdict(resolved_base_loss(node.node_id)),
            "representation_coefficient": 0.10,
            "representation_components": (
                {"jet": 0.40, "set": 0.60, "relation": 0.0}
                if node.strategy == RSET_STRATEGY
                else {"jet": 0.30, "set": 0.45, "relation": 0.25}
            ),
            "temperature": node.temperature,
            "passes": 60,
            "validation_checks": 60,
            "peak_learning_rate": 3.0e-4,
            "initialization": "fresh",
        })
    return with_content_hash({
        "contract": RECIPE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "parents": {
            "base_recipe": base_hash,
            "representation_recipe": representation_hash,
            "parent_graph_recipe_lock": require_sha256(
                parent_graph_recipe_lock_sha256, name="parent UJ graph/recipe lock",
            ),
            "integration_attestation": require_sha256(
                integration_attestation_sha256, name="source integration attestation",
            ),
        },
        "graph_sha256": GRAPH_SHA256,
        "class_weighting_policy": "unweighted_per_jet_population_mean_v1",
        "class_weights": [1.0] * 15,
        "training_passes": 60,
        "validation_every_passes": 1,
        "performance_early_stopping": False,
        "checkpoint_selection": [
            "highest_macro_ovr_auc", "lowest_cross_entropy",
            "highest_macro_mean_log_qcd_rejection_at_50pct_signal",
            "earliest_optimizer_update",
        ],
        "target_bank_count": TARGET_BANK_COUNT,
        "node_count": FIT_COUNT,
        "rows": rows,
        "final_test_accessed": False,
    })


def validate_recipe(
    value: Mapping[str, Any], *, base_recipe: Mapping[str, Any],
    representation_recipe: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RECIPE_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    rebuilt = build_recipe(
        base_recipe=base_recipe, representation_recipe=representation_recipe,
        parent_graph_recipe_lock_sha256=str(value.get("parents", {}).get("parent_graph_recipe_lock")),
        integration_attestation_sha256=str(value.get("parents", {}).get("integration_attestation")),
    )
    if dict(value) != rebuilt:
        raise ValueError("HCWDL-U-RKD combined recipe differs")
    return digest


__all__ = ["build_recipe", "validate_recipe"]
