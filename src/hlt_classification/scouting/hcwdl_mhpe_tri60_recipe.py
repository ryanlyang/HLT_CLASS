"""Exact optimization and loss overlay for HCWDL-MHPE-TRI60-FULL."""

from __future__ import annotations

from typing import Any, Mapping

from hlt_classification.data.cache_contracts import require_sha256

from .hcwdl_mhpe_tri60_contracts import RECIPE_CONTRACT, artifact, validate_artifact
from .hcwdl_mhpe_tri60_graph import (
    FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, REPRESENTATION_SOURCE_COMMIT,
)
from .hcwdl_representation_recipe import FROZEN_SCIENTIFIC_VALUES_SHA256


CHECKPOINT_SELECTION = "macro_auc_ce_logr50_earliest_update_v1"
CLASS_WEIGHTING = "unweighted_per_jet_population_mean_v1"
RESUME_POLICY = "disabled_restart_from_zero_v1"


def recipe_payload(
    *, base_recipe_sha256: str, representation_recipe_sha256: str,
    unified_balanced_recipe_sha256: str,
) -> dict[str, Any]:
    nodes = {node_id: NODE_REGISTRY[node_id].payload() for node_id in FIT_ORDER}
    return artifact({
        "graph_sha256": GRAPH_SHA256,
        "base_recipe_sha256": require_sha256(
            base_recipe_sha256, name="60-pass base recipe",
        ),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="representation recipe v5",
        ),
        "unified_balanced_recipe_sha256": require_sha256(
            unified_balanced_recipe_sha256, name="unified-balanced recipe",
        ),
        "representation_source_commit": REPRESENTATION_SOURCE_COMMIT,
        "representation_scientific_values_sha256": (
            FROZEN_SCIENTIFIC_VALUES_SHA256
        ),
        "training": {
            "passes": 60, "validation_every_passes": 1,
            "effective_batch_size": 256,
            "optimizer": "AdamW", "peak_learning_rate": 3.0e-4,
            "weight_decay": 0.01, "warmup_fraction": 0.05,
            "schedule": "linear_warmup_cosine_decay_v1",
            "learning_rate_floor_fraction": 0.05,
            "forward_precision": "bfloat16",
            "loss_precision": "float32", "metric_precision": "float32",
            "gradient_clipping": None,
            "performance_early_stopping": False,
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "class_weighting": CLASS_WEIGHTING,
        },
        "loss": {
            "view_changing": {
                "ce_weight": .25, "kd_weight": .75, "temperature": 2.0,
            },
            "compression": {
                "ce_weight": .10, "kd_weight": .90, "temperature": 1.0,
            },
            "representation_auxiliary_weight": .10,
            "rset": {
                "jet": .40, "set": .60, "relation": 0.0,
                "orthogonality": 1.0e-3,
            },
            "rrel": {
                "jet": .30, "set": .45, "relation": .25,
                "orthogonality": 1.0e-3,
                "relation_state": "raw_block2_fp32_normalized_v1",
            },
            "jet_set_ramp": {
                "zero_through_pass": 2, "linear_end_pass": 6,
            },
            "relation_ramp": {
                "zero_through_pass": 4, "linear_end_pass": 8,
            },
        },
        "representation_calibration": {
            "selection": "train_only_identity_hash_v1",
            "rows": 4096, "batches": 16, "minimum_rows_per_component": 12,
            "activation_passes": [2, 4],
            "scale_lower": 1.0e-4, "scale_upper": 1.0e4,
            "labels_forbidden": True, "validation_metrics_forbidden": True,
        },
        "probability_ensemble": {
            "space": "class_probability", "member_weighting": "uniform",
            "accumulation": "float64", "stored_dtype": "float32",
            "temperature_applied_once_by_consumer": True,
        },
        "persistence": {
            "probability_banks": "durable_compact_npz_v1",
            "representation_targets": "ram_only_never_persist_v1",
            "student_views": "ram_only_never_persist_v1",
            "selected_checkpoint": "durable_terminal_only",
            "final_checkpoint": "durable_terminal_only",
            "rolling_resume": False, "resume_policy": RESUME_POLICY,
            "partial_checkpoint_reuse": False,
        },
        "nodes": nodes,
        "fit_count": 32, "reducer_count": 12,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


def validate_recipe(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=RECIPE_CONTRACT)
    training = value.get("training", {})
    persistence = value.get("persistence", {})
    if (
        value.get("graph_sha256") != GRAPH_SHA256
        or value.get("representation_source_commit")
        != REPRESENTATION_SOURCE_COMMIT
        or value.get("representation_scientific_values_sha256")
        != FROZEN_SCIENTIFIC_VALUES_SHA256
        or value.get("fit_count") != 32
        or value.get("reducer_count") != 12
        or value.get("final_test_accessed") is not False
        or training.get("passes") != 60
        or training.get("validation_every_passes") != 1
        or training.get("effective_batch_size") != 256
        or training.get("checkpoint_selection") != CHECKPOINT_SELECTION
        or training.get("class_weighting") != CLASS_WEIGHTING
        or persistence.get("representation_targets")
        != "ram_only_never_persist_v1"
        or persistence.get("rolling_resume") is not False
        or persistence.get("resume_policy") != RESUME_POLICY
        or persistence.get("partial_checkpoint_reuse") is not False
    ):
        raise ValueError("HCWDL-MHPE-TRI60 recipe differs")
    expected = {key: NODE_REGISTRY[key].payload() for key in FIT_ORDER}
    if value.get("nodes") != expected:
        raise ValueError("HCWDL-MHPE-TRI60 per-node recipe differs")
    for key in (
        "base_recipe_sha256", "representation_recipe_sha256",
        "unified_balanced_recipe_sha256",
    ):
        require_sha256(value.get(key), name=key)
    return digest


__all__ = [
    "CHECKPOINT_SELECTION", "CLASS_WEIGHTING", "RESUME_POLICY",
    "recipe_payload", "validate_recipe",
]

