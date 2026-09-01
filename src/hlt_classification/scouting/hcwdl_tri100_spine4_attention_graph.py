"""Full four-spine Phase-I attention re-optimization graph."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_attention_reoptimization import DEFAULT_ATTENTION_RECIPE
from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_tri100_spine4_attention_contracts import (
    GRAPH_CONTRACT, RECIPE_CONTRACT, artifact,
)
from .hcwdl_tri100_spine4_bottleneck_graph import (
    ANCHOR_NODE, ANCHOR_NODE_ID, BRANCH_NODES, BRANCH_ORDER, BRANCH_PATHS,
    COORDINATES, DOWNSTREAM_FIT_ORDER, ENDPOINT_NODES, EXECUTION, FIT_ORDER,
    GRAPH_SHA256 as PERSISTENT_GRAPH_SHA256, LR_SCHEDULE, NODE_REGISTRY,
    PROBABILITY_COMPONENTS, REDUCER_ORDER, SOURCE_DISTRIBUTION,
    distribution_consumers,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI100-FOUR-SPINE-ATTENTION-REOPT-PERSISTENT-HLT"
RELATIONAL_CARRIERS: Final = MappingProxyType({
    node_id: NODE_REGISTRY[node_id].parent_node_id
    for node_id in DOWNSTREAM_FIT_ORDER
})


def recipe_payload() -> dict[str, object]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "persistent_graph_sha256": PERSISTENT_GRAPH_SHA256,
        "anchor_training": {
            "passes": 60, "validation_every_passes": 1,
            "effective_batch_size": 256, "optimizer": "AdamW",
            "peak_learning_rate": 3.0e-4, "weight_decay": 0.01,
            "warmup_fraction": 0.05,
            "schedule": "linear_warmup_cosine_decay_v1",
            "learning_rate_floor_fraction": 0.05,
            "forward_precision": "bfloat16",
            "performance_early_stopping": False,
            "restore_best_checkpoint": True,
        },
        "downstream_training": {
            "maximum_passes": 100, "validation_every_passes": 1,
            "effective_batch_size": 256, "optimizer": "AdamW",
            "peak_learning_rate": 3.0e-4, "weight_decay": 0.01,
            "adam_betas": [0.9, 0.999], "adam_epsilon": 1.0e-8,
            "forward_precision": "bfloat16",
            "stage0_learning_rate_schedule": dict(LR_SCHEDULE),
            "performance_early_stopping": False,
            "restore_best_checkpoint": True,
        },
        "task_loss": {
            "kind": "constant_ce_kd_v1", "ce_weight": 0.25,
            "kd_weight": 0.75, "temperature": 2.0,
        },
        "attention_reoptimization": DEFAULT_ATTENTION_RECIPE.payload(),
        "initialization": "fresh_per_fit",
        "teacher_policy": "immediate_parent_logits_and_live_relational_carrier_v1",
        "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "execution": dict(EXECUTION),
        "rolling_resume": False,
        "durable_dense_relational_targets": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT,
    "schema_version": 1,
    "campaign_label": CAMPAIGN_LABEL,
    "persistent_graph_sha256": PERSISTENT_GRAPH_SHA256,
    "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
    "branch_order": list(BRANCH_ORDER),
    "branch_paths": {name: list(BRANCH_PATHS[name]) for name in BRANCH_ORDER},
    "branch_nodes": {name: list(BRANCH_NODES[name]) for name in BRANCH_ORDER},
    "nodes": [NODE_REGISTRY[name].payload() for name in FIT_ORDER],
    "fit_order": list(FIT_ORDER),
    "reducer_order": list(REDUCER_ORDER),
    "endpoint_nodes": list(ENDPOINT_NODES),
    "source_distribution": SOURCE_DISTRIBUTION,
    "probability_components": {
        name: list(value) for name, value in PROBABILITY_COMPONENTS.items()
    },
    "relational_carriers": dict(RELATIONAL_CARRIERS),
    "stage0_attention_joint_at_every_downstream_rung": True,
    "fresh_fit_count": len(FIT_ORDER),
    "single_component_probability_banks": True,
    "immediate_parent_only": True,
    "ensembles": False,
    "weight_continuation": False,
    "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    value = with_content_hash(_GRAPH_BODY)
    if value["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("attention four-spine graph hash differs")
    return value


def validate_graph() -> str:
    DEFAULT_ATTENTION_RECIPE.validate()
    if (
        len(FIT_ORDER) != 30 or len(REDUCER_ORDER) != 26
        or tuple(map(len, (BRANCH_NODES[name] for name in BRANCH_ORDER)))
        != (1, 5, 8, 15)
        or set(RELATIONAL_CARRIERS) != set(DOWNSTREAM_FIT_ORDER)
        or any(
            RELATIONAL_CARRIERS[node_id] != NODE_REGISTRY[node_id].parent_node_id
            for node_id in DOWNSTREAM_FIT_ORDER
        )
        or recipe_payload()["downstream_training"]["maximum_passes"] != 100
    ):
        raise ValueError("attention four-spine graph differs")
    return GRAPH_SHA256


__all__ = [
    "ANCHOR_NODE", "ANCHOR_NODE_ID", "BRANCH_NODES", "BRANCH_ORDER",
    "BRANCH_PATHS", "CAMPAIGN_LABEL", "COORDINATES", "DOWNSTREAM_FIT_ORDER",
    "ENDPOINT_NODES", "EXECUTION", "FIT_ORDER", "GRAPH_SHA256", "LR_SCHEDULE",
    "NODE_REGISTRY", "PROBABILITY_COMPONENTS", "REDUCER_ORDER",
    "RELATIONAL_CARRIERS", "SOURCE_DISTRIBUTION", "distribution_consumers",
    "graph_payload", "recipe_payload", "validate_graph",
]
