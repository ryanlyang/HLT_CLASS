"""Controlled graph wrapper: identical four spines, new pairing lineage."""

from __future__ import annotations

from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_tri100_spine4_bottleneck_contracts import (
    GRAPH_CONTRACT,
    RECIPE_CONTRACT,
    SCHEMA_VERSION,
    artifact,
)
from .repair import PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
from .hcwdl_tri100_spine4_graph import (
    BRANCH_NODES,
    BRANCH_ORDER,
    BRANCH_PATHS,
    COORDINATES,
    EARLY_STOPPING,
    ENDPOINT_NODES,
    EXECUTION,
    FIT_ORDER,
    GRAPH_SHA256 as ESTABLISHED_GRAPH_SHA256,
    LR_SCHEDULE,
    NODE_REGISTRY,
    PROBABILITY_COMPONENTS,
    REDUCER_ORDER,
    SOURCE_DISTRIBUTION,
    distribution_consumers,
    recipe_payload as established_recipe_payload,
    validate_graph as validate_established_graph,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI100-FOUR-SPINE-FULLCARD-BOTTLENECK"
_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT,
    "schema_version": SCHEMA_VERSION,
    "campaign_label": CAMPAIGN_LABEL,
    "established_graph_sha256": ESTABLISHED_GRAPH_SHA256,
    "pairing_control": "full_cardinality_lexicographic_bottleneck_delta_r_v1",
    "matched_unclassified_hlt_policy": PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY,
    "branch_order": list(BRANCH_ORDER),
    "branch_paths": {name: list(BRANCH_PATHS[name]) for name in BRANCH_ORDER},
    "branch_nodes": {name: list(BRANCH_NODES[name]) for name in BRANCH_ORDER},
    "nodes": [NODE_REGISTRY[name].payload() for name in FIT_ORDER],
    "source_distribution": SOURCE_DISTRIBUTION,
    "probability_components": {
        name: list(value) for name, value in PROBABILITY_COMPONENTS.items()
    },
    "fit_order": list(FIT_ORDER),
    "reducer_order": list(REDUCER_ORDER),
    "endpoint_nodes": list(ENDPOINT_NODES),
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
        raise RuntimeError("bottleneck four-spine graph hash differs")
    return value


def recipe_payload() -> dict[str, object]:
    established = established_recipe_payload()
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "established_recipe_sha256": established["content_hash"],
        "training": established["training"],
        "loss": established["loss"],
        "initialization": established["initialization"],
        "teacher_policy": established["teacher_policy"],
        "execution": established["execution"],
        "same_coordinate_seed_policy": established["same_coordinate_seed_policy"],
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "only_changed_variable": "particle_pairing_foundation_lineage",
        "rolling_resume": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


def validate_graph() -> str:
    validate_established_graph()
    if (
        len(FIT_ORDER) != 29 or len(REDUCER_ORDER) != 25
        or dict(EXECUTION).get("world_size") != 1
        or recipe_payload()["training"]["maximum_passes"] != 100
        or recipe_payload()["loss"] != {
            "kind": "constant_ce_kd_v1", "ce_weight": .25,
            "kd_weight": .75, "temperature": 2.0,
        }
    ):
        raise ValueError("bottleneck four-spine controlled graph differs")
    return GRAPH_SHA256


__all__ = [
    "BRANCH_NODES", "BRANCH_ORDER", "BRANCH_PATHS", "CAMPAIGN_LABEL",
    "COORDINATES", "EARLY_STOPPING", "ENDPOINT_NODES", "EXECUTION",
    "FIT_ORDER", "GRAPH_SHA256", "LR_SCHEDULE", "NODE_REGISTRY",
    "PROBABILITY_COMPONENTS", "REDUCER_ORDER", "SOURCE_DISTRIBUTION",
    "distribution_consumers", "graph_payload", "recipe_payload", "validate_graph",
]
