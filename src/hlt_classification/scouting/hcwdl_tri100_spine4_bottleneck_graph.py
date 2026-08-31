"""Four spines with a fresh persistent-HLT-support U000 anchor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_tri100_spine4_bottleneck_contracts import (
    GRAPH_CONTRACT,
    RECIPE_CONTRACT,
    SCHEMA_VERSION,
    artifact,
)
from .repair import (
    PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY,
    PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY,
)
from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_mhpe_tri60_graph import (
    NODE_REGISTRY as TRI60_NODE_REGISTRY,
)
from .hcwdl_tri100_spine4_graph import (
    BRANCH_NODES as ESTABLISHED_BRANCH_NODES,
    BRANCH_ORDER,
    BRANCH_PATHS,
    COORDINATES,
    EARLY_STOPPING,
    ENDPOINT_NODES,
    EXECUTION,
    GRAPH_SHA256 as ESTABLISHED_GRAPH_SHA256,
    LR_SCHEDULE,
    NODE_REGISTRY as ESTABLISHED_NODE_REGISTRY,
    PROBABILITY_COMPONENTS as ESTABLISHED_PROBABILITY_COMPONENTS,
    recipe_payload as established_recipe_payload,
    validate_graph as validate_established_graph,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI100-FOUR-SPINE-PERSISTENT-HLT-SUPPORT"
ANCHOR_NODE_ID: Final = "SP4P_U000"
SOURCE_DISTRIBUTION: Final = "DIST_SP4P_U000"


@dataclass(frozen=True)
class PersistentAnchorNode:
    node_id: str = ANCHOR_NODE_ID
    track: str = "ROOT"
    coordinate_name: str = "U000"
    distribution_teacher_id: None = None
    distribution_teacher_kind: str = "none"
    representation_carrier_id: None = None
    auxiliary: str = "none"
    ce_weight: float = 1.0
    kd_weight: float = 0.0
    temperature: float = 1.0
    seed_alias: str = TRI60_NODE_REGISTRY["U000"].seed_alias
    representation_seed_alias: None = None
    training_passes: int = 60
    batch_size: int = 256
    initialization: str = "fresh"
    node_contract: str = TRI60_NODE_REGISTRY["U000"].node_contract
    output_distribution_id: str = SOURCE_DISTRIBUTION
    branch: str = "ANCHOR"
    path_index: int = -1
    parent_node_id: None = None
    parent_coordinate_name: None = None

    @property
    def coordinate(self):
        return COORDINATES["U000"]

    @property
    def deployable(self) -> bool:
        return False

    def payload(self) -> dict[str, object]:
        value = TRI60_NODE_REGISTRY["U000"].payload()
        value.update({
            "node_id": self.node_id,
            "output_distribution_id": self.output_distribution_id,
            "persistent_hlt_support_anchor": True,
        })
        return value


ANCHOR_NODE: Final = PersistentAnchorNode()

_nodes = {ANCHOR_NODE_ID: ANCHOR_NODE}
_branches = {}
for _branch in BRANCH_ORDER:
    _branch_nodes = []
    for _index, _node_id in enumerate(ESTABLISHED_BRANCH_NODES[_branch]):
        _node = ESTABLISHED_NODE_REGISTRY[_node_id]
        if _index == 0:
            _node = replace(
                _node, parent_node_id=ANCHOR_NODE_ID,
                distribution_teacher_id=SOURCE_DISTRIBUTION,
            )
        _nodes[_node_id] = _node
        _branch_nodes.append(_node_id)
    _branches[_branch] = tuple(_branch_nodes)
NODE_REGISTRY: Final = MappingProxyType(_nodes)
BRANCH_NODES: Final = MappingProxyType(_branches)
FIT_ORDER: Final = tuple(NODE_REGISTRY)
DOWNSTREAM_FIT_ORDER: Final = tuple(
    node_id for node_id in FIT_ORDER if node_id != ANCHOR_NODE_ID
)
PROBABILITY_COMPONENTS: Final = MappingProxyType({
    SOURCE_DISTRIBUTION: (ANCHOR_NODE_ID,),
    **dict(ESTABLISHED_PROBABILITY_COMPONENTS),
})
REDUCER_ORDER: Final = tuple(PROBABILITY_COMPONENTS)


def distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    consumers = tuple(
        node_id for node_id, node in NODE_REGISTRY.items()
        if node.distribution_teacher_id == distribution_id
    )
    if not consumers:
        raise KeyError(f"persistent-support distribution has no consumer: {distribution_id}")
    return consumers
_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT,
    "schema_version": SCHEMA_VERSION,
    "campaign_label": CAMPAIGN_LABEL,
    "established_graph_sha256": ESTABLISHED_GRAPH_SHA256,
    "pairing_control": "full_cardinality_lexicographic_bottleneck_delta_r_v1",
    "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
    "matched_unclassified_hlt_policy": PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY,
    "matched_unclassified_offline_policy": (
        PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY
    ),
    "branch_order": list(BRANCH_ORDER),
    "branch_paths": {name: list(BRANCH_PATHS[name]) for name in BRANCH_ORDER},
    "branch_nodes": {name: list(BRANCH_NODES[name]) for name in BRANCH_ORDER},
    "nodes": [NODE_REGISTRY[name].payload() for name in FIT_ORDER],
    "source_distribution": SOURCE_DISTRIBUTION,
    "experimental_anchor": ANCHOR_NODE_ID,
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
        "anchor_training": {
            "passes": 60, "validation_every_passes": 1,
            "effective_batch_size": 256, "optimizer": "AdamW",
            "peak_learning_rate": 3.0e-4, "weight_decay": .01,
            "warmup_fraction": .05,
            "schedule": "linear_warmup_cosine_decay_v1",
            "learning_rate_floor_fraction": .05,
            "forward_precision": "bfloat16",
            "performance_early_stopping": False,
            "restore_best_checkpoint": True,
        },
        "loss": established["loss"],
        "initialization": established["initialization"],
        "teacher_policy": established["teacher_policy"],
        "execution": established["execution"],
        "same_coordinate_seed_policy": established["same_coordinate_seed_policy"],
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "matched_unclassified_offline_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY
        ),
        "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "only_changed_variable": "persistent_hlt_skeleton_across_u",
        "rolling_resume": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


def validate_graph() -> str:
    validate_established_graph()
    if (
        len(FIT_ORDER) != 30 or len(REDUCER_ORDER) != 26
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
    "ANCHOR_NODE", "ANCHOR_NODE_ID", "BRANCH_NODES", "BRANCH_ORDER",
    "BRANCH_PATHS", "CAMPAIGN_LABEL",
    "COORDINATES", "EARLY_STOPPING", "ENDPOINT_NODES", "EXECUTION",
    "DOWNSTREAM_FIT_ORDER", "FIT_ORDER", "GRAPH_SHA256", "LR_SCHEDULE", "NODE_REGISTRY",
    "PROBABILITY_COMPONENTS", "REDUCER_ORDER", "SOURCE_DISTRIBUTION",
    "distribution_consumers", "graph_payload", "recipe_payload", "validate_graph",
]
