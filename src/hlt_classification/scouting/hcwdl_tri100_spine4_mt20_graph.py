"""Four persistent-HLT spines with exact all-ancestor MT20 supervision."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_mhpe_tri60_graph import NODE_REGISTRY as TRI60_NODE_REGISTRY
from .hcwdl_tri100_spine4_bottleneck_graph import (
    BRANCH_NODES as PERSISTENT_BRANCH_NODES,
    BRANCH_ORDER,
    BRANCH_PATHS,
    COORDINATES,
    EARLY_STOPPING,
    ENDPOINT_NODES,
    EXECUTION,
    GRAPH_SHA256 as PERSISTENT_GRAPH_SHA256,
    LR_SCHEDULE,
    NODE_REGISTRY as PERSISTENT_NODE_REGISTRY,
    PROBABILITY_COMPONENTS as PERSISTENT_PROBABILITY_COMPONENTS,
    SOURCE_DISTRIBUTION as PERSISTENT_SOURCE_DISTRIBUTION,
    recipe_payload as persistent_recipe_payload,
    validate_graph as validate_persistent_graph,
)
from .hcwdl_tri100_spine4_mt20_contracts import (
    GRAPH_CONTRACT,
    RECIPE_CONTRACT,
    SCHEMA_VERSION,
    artifact,
)
from .repair import (
    PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY,
    PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI100-FOUR-SPINE-PERSISTENT-HLT-MT20"
ANCHOR_NODE_ID: Final = "SP4MT20_U000"
SOURCE_DISTRIBUTION: Final = "DIST_SP4MT20_U000"
CE_WEIGHT: Final = Fraction(1, 5)
KD_WEIGHT: Final = Fraction(4, 5)
IMMEDIATE_WEIGHT: Final = Fraction(1, 2)
HISTORICAL_WEIGHT: Final = Fraction(3, 10)


def _rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def mt20_teacher_weights(count: int) -> tuple[Fraction, ...]:
    """Return nearest-first KD-loss contributions summing exactly to 4/5."""

    if count < 1:
        raise ValueError("MT20 requires at least one teacher")
    if count == 1:
        return (KD_WEIGHT,)
    raw = tuple(Fraction(1, 2**index) for index in range(1, count))
    normalizer = sum(raw, Fraction())
    weights = (
        IMMEDIATE_WEIGHT,
        *(HISTORICAL_WEIGHT * value / normalizer for value in raw),
    )
    if sum(weights, Fraction()) != KD_WEIGHT or any(value <= 0 for value in weights):
        raise RuntimeError("MT20 teacher weights differ")
    return weights


@dataclass(frozen=True)
class Mt20AnchorNode:
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
            "mt20_anchor": True,
        })
        return value


ANCHOR_NODE: Final = Mt20AnchorNode()

_nodes = {ANCHOR_NODE_ID: ANCHOR_NODE}
_branches: dict[str, tuple[str, ...]] = {}
_teacher_nodes: dict[str, tuple[str, ...]] = {}
_teacher_distributions: dict[str, tuple[str, ...]] = {}
_teacher_weights: dict[str, tuple[Fraction, ...]] = {}
for _branch in BRANCH_ORDER:
    _branch_nodes = []
    _ancestry = [ANCHOR_NODE_ID]
    for _index, _node_id in enumerate(PERSISTENT_BRANCH_NODES[_branch]):
        _node = PERSISTENT_NODE_REGISTRY[_node_id]
        if _index == 0:
            _node = replace(
                _node,
                parent_node_id=ANCHOR_NODE_ID,
                distribution_teacher_id=SOURCE_DISTRIBUTION,
            )
        _node = replace(_node, ce_weight=float(CE_WEIGHT), kd_weight=float(KD_WEIGHT))
        _nodes[_node_id] = _node
        nearest_first = tuple(reversed(_ancestry))
        distributions = tuple(
            _nodes[name].output_distribution_id for name in nearest_first
        )
        if any(value is None for value in distributions):
            raise RuntimeError("MT20 ancestry contains a non-teacher endpoint")
        _teacher_nodes[_node_id] = nearest_first
        _teacher_distributions[_node_id] = distributions
        _teacher_weights[_node_id] = mt20_teacher_weights(len(nearest_first))
        _branch_nodes.append(_node_id)
        _ancestry.append(_node_id)
    _branches[_branch] = tuple(_branch_nodes)

NODE_REGISTRY: Final = MappingProxyType(_nodes)
BRANCH_NODES: Final = MappingProxyType(_branches)
TEACHER_NODES: Final = MappingProxyType(_teacher_nodes)
TEACHER_DISTRIBUTIONS: Final = MappingProxyType(_teacher_distributions)
TEACHER_WEIGHTS: Final = MappingProxyType(_teacher_weights)
FIT_ORDER: Final = tuple(NODE_REGISTRY)
DOWNSTREAM_FIT_ORDER: Final = tuple(
    node_id for node_id in FIT_ORDER if node_id != ANCHOR_NODE_ID
)
PROBABILITY_COMPONENTS: Final = MappingProxyType({
    SOURCE_DISTRIBUTION: (ANCHOR_NODE_ID,),
    **{
        name: value for name, value in PERSISTENT_PROBABILITY_COMPONENTS.items()
        if name != PERSISTENT_SOURCE_DISTRIBUTION
    },
})
REDUCER_ORDER: Final = tuple(PROBABILITY_COMPONENTS)


def distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    consumers = tuple(
        node_id for node_id in DOWNSTREAM_FIT_ORDER
        if distribution_id in TEACHER_DISTRIBUTIONS[node_id]
    )
    if not consumers:
        raise KeyError(f"MT20 probability distribution has no consumer: {distribution_id}")
    return consumers


def teacher_registry(node_id: str) -> tuple[dict[str, object], ...]:
    if node_id not in TEACHER_NODES:
        raise KeyError(f"MT20 node has no teacher registry: {node_id}")
    return tuple({
        "teacher_node_id": teacher,
        "distribution_id": distribution,
        "weight": _rational(weight),
    } for teacher, distribution, weight in zip(
        TEACHER_NODES[node_id],
        TEACHER_DISTRIBUTIONS[node_id],
        TEACHER_WEIGHTS[node_id],
        strict=True,
    ))


_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT,
    "schema_version": SCHEMA_VERSION,
    "campaign_label": CAMPAIGN_LABEL,
    "persistent_graph_sha256": PERSISTENT_GRAPH_SHA256,
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
    "teacher_registries": {
        name: list(teacher_registry(name)) for name in DOWNSTREAM_FIT_ORDER
    },
    "fit_order": list(FIT_ORDER),
    "reducer_order": list(REDUCER_ORDER),
    "endpoint_nodes": list(ENDPOINT_NODES),
    "fresh_fit_count": len(FIT_ORDER),
    "single_component_probability_banks": True,
    "ram_only_weighted_teacher_mixtures": True,
    "all_prior_same_spine_teachers": True,
    "cross_spine_teachers": False,
    "immediate_parent_only": False,
    "ensembles": False,
    "weight_continuation": False,
    "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    value = with_content_hash(_GRAPH_BODY)
    if value["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("MT20 graph hash differs")
    return value


def recipe_payload() -> dict[str, object]:
    persistent = persistent_recipe_payload()
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "persistent_recipe_sha256": persistent["content_hash"],
        "training": persistent["training"],
        "anchor_training": persistent["anchor_training"],
        "loss": {
            "kind": "constant_ce_weighted_probability_kd_v1",
            "ce_weight": float(CE_WEIGHT),
            "kd_weight": float(KD_WEIGHT),
            "temperature": 2.0,
        },
        "teacher_policy": {
            "kind": "all_prior_same_spine_geometric_history_v1",
            "single_teacher_weight": _rational(KD_WEIGHT),
            "immediate_weight": _rational(IMMEDIATE_WEIGHT),
            "historical_total_weight": _rational(HISTORICAL_WEIGHT),
            "historical_decay_ratio": _rational(Fraction(1, 2)),
            "mixture_domain": "temperature_scaled_probabilities",
            "accumulation_dtype": "float64",
            "training_dtype": "float32",
            "durable_mixture_arrays": False,
        },
        "initialization": persistent["initialization"],
        "execution": persistent["execution"],
        "same_coordinate_seed_policy": persistent["same_coordinate_seed_policy"],
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "matched_unclassified_offline_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY
        ),
        "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "combined_intervention": ["c20p80", "all_prior_same_spine_teachers"],
        "rolling_resume": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


def validate_graph() -> str:
    validate_persistent_graph()
    if (
        len(FIT_ORDER) != 30
        or len(REDUCER_ORDER) != 26
        or dict(EXECUTION).get("world_size") != 1
        or recipe_payload()["training"]["maximum_passes"] != 100
        or recipe_payload()["loss"] != {
            "kind": "constant_ce_weighted_probability_kd_v1",
            "ce_weight": .20,
            "kd_weight": .80,
            "temperature": 2.0,
        }
        or any(
            sum(TEACHER_WEIGHTS[name], Fraction()) != KD_WEIGHT
            or TEACHER_NODES[name][0] != NODE_REGISTRY[name].parent_node_id
            for name in DOWNSTREAM_FIT_ORDER
        )
    ):
        raise ValueError("MT20 controlled graph differs")
    for branch in BRANCH_ORDER:
        ancestry = [ANCHOR_NODE_ID]
        for node_id in BRANCH_NODES[branch]:
            if TEACHER_NODES[node_id] != tuple(reversed(ancestry)):
                raise ValueError("MT20 same-spine ancestry differs")
            ancestry.append(node_id)
    return GRAPH_SHA256


__all__ = [
    "ANCHOR_NODE", "ANCHOR_NODE_ID", "BRANCH_NODES", "BRANCH_ORDER",
    "BRANCH_PATHS", "CAMPAIGN_LABEL", "CE_WEIGHT", "COORDINATES",
    "DOWNSTREAM_FIT_ORDER", "EARLY_STOPPING", "ENDPOINT_NODES", "EXECUTION",
    "FIT_ORDER", "GRAPH_SHA256", "HISTORICAL_WEIGHT", "IMMEDIATE_WEIGHT",
    "KD_WEIGHT", "LR_SCHEDULE", "NODE_REGISTRY", "PROBABILITY_COMPONENTS",
    "REDUCER_ORDER", "SOURCE_DISTRIBUTION", "TEACHER_DISTRIBUTIONS",
    "TEACHER_NODES", "TEACHER_WEIGHTS", "distribution_consumers",
    "graph_payload", "mt20_teacher_weights", "recipe_payload",
    "teacher_registry", "validate_graph",
]
