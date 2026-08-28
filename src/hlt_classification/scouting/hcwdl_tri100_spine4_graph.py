"""Immutable graph for four full-data, single-spine LOGIT-KD ladders."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_tri100_spine4_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, RECIPE_CONTRACT, artifact,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI100-FOUR-SPINE-LOGIT-FULL"
SEED_DOMAIN: Final = f"{CAMPAIGN_LABEL}/v1"
SOURCE_DISTRIBUTION: Final = "U000"
BRANCH_ORDER: Final = ("DIRECT", "COARSE", "DENSE", "ULTRADENSE")
BRANCH_PATHS: Final = MappingProxyType({
    "DIRECT": ("D000",),
    "COARSE": ("U050", "U100", "D066", "D033", "D000"),
    "DENSE": (
        "U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000",
    ),
    "ULTRADENSE": (
        "U020", "U040", "U060", "U080", "U100", "D090", "D080",
        "D070", "D060", "D050", "D040", "D030", "D020", "D010", "D000",
    ),
})


COORDINATES: Final = MappingProxyType({
    "U000": HomotopyCoordinate(0, 1, 0, 1),
    "U020": HomotopyCoordinate(1, 5, 0, 1),
    "U033": HomotopyCoordinate(1, 3, 0, 1),
    "U040": HomotopyCoordinate(2, 5, 0, 1),
    "U050": HomotopyCoordinate(1, 2, 0, 1),
    "U060": HomotopyCoordinate(3, 5, 0, 1),
    "U066": HomotopyCoordinate(2, 3, 0, 1),
    "U080": HomotopyCoordinate(4, 5, 0, 1),
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D090": HomotopyCoordinate(1, 1, 1, 10),
    "D080": HomotopyCoordinate(1, 1, 1, 5),
    "D070": HomotopyCoordinate(1, 1, 3, 10),
    "D066": HomotopyCoordinate(1, 1, 1, 3),
    "D060": HomotopyCoordinate(1, 1, 2, 5),
    "D050": HomotopyCoordinate(1, 1, 1, 2),
    "D040": HomotopyCoordinate(1, 1, 3, 5),
    "D033": HomotopyCoordinate(1, 1, 2, 3),
    "D030": HomotopyCoordinate(1, 1, 7, 10),
    "D020": HomotopyCoordinate(1, 1, 4, 5),
    "D010": HomotopyCoordinate(1, 1, 9, 10),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
})


LR_SCHEDULE: Final = MappingProxyType({
    "kind": "warmup_hold_cosine_floor_tail_v1",
    "warmup_passes": 3,
    "hold_through_pass": 45,
    "decay_through_pass": 60,
    "minimum_lr_fraction": .05,
})
EARLY_STOPPING: Final = MappingProxyType({
    "kind": "macro_auc_patience_v1",
    "minimum_passes": 60,
    "patience_passes": 15,
    "minimum_auc_delta": 5.0e-5,
})
EXECUTION: Final = MappingProxyType({
    "kind": "single_gpu_v1",
    "backend": "cuda",
    "world_size": 1,
    "nodes": 1,
    "ranks_per_node": 1,
    "global_batch_size": 256,
    "local_batch_size": 256,
    "partial_batch_policy": "native_single_process_v1",
    "validation_policy": "single_device_full_canonical_v1",
    "publication_policy": "single_process_v1",
})


@dataclass(frozen=True)
class SpineNode:
    node_id: str
    branch: str
    path_index: int
    coordinate_name: str
    parent_coordinate_name: str
    parent_node_id: str | None
    distribution_teacher_id: str
    output_distribution_id: str | None
    track: str = "LOGIT"
    distribution_teacher_kind: str = "probability_bank"
    representation_carrier_id: str | None = None
    auxiliary: str = "none"
    ce_weight: float = .25
    kd_weight: float = .75
    temperature: float = 2.0
    representation_seed_alias: str | None = None
    training_passes: int = 100
    batch_size: int = 256
    initialization: str = "fresh"
    node_contract: str = NODE_CONTRACT

    @property
    def coordinate(self) -> HomotopyCoordinate:
        return COORDINATES[self.coordinate_name]

    @property
    def seed_alias(self) -> str:
        # Deliberately matched across branches at the same coordinate.  Path
        # and teacher are the intended independent variables.
        return f"{SEED_DOMAIN}/view/{self.coordinate_name}/matched"

    @property
    def deployable(self) -> bool:
        return self.coordinate_name == "D000"

    def payload(self) -> dict[str, object]:
        return {
            "contract": self.node_contract,
            "node_id": self.node_id,
            "branch": self.branch,
            "path_index": self.path_index,
            "track": self.track,
            "coordinate_name": self.coordinate_name,
            "coordinate_exact": self.coordinate.payload(),
            "parent_coordinate_name": self.parent_coordinate_name,
            "parent_node_id": self.parent_node_id,
            "distribution_teacher_id": self.distribution_teacher_id,
            "distribution_teacher_kind": self.distribution_teacher_kind,
            "output_distribution_id": self.output_distribution_id,
            "representation_carrier_id": self.representation_carrier_id,
            "auxiliary": self.auxiliary,
            "ce_weight": self.ce_weight,
            "kd_weight": self.kd_weight,
            "temperature": self.temperature,
            "seed_alias": self.seed_alias,
            "representation_seed_alias": self.representation_seed_alias,
            "training_passes": self.training_passes,
            "validation_every_passes": 1,
            "batch_size": self.batch_size,
            "initialization": self.initialization,
            "deployable": self.deployable,
        }


def _node_id(branch: str, coordinate: str, parent: str) -> str:
    return f"SP4_{branch}_{coordinate}_from_{parent}"


def _distribution_id(node_id: str) -> str:
    return f"DIST_{node_id}"


def _build_nodes() -> tuple[Mapping[str, SpineNode], Mapping[str, tuple[str, ...]]]:
    nodes: dict[str, SpineNode] = {}
    branches: dict[str, tuple[str, ...]] = {}
    for branch in BRANCH_ORDER:
        path = BRANCH_PATHS[branch]
        parent_coordinate = "U000"
        parent_node_id = None
        teacher_distribution = SOURCE_DISTRIBUTION
        branch_nodes = []
        for index, coordinate in enumerate(path):
            node_id = _node_id(branch, coordinate, parent_coordinate)
            output_distribution = (
                None if index == len(path) - 1 else _distribution_id(node_id)
            )
            node = SpineNode(
                node_id=node_id, branch=branch, path_index=index,
                coordinate_name=coordinate,
                parent_coordinate_name=parent_coordinate,
                parent_node_id=parent_node_id,
                distribution_teacher_id=teacher_distribution,
                output_distribution_id=output_distribution,
            )
            if node_id in nodes:
                raise RuntimeError("TRI100 four-spine node identity repeats")
            nodes[node_id] = node
            branch_nodes.append(node_id)
            parent_coordinate = coordinate
            parent_node_id = node_id
            if output_distribution is not None:
                teacher_distribution = output_distribution
        branches[branch] = tuple(branch_nodes)
    return MappingProxyType(nodes), MappingProxyType(branches)


NODE_REGISTRY, BRANCH_NODES = _build_nodes()
FIT_ORDER: Final = tuple(NODE_REGISTRY)
ENDPOINT_NODES: Final = tuple(BRANCH_NODES[name][-1] for name in BRANCH_ORDER)
PROBABILITY_COMPONENTS: Final = MappingProxyType({
    node.output_distribution_id: (node.node_id,)
    for node in NODE_REGISTRY.values()
    if node.output_distribution_id is not None
})
REDUCER_ORDER: Final = tuple(PROBABILITY_COMPONENTS)


def distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    consumers = tuple(
        node_id for node_id, node in NODE_REGISTRY.items()
        if node.distribution_teacher_id == distribution_id
    )
    if not consumers:
        raise KeyError(f"TRI100 four-spine distribution has no consumer: {distribution_id}")
    return consumers


def recipe_payload() -> dict[str, object]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "training": {
            "maximum_passes": 100,
            "minimum_passes": 60,
            "validation_every_passes": 1,
            "effective_batch_size": 256,
            "peak_learning_rate": 3.0e-4,
            "weight_decay": .01,
            "adam_betas": [.9, .999],
            "adam_epsilon": 1.0e-8,
            "forward_precision": "bfloat16",
            "learning_rate_schedule": dict(LR_SCHEDULE),
            "early_stopping": dict(EARLY_STOPPING),
            "restore_best_checkpoint": True,
        },
        "loss": {
            "kind": "constant_ce_kd_v1", "ce_weight": .25,
            "kd_weight": .75, "temperature": 2.0,
        },
        "initialization": "fresh_per_fit",
        "teacher_policy": "immediate_parent_only_selected_checkpoint_logits_v1",
        "execution": dict(EXECUTION),
        "same_coordinate_seed_policy": "matched_across_branches_v1",
        "rolling_resume": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT,
    "schema_version": 1,
    "campaign_label": CAMPAIGN_LABEL,
    "coordinates": {name: value.payload() for name, value in COORDINATES.items()},
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
    "ensembles": False,
    "weight_continuation": False,
    "immediate_parent_only": True,
    "source_outputs_mutated": False,
    "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    value = with_content_hash(_GRAPH_BODY)
    if value["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("TRI100 four-spine graph hash differs")
    return value


def validate_graph() -> str:
    if len(FIT_ORDER) != 29 or len(REDUCER_ORDER) != 25:
        raise ValueError("TRI100 four-spine graph counts differ")
    if tuple(map(len, (BRANCH_NODES[name] for name in BRANCH_ORDER))) != (1, 5, 8, 15):
        raise ValueError("TRI100 four-spine branch lengths differ")
    if dict(EXECUTION) != {
        "kind": "single_gpu_v1", "backend": "cuda",
        "world_size": 1, "nodes": 1, "ranks_per_node": 1,
        "global_batch_size": 256, "local_batch_size": 256,
        "partial_batch_policy": "native_single_process_v1",
        "validation_policy": "single_device_full_canonical_v1",
        "publication_policy": "single_process_v1",
    }:
        raise ValueError("TRI100 four-spine single-GPU execution differs")
    for branch in BRANCH_ORDER:
        previous = None
        for index, node_id in enumerate(BRANCH_NODES[branch]):
            node = NODE_REGISTRY[node_id]
            if node.branch != branch or node.path_index != index:
                raise ValueError("TRI100 four-spine path identity differs")
            if node.parent_node_id != previous:
                raise ValueError("TRI100 four-spine causal parent differs")
            if previous is None:
                if node.distribution_teacher_id != SOURCE_DISTRIBUTION:
                    raise ValueError("TRI100 four-spine source teacher differs")
            else:
                parent = NODE_REGISTRY[previous]
                if (
                    parent.output_distribution_id is None
                    or node.distribution_teacher_id
                    != parent.output_distribution_id
                    or PROBABILITY_COMPONENTS[parent.output_distribution_id]
                    != (previous,)
                ):
                    raise ValueError("TRI100 four-spine immediate teacher differs")
            if node.training_passes != 100 or node.batch_size != 256:
                raise ValueError("TRI100 four-spine fit budget differs")
            if node.initialization != "fresh" or node.auxiliary != "none":
                raise ValueError("TRI100 four-spine model semantics differ")
            if node.ce_weight != .25 or node.kd_weight != .75 or node.temperature != 2.0:
                raise ValueError("TRI100 four-spine KD recipe differs")
            previous = node_id
    for distribution, components in PROBABILITY_COMPONENTS.items():
        if len(components) != 1 or distribution_consumers(distribution) == ():
            raise ValueError("TRI100 four-spine probability graph differs")
    if len({NODE_REGISTRY[name].seed_alias for name in ENDPOINT_NODES}) != 1:
        raise ValueError("TRI100 four-spine endpoints are not seed matched")
    return GRAPH_SHA256


__all__ = [
    "BRANCH_NODES", "BRANCH_ORDER", "BRANCH_PATHS", "CAMPAIGN_LABEL",
    "COORDINATES", "EARLY_STOPPING", "ENDPOINT_NODES", "EXECUTION", "FIT_ORDER",
    "GRAPH_SHA256", "LR_SCHEDULE", "NODE_REGISTRY", "PROBABILITY_COMPONENTS",
    "REDUCER_ORDER", "SEED_DOMAIN", "SOURCE_DISTRIBUTION", "SpineNode",
    "distribution_consumers", "graph_payload", "recipe_payload", "validate_graph",
]
