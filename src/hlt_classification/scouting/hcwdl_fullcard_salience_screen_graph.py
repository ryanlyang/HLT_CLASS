"""Frozen matched U100 endpoint screen for full-cardinality salience pairing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_fullcard_salience_contracts import (
    CANDIDATES,
    SCREEN_GRAPH_CONTRACT,
    SCREEN_RECIPE_CONTRACT,
    SCHEMA_VERSION,
)
from .hcwdl_tri100_spine4_graph import COORDINATES, EARLY_STOPPING, LR_SCHEDULE


BOTTLENECK_REFERENCE: Final = "BOTTLENECK_REFERENCE"
SCREEN_ORDER: Final = (BOTTLENECK_REFERENCE, *CANDIDATES)
CAMPAIGN_LABEL: Final = "HCWDL-FULLCARD-SALIENCE-U100-ENDPOINT-SCREEN"
SEED_ALIAS: Final = f"{CAMPAIGN_LABEL}/matched/U100"


@dataclass(frozen=True)
class ScreenNode:
    candidate: str

    @property
    def node_id(self) -> str:
        return f"SCREEN_U100_{self.candidate}"

    track: str = "CE_SCREEN"
    coordinate_name: str = "U100"
    distribution_teacher_id: None = None
    distribution_teacher_kind: str = "none"
    representation_carrier_id: None = None
    auxiliary: str = "none"
    ce_weight: float = 1.0
    kd_weight: float = 0.0
    temperature: float = 1.0
    representation_seed_alias: None = None
    training_passes: int = 100
    batch_size: int = 256
    initialization: str = "fresh"
    node_contract: str = "HCWDL_FULLCARD_SALIENCE_SCREEN_NODE/v1"
    output_distribution_id: None = None

    @property
    def coordinate(self):
        return COORDINATES["U100"]

    @property
    def seed_alias(self) -> str:
        return SEED_ALIAS

    @property
    def deployable(self) -> bool:
        return False

    def payload(self) -> dict[str, object]:
        return {
            "contract": self.node_contract,
            "node_id": self.node_id,
            "candidate": self.candidate,
            "track": self.track,
            "coordinate_name": self.coordinate_name,
            "coordinate_exact": self.coordinate.payload(),
            "distribution_teacher_id": None,
            "distribution_teacher_kind": "none",
            "representation_carrier_id": None,
            "auxiliary": "none",
            "ce_weight": 1.0,
            "kd_weight": 0.0,
            "temperature": 1.0,
            "seed_alias": self.seed_alias,
            "representation_seed_alias": None,
            "training_passes": 100,
            "validation_every_passes": 1,
            "batch_size": 256,
            "initialization": "fresh",
            "deployable": False,
        }


NODE_REGISTRY: Final = MappingProxyType({
    candidate: ScreenNode(candidate) for candidate in SCREEN_ORDER
})

_GRAPH_BODY: Final = {
    "contract": SCREEN_GRAPH_CONTRACT,
    "schema_version": SCHEMA_VERSION,
    "campaign_label": CAMPAIGN_LABEL,
    "candidate_order": list(SCREEN_ORDER),
    "eligible_salience_candidates": list(CANDIDATES),
    "nodes": [NODE_REGISTRY[name].payload() for name in SCREEN_ORDER],
    "endpoint": "persistent_support_U100",
    "full_train_role": True,
    "validation_checkpoint_fraction": .75,
    "validation_selection_fraction": .25,
    "matched_initialization_seed": SEED_ALIAS,
    "ce_only": True,
    "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    value = with_content_hash(_GRAPH_BODY)
    if value["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("salience screen graph hash differs")
    return value


def recipe_payload() -> dict[str, object]:
    return with_content_hash({
        "contract": SCREEN_RECIPE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "campaign_label": CAMPAIGN_LABEL,
        "loss": {"kind": "ce_only_v1", "ce_weight": 1.0, "kd_weight": 0.0},
        "training": {
            "maximum_passes": 100,
            "minimum_passes": 60,
            "batch_size": 256,
            "optimizer": "AdamW",
            "peak_learning_rate": 3.0e-4,
            "weight_decay": .01,
            "forward_precision": "bfloat16",
            "learning_rate_schedule": dict(LR_SCHEDULE),
            "early_stopping": dict(EARLY_STOPPING),
            "restore_best_checkpoint": True,
        },
        "execution": {
            "kind": "single_gpu_v1", "gpu": "GH200", "world_size": 1,
            "global_batch_size": 256,
        },
        "rolling_resume": False,
        "final_test_accessed": False,
    })


def validate_graph() -> str:
    if (
        tuple(NODE_REGISTRY) != SCREEN_ORDER
        or tuple(NODE_REGISTRY)[1:] != CANDIDATES
        or any(node.seed_alias != SEED_ALIAS for node in NODE_REGISTRY.values())
        or any(node.coordinate_name != "U100" for node in NODE_REGISTRY.values())
    ):
        raise ValueError("salience endpoint screen graph differs")
    return GRAPH_SHA256


__all__ = [
    "BOTTLENECK_REFERENCE", "CAMPAIGN_LABEL", "GRAPH_SHA256", "NODE_REGISTRY",
    "SCREEN_ORDER", "SEED_ALIAS", "graph_payload", "recipe_payload",
    "validate_graph",
]
