"""Immutable matched-seed graph for the TRI60 D000 ablation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_graph import Tri60Node
from .hcwdl_tri60_ce5_graph import (
    NODE_REGISTRY as CE5_NODE_REGISTRY,
    TEACHER_IDS as CE5_TEACHER_IDS,
)
from .hcwdl_tri60_d000_sd5_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, artifact, validate_artifact,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI60-D000-SD5-ABLATION"
ENSEMBLE_ID: Final = "SD5_LOGIT_D000E"
SOURCE_TEACHERS: Final = (
    "U000", "LOGIT_U050E", "LOGIT_U100E", "LOGIT_D066E", "LOGIT_D033E",
)
SEED_MATCH: Final = MappingProxyType(dict(zip(SOURCE_TEACHERS, CE5_TEACHER_IDS)))


def _node_id(teacher_id: str) -> str:
    return f"SD5_LOGIT_D000_from_{teacher_id.removeprefix('LOGIT_')}"


def _node(teacher_id: str) -> Tri60Node:
    ce5_id = SEED_MATCH[teacher_id]
    return Tri60Node(
        node_id=_node_id(teacher_id), track="LOGIT_SD5",
        coordinate_name="D000", distribution_teacher_id=teacher_id,
        distribution_teacher_kind="probability_bank",
        representation_carrier_id=None, auxiliary="none",
        ce_weight=.25, kd_weight=.75, temperature=2.0,
        seed_alias=CE5_NODE_REGISTRY[ce5_id].seed_alias,
        representation_seed_alias=None, training_passes=60, batch_size=256,
        initialization="fresh", node_contract=NODE_CONTRACT,
    )


NODE_REGISTRY: Final[Mapping[str, Tri60Node]] = MappingProxyType({
    node.node_id: node for node in map(_node, SOURCE_TEACHERS)
})
FIT_ORDER: Final = tuple(NODE_REGISTRY)
ENSEMBLE_COMPONENTS: Final = MappingProxyType({ENSEMBLE_ID: FIT_ORDER})


def graph_payload() -> dict[str, object]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "fit_order": list(FIT_ORDER),
        "nodes": {
            node_id: NODE_REGISTRY[node_id].payload() for node_id in FIT_ORDER
        },
        "source_teacher_order": list(SOURCE_TEACHERS),
        "source_teacher_to_fit": {
            teacher: _node_id(teacher) for teacher in SOURCE_TEACHERS
        },
        "ce5_seed_match": dict(SEED_MATCH),
        "ce5_seed_aliases": {
            teacher: CE5_NODE_REGISTRY[SEED_MATCH[teacher]].seed_alias
            for teacher in SOURCE_TEACHERS
        },
        "ensemble_id": ENSEMBLE_ID,
        "ensemble_components": list(FIT_ORDER),
        "ensemble_weight": [1, 5],
        "ensemble_space": "temperature_one_class_probability",
        "fresh_fit_count": 5,
        "reducer_count": 1,
        "validation_selects_graph": False,
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def validate_graph() -> str:
    payload = graph_payload()
    digest = validate_artifact(payload, contract=GRAPH_CONTRACT)
    seed_aliases = [NODE_REGISTRY[name].seed_alias for name in FIT_ORDER]
    if (
        len(FIT_ORDER) != 5
        or len(set(seed_aliases)) != 5
        or tuple(SEED_MATCH) != SOURCE_TEACHERS
        or tuple(SEED_MATCH.values()) != CE5_TEACHER_IDS
        or any(
            node.distribution_teacher_id != teacher
            or node.coordinate_name != "D000"
            or node.track != "LOGIT_SD5"
            or node.auxiliary != "none"
            or node.initialization != "fresh"
            or (node.ce_weight, node.kd_weight, node.temperature) != (.25, .75, 2.0)
            or node.seed_alias != CE5_NODE_REGISTRY[SEED_MATCH[teacher]].seed_alias
            for teacher, node in zip(SOURCE_TEACHERS, NODE_REGISTRY.values())
        )
        or canonical_sha256(payload) != canonical_sha256(graph_payload())
    ):
        raise ValueError("TRI60 D000 SD5 graph semantics differ")
    return digest


__all__ = [
    "CAMPAIGN_LABEL", "ENSEMBLE_COMPONENTS", "ENSEMBLE_ID", "FIT_ORDER",
    "GRAPH_SHA256", "NODE_REGISTRY", "SEED_MATCH", "SOURCE_TEACHERS",
    "graph_payload", "validate_graph",
]
