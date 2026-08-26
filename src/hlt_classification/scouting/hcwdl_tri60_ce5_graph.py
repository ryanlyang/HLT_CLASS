"""Immutable graph for the full-data TRI60 CE5 reviewer study."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_graph import Tri60Node
from .hcwdl_tri60_ce5_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, artifact, validate_artifact,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI60-CE5-SEED-ENSEMBLE-REVIEWER"
SEED_DOMAIN: Final = "HCWDL-TRI60-CE5-REVIEWER/v1"
TEACHER_IDS: Final = tuple(f"CE5_S{index:02d}" for index in range(1, 6))
ENSEMBLE_ID: Final = "CE5E"
KD_STUDENT_ID: Final = "CE5_KD"
CONTROL_STUDENT_ID: Final = "CE5_CONTROL"
STUDENT_SEED_ALIAS: Final = f"{SEED_DOMAIN}/student_pair"


def _teacher(index: int) -> Tri60Node:
    return Tri60Node(
        node_id=f"CE5_S{index:02d}", track="CE_TEACHER",
        coordinate_name="D000", distribution_teacher_id=None,
        distribution_teacher_kind="none", representation_carrier_id=None,
        auxiliary="none", ce_weight=1.0, kd_weight=0.0, temperature=1.0,
        seed_alias=f"{SEED_DOMAIN}/teacher/{index:02d}",
        representation_seed_alias=None, training_passes=60, batch_size=256,
        initialization="fresh", node_contract=NODE_CONTRACT,
    )


def _kd_student() -> Tri60Node:
    return Tri60Node(
        node_id=KD_STUDENT_ID, track="CE5_KD", coordinate_name="D000",
        distribution_teacher_id=ENSEMBLE_ID,
        distribution_teacher_kind="probability_bank",
        representation_carrier_id=None, auxiliary="none",
        ce_weight=.10, kd_weight=.90, temperature=1.0,
        seed_alias=STUDENT_SEED_ALIAS, representation_seed_alias=None,
        training_passes=60, batch_size=256, initialization="fresh",
        node_contract=NODE_CONTRACT,
    )


def _control_student() -> Tri60Node:
    return Tri60Node(
        node_id=CONTROL_STUDENT_ID, track="PAIRED_CE_CONTROL",
        coordinate_name="D000", distribution_teacher_id=None,
        distribution_teacher_kind="none", representation_carrier_id=None,
        auxiliary="none", ce_weight=1.0, kd_weight=0.0, temperature=1.0,
        seed_alias=STUDENT_SEED_ALIAS, representation_seed_alias=None,
        training_passes=60, batch_size=256, initialization="fresh",
        node_contract=NODE_CONTRACT,
    )


NODE_REGISTRY: Final[Mapping[str, Tri60Node]] = MappingProxyType({
    **{node.node_id: node for node in (_teacher(index) for index in range(1, 6))},
    KD_STUDENT_ID: _kd_student(),
    CONTROL_STUDENT_ID: _control_student(),
})
FIT_ORDER: Final = (*TEACHER_IDS, KD_STUDENT_ID, CONTROL_STUDENT_ID)
ENSEMBLE_COMPONENTS: Final = MappingProxyType({ENSEMBLE_ID: TEACHER_IDS})


def graph_payload() -> dict[str, object]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "seed_domain": SEED_DOMAIN,
        "fit_order": list(FIT_ORDER),
        "nodes": {
            node_id: NODE_REGISTRY[node_id].payload() for node_id in FIT_ORDER
        },
        "ensemble_id": ENSEMBLE_ID,
        "ensemble_components": list(TEACHER_IDS),
        "ensemble_weight": [1, 5],
        "ensemble_space": "temperature_one_class_probability",
        "paired_students": [KD_STUDENT_ID, CONTROL_STUDENT_ID],
        "paired_student_seed_alias": STUDENT_SEED_ALIAS,
        "fresh_fit_count": 7,
        "reducer_count": 1,
        "validation_selects_graph": False,
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def validate_graph() -> str:
    payload = graph_payload()
    digest = validate_artifact(payload, contract=GRAPH_CONTRACT)
    teacher_aliases = [NODE_REGISTRY[name].seed_alias for name in TEACHER_IDS]
    kd = NODE_REGISTRY[KD_STUDENT_ID]
    control = NODE_REGISTRY[CONTROL_STUDENT_ID]
    if (
        tuple(NODE_REGISTRY) != FIT_ORDER
        or len(set(teacher_aliases)) != 5
        or any(NODE_REGISTRY[name].kd_weight != 0 for name in TEACHER_IDS)
        or any(NODE_REGISTRY[name].ce_weight != 1 for name in TEACHER_IDS)
        or kd.seed_alias != control.seed_alias
        or kd.seed_alias != STUDENT_SEED_ALIAS
        or (kd.ce_weight, kd.kd_weight, kd.temperature) != (.10, .90, 1.0)
        or kd.distribution_teacher_id != ENSEMBLE_ID
        or (control.ce_weight, control.kd_weight) != (1.0, 0.0)
        or control.distribution_teacher_id is not None
        or any(node.coordinate_name != "D000" for node in NODE_REGISTRY.values())
        or any(node.auxiliary != "none" for node in NODE_REGISTRY.values())
        or canonical_sha256(payload) != canonical_sha256(graph_payload())
    ):
        raise ValueError("TRI60 CE5 graph semantics differ")
    return digest


__all__ = [
    "CAMPAIGN_LABEL", "CONTROL_STUDENT_ID", "ENSEMBLE_COMPONENTS",
    "ENSEMBLE_ID", "FIT_ORDER", "GRAPH_SHA256", "KD_STUDENT_ID",
    "NODE_REGISTRY", "SEED_DOMAIN", "STUDENT_SEED_ALIAS", "TEACHER_IDS",
    "graph_payload", "validate_graph",
]
