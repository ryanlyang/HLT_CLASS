"""Five-fit direct native-offline-to-HLT KD ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_ladder import NodeSpec, TeacherSpec
from .hcwdl_representation_graph import RREL_STRATEGY, RSET_STRATEGY


GRAPH_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_NODE/v1"
CAMPAIGN_LABEL: Final = "HCWDL_DIRECT_OFFLINE_KD"
ROLE_COUNTS: Final = {"train": 300_000, "validation": 100_000, "final_test": 0}
HLT_SEED_ALIAS: Final = "direct_hlt_pair_v1"
TOFF_SEED_ALIAS: Final = "direct_toff_root_v1"
AUTHORIZATION_PHRASE: Final = "AUTHORIZE HCWDL DIRECT OFFLINE KD 300K EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL DIRECT OFFLINE KD 300K EXACT DAG"


@dataclass(frozen=True)
class DirectRepresentationNodeSpec:
    node_id: str
    strategy: str
    track: str = "direct_offline_to_hlt"
    student_domain: str = "hlt"
    teacher_node_id: str = "TOFF_CE"
    teacher_domain: str = "toff"
    seed_alias: str = HLT_SEED_ALIAS
    initialization: str = "fresh"
    deployable: bool = True

    def payload(self) -> dict[str, object]:
        return {
            "contract": NODE_CONTRACT,
            "schema_version": 1,
            **asdict(self),
            "loss": "unweighted_ce_0p25_plus_toff_kd_0p75_t2_plus_repr_0p10",
            "training_passes": 60,
            "validation_every_passes": 1,
            "final_test_accessed": False,
        }


BASE_NODE_REGISTRY: Final[Mapping[str, NodeSpec]] = MappingProxyType({
    "HLT_CE": NodeSpec(
        "HLT_CE", "direct", "root", "hlt", "fresh", None, (), "ce", True,
    ),
    "TOFF_CE": NodeSpec(
        "TOFF_CE", "direct", "root", "toff", "fresh", None, (), "ce", False,
    ),
    "HLT_LOGIT": NodeSpec(
        "HLT_LOGIT", "direct", "student", "hlt", "fresh", None,
        (TeacherSpec("TOFF_CE", "toff", "sole"),), "ce_kd", True,
    ),
})

REPRESENTATION_NODE_REGISTRY: Final[
    Mapping[str, DirectRepresentationNodeSpec]
] = MappingProxyType({
    "HLT_RSET": DirectRepresentationNodeSpec("HLT_RSET", RSET_STRATEGY),
    "HLT_RREL": DirectRepresentationNodeSpec("HLT_RREL", RREL_STRATEGY),
})

NODE_ORDER: Final = ("HLT_CE", "TOFF_CE", "HLT_LOGIT", "HLT_RSET", "HLT_RREL")


def _base_node_payload(node_id: str) -> dict[str, object]:
    """Return the canonical JSON-native form of an imported ladder node."""
    payload = {
        **BASE_NODE_REGISTRY[node_id].payload(),
        "contract": NODE_CONTRACT,
    }
    # NodeSpec is reusable elsewhere with tuple-valued teachers, but direct
    # campaign artifacts cross a JSON persistence boundary before submission.
    # Emit the persisted representation up front so create-time and reload-time
    # graph validation compare the same Python value types.
    payload["teachers"] = list(payload["teachers"])
    return payload


def graph_payload() -> dict[str, object]:
    return {
        "contract": GRAPH_CONTRACT,
        "schema_version": 1,
        "fit_count": 5,
        "node_order": list(NODE_ORDER),
        "base_nodes": [
            _base_node_payload(node)
            for node in ("HLT_CE", "TOFF_CE", "HLT_LOGIT")
        ],
        "representation_nodes": [
            REPRESENTATION_NODE_REGISTRY[node].payload()
            for node in ("HLT_RSET", "HLT_RREL")
        ],
        "shared_teacher_target_bank": {
            "teacher": "TOFF_CE",
            "consumers": ["HLT_LOGIT", "HLT_RSET", "HLT_RREL"],
        },
        "paired_hlt_seed_alias": HLT_SEED_ALIAS,
        "fresh_toff_seed_alias": TOFF_SEED_ALIAS,
        "final_test_accessed": False,
    }


def validate_graph() -> str:
    if set(BASE_NODE_REGISTRY) != {"HLT_CE", "TOFF_CE", "HLT_LOGIT"}:
        raise ValueError("direct KD base-node registry differs")
    if set(REPRESENTATION_NODE_REGISTRY) != {"HLT_RSET", "HLT_RREL"}:
        raise ValueError("direct KD representation-node registry differs")
    logit = BASE_NODE_REGISTRY["HLT_LOGIT"]
    if logit.teachers != (TeacherSpec("TOFF_CE", "toff", "sole"),):
        raise ValueError("direct logit student teacher differs")
    if any(
        node.teacher_node_id != "TOFF_CE"
        or node.teacher_domain != "toff"
        or node.student_domain != "hlt"
        or node.initialization != "fresh"
        or not node.deployable
        for node in REPRESENTATION_NODE_REGISTRY.values()
    ):
        raise ValueError("direct representation student semantics differ")
    return canonical_sha256(graph_payload())


GRAPH_SHA256: Final = validate_graph()


def graph_artifact() -> dict[str, object]:
    return with_content_hash(graph_payload())


__all__ = [
    "AUTHORIZATION_PHRASE", "BASE_NODE_REGISTRY", "CAMPAIGN_LABEL",
    "DirectRepresentationNodeSpec",
    "GRAPH_CONTRACT", "GRAPH_SHA256", "HLT_SEED_ALIAS", "NODE_CONTRACT",
    "NODE_ORDER", "REPRESENTATION_NODE_REGISTRY", "ROLE_COUNTS",
    "SUBMISSION_PHRASE", "TOFF_SEED_ALIAS", "graph_artifact", "validate_graph",
]
