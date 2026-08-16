"""Immutable HCWDL multi-horizon projection-ensemble graph."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_homotopy import HomotopyCoordinate


GRAPH_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v1"
CAMPAIGN_LABEL: Final = "HCWDL-MHPE-FULL"
GRAPH_CONTRACT_C10P90: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v2"
NODE_CONTRACT_C10P90: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v2"
CAMPAIGN_LABEL_C10P90: Final = "HCWDL-MHPE-C10P90-FULL"
PROFILE_C25P75: Final = "C25P75"
PROFILE_C10P90: Final = "C10P90"
SUPPORTED_PROFILES: Final = (PROFILE_C25P75, PROFILE_C10P90)

COORDINATES: Final = MappingProxyType({
    "U000": HomotopyCoordinate(0, 1, 0, 1),
    "U050": HomotopyCoordinate(1, 2, 0, 1),
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D066": HomotopyCoordinate(1, 1, 1, 3),
    "D033": HomotopyCoordinate(1, 1, 2, 3),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
})

STAGE_TEACHERS: Final = MappingProxyType({
    "U050": ("U000",),
    "U100": ("U000", "U050"),
    "D066": ("U000", "U050", "U100E"),
    "D033": ("U000", "U050", "U100E", "D066E"),
    "D000": ("U000", "U050", "U100E", "D066E", "D033E"),
})
ENSEMBLE_COMPONENTS: Final = MappingProxyType({
    f"{stage}E": tuple(sorted(f"{stage}_from_{teacher}" for teacher in teachers))
    for stage, teachers in STAGE_TEACHERS.items() if stage != "U050"
})


@dataclass(frozen=True)
class MhpeNode:
    node_id: str
    coordinate_name: str
    teacher_id: str
    seed_alias: str
    ce_weight: float
    kd_weight: float
    temperature: float
    teacher_kind: str = "logits"
    contract: str = NODE_CONTRACT

    @property
    def coordinate(self) -> HomotopyCoordinate:
        return COORDINATES[self.coordinate_name]

    @property
    def input_domain(self) -> str:
        return "hlt" if self.coordinate_name == "D000" or self.node_id == "M1" else "homotopy"

    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "node_id": self.node_id,
            "coordinate_name": self.coordinate_name,
            "coordinate_exact": self.coordinate.payload(),
            "teacher_id": self.teacher_id,
            "teacher_kind": self.teacher_kind,
            "seed_alias": self.seed_alias,
            "input_domain": self.input_domain,
            "initialization": "fresh",
            "ce_weight": self.ce_weight,
            "kd_weight": self.kd_weight,
            "temperature": self.temperature,
            "training_passes": 20,
        }


def _build_registry(profile: str = PROFILE_C25P75) -> Mapping[str, MhpeNode]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError("unknown HCWDL-MHPE recipe profile")
    specialist_weights = {
        PROFILE_C25P75: (.25, .75),
        PROFILE_C10P90: (.10, .90),
    }[profile]
    node_contract = (
        NODE_CONTRACT if profile == PROFILE_C25P75 else NODE_CONTRACT_C10P90
    )
    nodes: dict[str, MhpeNode] = {}
    for stage, teachers in STAGE_TEACHERS.items():
        for teacher in teachers:
            node_id = f"{stage}_from_{teacher}"
            nodes[node_id] = MhpeNode(
                node_id=node_id, coordinate_name=stage, teacher_id=teacher,
                seed_alias=f"HCWDL-MHPE-FULL/v1/{stage}/paired",
                ce_weight=specialist_weights[0], kd_weight=specialist_weights[1],
                temperature=2.0, contract=node_contract,
                teacher_kind="probabilities" if teacher.endswith("E") else "logits",
            )
    nodes["M1"] = MhpeNode(
        node_id="M1", coordinate_name="D000", teacher_id="D000E",
        seed_alias="HCWDL-MHPE-FULL/v1/M1", ce_weight=.10, kd_weight=.90,
        temperature=1.0, teacher_kind="probabilities", contract=node_contract,
    )
    if len(nodes) != 16:
        raise RuntimeError("HCWDL-MHPE graph must contain exactly 16 fresh fits")
    return MappingProxyType(dict(sorted(nodes.items())))


NODE_REGISTRY: Final = _build_registry(PROFILE_C25P75)
C10P90_NODE_REGISTRY: Final = _build_registry(PROFILE_C10P90)
STAGES: Final = ("U050", "U100", "D066", "D033", "D000", "M1")
FINALISTS: Final = (
    "M0paired",
    "D000_from_U000", "D000_from_U050", "D000_from_U100E",
    "D000_from_D066E", "D000_from_D033E", "D000E", "M1",
)
GRAPH_SHA256: Final = canonical_sha256({
    "contract": GRAPH_CONTRACT,
    "imported": ["U000", "M0paired"],
    "nodes": [NODE_REGISTRY[key].payload() for key in NODE_REGISTRY],
    "ensemble_components": dict(ENSEMBLE_COMPONENTS),
    "finalists": list(FINALISTS),
})
C10P90_GRAPH_SHA256: Final = canonical_sha256({
    "contract": GRAPH_CONTRACT_C10P90,
    "recipe_profile": PROFILE_C10P90,
    "imported": ["U000", "M0paired"],
    "nodes": [C10P90_NODE_REGISTRY[key].payload() for key in C10P90_NODE_REGISTRY],
    "ensemble_components": dict(ENSEMBLE_COMPONENTS),
    "finalists": list(FINALISTS),
})


def node_registry(profile: str = PROFILE_C25P75) -> Mapping[str, MhpeNode]:
    if profile == PROFILE_C25P75:
        return NODE_REGISTRY
    if profile == PROFILE_C10P90:
        return C10P90_NODE_REGISTRY
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def graph_sha256(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return GRAPH_SHA256
    if profile == PROFILE_C10P90:
        return C10P90_GRAPH_SHA256
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def graph_contract(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return GRAPH_CONTRACT
    if profile == PROFILE_C10P90:
        return GRAPH_CONTRACT_C10P90
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def campaign_label(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return CAMPAIGN_LABEL
    if profile == PROFILE_C10P90:
        return CAMPAIGN_LABEL_C10P90
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def validate_graph(profile: str = PROFILE_C25P75) -> str:
    registry = node_registry(profile)
    if set(ENSEMBLE_COMPONENTS) != {"U100E", "D066E", "D033E", "D000E"}:
        raise ValueError("HCWDL-MHPE ensemble registry differs")
    for ensemble, components in ENSEMBLE_COMPONENTS.items():
        if tuple(sorted(components)) != components:
            raise ValueError("HCWDL-MHPE components are not lexical")
        if any(component not in registry for component in components):
            raise ValueError("HCWDL-MHPE component is absent")
    return graph_sha256(profile)


def training_registry(profile: str = PROFILE_C25P75):
    from .hcwdl_ladder import NodeSpec, TeacherSpec
    result = {}
    for node_id, node in node_registry(profile).items():
        result[node_id] = NodeSpec(
            node_id=node_id, track="mhpe", stage="compression" if node_id == "M1" else "specialist",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh", initialization_parent=None,
            teachers=(TeacherSpec(node.teacher_id, "privileged", "parent"),),
            loss_kind="ce_kd", deployable=node.input_domain == "hlt",
        )
    return MappingProxyType(result)


__all__ = [
    "CAMPAIGN_LABEL", "CAMPAIGN_LABEL_C10P90", "C10P90_GRAPH_SHA256",
    "C10P90_NODE_REGISTRY", "COORDINATES", "ENSEMBLE_COMPONENTS", "FINALISTS",
    "GRAPH_CONTRACT", "GRAPH_CONTRACT_C10P90", "GRAPH_SHA256", "MhpeNode",
    "NODE_CONTRACT", "NODE_CONTRACT_C10P90", "NODE_REGISTRY",
    "PROFILE_C10P90", "PROFILE_C25P75", "STAGES", "STAGE_TEACHERS",
    "SUPPORTED_PROFILES", "campaign_label", "graph_contract", "graph_sha256",
    "node_registry", "validate_graph", "training_registry",
]
