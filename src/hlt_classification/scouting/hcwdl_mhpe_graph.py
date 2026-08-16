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

    @property
    def coordinate(self) -> HomotopyCoordinate:
        return COORDINATES[self.coordinate_name]

    @property
    def input_domain(self) -> str:
        return "hlt" if self.coordinate_name == "D000" or self.node_id == "M1" else "homotopy"

    def payload(self) -> dict[str, object]:
        return {
            "contract": NODE_CONTRACT,
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


def _build_registry() -> Mapping[str, MhpeNode]:
    nodes: dict[str, MhpeNode] = {}
    for stage, teachers in STAGE_TEACHERS.items():
        for teacher in teachers:
            node_id = f"{stage}_from_{teacher}"
            nodes[node_id] = MhpeNode(
                node_id=node_id, coordinate_name=stage, teacher_id=teacher,
                seed_alias=f"HCWDL-MHPE-FULL/v1/{stage}/paired",
                ce_weight=.25, kd_weight=.75, temperature=2.0,
                teacher_kind="probabilities" if teacher.endswith("E") else "logits",
            )
    nodes["M1"] = MhpeNode(
        node_id="M1", coordinate_name="D000", teacher_id="D000E",
        seed_alias="HCWDL-MHPE-FULL/v1/M1", ce_weight=.10, kd_weight=.90,
        temperature=1.0, teacher_kind="probabilities",
    )
    if len(nodes) != 16:
        raise RuntimeError("HCWDL-MHPE graph must contain exactly 16 fresh fits")
    return MappingProxyType(dict(sorted(nodes.items())))


NODE_REGISTRY: Final = _build_registry()
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


def validate_graph() -> str:
    if set(ENSEMBLE_COMPONENTS) != {"U100E", "D066E", "D033E", "D000E"}:
        raise ValueError("HCWDL-MHPE ensemble registry differs")
    for ensemble, components in ENSEMBLE_COMPONENTS.items():
        if tuple(sorted(components)) != components:
            raise ValueError("HCWDL-MHPE components are not lexical")
        if any(component not in NODE_REGISTRY for component in components):
            raise ValueError("HCWDL-MHPE component is absent")
    return GRAPH_SHA256


def training_registry():
    from .hcwdl_ladder import NodeSpec, TeacherSpec
    result = {}
    for node_id, node in NODE_REGISTRY.items():
        result[node_id] = NodeSpec(
            node_id=node_id, track="mhpe", stage="compression" if node_id == "M1" else "specialist",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh", initialization_parent=None,
            teachers=(TeacherSpec(node.teacher_id, "privileged", "parent"),),
            loss_kind="ce_kd", deployable=node.input_domain == "hlt",
        )
    return MappingProxyType(result)


__all__ = [
    "CAMPAIGN_LABEL", "COORDINATES", "ENSEMBLE_COMPONENTS", "FINALISTS",
    "GRAPH_CONTRACT", "GRAPH_SHA256", "MhpeNode", "NODE_CONTRACT",
    "NODE_REGISTRY", "STAGES", "STAGE_TEACHERS", "validate_graph",
    "training_registry",
]
