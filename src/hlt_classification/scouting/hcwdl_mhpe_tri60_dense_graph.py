"""Immutable additive graph for the independent TRI60 dense extension.

The source TRI60 graph is never modified.  Components without the ``DX_``
prefix are immutable imports from that graph; every fresh fit and every
expanded probability ensemble has a new dense-extension identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_mhpe_tri60_dense_contracts import GRAPH_CONTRACT, NODE_CONTRACT
from .hcwdl_mhpe_tri60_graph import GRAPH_SHA256 as SOURCE_GRAPH_SHA256


CAMPAIGN_LABEL: Final = "HCWDL-MHPE-TRI60-DENSE-EXTENSION"
SEED_DOMAIN: Final = f"{CAMPAIGN_LABEL}/v1"

COORDINATES: Final = MappingProxyType({
    "U000": HomotopyCoordinate(0, 1, 0, 1),
    "U050": HomotopyCoordinate(1, 2, 0, 1),
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D083": HomotopyCoordinate(1, 1, 1, 6),
    "D075": HomotopyCoordinate(1, 1, 1, 4),
    "D066": HomotopyCoordinate(1, 1, 1, 3),
    "D050": HomotopyCoordinate(1, 1, 1, 2),
    "D033": HomotopyCoordinate(1, 1, 2, 3),
    "D025": HomotopyCoordinate(1, 1, 3, 4),
    "D017": HomotopyCoordinate(1, 1, 5, 6),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
})


@dataclass(frozen=True)
class DenseNode:
    node_id: str
    track: str
    coordinate_name: str
    distribution_teacher_id: str | None
    distribution_teacher_kind: str
    representation_carrier_id: str | None
    auxiliary: str
    ce_weight: float
    kd_weight: float
    temperature: float
    seed_alias: str
    representation_seed_alias: str | None
    training_passes: int = 60
    batch_size: int = 256
    initialization: str = "fresh"
    node_contract: str = NODE_CONTRACT

    @property
    def coordinate(self) -> HomotopyCoordinate:
        return COORDINATES[self.coordinate_name]

    @property
    def deployable(self) -> bool:
        return self.coordinate_name == "D000"

    def payload(self) -> dict[str, object]:
        return {
            "contract": self.node_contract, "node_id": self.node_id,
            "track": self.track, "coordinate_name": self.coordinate_name,
            "coordinate_exact": self.coordinate.payload(),
            "distribution_teacher_id": self.distribution_teacher_id,
            "distribution_teacher_kind": self.distribution_teacher_kind,
            "representation_carrier_id": self.representation_carrier_id,
            "auxiliary": self.auxiliary, "ce_weight": self.ce_weight,
            "kd_weight": self.kd_weight, "temperature": self.temperature,
            "seed_alias": self.seed_alias,
            "representation_seed_alias": self.representation_seed_alias,
            "training_passes": self.training_passes,
            "validation_every_passes": 1, "batch_size": self.batch_size,
            "initialization": self.initialization, "deployable": self.deployable,
        }


SOURCE_DISTRIBUTIONS: Final = (
    "U000", "LOGIT_U050E", "LOGIT_U100E", "RSET_U100E", "RREL_U100E",
)

# Imports that remain exact components of expanded ensembles.  Lower imports
# are authenticated only after the source campaign's exact completion job.
EARLY_SOURCE_NODES: Final = (
    "U000", "LOGIT_U050_from_U000", "LOGIT_U100_from_U000",
    "LOGIT_U100_from_U050E", "RSET_U100_from_U000", "RREL_U100_from_U000",
)
LATE_SOURCE_NODES: Final = (
    "LOGIT_D066_from_U000", "LOGIT_D066_from_U050E",
    "LOGIT_D066_from_U100E", "LOGIT_D033_from_U000",
    "LOGIT_D033_from_U050E", "LOGIT_D033_from_U100E",
    "LOGIT_D000_from_U000", "LOGIT_D000_from_U050E",
    "LOGIT_D000_from_U100E",
    "RSET_D050_from_U000", "RSET_D050_from_U100E",
    "RSET_D000_from_U000", "RSET_D000_from_U100E",
    "RREL_D050_from_U000", "RREL_D050_from_U100E",
    "RREL_D000_from_U000", "RREL_D000_from_U100E",
)
SOURCE_NODES: Final = EARLY_SOURCE_NODES + LATE_SOURCE_NODES


def _suffix(track: str, teacher: str) -> str:
    prefix = f"DX_{track}_"
    return teacher[len(prefix):] if teacher.startswith(prefix) else teacher


def _specialist(
    *, track: str, coordinate: str, teacher: str, carrier: str | None = None,
) -> DenseNode:
    auxiliary = track.lower() if track in {"RSET", "RREL"} else "none"
    node_id = f"DX_{track}_{coordinate}_from_{_suffix(track, teacher)}"
    if auxiliary != "none" and carrier is None:
        raise ValueError("dense representation specialist lacks its carrier")
    return DenseNode(
        node_id=node_id, track=track, coordinate_name=coordinate,
        distribution_teacher_id=teacher,
        distribution_teacher_kind=(
            "source_probability_bank" if teacher in SOURCE_DISTRIBUTIONS
            else "dense_probability_bank"
        ),
        representation_carrier_id=carrier, auxiliary=auxiliary,
        ce_weight=.25, kd_weight=.75, temperature=2.0,
        seed_alias=f"{SEED_DOMAIN}/view/{coordinate}/paired",
        representation_seed_alias=(
            None if auxiliary == "none"
            else f"{SEED_DOMAIN}/{auxiliary}/{coordinate}/representation"
        ),
    )


LOGIT_TEACHERS: Final = MappingProxyType({
    "D083": ("U000", "LOGIT_U050E", "LOGIT_U100E"),
    "D066": ("DX_LOGIT_D083E",),
    "D050": (
        "U000", "LOGIT_U050E", "LOGIT_U100E", "DX_LOGIT_D083E",
        "DX_LOGIT_D066E",
    ),
    "D033": ("DX_LOGIT_D083E", "DX_LOGIT_D066E", "DX_LOGIT_D050E"),
    "D017": (
        "U000", "LOGIT_U050E", "LOGIT_U100E", "DX_LOGIT_D083E",
        "DX_LOGIT_D066E", "DX_LOGIT_D050E", "DX_LOGIT_D033E",
    ),
    "D000": (
        "DX_LOGIT_D083E", "DX_LOGIT_D066E", "DX_LOGIT_D050E",
        "DX_LOGIT_D033E", "DX_LOGIT_D017E",
    ),
})

REP_TEACHERS: Final = MappingProxyType({
    "D075": ("U000", "{track}_U100E"),
    "D050": ("DX_{track}_D075E",),
    "D025": ("U000", "{track}_U100E", "DX_{track}_D075E", "DX_{track}_D050E"),
    "D000": ("DX_{track}_D075E", "DX_{track}_D050E", "DX_{track}_D025E"),
})

REPRESENTATION_CARRIERS: Final = MappingProxyType({
    "U000": "U000",
    "RSET_U100E": "RSET_U100_from_U000",
    "RREL_U100E": "RREL_U100_from_U000",
    "DX_RSET_D075E": "DX_RSET_D075_from_RSET_U100E",
    "DX_RSET_D050E": "DX_RSET_D050_from_D075E",
    "DX_RSET_D025E": "DX_RSET_D025_from_D050E",
    "DX_RSET_D000E": "DX_RSET_D000_from_D025E",
    "DX_RREL_D075E": "DX_RREL_D075_from_RREL_U100E",
    "DX_RREL_D050E": "DX_RREL_D050_from_D075E",
    "DX_RREL_D025E": "DX_RREL_D025_from_D050E",
    "DX_RREL_D000E": "DX_RREL_D000_from_D025E",
})


def _build_nodes() -> Mapping[str, DenseNode]:
    nodes: dict[str, DenseNode] = {}
    for coordinate, teachers in LOGIT_TEACHERS.items():
        for teacher in teachers:
            node = _specialist(track="LOGIT", coordinate=coordinate, teacher=teacher)
            nodes[node.node_id] = node
    nodes["DX_M1_LOGIT"] = DenseNode(
        node_id="DX_M1_LOGIT", track="LOGIT", coordinate_name="D000",
        distribution_teacher_id="DX_LOGIT_D000E",
        distribution_teacher_kind="dense_probability_bank",
        representation_carrier_id=None, auxiliary="none", ce_weight=.10,
        kd_weight=.90, temperature=1.0,
        seed_alias=f"{SEED_DOMAIN}/M1/paired", representation_seed_alias=None,
    )
    for track in ("RSET", "RREL"):
        for coordinate, templates in REP_TEACHERS.items():
            for template in templates:
                teacher = template.format(track=track)
                carrier = REPRESENTATION_CARRIERS[teacher]
                node = _specialist(
                    track=track, coordinate=coordinate, teacher=teacher,
                    carrier=carrier,
                )
                nodes[node.node_id] = node
        nodes[f"DX_M1_{track}"] = DenseNode(
            node_id=f"DX_M1_{track}", track=track, coordinate_name="D000",
            distribution_teacher_id=f"DX_{track}_D000E",
            distribution_teacher_kind="dense_probability_bank",
            representation_carrier_id=REPRESENTATION_CARRIERS[f"DX_{track}_D000E"],
            auxiliary=track.lower(), ce_weight=.10, kd_weight=.90,
            temperature=1.0, seed_alias=f"{SEED_DOMAIN}/M1/paired",
            representation_seed_alias=f"{SEED_DOMAIN}/{track.lower()}/M1/representation",
        )
    nodes["DX_M2"] = DenseNode(
        node_id="DX_M2", track="TERMINAL", coordinate_name="D000",
        distribution_teacher_id="DX_M1E",
        distribution_teacher_kind="dense_probability_bank",
        representation_carrier_id=None, auxiliary="none", ce_weight=.10,
        kd_weight=.90, temperature=1.0,
        seed_alias=f"{SEED_DOMAIN}/M2", representation_seed_alias=None,
    )
    if len(nodes) != 48:
        raise RuntimeError(f"dense extension must contain 48 fits, found {len(nodes)}")
    return MappingProxyType(dict(nodes))


NODE_REGISTRY: Final = _build_nodes()


def _new_components(track: str, coordinate: str) -> tuple[str, ...]:
    teachers = LOGIT_TEACHERS[coordinate] if track == "LOGIT" else tuple(
        item.format(track=track) for item in REP_TEACHERS[coordinate]
    )
    return tuple(f"DX_{track}_{coordinate}_from_{_suffix(track, teacher)}" for teacher in teachers)


ENSEMBLE_COMPONENTS: Final = MappingProxyType({
    "DX_LOGIT_D083E": _new_components("LOGIT", "D083"),
    "DX_LOGIT_D066E": (
        "LOGIT_D066_from_U000", "LOGIT_D066_from_U050E",
        "LOGIT_D066_from_U100E", *_new_components("LOGIT", "D066"),
    ),
    "DX_LOGIT_D050E": _new_components("LOGIT", "D050"),
    "DX_LOGIT_D033E": (
        "LOGIT_D033_from_U000", "LOGIT_D033_from_U050E",
        "LOGIT_D033_from_U100E", *_new_components("LOGIT", "D033"),
    ),
    "DX_LOGIT_D017E": _new_components("LOGIT", "D017"),
    "DX_LOGIT_D000E": (
        "LOGIT_D000_from_U000", "LOGIT_D000_from_U050E",
        "LOGIT_D000_from_U100E", *_new_components("LOGIT", "D000"),
    ),
    **{
        f"DX_{track}_D075E": _new_components(track, "D075")
        for track in ("RSET", "RREL")
    },
    **{
        f"DX_{track}_D050E": (
            f"{track}_D050_from_U000", f"{track}_D050_from_U100E",
            *_new_components(track, "D050"),
        ) for track in ("RSET", "RREL")
    },
    **{
        f"DX_{track}_D025E": _new_components(track, "D025")
        for track in ("RSET", "RREL")
    },
    **{
        f"DX_{track}_D000E": (
            f"{track}_D000_from_U000", f"{track}_D000_from_U100E",
            *_new_components(track, "D000"),
        ) for track in ("RSET", "RREL")
    },
    "DX_M1E": ("DX_M1_LOGIT", "DX_M1_RSET", "DX_M1_RREL"),
})

FIT_ORDER: Final = tuple(NODE_REGISTRY)
REDUCER_ORDER: Final = tuple(ENSEMBLE_COMPONENTS)


def distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    consumers = tuple(
        node_id for node_id in FIT_ORDER
        if NODE_REGISTRY[node_id].distribution_teacher_id == distribution_id
    )
    if not consumers:
        raise KeyError(f"dense distribution has no consumer: {distribution_id}")
    return consumers


def source_distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    if distribution_id not in SOURCE_DISTRIBUTIONS:
        raise KeyError(f"unknown dense source distribution: {distribution_id}")
    consumers = tuple(
        node_id for node_id in FIT_ORDER
        if NODE_REGISTRY[node_id].distribution_teacher_id == distribution_id
    )
    if not consumers:
        raise KeyError(f"dense source distribution has no consumer: {distribution_id}")
    return consumers


def component_origin(node_id: str) -> str:
    if node_id in NODE_REGISTRY:
        return "dense"
    if node_id in SOURCE_NODES:
        return "source"
    raise KeyError(f"unknown dense component: {node_id}")


_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT, "schema_version": 1,
    "campaign_label": CAMPAIGN_LABEL, "source_graph_sha256": SOURCE_GRAPH_SHA256,
    "coordinates": {name: value.payload() for name, value in COORDINATES.items()},
    "nodes": [NODE_REGISTRY[name].payload() for name in FIT_ORDER],
    "source_distributions": list(SOURCE_DISTRIBUTIONS),
    "early_source_nodes": list(EARLY_SOURCE_NODES),
    "late_source_nodes": list(LATE_SOURCE_NODES),
    "reducers": {name: list(value) for name, value in ENSEMBLE_COMPONENTS.items()},
    "representation_carriers": dict(REPRESENTATION_CARRIERS),
    "fit_order": list(FIT_ORDER), "reducer_order": list(REDUCER_ORDER),
    "uniform_probability_ensembles": True,
    "terminal": {"ensemble": "DX_M1E", "student": "DX_M2"},
    "source_campaign_outputs_mutated": False, "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    value = with_content_hash(_GRAPH_BODY)
    if value["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("dense graph hash differs")
    return value


def validate_graph() -> str:
    if len(NODE_REGISTRY) != 48 or len(ENSEMBLE_COMPONENTS) != 15:
        raise ValueError("dense graph counts differ")
    for distribution, components in ENSEMBLE_COMPONENTS.items():
        if not components or len(set(components)) != len(components):
            raise ValueError(f"dense ensemble components differ: {distribution}")
        for component in components:
            component_origin(component)
    for node in NODE_REGISTRY.values():
        if node.training_passes != 60 or node.batch_size != 256:
            raise ValueError("dense node budget differs")
        if node.auxiliary == "none" and node.representation_carrier_id is not None:
            raise ValueError("dense LOGIT node has representation carrier")
        if node.auxiliary != "none" and node.representation_carrier_id is None:
            raise ValueError("dense representation node lacks carrier")
    return GRAPH_SHA256


__all__ = [
    "CAMPAIGN_LABEL", "COORDINATES", "DenseNode", "EARLY_SOURCE_NODES",
    "ENSEMBLE_COMPONENTS", "FIT_ORDER", "GRAPH_SHA256", "LATE_SOURCE_NODES",
    "NODE_REGISTRY", "REDUCER_ORDER", "REPRESENTATION_CARRIERS",
    "SOURCE_DISTRIBUTIONS", "SOURCE_GRAPH_SHA256", "SOURCE_NODES",
    "component_origin", "distribution_consumers", "graph_payload",
    "source_distribution_consumers", "validate_graph",
]
