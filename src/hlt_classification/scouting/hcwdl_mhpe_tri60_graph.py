"""Immutable graph for the full-data, three-track, 60-pass MHPE campaign."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_homotopy import HomotopyCoordinate


GRAPH_CONTRACT: Final = "HCWDL_MHPE_THREE_TRACK_60E_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_MHPE_THREE_TRACK_60E_NODE_SPEC/v1"
CAMPAIGN_LABEL: Final = "HCWDL-MHPE-THREE-TRACK-60E-FULL"
SEED_DOMAIN: Final = "HCWDL-MHPE-THREE-TRACK-60E-FULL/v1"
REPRESENTATION_SOURCE_COMMIT: Final = (
    "acecf9f74dab3d4ac675d8160cfb5decf83ba680"
)

COORDINATES: Final = MappingProxyType({
    "U000": HomotopyCoordinate(0, 1, 0, 1),
    "U050": HomotopyCoordinate(1, 2, 0, 1),
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D066": HomotopyCoordinate(1, 1, 1, 3),
    "D050": HomotopyCoordinate(1, 1, 1, 2),
    "D033": HomotopyCoordinate(1, 1, 2, 3),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
})

LOGIT_STAGES: Final = MappingProxyType({
    "U050": ("U000",),
    "U100": ("U000", "LOGIT_U050E"),
    "D066": ("U000", "LOGIT_U050E", "LOGIT_U100E"),
    "D033": ("U000", "LOGIT_U050E", "LOGIT_U100E", "LOGIT_D066E"),
    "D000": (
        "U000", "LOGIT_U050E", "LOGIT_U100E",
        "LOGIT_D066E", "LOGIT_D033E",
    ),
})
REP_STAGES: Final = MappingProxyType({
    "U100": ("U000",),
    "D050": ("U000", "{track}_U100E"),
    "D000": ("U000", "{track}_U100E", "{track}_D050E"),
})
REPRESENTATION_CARRIERS: Final = MappingProxyType({
    "U000": "U000",
    "RSET_U100E": "RSET_U100_from_U000",
    "RSET_D050E": "RSET_D050_from_U100E",
    "RSET_D000E": "RSET_D000_from_D050E",
    "RREL_U100E": "RREL_U100_from_U000",
    "RREL_D050E": "RREL_D050_from_U100E",
    "RREL_D000E": "RREL_D000_from_D050E",
})


@dataclass(frozen=True)
class Tri60Node:
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
            "contract": self.node_contract,
            "node_id": self.node_id,
            "track": self.track,
            "coordinate_name": self.coordinate_name,
            "coordinate_exact": self.coordinate.payload(),
            "distribution_teacher_id": self.distribution_teacher_id,
            "distribution_teacher_kind": self.distribution_teacher_kind,
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


def _specialist(
    *, node_id: str, track: str, coordinate: str, teacher: str,
    carrier: str | None = None,
) -> Tri60Node:
    auxiliary = track.lower() if track in {"RSET", "RREL"} else "none"
    if auxiliary != "none" and carrier is None:
        raise ValueError("representation specialist requires an explicit carrier")
    return Tri60Node(
        node_id=node_id, track=track, coordinate_name=coordinate,
        distribution_teacher_id=teacher,
        distribution_teacher_kind=(
            "checkpoint" if teacher == "U000" else "probability_bank"
        ),
        representation_carrier_id=carrier, auxiliary=auxiliary,
        ce_weight=.25, kd_weight=.75, temperature=2.0,
        seed_alias=f"{SEED_DOMAIN}/view/{coordinate}/paired",
        representation_seed_alias=(
            None if auxiliary == "none"
            else f"{SEED_DOMAIN}/{auxiliary}/{coordinate}/representation"
        ),
    )


def _teacher_suffix(track: str, teacher: str) -> str:
    prefix = f"{track}_"
    return teacher[len(prefix):] if teacher.startswith(prefix) else teacher


def _build_registry() -> Mapping[str, Tri60Node]:
    nodes: dict[str, Tri60Node] = {
        "U000": Tri60Node(
            node_id="U000", track="ROOT", coordinate_name="U000",
            distribution_teacher_id=None, distribution_teacher_kind="none",
            representation_carrier_id=None, auxiliary="none",
            ce_weight=1.0, kd_weight=0.0, temperature=1.0,
            seed_alias=f"{SEED_DOMAIN}/U000", representation_seed_alias=None,
        ),
    }
    for coordinate, teachers in LOGIT_STAGES.items():
        for teacher in teachers:
            node_id = f"LOGIT_{coordinate}_from_{_teacher_suffix('LOGIT', teacher)}"
            nodes[node_id] = _specialist(
                node_id=node_id, track="LOGIT", coordinate=coordinate,
                teacher=teacher,
            )
    nodes["M1_LOGIT"] = Tri60Node(
        node_id="M1_LOGIT", track="LOGIT", coordinate_name="D000",
        distribution_teacher_id="LOGIT_D000E",
        distribution_teacher_kind="probability_bank",
        representation_carrier_id=None, auxiliary="none",
        ce_weight=.10, kd_weight=.90, temperature=1.0,
        seed_alias=f"{SEED_DOMAIN}/M1/paired", representation_seed_alias=None,
    )
    for track in ("RSET", "RREL"):
        for coordinate, templates in REP_STAGES.items():
            teachers = tuple(item.format(track=track) for item in templates)
            for teacher in teachers:
                node_id = f"{track}_{coordinate}_from_{_teacher_suffix(track, teacher)}"
                carrier = REPRESENTATION_CARRIERS[teacher]
                nodes[node_id] = _specialist(
                    node_id=node_id, track=track, coordinate=coordinate,
                    teacher=teacher, carrier=carrier,
                )
        endpoint_teacher = f"{track}_D000E"
        nodes[f"M1_{track}"] = Tri60Node(
            node_id=f"M1_{track}", track=track, coordinate_name="D000",
            distribution_teacher_id=endpoint_teacher,
            distribution_teacher_kind="probability_bank",
            representation_carrier_id=REPRESENTATION_CARRIERS[endpoint_teacher],
            auxiliary=track.lower(), ce_weight=.10, kd_weight=.90,
            temperature=1.0, seed_alias=f"{SEED_DOMAIN}/M1/paired",
            representation_seed_alias=(
                f"{SEED_DOMAIN}/{track.lower()}/M1/representation"
            ),
        )
    nodes["M2"] = Tri60Node(
        node_id="M2", track="TERMINAL", coordinate_name="D000",
        distribution_teacher_id="M1E",
        distribution_teacher_kind="probability_bank",
        representation_carrier_id=None, auxiliary="none",
        ce_weight=.10, kd_weight=.90, temperature=1.0,
        seed_alias=f"{SEED_DOMAIN}/M2", representation_seed_alias=None,
    )
    if len(nodes) != 32:
        raise RuntimeError("HCWDL-MHPE-TRI60 graph must contain exactly 32 fits")
    return MappingProxyType(dict(sorted(nodes.items())))


NODE_REGISTRY: Final = _build_registry()


def _components(track: str, coordinate: str, teachers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{track}_{coordinate}_from_{_teacher_suffix(track, teacher)}"
        for teacher in teachers
    )


ENSEMBLE_COMPONENTS: Final = MappingProxyType({
    **{
        f"LOGIT_{coordinate}E": _components("LOGIT", coordinate, teachers)
        for coordinate, teachers in LOGIT_STAGES.items()
    },
    **{
        f"{track}_{coordinate}E": _components(
            track, coordinate,
            tuple(item.format(track=track) for item in templates),
        )
        for track in ("RSET", "RREL")
        for coordinate, templates in REP_STAGES.items()
    },
    "M1E": ("M1_LOGIT", "M1_RSET", "M1_RREL"),
})
REDUCER_ORDER: Final = (
    "LOGIT_U050E", "LOGIT_U100E", "LOGIT_D066E",
    "LOGIT_D033E", "LOGIT_D000E",
    "RSET_U100E", "RSET_D050E", "RSET_D000E",
    "RREL_U100E", "RREL_D050E", "RREL_D000E", "M1E",
)
FIT_ORDER: Final = (
    "U000",
    "LOGIT_U050_from_U000",
    "LOGIT_U100_from_U000", "LOGIT_U100_from_U050E",
    "LOGIT_D066_from_U000", "LOGIT_D066_from_U050E",
    "LOGIT_D066_from_U100E",
    "LOGIT_D033_from_U000", "LOGIT_D033_from_U050E",
    "LOGIT_D033_from_U100E", "LOGIT_D033_from_D066E",
    "LOGIT_D000_from_U000", "LOGIT_D000_from_U050E",
    "LOGIT_D000_from_U100E", "LOGIT_D000_from_D066E",
    "LOGIT_D000_from_D033E", "M1_LOGIT",
    "RSET_U100_from_U000",
    "RSET_D050_from_U000", "RSET_D050_from_U100E",
    "RSET_D000_from_U000", "RSET_D000_from_U100E",
    "RSET_D000_from_D050E", "M1_RSET",
    "RREL_U100_from_U000",
    "RREL_D050_from_U000", "RREL_D050_from_U100E",
    "RREL_D000_from_U000", "RREL_D000_from_U100E",
    "RREL_D000_from_D050E", "M1_RREL", "M2",
)
if set(FIT_ORDER) != set(NODE_REGISTRY) or len(FIT_ORDER) != 32:
    raise RuntimeError("canonical HCWDL-MHPE-TRI60 fit order differs")
if tuple(ENSEMBLE_COMPONENTS) != REDUCER_ORDER or len(REDUCER_ORDER) != 12:
    raise RuntimeError("canonical HCWDL-MHPE-TRI60 reducer order differs")

_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT,
    "schema_version": 1,
    "campaign_label": CAMPAIGN_LABEL,
    "representation_source_commit": REPRESENTATION_SOURCE_COMMIT,
    "coordinates": {key: value.payload() for key, value in COORDINATES.items()},
    "nodes": [NODE_REGISTRY[key].payload() for key in FIT_ORDER],
    "reducers": {
        key: list(ENSEMBLE_COMPONENTS[key]) for key in REDUCER_ORDER
    },
    "representation_carriers": dict(REPRESENTATION_CARRIERS),
    "fit_order": list(FIT_ORDER), "reducer_order": list(REDUCER_ORDER),
    "finalist": "M2", "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    payload = with_content_hash(_GRAPH_BODY)
    if payload["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("HCWDL-MHPE-TRI60 graph payload hash differs")
    return payload


def validate_graph() -> str:
    if len(NODE_REGISTRY) != 32 or len(ENSEMBLE_COMPONENTS) != 12:
        raise ValueError("HCWDL-MHPE-TRI60 graph counts differ")
    for ensemble_id, components in ENSEMBLE_COMPONENTS.items():
        if not components or any(component not in NODE_REGISTRY for component in components):
            raise ValueError(f"invalid reducer components for {ensemble_id}")
    for node in NODE_REGISTRY.values():
        if node.initialization != "fresh" or node.training_passes != 60:
            raise ValueError("HCWDL-MHPE-TRI60 fit policy differs")
        if node.auxiliary in {"rset", "rrel"} and node.representation_carrier_id is None:
            raise ValueError("representation carrier is absent")
        if node.auxiliary == "none" and node.representation_carrier_id is not None:
            raise ValueError("LOGIT-only node has a representation carrier")
    return GRAPH_SHA256


def distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    consumers = tuple(
        node_id for node_id in FIT_ORDER
        if NODE_REGISTRY[node_id].distribution_teacher_id == distribution_id
    )
    if not consumers:
        raise KeyError(f"TRI60 distribution has no registered consumer: {distribution_id}")
    return consumers


__all__ = [
    "CAMPAIGN_LABEL", "COORDINATES", "ENSEMBLE_COMPONENTS", "FIT_ORDER",
    "GRAPH_CONTRACT", "GRAPH_SHA256", "LOGIT_STAGES", "NODE_CONTRACT",
    "NODE_REGISTRY", "REDUCER_ORDER", "REPRESENTATION_CARRIERS",
    "REPRESENTATION_SOURCE_COMMIT", "REP_STAGES", "SEED_DOMAIN",
    "Tri60Node", "validate_graph",
    "distribution_consumers", "graph_payload",
]
