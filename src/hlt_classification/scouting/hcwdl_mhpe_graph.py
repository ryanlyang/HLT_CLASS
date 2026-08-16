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
GRAPH_CONTRACT_C25P75_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v3"
NODE_CONTRACT_C25P75_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v3"
CAMPAIGN_LABEL_C25P75_300K60: Final = "HCWDL-MHPE-C25P75-300K60"
GRAPH_CONTRACT_C10P90_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v4"
NODE_CONTRACT_C10P90_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v4"
CAMPAIGN_LABEL_C10P90_300K60: Final = "HCWDL-MHPE-C10P90-300K60"
GRAPH_CONTRACT_DENSE_ANCHOR50_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v5"
NODE_CONTRACT_DENSE_ANCHOR50_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v5"
CAMPAIGN_LABEL_DENSE_ANCHOR50_300K60: Final = "HCWDL-MHPE-DENSE-ANCHOR50-300K60"
PROFILE_C25P75: Final = "C25P75"
PROFILE_C10P90: Final = "C10P90"
PROFILE_C25P75_300K60: Final = "C25P75_300K60"
PROFILE_C10P90_300K60: Final = "C10P90_300K60"
PROFILE_DENSE_ANCHOR50_300K60: Final = "C10P90_DENSE_ANCHOR50_300K60"
SUPPORTED_PROFILES: Final = (
    PROFILE_C25P75, PROFILE_C10P90,
    PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
    PROFILE_DENSE_ANCHOR50_300K60,
)

COORDINATES: Final = MappingProxyType({
    "U000": HomotopyCoordinate(0, 1, 0, 1),
    "U050": HomotopyCoordinate(1, 2, 0, 1),
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D066": HomotopyCoordinate(1, 1, 1, 3),
    "D033": HomotopyCoordinate(1, 1, 2, 3),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
    "U033": HomotopyCoordinate(1, 3, 0, 1),
    "U066": HomotopyCoordinate(2, 3, 0, 1),
    "D75": HomotopyCoordinate(1, 1, 1, 4),
    "D50": HomotopyCoordinate(1, 1, 1, 2),
    "D25": HomotopyCoordinate(1, 1, 3, 4),
    "D0": HomotopyCoordinate(1, 1, 1, 1),
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

DENSE_STAGE_TEACHERS: Final = MappingProxyType({
    "U033": ("U000",),
    "U066": ("U000", "U033"),
    "U100": ("U000", "U033", "U066E"),
    "D75": ("U000", "U033", "U066E", "U100E"),
    "D50": ("U000", "U033", "U066E", "U100E", "D75E"),
    "D25": ("U000", "U033", "U066E", "U100E", "D75E", "D50E"),
    "D0": ("U000", "U033", "U066E", "U100E", "D75E", "D50E", "D25E"),
})
DENSE_ENSEMBLE_COMPONENTS: Final = MappingProxyType({
    f"{stage}E": tuple(sorted(f"{stage}_from_{teacher}" for teacher in teachers))
    for stage, teachers in DENSE_STAGE_TEACHERS.items() if stage != "U033"
})
DENSE_LOCAL_TEACHERS: Final = MappingProxyType({
    "U066E": "U033", "U100E": "U066E", "D75E": "U100E",
    "D50E": "D75E", "D25E": "D50E", "D0E": "D25E",
})


def stage_teachers(profile: str = PROFILE_C25P75) -> Mapping[str, tuple[str, ...]]:
    return DENSE_STAGE_TEACHERS if profile == PROFILE_DENSE_ANCHOR50_300K60 else STAGE_TEACHERS


def ensemble_components(profile: str = PROFILE_C25P75) -> Mapping[str, tuple[str, ...]]:
    return (
        DENSE_ENSEMBLE_COMPONENTS
        if profile == PROFILE_DENSE_ANCHOR50_300K60 else ENSEMBLE_COMPONENTS
    )


def local_teacher(profile: str, ensemble_id: str) -> str:
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        try:
            return DENSE_LOCAL_TEACHERS[ensemble_id]
        except KeyError as error:
            raise ValueError("unknown dense HCWDL-MHPE ensemble") from error
    legacy = {
        "U100E": "U050", "D066E": "U100E",
        "D033E": "D066E", "D000E": "D033E",
    }
    try:
        return legacy[ensemble_id]
    except KeyError as error:
        raise ValueError("unknown HCWDL-MHPE ensemble") from error


def ensemble_weight_rationals(
    profile: str, ensemble_id: str,
) -> dict[str, list[int]]:
    components = ensemble_components(profile).get(ensemble_id)
    if components is None:
        raise ValueError("unknown HCWDL-MHPE ensemble")
    if profile != PROFILE_DENSE_ANCHOR50_300K60:
        return {name: [1, len(components)] for name in components}
    local = f"{ensemble_id[:-1]}_from_{local_teacher(profile, ensemble_id)}"
    if local not in components or len(components) < 2:
        raise ValueError("dense anchor-50 local component differs")
    return {
        name: ([1, 2] if name == local else [1, 2 * (len(components) - 1)])
        for name in components
    }


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
    training_passes: int = 20

    @property
    def coordinate(self) -> HomotopyCoordinate:
        return COORDINATES[self.coordinate_name]

    @property
    def input_domain(self) -> str:
        coordinate = self.coordinate
        return "hlt" if (
            coordinate.feature_numerator == coordinate.feature_denominator
            or self.node_id == "M1"
        ) else "homotopy"

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
            "training_passes": self.training_passes,
        }


def _build_registry(profile: str = PROFILE_C25P75) -> Mapping[str, MhpeNode]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError("unknown HCWDL-MHPE recipe profile")
    specialist_weights = {
        PROFILE_C25P75: (.25, .75),
        PROFILE_C10P90: (.10, .90),
        PROFILE_C25P75_300K60: (.25, .75),
        PROFILE_C10P90_300K60: (.10, .90),
        PROFILE_DENSE_ANCHOR50_300K60: (.10, .90),
    }[profile]
    node_contract = {
        PROFILE_C25P75: NODE_CONTRACT,
        PROFILE_C10P90: NODE_CONTRACT_C10P90,
        PROFILE_C25P75_300K60: NODE_CONTRACT_C25P75_300K60,
        PROFILE_C10P90_300K60: NODE_CONTRACT_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60: NODE_CONTRACT_DENSE_ANCHOR50_300K60,
    }[profile]
    passes = 60 if profile in {
        PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60,
    } else 20
    seed_domain = (
        "HCWDL-MHPE-DENSE-ANCHOR50-300K60/v1"
        if profile == PROFILE_DENSE_ANCHOR50_300K60 else
        "HCWDL-MHPE-300K60/v1" if passes == 60 else "HCWDL-MHPE-FULL/v1"
    )
    nodes: dict[str, MhpeNode] = {}
    for stage, teachers in stage_teachers(profile).items():
        for teacher in teachers:
            node_id = f"{stage}_from_{teacher}"
            nodes[node_id] = MhpeNode(
                node_id=node_id, coordinate_name=stage, teacher_id=teacher,
                seed_alias=f"{seed_domain}/{stage}/paired",
                ce_weight=specialist_weights[0], kd_weight=specialist_weights[1],
                temperature=2.0, contract=node_contract,
                teacher_kind="probabilities" if teacher.endswith("E") else "logits",
                training_passes=passes,
            )
    endpoint = "D0" if profile == PROFILE_DENSE_ANCHOR50_300K60 else "D000"
    nodes["M1"] = MhpeNode(
        node_id="M1", coordinate_name=endpoint, teacher_id=f"{endpoint}E",
        seed_alias=f"{seed_domain}/M1", ce_weight=.10, kd_weight=.90,
        temperature=1.0, teacher_kind="probabilities", contract=node_contract,
        training_passes=passes,
    )
    expected_fits = 29 if profile == PROFILE_DENSE_ANCHOR50_300K60 else 16
    if len(nodes) != expected_fits:
        raise RuntimeError(f"HCWDL-MHPE graph must contain exactly {expected_fits} fresh fits")
    return MappingProxyType(dict(sorted(nodes.items())))


NODE_REGISTRY: Final = _build_registry(PROFILE_C25P75)
C10P90_NODE_REGISTRY: Final = _build_registry(PROFILE_C10P90)
C25P75_300K60_NODE_REGISTRY: Final = _build_registry(PROFILE_C25P75_300K60)
C10P90_300K60_NODE_REGISTRY: Final = _build_registry(PROFILE_C10P90_300K60)
DENSE_ANCHOR50_300K60_NODE_REGISTRY: Final = _build_registry(
    PROFILE_DENSE_ANCHOR50_300K60,
)
STAGES: Final = ("U050", "U100", "D066", "D033", "D000", "M1")
FINALISTS: Final = (
    "M0paired",
    "D000_from_U000", "D000_from_U050", "D000_from_U100E",
    "D000_from_D066E", "D000_from_D033E", "D000E", "M1",
)
DENSE_FINALISTS: Final = (
    "M0paired",
    "D0_from_U000", "D0_from_U033", "D0_from_U066E",
    "D0_from_U100E", "D0_from_D75E", "D0_from_D50E",
    "D0_from_D25E", "D0E", "M1",
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
C25P75_300K60_GRAPH_SHA256: Final = canonical_sha256({
    "contract": GRAPH_CONTRACT_C25P75_300K60,
    "recipe_profile": PROFILE_C25P75_300K60,
    "population_profile": "pilot_300k_60pass",
    "imported": ["U000", "M0paired"],
    "nodes": [C25P75_300K60_NODE_REGISTRY[key].payload()
              for key in C25P75_300K60_NODE_REGISTRY],
    "ensemble_components": dict(ENSEMBLE_COMPONENTS),
    "finalists": list(FINALISTS),
})
C10P90_300K60_GRAPH_SHA256: Final = canonical_sha256({
    "contract": GRAPH_CONTRACT_C10P90_300K60,
    "recipe_profile": PROFILE_C10P90_300K60,
    "population_profile": "pilot_300k_60pass",
    "imported": ["U000", "M0paired"],
    "nodes": [C10P90_300K60_NODE_REGISTRY[key].payload()
              for key in C10P90_300K60_NODE_REGISTRY],
    "ensemble_components": dict(ENSEMBLE_COMPONENTS),
    "finalists": list(FINALISTS),
})
DENSE_ANCHOR50_300K60_GRAPH_SHA256: Final = canonical_sha256({
    "contract": GRAPH_CONTRACT_DENSE_ANCHOR50_300K60,
    "recipe_profile": PROFILE_DENSE_ANCHOR50_300K60,
    "population_profile": "pilot_300k_60pass",
    "ensemble_policy": "local_predecessor_half_skip_half_exact_rational_v1",
    "imported": ["U000", "M0paired"],
    "nodes": [DENSE_ANCHOR50_300K60_NODE_REGISTRY[key].payload()
              for key in DENSE_ANCHOR50_300K60_NODE_REGISTRY],
    "ensemble_components": dict(DENSE_ENSEMBLE_COMPONENTS),
    "ensemble_weights": {
        key: ensemble_weight_rationals(PROFILE_DENSE_ANCHOR50_300K60, key)
        for key in DENSE_ENSEMBLE_COMPONENTS
    },
    "finalists": list(DENSE_FINALISTS),
})


def node_registry(profile: str = PROFILE_C25P75) -> Mapping[str, MhpeNode]:
    if profile == PROFILE_C25P75:
        return NODE_REGISTRY
    if profile == PROFILE_C10P90:
        return C10P90_NODE_REGISTRY
    if profile == PROFILE_C25P75_300K60:
        return C25P75_300K60_NODE_REGISTRY
    if profile == PROFILE_C10P90_300K60:
        return C10P90_300K60_NODE_REGISTRY
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        return DENSE_ANCHOR50_300K60_NODE_REGISTRY
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def graph_sha256(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return GRAPH_SHA256
    if profile == PROFILE_C10P90:
        return C10P90_GRAPH_SHA256
    if profile == PROFILE_C25P75_300K60:
        return C25P75_300K60_GRAPH_SHA256
    if profile == PROFILE_C10P90_300K60:
        return C10P90_300K60_GRAPH_SHA256
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        return DENSE_ANCHOR50_300K60_GRAPH_SHA256
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def graph_contract(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return GRAPH_CONTRACT
    if profile == PROFILE_C10P90:
        return GRAPH_CONTRACT_C10P90
    if profile == PROFILE_C25P75_300K60:
        return GRAPH_CONTRACT_C25P75_300K60
    if profile == PROFILE_C10P90_300K60:
        return GRAPH_CONTRACT_C10P90_300K60
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        return GRAPH_CONTRACT_DENSE_ANCHOR50_300K60
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def campaign_label(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return CAMPAIGN_LABEL
    if profile == PROFILE_C10P90:
        return CAMPAIGN_LABEL_C10P90
    if profile == PROFILE_C25P75_300K60:
        return CAMPAIGN_LABEL_C25P75_300K60
    if profile == PROFILE_C10P90_300K60:
        return CAMPAIGN_LABEL_C10P90_300K60
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        return CAMPAIGN_LABEL_DENSE_ANCHOR50_300K60
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def validate_graph(profile: str = PROFILE_C25P75) -> str:
    registry = node_registry(profile)
    components_by_ensemble = ensemble_components(profile)
    expected_ensembles = (
        {"U066E", "U100E", "D75E", "D50E", "D25E", "D0E"}
        if profile == PROFILE_DENSE_ANCHOR50_300K60
        else {"U100E", "D066E", "D033E", "D000E"}
    )
    if set(components_by_ensemble) != expected_ensembles:
        raise ValueError("HCWDL-MHPE ensemble registry differs")
    for ensemble, components in components_by_ensemble.items():
        if tuple(sorted(components)) != components:
            raise ValueError("HCWDL-MHPE components are not lexical")
        if any(component not in registry for component in components):
            raise ValueError("HCWDL-MHPE component is absent")
    return graph_sha256(profile)


def stages(profile: str = PROFILE_C25P75) -> tuple[str, ...]:
    return tuple(stage_teachers(profile)) + ("M1",)


def finalists(profile: str = PROFILE_C25P75) -> tuple[str, ...]:
    return DENSE_FINALISTS if profile == PROFILE_DENSE_ANCHOR50_300K60 else FINALISTS


def endpoint_ensemble(profile: str = PROFILE_C25P75) -> str:
    return "D0E" if profile == PROFILE_DENSE_ANCHOR50_300K60 else "D000E"


def direct_model_teacher(profile: str = PROFILE_C25P75) -> str:
    return "U033" if profile == PROFILE_DENSE_ANCHOR50_300K60 else "U050"


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
    "CAMPAIGN_LABEL", "CAMPAIGN_LABEL_C10P90",
    "CAMPAIGN_LABEL_C25P75_300K60", "CAMPAIGN_LABEL_C10P90_300K60",
    "CAMPAIGN_LABEL_DENSE_ANCHOR50_300K60",
    "C10P90_GRAPH_SHA256", "C25P75_300K60_GRAPH_SHA256",
    "C10P90_300K60_GRAPH_SHA256",
    "DENSE_ANCHOR50_300K60_GRAPH_SHA256",
    "C10P90_NODE_REGISTRY", "C25P75_300K60_NODE_REGISTRY",
    "C10P90_300K60_NODE_REGISTRY", "COORDINATES", "ENSEMBLE_COMPONENTS", "FINALISTS",
    "DENSE_ANCHOR50_300K60_NODE_REGISTRY", "DENSE_ENSEMBLE_COMPONENTS",
    "DENSE_FINALISTS",
    "GRAPH_CONTRACT", "GRAPH_CONTRACT_C10P90",
    "GRAPH_CONTRACT_C25P75_300K60", "GRAPH_CONTRACT_C10P90_300K60",
    "GRAPH_CONTRACT_DENSE_ANCHOR50_300K60",
    "GRAPH_SHA256", "MhpeNode", "NODE_CONTRACT", "NODE_CONTRACT_C10P90",
    "NODE_CONTRACT_C25P75_300K60", "NODE_CONTRACT_C10P90_300K60",
    "NODE_CONTRACT_DENSE_ANCHOR50_300K60",
    "NODE_REGISTRY",
    "PROFILE_C10P90", "PROFILE_C25P75", "PROFILE_C25P75_300K60",
    "PROFILE_C10P90_300K60", "STAGES", "STAGE_TEACHERS",
    "PROFILE_DENSE_ANCHOR50_300K60",
    "SUPPORTED_PROFILES", "campaign_label", "graph_contract", "graph_sha256",
    "direct_model_teacher", "endpoint_ensemble", "ensemble_components",
    "ensemble_weight_rationals", "finalists", "local_teacher", "node_registry",
    "stage_teachers", "stages", "validate_graph", "training_registry",
]
