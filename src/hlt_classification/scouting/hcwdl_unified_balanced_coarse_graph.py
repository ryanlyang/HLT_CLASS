"""Exact coarse factorized/joint full-data HCWDL-UB registry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_unified_balanced_graph import UnifiedBalancedNodeSpec


CAMPAIGN_LABEL: Final = "HCWDL-UB-FULLCOARSE3"
SHARED_ARM: Final = "shared"
ARM_IDS: Final = ("C25P75", "C10P90", "C10P75G15")
ARM_WEIGHTS: Final = MappingProxyType({
    "C25P75": (.25, .75, 0.0),
    "C10P90": (.10, .90, 0.0),
    "C10P75G15": (.10, .75, .15),
})

# The labels are rounded display names. Coordinates below are exact rationals.
FACTORIZED_NODES: Final = (
    "U033", "U067", "U100", "D67F", "D33F", "D0F",
)
JOINT_NODES: Final = (
    "J017", "J033", "J050", "J067", "J083", "J100",
)
PATHS: Final = MappingProxyType({
    "factorized": FACTORIZED_NODES,
    "joint": JOINT_NODES,
})

_COORDINATES: Final = MappingProxyType({
    "U033": HomotopyCoordinate(1, 3, 0, 1),
    "U067": HomotopyCoordinate(2, 3, 0, 1),
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D67F": HomotopyCoordinate(1, 1, 1, 3),
    "D33F": HomotopyCoordinate(1, 1, 2, 3),
    "D0F": HomotopyCoordinate(1, 1, 1, 1),
    "J017": HomotopyCoordinate(1, 6, 1, 6),
    "J033": HomotopyCoordinate(1, 3, 1, 3),
    "J050": HomotopyCoordinate(1, 2, 1, 2),
    "J067": HomotopyCoordinate(2, 3, 2, 3),
    "J083": HomotopyCoordinate(5, 6, 5, 6),
    "J100": HomotopyCoordinate(1, 1, 1, 1),
})


def _node(
    arm_id: str, node_id: str, *, parent_id: str,
    grandparent_id: str | None, transition: int,
) -> UnifiedBalancedNodeSpec:
    ce, parent, grandparent = ARM_WEIGHTS[arm_id]
    if grandparent and grandparent_id is None:
        parent += grandparent
        grandparent = 0.0
    coordinate = _COORDINATES[node_id]
    exact_hlt = (
        coordinate.structural_numerator == coordinate.structural_denominator
        and coordinate.feature_numerator == coordinate.feature_denominator
    )
    return UnifiedBalancedNodeSpec(
        arm_id=arm_id,
        node_id=node_id,
        input_domain="hlt" if exact_hlt else "homotopy",
        coordinate=coordinate,
        parent_id=parent_id,
        grandparent_id=grandparent_id,
        ce_weight=ce,
        parent_kd_weight=parent,
        grandparent_kd_weight=grandparent,
        parent_temperature=2.0,
        grandparent_temperature=2.0,
        # Same transition number is paired across path geometry and recipe.
        seed_alias=f"HCWDL-UB-FULLCOARSE3/v1/transition_{transition:02d}/paired",
        behavior="balanced_uniform",
    )


def arm_registry(arm_id: str) -> Mapping[str, UnifiedBalancedNodeSpec]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB-FULLCOARSE3 arm")
    result: dict[str, UnifiedBalancedNodeSpec] = {}
    for nodes in PATHS.values():
        history = ["shared/U000"]
        for transition, node_id in enumerate(nodes, start=1):
            node = _node(
                arm_id,
                node_id,
                parent_id=history[-1],
                grandparent_id=history[-2] if len(history) >= 2 else None,
                transition=transition,
            )
            result[node_id] = node
            history.append(node.canonical_id)
    validate_arm_registry(arm_id, result)
    return MappingProxyType(result)


def validate_arm_registry(
    arm_id: str, registry: Mapping[str, UnifiedBalancedNodeSpec],
) -> None:
    expected = set(FACTORIZED_NODES) | set(JOINT_NODES)
    if arm_id not in ARM_IDS or set(registry) != expected or len(registry) != 12:
        raise ValueError("HCWDL-UB-FULLCOARSE3 arm registry differs")
    for node_id, node in registry.items():
        if node.canonical_id != f"{arm_id}/{node_id}":
            raise ValueError("HCWDL-UB-FULLCOARSE3 node ownership differs")
        if any(
            teacher.split("/", 1)[0] not in {SHARED_ARM, arm_id}
            for teacher in node.teachers
        ):
            raise PermissionError("HCWDL-UB-FULLCOARSE3 teacher crosses arms")
        if abs(
            node.ce_weight + node.parent_kd_weight + node.grandparent_kd_weight - 1
        ) > 1e-12:
            raise ValueError("HCWDL-UB-FULLCOARSE3 loss weights differ")
        if node.grandparent_kd_weight and node.grandparent_id is None:
            raise ValueError("HCWDL-UB-FULLCOARSE3 grandparent is unavailable")


def meta_registry() -> Mapping[str, UnifiedBalancedNodeSpec]:
    result = {
        f"{arm_id}/{node_id}": node
        for arm_id in ARM_IDS
        for node_id, node in arm_registry(arm_id).items()
    }
    if len(result) != 36:
        raise RuntimeError("HCWDL-UB-FULLCOARSE3 registry is not 36 fits")
    return MappingProxyType(dict(sorted(result.items())))


def training_registry_for_arm(arm_id: str):
    from .hcwdl_ladder import NodeSpec, TeacherSpec

    registry = arm_registry(arm_id)
    result = {}
    for node_id, node in registry.items():
        teachers = []
        for teacher_id, kind, weight in (
            (node.parent_id, "parent", node.parent_kd_weight),
            (node.grandparent_id, "grandparent", node.grandparent_kd_weight),
        ):
            if teacher_id is None or weight <= 0:
                continue
            if teacher_id == "shared/U000":
                domain = "privileged"
            else:
                owner, local_id = teacher_id.split("/", 1)
                if owner != arm_id:
                    raise PermissionError("HCWDL-UB-FULLCOARSE3 teacher crosses arms")
                domain = (
                    "hlt" if registry[local_id].input_domain == "hlt" else "privileged"
                )
            teachers.append(TeacherSpec(teacher_id, domain, kind))
        result[node_id] = NodeSpec(
            node_id=node_id,
            track="coarse_factorized" if node_id in FACTORIZED_NODES else "coarse_joint",
            stage="child",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh",
            initialization_parent=None,
            teachers=tuple(teachers),
            loss_kind="ce_kd" if len(teachers) == 1 else "ce_two_kd",
            deployable=node.input_domain == "hlt",
        )
    return MappingProxyType(result)


def idealized_u000_ancestry(arm_id: str) -> Mapping[str, float]:
    values = {"shared/U000": 1.0}
    result: dict[str, float] = {}
    for nodes in PATHS.values():
        for node_id in nodes:
            node = arm_registry(arm_id)[node_id]
            parent = values[node.parent_id]
            grandparent = 0.0 if node.grandparent_id is None else values[node.grandparent_id]
            value = (
                node.parent_kd_weight * parent
                + node.grandparent_kd_weight * grandparent
            )
            values[node.canonical_id] = value
            result[node_id] = value
    return MappingProxyType(result)


META_REGISTRY: Final = meta_registry()
META_GRAPH_SHA256: Final = canonical_sha256({
    "contract": "HCWDL_UNIFIED_BALANCED_FULL_COARSE_GRAPH/v1",
    "imported_anchors": ["shared/U000", "shared/M0paired"],
    "nodes": [META_REGISTRY[key].payload() for key in sorted(META_REGISTRY)],
})


__all__ = [
    "ARM_IDS", "ARM_WEIGHTS", "CAMPAIGN_LABEL", "FACTORIZED_NODES",
    "JOINT_NODES", "META_GRAPH_SHA256", "META_REGISTRY", "PATHS",
    "SHARED_ARM", "arm_registry", "idealized_u000_ancestry",
    "meta_registry", "training_registry_for_arm", "validate_arm_registry",
]
