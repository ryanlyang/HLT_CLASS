"""Exact 38-fit registry for the all-mapped HCWDL-UB three-arm scale-up."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_unified_balanced_graph import UnifiedBalancedNodeSpec
from .hcwdl_homotopy import HomotopyCoordinate


CAMPAIGN_LABEL: Final = "HCWDL-UB-FULL3"
SHARED_ARM: Final = "shared"
ARM_IDS: Final = ("C25P75", "C10P90", "C10P75G15")
ARM_WEIGHTS: Final = MappingProxyType({
    "C25P75": (.25, .75, 0.0),
    "C10P90": (.10, .90, 0.0),
    "C10P75G15": (.10, .75, .15),
})
FACTORIZED_NODES: Final = (
    "U020", "U040", "U060", "U080", "U100",
    "D80F", "D60F", "D40F", "D20F", "D0F", "M1F",
)
DIRECT_NODE: Final = "D100direct"


def _coordinate(node_id: str) -> HomotopyCoordinate:
    if node_id == "U000":
        return HomotopyCoordinate(0, 1, 0, 1)
    if node_id in {"M0paired", "M1F"}:
        return HomotopyCoordinate(1, 1, 1, 1)
    if node_id == DIRECT_NODE:
        return HomotopyCoordinate(1, 1, 0, 1)
    if node_id.startswith("U"):
        return HomotopyCoordinate(int(node_id[1:]), 100, 0, 1)
    if node_id.startswith("D") and node_id.endswith("F"):
        offline = int(node_id[1:-1])
        return HomotopyCoordinate(1, 1, 100 - offline, 100)
    raise KeyError(f"unknown HCWDL-UB-FULL3 node {node_id}")


def shared_registry() -> Mapping[str, UnifiedBalancedNodeSpec]:
    return MappingProxyType({
        "U000": UnifiedBalancedNodeSpec(
            SHARED_ARM, "U000", "p0", _coordinate("U000"), None, None,
            1.0, 0.0, 0.0, 1.0, 1.0,
            "HCWDL-UB-FULL3/v1/shared-root/paired", "balanced_uniform",
        ),
        "M0paired": UnifiedBalancedNodeSpec(
            SHARED_ARM, "M0paired", "hlt", _coordinate("M0paired"), None, None,
            1.0, 0.0, 0.0, 1.0, 1.0,
            "HCWDL-UB-FULL3/v1/shared-root/paired", "balanced_uniform",
        ),
    })


def _node(
    arm_id: str, node_id: str, *, parent_id: str,
    grandparent_id: str | None, seed_node_id: str | None = None,
) -> UnifiedBalancedNodeSpec:
    if node_id == "M1F":
        ce, parent, grandparent, temperature = .25, .75, 0.0, 1.0
    else:
        ce, parent, grandparent = ARM_WEIGHTS[arm_id]
        if grandparent and grandparent_id is None:
            parent += grandparent
            grandparent = 0.0
        temperature = 2.0
    coordinate = _coordinate(node_id)
    exact_hlt = (
        coordinate.structural_numerator == coordinate.structural_denominator
        and coordinate.feature_numerator == coordinate.feature_denominator
    )
    return UnifiedBalancedNodeSpec(
        arm_id=arm_id, node_id=node_id,
        input_domain="hlt" if exact_hlt else "homotopy",
        coordinate=coordinate, parent_id=parent_id,
        grandparent_id=grandparent_id, ce_weight=ce,
        parent_kd_weight=parent, grandparent_kd_weight=grandparent,
        parent_temperature=temperature, grandparent_temperature=temperature,
        seed_alias=f"HCWDL-UB-FULL3/v1/{seed_node_id or node_id}/paired",
        behavior="balanced_uniform",
    )


def arm_registry(arm_id: str) -> Mapping[str, UnifiedBalancedNodeSpec]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB-FULL3 arm")
    result: dict[str, UnifiedBalancedNodeSpec] = {}
    history = ["shared/U000"]
    for node_id in FACTORIZED_NODES:
        node = _node(
            arm_id, node_id, parent_id=history[-1],
            grandparent_id=history[-2] if len(history) >= 2 else None,
        )
        result[node_id] = node
        history.append(node.canonical_id)
    direct = _node(
        arm_id, DIRECT_NODE, parent_id="shared/U000", grandparent_id=None,
        seed_node_id="U100",
    )
    result[DIRECT_NODE] = direct
    validate_arm_registry(arm_id, result)
    return MappingProxyType(result)


def validate_arm_registry(
    arm_id: str, registry: Mapping[str, UnifiedBalancedNodeSpec],
) -> None:
    expected_ids = set(FACTORIZED_NODES) | {DIRECT_NODE}
    if arm_id not in ARM_IDS or set(registry) != expected_ids or len(registry) != 12:
        raise ValueError("HCWDL-UB-FULL3 arm registry differs")
    for node_id, node in registry.items():
        if node.canonical_id != f"{arm_id}/{node_id}":
            raise ValueError("HCWDL-UB-FULL3 node ownership differs")
        if any(teacher.split("/", 1)[0] not in {SHARED_ARM, arm_id} for teacher in node.teachers):
            raise PermissionError("HCWDL-UB-FULL3 teacher crosses arms")
        if abs(node.ce_weight + node.parent_kd_weight + node.grandparent_kd_weight - 1) > 1e-12:
            raise ValueError("HCWDL-UB-FULL3 loss weights differ")
        if node.grandparent_kd_weight and node.grandparent_id is None:
            raise ValueError("HCWDL-UB-FULL3 grandparent is unavailable")


def meta_registry() -> Mapping[str, UnifiedBalancedNodeSpec]:
    result = {f"shared/{key}": value for key, value in shared_registry().items()}
    for arm_id in ARM_IDS:
        result.update({
            f"{arm_id}/{key}": value for key, value in arm_registry(arm_id).items()
        })
    if len(result) != 38:
        raise RuntimeError("HCWDL-UB-FULL3 registry is not 38 fits")
    return MappingProxyType(dict(sorted(result.items())))


def training_registry_for_arm(arm_id: str):
    from .hcwdl_ladder import NodeSpec, TeacherSpec

    registry = arm_registry(arm_id)
    result = {}
    for node_id, node in registry.items():
        teachers = []
        if node.parent_id is not None and node.parent_kd_weight > 0:
            teachers.append(TeacherSpec(
                node.parent_id,
                "hlt" if _node_domain(node.parent_id, arm_id) == "hlt" else "privileged",
                "parent",
            ))
        if node.grandparent_id is not None and node.grandparent_kd_weight > 0:
            teachers.append(TeacherSpec(
                node.grandparent_id,
                "hlt" if _node_domain(node.grandparent_id, arm_id) == "hlt" else "privileged",
                "grandparent",
            ))
        result[node_id] = NodeSpec(
            node_id=node_id, track="full_factorized", stage="child",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh", initialization_parent=None,
            teachers=tuple(teachers),
            loss_kind="ce_kd" if len(teachers) == 1 else "ce_two_kd",
            deployable=node.input_domain == "hlt",
        )
    return MappingProxyType(result)


def _node_domain(canonical_id: str, arm_id: str) -> str:
    owner, node_id = canonical_id.split("/", 1)
    if owner == SHARED_ARM:
        return shared_registry()[node_id].input_domain
    if owner != arm_id:
        raise PermissionError("HCWDL-UB-FULL3 teacher belongs to another arm")
    return arm_registry(owner)[node_id].input_domain


def shared_training_registry():
    from .hcwdl_ladder import NodeSpec

    return MappingProxyType({
        node_id: NodeSpec(
            node_id=node_id, track="full_shared", stage="root",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh", initialization_parent=None, teachers=(),
            loss_kind="ce", deployable=node.input_domain == "hlt",
        )
        for node_id, node in shared_registry().items()
    })


def idealized_u000_ancestry(arm_id: str) -> Mapping[str, float]:
    values = {"shared/U000": 1.0}
    remaining = dict(arm_registry(arm_id))
    while remaining:
        progressed = False
        for node_id, node in list(remaining.items()):
            if all(teacher in values for teacher in node.teachers):
                parent = values[node.parent_id] if node.parent_id else 0.0
                grandparent = values[node.grandparent_id] if node.grandparent_id else 0.0
                values[node.canonical_id] = (
                    node.parent_kd_weight * parent
                    + node.grandparent_kd_weight * grandparent
                )
                del remaining[node_id]
                progressed = True
        if not progressed:
            raise RuntimeError("HCWDL-UB-FULL3 ancestry graph is cyclic")
    return MappingProxyType({
        node_id: values[f"{arm_id}/{node_id}"] for node_id in arm_registry(arm_id)
    })


META_REGISTRY: Final = meta_registry()
META_GRAPH_SHA256: Final = canonical_sha256({
    "contract": "HCWDL_UNIFIED_BALANCED_FULL_GRAPH/v1",
    "nodes": [META_REGISTRY[key].payload() for key in sorted(META_REGISTRY)],
})


__all__ = [
    "ARM_IDS", "ARM_WEIGHTS", "CAMPAIGN_LABEL", "DIRECT_NODE",
    "FACTORIZED_NODES", "META_GRAPH_SHA256", "META_REGISTRY", "SHARED_ARM",
    "arm_registry", "idealized_u000_ancestry", "meta_registry",
    "shared_registry", "shared_training_registry", "training_registry_for_arm",
    "validate_arm_registry",
]
