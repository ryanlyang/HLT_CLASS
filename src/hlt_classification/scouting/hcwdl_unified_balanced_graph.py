"""Exact shared-foundation and six-arm HCWDL-UB scientific registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_homotopy import HomotopyCoordinate


SHARED_ARM: Final = "shared"
ARM_IDS: Final = (
    "C25P75", "C10P90", "C05P95", "C10P75G15", "C05P80G15", "C00P100",
)
REFERENCE_ARM: Final = "C25P75"
ARM_WEIGHTS: Final = MappingProxyType({
    "C25P75": (.25, .75, 0.0),
    "C10P90": (.10, .90, 0.0),
    "C05P95": (.05, .95, 0.0),
    "C10P75G15": (.10, .75, .15),
    "C05P80G15": (.05, .80, .15),
    "C00P100": (0.0, 1.0, 0.0),
})

FACTORIZED_NODES: Final = (
    "U020", "U040", "U060", "U080", "U100",
    "D80F", "D60F", "D40F", "D20F", "D0F", "M1F",
)
JOINT_NODES: Final = (
    "J010", "J020", "J030", "J040", "J050", "J060",
    "J070", "J080", "J090", "J100", "M1J",
)
LEGACY_U_NODES: Final = tuple(f"U{value:03d}_legacycdf" for value in (20, 40, 60, 80, 100))
LEGACY_D_NODES: Final = tuple(
    [f"D{value}F_legacywarp" for value in (80, 60, 40, 20, 0)]
    + ["M1F_legacywarp"]
)


@dataclass(frozen=True)
class UnifiedBalancedNodeSpec:
    arm_id: str
    node_id: str
    input_domain: str
    coordinate: HomotopyCoordinate
    parent_id: str | None
    grandparent_id: str | None
    ce_weight: float
    parent_kd_weight: float
    grandparent_kd_weight: float
    parent_temperature: float
    grandparent_temperature: float
    seed_alias: str
    behavior: str

    @property
    def canonical_id(self) -> str:
        return f"{self.arm_id}/{self.node_id}"

    @property
    def teachers(self) -> tuple[str, ...]:
        rows = []
        if self.parent_id is not None and self.parent_kd_weight > 0:
            rows.append(self.parent_id)
        if self.grandparent_id is not None and self.grandparent_kd_weight > 0:
            rows.append(self.grandparent_id)
        return tuple(rows)

    def payload(self) -> dict[str, object]:
        value = asdict(self)
        value["coordinate"] = self.coordinate.payload()
        value["canonical_id"] = self.canonical_id
        return value


def _coordinate(node_id: str) -> HomotopyCoordinate:
    base = node_id.removesuffix("_legacycdf").removesuffix("_legacywarp")
    if base in {"U000", "M0paired"}:
        return HomotopyCoordinate(0, 1, 0, 1) if base == "U000" else HomotopyCoordinate(1, 1, 1, 1)
    if base == "D100direct":
        return HomotopyCoordinate(1, 1, 0, 1)
    if base.startswith("U"):
        value = int(base[1:]); return HomotopyCoordinate(value, 100, 0, 1)
    if base.startswith("D"):
        offline = int(base[1:-1]); return HomotopyCoordinate(1, 1, 100 - offline, 100)
    if base.startswith("J"):
        value = int(base[1:]); return HomotopyCoordinate(value, 100, value, 100)
    if base.startswith("M1"):
        return HomotopyCoordinate(1, 1, 1, 1)
    raise KeyError(f"unknown HCWDL-UB coordinate node {node_id}")


def _resolved_weights(
    arm_id: str, *, has_parent: bool, has_grandparent: bool, fixed_m1: bool = False,
) -> tuple[float, float, float, float]:
    if not has_parent:
        return 1.0, 0.0, 0.0, 1.0
    if fixed_m1:
        return .25, .75, 0.0, 1.0
    ce, parent, grandparent = ARM_WEIGHTS[arm_id]
    if grandparent and not has_grandparent:
        parent += grandparent; grandparent = 0.0
    return ce, parent, grandparent, 2.0


def _node(
    *, arm_id: str, node_id: str, input_domain: str,
    parent_id: str | None, grandparent_id: str | None,
    behavior: str, seed_node_id: str | None = None,
) -> UnifiedBalancedNodeSpec:
    fixed_m1 = node_id.startswith("M1")
    ce, parent, grandparent, temperature = _resolved_weights(
        arm_id, has_parent=parent_id is not None,
        has_grandparent=grandparent_id is not None, fixed_m1=fixed_m1,
    )
    return UnifiedBalancedNodeSpec(
        arm_id=arm_id, node_id=node_id, input_domain=input_domain,
        coordinate=_coordinate(node_id), parent_id=parent_id,
        grandparent_id=grandparent_id, ce_weight=ce,
        parent_kd_weight=parent, grandparent_kd_weight=grandparent,
        parent_temperature=temperature, grandparent_temperature=temperature,
        seed_alias=f"HCWDL-UB/v1/{seed_node_id or node_id}/paired",
        behavior=behavior,
    )


def shared_registry() -> Mapping[str, UnifiedBalancedNodeSpec]:
    nodes = {
        "U000": UnifiedBalancedNodeSpec(
            SHARED_ARM, "U000", "p0", _coordinate("U000"), None, None,
            1.0, 0.0, 0.0, 1.0, 1.0,
            "HCWDL-UB/v1/shared-root/paired", "balanced_uniform",
        ),
        "M0paired": UnifiedBalancedNodeSpec(
            SHARED_ARM, "M0paired", "hlt", _coordinate("M0paired"), None, None,
            1.0, 0.0, 0.0, 1.0, 1.0,
            "HCWDL-UB/v1/shared-root/paired", "balanced_uniform",
        ),
    }
    return MappingProxyType(nodes)


def _path_nodes(
    arm_id: str, node_ids: tuple[str, ...], *, behavior: str,
    root_id: str, input_domain: str = "homotopy",
) -> list[UnifiedBalancedNodeSpec]:
    history = [root_id]
    result = []
    for node_id in node_ids:
        parent = history[-1]
        grandparent = history[-2] if len(history) >= 2 else None
        coordinate = _coordinate(node_id)
        exact_hlt = (
            coordinate.structural_numerator == coordinate.structural_denominator
            and coordinate.feature_numerator == coordinate.feature_denominator
        )
        domain = "hlt" if node_id.startswith("M1") or exact_hlt else input_domain
        seed_name = node_id.removesuffix("_legacycdf").removesuffix("_legacywarp")
        result.append(_node(
            arm_id=arm_id, node_id=node_id, input_domain=domain,
            parent_id=parent, grandparent_id=grandparent,
            behavior=behavior, seed_node_id=seed_name,
        ))
        history.append(f"{arm_id}/{node_id}")
    return result


def arm_registry(arm_id: str) -> Mapping[str, UnifiedBalancedNodeSpec]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB recipe arm")
    nodes = {}
    for node in _path_nodes(
        arm_id, FACTORIZED_NODES, behavior="balanced_uniform", root_id="shared/U000",
    ):
        nodes[node.node_id] = node
    for node in _path_nodes(
        arm_id, JOINT_NODES, behavior="balanced_uniform", root_id="shared/U000",
    ):
        nodes[node.node_id] = node
    direct = _node(
        arm_id=arm_id, node_id="D100direct", input_domain="homotopy",
        parent_id="shared/U000", grandparent_id=None,
        behavior="balanced_uniform", seed_node_id="U100",
    )
    nodes[direct.node_id] = direct
    if arm_id == REFERENCE_ARM:
        for node in _path_nodes(
            arm_id, LEGACY_U_NODES, behavior="legacycdf_uniform",
            root_id="shared/U000",
        ):
            nodes[node.node_id] = node
        # The legacy D path is paired to and rooted at the reference arm's
        # primary U100 endpoint, exactly as the scientific plan declares.
        for node in _path_nodes(
            arm_id, LEGACY_D_NODES, behavior="balanced_legacywarp",
            root_id=f"{arm_id}/U100",
        ):
            nodes[node.node_id] = node
    validate_arm_registry(arm_id, nodes)
    return MappingProxyType(dict(sorted(nodes.items())))


def validate_arm_registry(
    arm_id: str, registry: Mapping[str, UnifiedBalancedNodeSpec],
) -> None:
    expected = 34 if arm_id == REFERENCE_ARM else 23
    if len(registry) != expected or set(registry) != {node.node_id for node in registry.values()}:
        raise ValueError("HCWDL-UB arm registry count/identity differs")
    for node_id, node in registry.items():
        if node.arm_id != arm_id or node.canonical_id != f"{arm_id}/{node_id}":
            raise ValueError("HCWDL-UB arm node ownership differs")
        if any(
            teacher.split("/", 1)[0] not in {SHARED_ARM, arm_id}
            for teacher in node.teachers
        ):
            raise PermissionError("HCWDL-UB teacher crosses recipe arms")
        if abs(node.ce_weight + node.parent_kd_weight + node.grandparent_kd_weight - 1.0) > 1e-12:
            raise ValueError("HCWDL-UB node loss weights do not sum to one")
        if node.parent_id is None and (node.parent_kd_weight or node.grandparent_kd_weight):
            raise ValueError("HCWDL-UB root has an unavailable KD teacher")
        if node.grandparent_id is None and node.grandparent_kd_weight:
            raise ValueError("HCWDL-UB node has an unavailable grandparent")
        if node.node_id.startswith("M1") and (
            (node.ce_weight, node.parent_kd_weight, node.grandparent_kd_weight)
            != (.25, .75, 0.0)
            or node.parent_temperature != 1.0
        ):
            raise ValueError("HCWDL-UB M1 loss route differs")


def meta_registry() -> Mapping[str, UnifiedBalancedNodeSpec]:
    result = {f"shared/{name}": node for name, node in shared_registry().items()}
    for arm_id in ARM_IDS:
        for node_id, node in arm_registry(arm_id).items():
            key = f"{arm_id}/{node_id}"
            if key in result:
                raise RuntimeError("HCWDL-UB meta registry contains a duplicate")
            result[key] = node
    if len(result) != 151:
        raise RuntimeError("HCWDL-UB default meta registry is not 151 fits")
    return MappingProxyType(dict(sorted(result.items())))


def idealized_u000_ancestry(arm_id: str) -> Mapping[str, float]:
    """Return the predeclared linear U000-teacher ancestry diagnostic."""

    registry = arm_registry(arm_id); values = {"shared/U000": 1.0}
    remaining = dict(registry)
    while remaining:
        progressed = False
        for node_id, node in list(remaining.items()):
            if all(teacher in values for teacher in node.teachers):
                parent = 0.0 if node.parent_id is None else values[node.parent_id]
                grand = 0.0 if node.grandparent_id is None else values[node.grandparent_id]
                values[node.canonical_id] = (
                    node.parent_kd_weight * parent
                    + node.grandparent_kd_weight * grand
                )
                del remaining[node_id]; progressed = True
        if not progressed:
            raise RuntimeError("HCWDL-UB ancestry graph is cyclic or incomplete")
    return MappingProxyType({
        node_id: values[f"{arm_id}/{node_id}"] for node_id in registry
    })


def training_registry_for_arm(arm_id: str):
    """Adapt the UB graph to the generic cold-start HCWDL engine interface."""

    from .hcwdl_ladder import NodeSpec, TeacherSpec

    registry = {}
    def teacher_domain(canonical_id: str) -> str:
        owner, teacher_id = canonical_id.split("/", 1)
        teacher = (
            shared_registry()[teacher_id]
            if owner == SHARED_ARM else arm_registry(owner)[teacher_id]
        )
        return "hlt" if teacher.input_domain == "hlt" else "privileged"

    for node_id, node in arm_registry(arm_id).items():
        teachers = []
        if node.parent_id is not None and node.parent_kd_weight > 0:
            teachers.append(TeacherSpec(
                node.parent_id, teacher_domain(node.parent_id), "parent",
            ))
        if node.grandparent_id is not None and node.grandparent_kd_weight > 0:
            teachers.append(TeacherSpec(
                node.grandparent_id, teacher_domain(node.grandparent_id),
                "grandparent",
            ))
        registry[node_id] = NodeSpec(
            node_id=node_id, track=node.behavior,
            stage="root" if not teachers else "child",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh", initialization_parent=None,
            teachers=tuple(teachers),
            loss_kind="ce" if not teachers else "ce_kd" if len(teachers) == 1 else "ce_two_kd",
            deployable=node.input_domain == "hlt",
        )
    return MappingProxyType(registry)


def shared_training_registry():
    from .hcwdl_ladder import NodeSpec

    return MappingProxyType({
        node_id: NodeSpec(
            node_id=node_id, track="shared", stage="root",
            student_domain="hlt" if node.input_domain == "hlt" else "privileged",
            initialization="fresh", initialization_parent=None, teachers=(),
            loss_kind="ce", deployable=node.input_domain == "hlt",
        )
        for node_id, node in shared_registry().items()
    })


META_REGISTRY: Final = meta_registry()
META_GRAPH_SHA256: Final = canonical_sha256({
    "contract": "HCWDL_UNIFIED_BALANCED_GRAPH/v1",
    "nodes": [META_REGISTRY[key].payload() for key in sorted(META_REGISTRY)],
})


__all__ = [
    "ARM_IDS", "ARM_WEIGHTS", "FACTORIZED_NODES", "JOINT_NODES",
    "LEGACY_D_NODES", "LEGACY_U_NODES", "META_GRAPH_SHA256",
    "META_REGISTRY", "REFERENCE_ARM", "SHARED_ARM", "UnifiedBalancedNodeSpec",
    "arm_registry", "idealized_u000_ancestry", "meta_registry", "shared_registry", "shared_training_registry",
    "validate_arm_registry",
    "training_registry_for_arm",
]
