"""Immutable 23-node HCWDL graph and exact domain/teacher semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256


NODE_SPEC_CONTRACT: Final = "HCWDL_NODE_SPEC/v1"
GRAPH_CONTRACT: Final = "HCWDL_GRAPH/v1"
DOMAINS: Final = MappingProxyType({
    "hlt": {"input": "hlt", "alpha": 0.0, "deployable": True},
    "d25": {"input": "privileged", "alpha": 0.25, "deployable": False},
    "d50": {"input": "privileged", "alpha": 0.50, "deployable": False},
    "d75": {"input": "privileged", "alpha": 0.75, "deployable": False},
    "d100": {"input": "privileged", "alpha": 1.0, "deployable": False},
    "toff": {"input": "toff", "alpha": None, "deployable": False},
})


@dataclass(frozen=True)
class TeacherSpec:
    node_id: str
    domain: str
    role: str


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    track: str
    stage: str
    student_domain: str
    initialization: str
    initialization_parent: str | None
    teachers: tuple[TeacherSpec, ...]
    loss_kind: str
    deployable: bool

    def payload(self) -> dict[str, object]:
        return {
            "contract": NODE_SPEC_CONTRACT,
            "schema_version": 1,
            **asdict(self),
        }


def _teacher(node: str, domain: str, role: str) -> TeacherSpec:
    return TeacherSpec(node, domain, role)


def _build_registry() -> dict[str, NodeSpec]:
    nodes: dict[str, NodeSpec] = {}

    def add(
        node_id: str, track: str, stage: str, domain: str,
        initialization: str = "fresh", parent: str | None = None,
        teachers: tuple[TeacherSpec, ...] = (), deployable: bool = False,
    ) -> None:
        loss = "ce" if not teachers else "ce_kd" if len(teachers) == 1 else "ce_two_kd"
        nodes[node_id] = NodeSpec(
            node_id, track, stage, domain, initialization, parent,
            teachers, loss, deployable,
        )

    add("M0", "shared", "root", "hlt", deployable=True)
    add("D100", "shared", "root", "d100")
    add("TOFF", "shared", "root", "toff")
    for track, suffix in (("cold", "c"), ("warm", "w")):
        previous = "D100"
        previous_domain = "d100"
        for level, domain in ((75, "d75"), (50, "d50"), (25, "d25"), (0, "hlt")):
            node = f"D{level}{suffix}"
            initialization = "fresh" if track == "cold" else "warm"
            add(
                node, track, "down", domain, initialization,
                previous if track == "warm" else None,
                (_teacher(previous, previous_domain, "sole"),),
                deployable=False,
            )
            previous, previous_domain = node, domain
        dnodes = {25: f"D25{suffix}", 50: f"D50{suffix}", 75: f"D75{suffix}"}
        predecessor = f"D0{suffix}"
        for generation in range(1, 7):
            node = f"M{generation}{suffix}"
            if generation == 1:
                teachers = (_teacher(predecessor, "hlt", "sole"),)
            else:
                privileged = {
                    2: (dnodes[25], "d25"),
                    3: (dnodes[50], "d50"),
                    4: (dnodes[75], "d75"),
                    5: ("D100", "d100"),
                    6: ("TOFF", "toff"),
                }[generation]
                teachers = (
                    _teacher(predecessor, "hlt", "predecessor"),
                    _teacher(privileged[0], privileged[1], "privileged"),
                )
            add(
                node, track, "up", "hlt",
                "fresh" if track == "cold" else "warm",
                predecessor if track == "warm" else None,
                teachers, deployable=True,
            )
            predecessor = node
    return nodes


NODE_REGISTRY: Final[Mapping[str, NodeSpec]] = MappingProxyType(_build_registry())


def validate_ladder_graph(registry: Mapping[str, NodeSpec] = NODE_REGISTRY) -> str:
    if len(registry) != 23 or set(registry) != {node.node_id for node in registry.values()}:
        raise ValueError("HCWDL primary registry must contain exactly 23 unique nodes")
    if registry["M1c"].initialization_parent is not None or registry["M1c"].teachers[0].node_id != "D0c":
        raise ValueError("cold M1 bottom rung differs")
    if registry["M1w"].initialization_parent != "D0w" or registry["M1w"].teachers[0].node_id != "D0w":
        raise ValueError("warm M1 bottom rung differs")
    for node in registry.values():
        if node.student_domain not in DOMAINS:
            raise ValueError(f"unknown student domain for {node.node_id}")
        if node.deployable and node.student_domain != "hlt":
            raise ValueError(f"deployable node {node.node_id} is not HLT-only")
        if node.node_id != "M0" and any(teacher.node_id == "M0" for teacher in node.teachers):
            raise ValueError("M0 cannot teach a primary HCWDL node")
        if len(node.teachers) != {"ce": 0, "ce_kd": 1, "ce_two_kd": 2}[node.loss_kind]:
            raise ValueError(f"teacher/loss arity differs for {node.node_id}")
        for teacher in node.teachers:
            if teacher.node_id not in registry or teacher.domain not in DOMAINS:
                raise ValueError(f"teacher edge differs for {node.node_id}")
            if node.track in {"cold", "warm"} and teacher.node_id[-1:] in {"c", "w"}:
                expected = "c" if node.track == "cold" else "w"
                if teacher.node_id[-1] != expected:
                    raise ValueError(f"cross-track teacher leakage for {node.node_id}")
        if node.initialization == "warm":
            if node.initialization_parent not in registry:
                raise ValueError(f"warm parent differs for {node.node_id}")
        elif node.initialization_parent is not None:
            raise ValueError(f"fresh node {node.node_id} has an initialization parent")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("HCWDL graph contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        node = registry[node_id]
        parents = [teacher.node_id for teacher in node.teachers]
        if node.initialization_parent is not None:
            parents.append(node.initialization_parent)
        for parent in parents:
            visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in registry:
        visit(node_id)
    payload = {
        "contract": GRAPH_CONTRACT,
        "schema_version": 1,
        "nodes": [registry[name].payload() for name in sorted(registry)],
    }
    return canonical_sha256(payload)


GRAPH_SHA256: Final = validate_ladder_graph()


__all__ = [
    "DOMAINS", "GRAPH_CONTRACT", "GRAPH_SHA256", "NODE_REGISTRY",
    "NODE_SPEC_CONTRACT", "NodeSpec", "TeacherSpec", "validate_ladder_graph",
]
