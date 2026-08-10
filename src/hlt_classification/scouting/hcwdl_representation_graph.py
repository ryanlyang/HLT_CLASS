"""Frozen four-track dense HCWDL matching-free representation-KD descent.

The graph has two strategy-specific D100 roots taught by the native offline
model.  Each root fans out into a cold and a warm five-percentage-point
descent from D95 through D0 and terminates at M1.  Cold children are freshly
initialized; warm children load only their immediate same-strategy
predecessor and reset all optimizer state.

The historical M1--M6 ascent remains a distinct retired contract.  This
module publishes ``HCWDL_REPRESENTATION_DENSE_DESCENT_GRAPH/v1``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_representation_contracts import (
    REPRESENTATION_CONTROL_REGISTRY_CONTRACT,
    REPRESENTATION_DESCENT_GRAPH_CONTRACT,
    build_versioned_artifact,
    validate_versioned_artifact,
)


# Compatibility names are retained for internal imports while their values
# identify the new scientific artifact explicitly.
ASCENT_GRAPH_CONTRACT: Final = REPRESENTATION_DESCENT_GRAPH_CONTRACT
DESCENT_GRAPH_CONTRACT: Final = REPRESENTATION_DESCENT_GRAPH_CONTRACT
CONTROL_REGISTRY_CONTRACT: Final = REPRESENTATION_CONTROL_REGISTRY_CONTRACT
SCHEMA_VERSION: Final = 1

RSET_STRATEGY: Final = "HCWDL_REP_SET/v1"
RREL_STRATEGY: Final = "HCWDL_REP_REL/v1"
STRATEGIES: Final = (RSET_STRATEGY, RREL_STRATEGY)
TRACKS: Final = ("cold", "warm")
DESCENT_LEVELS: Final = tuple(range(100, -1, -5))
TRACKED_LEVELS: Final = DESCENT_LEVELS[1:]
TERMINAL_STEP: Final = len(TRACKED_LEVELS) + 1


def domain_for_level(level: int) -> str:
    if level not in DESCENT_LEVELS:
        raise ValueError(f"unknown representation privilege level {level}")
    return "hlt" if level == 0 else f"d{level}"


@dataclass(frozen=True)
class RepresentationNodeSpec:
    """One explicit dense-descent student.

    ``rung`` is the canonical step coordinate: D100 is zero, D95 is one,
    D0 is twenty, and M1 is twenty-one.  Runtime routing never derives these
    semantics from the node name.
    """

    node_id: str
    branch_id: str
    strategy: str
    track: str
    rung: int
    stage: str
    privilege_percent: int
    student_domain: str
    initialization: str
    initialization_parent: str | None
    predecessor_logit_teacher: str | None
    representation_logit_teacher: str
    representation_teacher_domain: str
    parent_counterpart: str | None
    target_bank_identity: str
    deployable: bool

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RepresentationControlSpec:
    """Type surface retained for future explicitly versioned controls."""

    control_id: str
    paired_primary_node: str
    strategy: str
    track: str
    rung: int
    student_domain: str
    initialization: str
    initialization_parent: None
    predecessor_logit_teacher: str
    representation_logit_teacher: str
    representation_teacher_domain: str
    parent_counterpart: str
    target_bank_identity: str
    component_allocation: tuple[tuple[str, float], ...]
    shuffled_representation_targets: bool
    shuffle_map_contract: str | None
    owns_gradient_calibration: bool
    disposition: str
    deployable: bool
    finalist_eligible: bool
    confirmation_eligible: bool
    descendants: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        result = asdict(self)
        result["component_allocation"] = dict(self.component_allocation)
        result["descendants"] = list(self.descendants)
        return result


def _strategy_prefix(strategy: str) -> str:
    if strategy == RSET_STRATEGY:
        return "RSET"
    if strategy == RREL_STRATEGY:
        return "RREL"
    raise ValueError(f"unknown representation strategy {strategy!r}")


def _track_suffix(track: str) -> str:
    if track == "cold":
        return "c"
    if track == "warm":
        return "w"
    raise ValueError(f"unknown representation track {track!r}")


def _build_node_registry() -> dict[str, RepresentationNodeSpec]:
    nodes: dict[str, RepresentationNodeSpec] = {}
    for strategy in STRATEGIES:
        prefix = _strategy_prefix(strategy)
        root_id = f"{prefix}_D100"
        nodes[root_id] = RepresentationNodeSpec(
            node_id=root_id,
            branch_id=f"{prefix}-root",
            strategy=strategy,
            track="shared",
            rung=0,
            stage="offline_to_d100",
            privilege_percent=100,
            student_domain="d100",
            initialization="fresh",
            initialization_parent=None,
            predecessor_logit_teacher=None,
            representation_logit_teacher="TOFF",
            representation_teacher_domain="toff",
            parent_counterpart="D100",
            target_bank_identity="TOFF",
            deployable=False,
        )
        for track in TRACKS:
            suffix = _track_suffix(track)
            branch_id = f"{prefix}-{track}"
            predecessor = root_id
            predecessor_domain = "d100"
            for step, level in enumerate(TRACKED_LEVELS, start=1):
                node_id = f"{prefix}_D{level}{suffix}"
                nodes[node_id] = RepresentationNodeSpec(
                    node_id=node_id,
                    branch_id=branch_id,
                    strategy=strategy,
                    track=track,
                    rung=step,
                    stage="down",
                    privilege_percent=level,
                    student_domain=domain_for_level(level),
                    initialization="fresh" if track == "cold" else "warm",
                    initialization_parent=(
                        None if track == "cold" else predecessor
                    ),
                    # The predecessor's target bank supplies both the sole
                    # logit teacher and the representation teacher.
                    predecessor_logit_teacher=None,
                    representation_logit_teacher=predecessor,
                    representation_teacher_domain=predecessor_domain,
                    # This is a stochastic-comparison coordinate, not a
                    # claim that a separately deployable base model exists.
                    # Every five-point rung therefore has an exact paired
                    # coordinate even when it is only a training
                    # intermediate in this dense descent.
                    parent_counterpart=f"D{level}{suffix}",
                    target_bank_identity=predecessor,
                    deployable=level == 0,
                )
                predecessor = node_id
                predecessor_domain = domain_for_level(level)
            terminal_id = f"{prefix}_M1{suffix}"
            nodes[terminal_id] = RepresentationNodeSpec(
                node_id=terminal_id,
                branch_id=branch_id,
                strategy=strategy,
                track=track,
                rung=TERMINAL_STEP,
                stage="terminal_m1",
                privilege_percent=0,
                student_domain="hlt",
                initialization="fresh" if track == "cold" else "warm",
                initialization_parent=None if track == "cold" else predecessor,
                predecessor_logit_teacher=None,
                representation_logit_teacher=predecessor,
                representation_teacher_domain="hlt",
                parent_counterpart=f"M1{suffix}",
                target_bank_identity=predecessor,
                deployable=True,
            )
    return nodes


NODE_REGISTRY: Final[Mapping[str, RepresentationNodeSpec]] = MappingProxyType(
    _build_node_registry()
)
# The old M5-only mechanisms do not have a valid dense-descent meaning.
CONTROL_REGISTRY: Final[Mapping[str, RepresentationControlSpec]] = (
    MappingProxyType({})
)


def _expected_primary_ids() -> set[str]:
    expected: set[str] = set()
    for prefix in ("RSET", "RREL"):
        expected.add(f"{prefix}_D100")
        for suffix in ("c", "w"):
            expected.update(
                f"{prefix}_D{level}{suffix}" for level in TRACKED_LEVELS
            )
            expected.add(f"{prefix}_M1{suffix}")
    return expected


def _assert_acyclic(registry: Mapping[str, RepresentationNodeSpec]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("representation descent graph contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        node = registry[node_id]
        for parent in (
            node.representation_logit_teacher,
            node.initialization_parent,
            node.predecessor_logit_teacher,
        ):
            if parent in registry:
                visit(str(parent))
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in registry:
        visit(node_id)


def validate_ascent_graph(
    registry: Mapping[str, RepresentationNodeSpec] = NODE_REGISTRY,
) -> str:
    """Validate the dense descent and return its canonical semantic hash."""

    if len(registry) != 86 or set(registry) != _expected_primary_ids():
        raise ValueError("representation descent registry must contain exactly 86 nodes")
    if set(registry) != {node.node_id for node in registry.values()}:
        raise ValueError("representation descent node IDs differ from registry keys")

    for strategy in STRATEGIES:
        prefix = _strategy_prefix(strategy)
        root = registry[f"{prefix}_D100"]
        if (
            root.branch_id != f"{prefix}-root"
            or root.strategy != strategy
            or root.track != "shared"
            or root.rung != 0
            or root.stage != "offline_to_d100"
            or root.privilege_percent != 100
            or root.student_domain != "d100"
            or root.initialization != "fresh"
            or root.initialization_parent is not None
            or root.predecessor_logit_teacher is not None
            or root.representation_logit_teacher != "TOFF"
            or root.representation_teacher_domain != "toff"
            or root.parent_counterpart != "D100"
            or root.target_bank_identity != "TOFF"
            or root.deployable
        ):
            raise ValueError(f"offline-to-D100 root differs for {prefix}")

        for track in TRACKS:
            suffix = _track_suffix(track)
            predecessor = root.node_id
            predecessor_domain = "d100"
            for step, level in enumerate(TRACKED_LEVELS, start=1):
                node_id = f"{prefix}_D{level}{suffix}"
                node = registry[node_id]
                if (
                    node.branch_id != f"{prefix}-{track}"
                    or node.strategy != strategy
                    or node.track != track
                    or node.rung != step
                    or node.stage != "down"
                    or node.privilege_percent != level
                    or node.student_domain != domain_for_level(level)
                    or node.predecessor_logit_teacher is not None
                    or node.representation_logit_teacher != predecessor
                    or node.representation_teacher_domain != predecessor_domain
                    or node.target_bank_identity != predecessor
                    or node.parent_counterpart != f"D{level}{suffix}"
                    or node.deployable != (level == 0)
                ):
                    raise ValueError(f"dense descent routing differs for {node_id}")
                if track == "cold":
                    if node.initialization != "fresh" or node.initialization_parent is not None:
                        raise ValueError(f"cold initialization differs for {node_id}")
                elif node.initialization != "warm" or node.initialization_parent != predecessor:
                    raise ValueError(f"warm initialization differs for {node_id}")
                predecessor = node_id
                predecessor_domain = domain_for_level(level)

            terminal_id = f"{prefix}_M1{suffix}"
            terminal = registry[terminal_id]
            if (
                terminal.branch_id != f"{prefix}-{track}"
                or terminal.strategy != strategy
                or terminal.track != track
                or terminal.rung != TERMINAL_STEP
                or terminal.stage != "terminal_m1"
                or terminal.privilege_percent != 0
                or terminal.student_domain != "hlt"
                or terminal.predecessor_logit_teacher is not None
                or terminal.representation_logit_teacher != predecessor
                or terminal.representation_teacher_domain != "hlt"
                or terminal.target_bank_identity != predecessor
                or terminal.parent_counterpart != f"M1{suffix}"
                or not terminal.deployable
            ):
                raise ValueError(f"terminal M1 routing differs for {terminal_id}")
            if track == "cold":
                if terminal.initialization != "fresh" or terminal.initialization_parent is not None:
                    raise ValueError(f"cold terminal initialization differs for {terminal_id}")
            elif terminal.initialization != "warm" or terminal.initialization_parent != predecessor:
                raise ValueError(f"warm terminal initialization differs for {terminal_id}")

    _assert_acyclic(registry)
    return canonical_sha256({
        "contract": ASCENT_GRAPH_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "nodes": [registry[node_id].payload() for node_id in sorted(registry)],
    })


def validate_control_registry(
    registry: Mapping[str, RepresentationControlSpec] = CONTROL_REGISTRY,
    *,
    primary_registry: Mapping[str, RepresentationNodeSpec] = NODE_REGISTRY,
) -> str:
    """Dense descent v1 deliberately has no inherited M5-only controls."""

    validate_ascent_graph(primary_registry)
    if registry:
        raise ValueError("dense descent control registry must be empty")
    return canonical_sha256({
        "contract": CONTROL_REGISTRY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "descent_graph_sha256": ASCENT_GRAPH_SHA256,
        "controls": [],
    })


def ascent_graph_artifact(*, parents: Mapping[str, str]) -> dict[str, object]:
    """Build the reusable dense-descent graph artifact."""

    if set(parents) != {"parent_graph", "parent_import"}:
        raise ValueError("representation descent graph parent keys differ")
    return build_versioned_artifact(
        ASCENT_GRAPH_CONTRACT,
        parents=parents,
        payload={
            "semantic_graph_sha256": ASCENT_GRAPH_SHA256,
            "nodes": [NODE_REGISTRY[node_id].payload() for node_id in sorted(NODE_REGISTRY)],
        },
    )


def validate_ascent_graph_artifact(
    value: Mapping[str, object], *, expected_parents: Mapping[str, str] | None = None,
) -> str:
    digest = validate_versioned_artifact(
        value,
        expected_contract=ASCENT_GRAPH_CONTRACT,
        expected_parents=expected_parents,
        required_payload_keys=("semantic_graph_sha256", "nodes"),
    )
    payload = value["payload"]
    if set(payload) != {"semantic_graph_sha256", "nodes"} or payload != {
        "semantic_graph_sha256": ASCENT_GRAPH_SHA256,
        "nodes": [NODE_REGISTRY[node_id].payload() for node_id in sorted(NODE_REGISTRY)],
    }:
        raise ValueError("representation descent graph artifact differs")
    return digest


def control_registry_artifact(*, ascent_graph_artifact_sha256: str) -> dict[str, object]:
    return build_versioned_artifact(
        CONTROL_REGISTRY_CONTRACT,
        parents={"representation_descent_graph": ascent_graph_artifact_sha256},
        payload={
            "semantic_registry_sha256": CONTROL_REGISTRY_SHA256,
            "controls": [],
        },
    )


def validate_control_registry_artifact(
    value: Mapping[str, object], *, ascent_graph_artifact_sha256: str,
) -> str:
    digest = validate_versioned_artifact(
        value,
        expected_contract=CONTROL_REGISTRY_CONTRACT,
        expected_parents={"representation_descent_graph": ascent_graph_artifact_sha256},
        required_payload_keys=("semantic_registry_sha256", "controls"),
    )
    payload = value["payload"]
    if set(payload) != {"semantic_registry_sha256", "controls"} or payload != {
        "semantic_registry_sha256": CONTROL_REGISTRY_SHA256,
        "controls": [],
    }:
        raise ValueError("dense descent control registry artifact differs")
    return digest


ASCENT_GRAPH_SHA256: Final = validate_ascent_graph()
DESCENT_GRAPH_SHA256: Final = ASCENT_GRAPH_SHA256
CONTROL_REGISTRY_SHA256: Final = validate_control_registry()


__all__ = [
    "ASCENT_GRAPH_CONTRACT",
    "ASCENT_GRAPH_SHA256",
    "CONTROL_REGISTRY",
    "CONTROL_REGISTRY_CONTRACT",
    "CONTROL_REGISTRY_SHA256",
    "DESCENT_GRAPH_CONTRACT",
    "DESCENT_GRAPH_SHA256",
    "DESCENT_LEVELS",
    "NODE_REGISTRY",
    "RREL_STRATEGY",
    "RSET_STRATEGY",
    "RepresentationControlSpec",
    "RepresentationNodeSpec",
    "TERMINAL_STEP",
    "TRACKED_LEVELS",
    "ascent_graph_artifact",
    "control_registry_artifact",
    "domain_for_level",
    "validate_ascent_graph",
    "validate_ascent_graph_artifact",
    "validate_control_registry",
    "validate_control_registry_artifact",
]
