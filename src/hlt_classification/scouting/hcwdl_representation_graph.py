"""Frozen HCWDL matching-free representation-KD ascent and control graphs.

This module is deliberately additive.  It authenticates the 24 representation
students and four terminal controls without changing ``HCWDL_GRAPH/v1`` or
deriving scientific semantics from node-name suffixes at runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_representation_contracts import (
    REPRESENTATION_ASCENT_GRAPH_CONTRACT,
    REPRESENTATION_CONTROL_REGISTRY_CONTRACT,
    build_versioned_artifact,
    validate_versioned_artifact,
)


ASCENT_GRAPH_CONTRACT: Final = REPRESENTATION_ASCENT_GRAPH_CONTRACT
CONTROL_REGISTRY_CONTRACT: Final = REPRESENTATION_CONTROL_REGISTRY_CONTRACT
SCHEMA_VERSION: Final = 1

RSET_STRATEGY: Final = "HCWDL_REP_SET/v1"
RREL_STRATEGY: Final = "HCWDL_REP_REL/v1"
STRATEGIES: Final = (RSET_STRATEGY, RREL_STRATEGY)
TRACKS: Final = ("cold", "warm")

_PRIVILEGED_BY_RUNG: Final = MappingProxyType({
    1: ("D0", "hlt"),
    2: ("D25", "d25"),
    3: ("D50", "d50"),
    4: ("D75", "d75"),
    5: ("D100", "d100"),
    6: ("TOFF", "toff"),
})


@dataclass(frozen=True)
class RepresentationNodeSpec:
    """One explicit primary ascent node.

    Every routing field is stored rather than inferred from ``node_id``.
    ``initialization_parent`` names a deployable checkpoint source; cold nodes
    intentionally leave it null and instead bind ``parent_counterpart`` for
    paired initialization seeds.
    """

    node_id: str
    branch_id: str
    strategy: str
    track: str
    rung: int
    student_domain: str
    initialization: str
    initialization_parent: str | None
    predecessor_logit_teacher: str | None
    representation_logit_teacher: str
    representation_teacher_domain: str
    parent_counterpart: str
    target_bank_identity: str
    deployable: bool

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RepresentationControlSpec:
    """One registered validation-only M5 control."""

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
        # JSON publication canonicalizes tuples to arrays.  Emit the canonical
        # list form up front so an artifact validates identically before and
        # after its immutable-file round trip.
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


def _teacher_for(track: str, rung: int) -> tuple[str, str, str]:
    base, domain = _PRIVILEGED_BY_RUNG[rung]
    suffix = _track_suffix(track)
    if base in {"D0", "D25", "D50", "D75"}:
        teacher = f"{base}{suffix}"
    else:
        teacher = base
    return teacher, domain, teacher


def _build_node_registry() -> dict[str, RepresentationNodeSpec]:
    nodes: dict[str, RepresentationNodeSpec] = {}
    for strategy in STRATEGIES:
        prefix = _strategy_prefix(strategy)
        for track in TRACKS:
            suffix = _track_suffix(track)
            branch_id = f"{prefix}-{track}"
            predecessor: str | None = None
            for rung in range(1, 7):
                node_id = f"{prefix}_M{rung}{suffix}"
                teacher, teacher_domain, bank = _teacher_for(track, rung)
                initialization_parent = (
                    None
                    if track == "cold"
                    else teacher if rung == 1 else predecessor
                )
                nodes[node_id] = RepresentationNodeSpec(
                    node_id=node_id,
                    branch_id=branch_id,
                    strategy=strategy,
                    track=track,
                    rung=rung,
                    student_domain="hlt",
                    initialization="fresh" if track == "cold" else "warm",
                    initialization_parent=initialization_parent,
                    predecessor_logit_teacher=None if rung == 1 else predecessor,
                    representation_logit_teacher=teacher,
                    representation_teacher_domain=teacher_domain,
                    parent_counterpart=f"M{rung}{suffix}",
                    target_bank_identity=bank,
                    deployable=True,
                )
                predecessor = node_id
    return nodes


NODE_REGISTRY: Final[Mapping[str, RepresentationNodeSpec]] = MappingProxyType(
    _build_node_registry()
)


def _control(
    control_id: str,
    *,
    primary: str,
    strategy: str,
    allocation: Mapping[str, float],
    shuffled: bool,
) -> RepresentationControlSpec:
    prefix = _strategy_prefix(strategy)
    return RepresentationControlSpec(
        control_id=control_id,
        paired_primary_node=primary,
        strategy=strategy,
        track="cold",
        rung=5,
        student_domain="hlt",
        initialization="fresh",
        initialization_parent=None,
        predecessor_logit_teacher=f"{prefix}_M4c",
        representation_logit_teacher="D100",
        representation_teacher_domain="d100",
        parent_counterpart="M5c",
        target_bank_identity="D100",
        component_allocation=tuple((name, float(allocation[name])) for name in (
            "jet", "set", "relation"
        )),
        shuffled_representation_targets=shuffled,
        shuffle_map_contract=(
            "HCWDL_REPRESENTATION_SHUFFLE_MAP/v1" if shuffled else None
        ),
        owns_gradient_calibration=True,
        disposition="validation_only_terminal",
        deployable=True,
        finalist_eligible=False,
        confirmation_eligible=False,
        descendants=(),
    )


CONTROL_REGISTRY: Final[Mapping[str, RepresentationControlSpec]] = MappingProxyType({
    "RSET_M5c_JET_ONLY_REP": _control(
        "RSET_M5c_JET_ONLY_REP",
        primary="RSET_M5c",
        strategy=RSET_STRATEGY,
        allocation={"jet": 1.0, "set": 0.0, "relation": 0.0},
        shuffled=False,
    ),
    "RREL_M5c_NO_REL_REP": _control(
        "RREL_M5c_NO_REL_REP",
        primary="RREL_M5c",
        strategy=RREL_STRATEGY,
        allocation={"jet": 0.4, "set": 0.6, "relation": 0.0},
        shuffled=False,
    ),
    "RSET_M5c_WITHIN_CLASS_SHUFFLED_REP": _control(
        "RSET_M5c_WITHIN_CLASS_SHUFFLED_REP",
        primary="RSET_M5c",
        strategy=RSET_STRATEGY,
        allocation={"jet": 0.4, "set": 0.6, "relation": 0.0},
        shuffled=True,
    ),
    "RREL_M5c_WITHIN_CLASS_SHUFFLED_REP": _control(
        "RREL_M5c_WITHIN_CLASS_SHUFFLED_REP",
        primary="RREL_M5c",
        strategy=RREL_STRATEGY,
        allocation={"jet": 0.3, "set": 0.45, "relation": 0.25},
        shuffled=True,
    ),
})


def _expected_primary_ids() -> set[str]:
    return {
        f"{prefix}_M{rung}{suffix}"
        for prefix in ("RSET", "RREL")
        for suffix in ("c", "w")
        for rung in range(1, 7)
    }


def _assert_acyclic(registry: Mapping[str, RepresentationNodeSpec]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("representation ascent graph contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        predecessor = registry[node_id].predecessor_logit_teacher
        if predecessor is not None:
            if predecessor not in registry:
                raise ValueError(f"missing predecessor for {node_id}")
            visit(predecessor)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in registry:
        visit(node_id)


def validate_ascent_graph(
    registry: Mapping[str, RepresentationNodeSpec] = NODE_REGISTRY,
) -> str:
    """Validate every frozen primary invariant and return its graph hash."""

    if len(registry) != 24 or set(registry) != _expected_primary_ids():
        raise ValueError("representation ascent registry must contain exactly 24 nodes")
    if set(registry) != {node.node_id for node in registry.values()}:
        raise ValueError("representation ascent node IDs differ from registry keys")

    for node_id, node in registry.items():
        expected_prefix = _strategy_prefix(node.strategy)
        expected_suffix = _track_suffix(node.track)
        if node.branch_id != f"{expected_prefix}-{node.track}":
            raise ValueError(f"branch identity differs for {node_id}")
        if node.student_domain != "hlt" or not node.deployable:
            raise ValueError(f"representation student {node_id} is not HLT-only deployable")
        if node.rung not in range(1, 7):
            raise ValueError(f"invalid representation rung for {node_id}")
        if node.parent_counterpart != f"M{node.rung}{expected_suffix}":
            raise ValueError(f"parent counterpart differs for {node_id}")

        teacher, domain, bank = _teacher_for(node.track, node.rung)
        if (
            node.representation_logit_teacher != teacher
            or node.representation_teacher_domain != domain
            or node.target_bank_identity != bank
        ):
            raise ValueError(f"privileged teacher or target bank differs for {node_id}")

        expected_predecessor = (
            None
            if node.rung == 1
            else f"{expected_prefix}_M{node.rung - 1}{expected_suffix}"
        )
        if node.predecessor_logit_teacher != expected_predecessor:
            raise ValueError(f"same-branch predecessor differs for {node_id}")
        if node.rung > 1 and node.predecessor_logit_teacher == teacher:
            raise ValueError(f"predecessor supplies representation targets for {node_id}")

        if node.track == "cold":
            if node.initialization != "fresh" or node.initialization_parent is not None:
                raise ValueError(f"cold initialization differs for {node_id}")
        else:
            expected_parent = teacher if node.rung == 1 else expected_predecessor
            if node.initialization != "warm" or node.initialization_parent != expected_parent:
                raise ValueError(f"warm initialization differs for {node_id}")

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
    """Validate the exact four terminal controls and return the registry hash."""

    expected = {
        "RSET_M5c_JET_ONLY_REP",
        "RREL_M5c_NO_REL_REP",
        "RSET_M5c_WITHIN_CLASS_SHUFFLED_REP",
        "RREL_M5c_WITHIN_CLASS_SHUFFLED_REP",
    }
    if len(registry) != 4 or set(registry) != expected:
        raise ValueError("representation control registry must contain exactly four controls")
    if set(registry) != {control.control_id for control in registry.values()}:
        raise ValueError("representation control IDs differ from registry keys")

    for control_id, control in registry.items():
        if control.paired_primary_node not in primary_registry:
            raise ValueError(f"unknown paired primary for {control_id}")
        primary = primary_registry[control.paired_primary_node]
        if (
            primary.track != "cold"
            or primary.rung != 5
            or control.strategy != primary.strategy
            or control.predecessor_logit_teacher != primary.predecessor_logit_teacher
            or control.representation_logit_teacher != "D100"
            or control.target_bank_identity != "D100"
            or control.parent_counterpart != "M5c"
        ):
            raise ValueError(f"control lineage differs for {control_id}")
        if (
            control.student_domain != "hlt"
            or control.initialization != "fresh"
            or control.initialization_parent is not None
            or control.disposition != "validation_only_terminal"
            or not control.deployable
            or control.finalist_eligible
            or control.confirmation_eligible
            or control.descendants
            or not control.owns_gradient_calibration
        ):
            raise ValueError(f"control disposition differs for {control_id}")
        allocation = dict(control.component_allocation)
        if set(allocation) != {"jet", "set", "relation"}:
            raise ValueError(f"control allocation fields differ for {control_id}")
        if abs(sum(allocation.values()) - 1.0) > 1e-12:
            raise ValueError(f"control component allocation differs for {control_id}")
        if control.shuffled_representation_targets != ("SHUFFLED" in control_id):
            raise ValueError(f"shuffle disposition differs for {control_id}")
        if control.shuffled_representation_targets != (control.shuffle_map_contract is not None):
            raise ValueError(f"shuffle-map contract differs for {control_id}")

    return canonical_sha256({
        "contract": CONTROL_REGISTRY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "ascent_graph_sha256": ASCENT_GRAPH_SHA256,
        "controls": [registry[control_id].payload() for control_id in sorted(registry)],
    })


def ascent_graph_artifact(*, parents: Mapping[str, str]) -> dict[str, object]:
    """Build a reusable graph artifact with its authenticated parent lineage."""

    if set(parents) != {"parent_graph", "parent_import"}:
        raise ValueError("representation ascent graph parent keys differ")
    return build_versioned_artifact(
        ASCENT_GRAPH_CONTRACT,
        parents=parents,
        payload={
            "semantic_graph_sha256": ASCENT_GRAPH_SHA256,
            "nodes": [
                NODE_REGISTRY[node_id].payload() for node_id in sorted(NODE_REGISTRY)
            ],
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
        raise ValueError("representation ascent graph artifact differs")
    return digest


def control_registry_artifact(*, ascent_graph_artifact_sha256: str) -> dict[str, object]:
    """Build the separate four-control artifact; controls never enter the graph."""

    return build_versioned_artifact(
        CONTROL_REGISTRY_CONTRACT,
        parents={"representation_ascent_graph": ascent_graph_artifact_sha256},
        payload={
            "semantic_registry_sha256": CONTROL_REGISTRY_SHA256,
            "controls": [
                CONTROL_REGISTRY[control_id].payload()
                for control_id in sorted(CONTROL_REGISTRY)
            ],
        },
    )


def validate_control_registry_artifact(
    value: Mapping[str, object], *, ascent_graph_artifact_sha256: str,
) -> str:
    digest = validate_versioned_artifact(
        value,
        expected_contract=CONTROL_REGISTRY_CONTRACT,
        expected_parents={"representation_ascent_graph": ascent_graph_artifact_sha256},
        required_payload_keys=("semantic_registry_sha256", "controls"),
    )
    payload = value["payload"]
    if set(payload) != {"semantic_registry_sha256", "controls"} or payload != {
        "semantic_registry_sha256": CONTROL_REGISTRY_SHA256,
        "controls": [
            CONTROL_REGISTRY[control_id].payload()
            for control_id in sorted(CONTROL_REGISTRY)
        ],
    }:
        raise ValueError("representation control registry artifact differs")
    return digest


ASCENT_GRAPH_SHA256: Final = validate_ascent_graph()
CONTROL_REGISTRY_SHA256: Final = validate_control_registry()


__all__ = [
    "ASCENT_GRAPH_CONTRACT",
    "ASCENT_GRAPH_SHA256",
    "CONTROL_REGISTRY",
    "CONTROL_REGISTRY_CONTRACT",
    "CONTROL_REGISTRY_SHA256",
    "NODE_REGISTRY",
    "RREL_STRATEGY",
    "RSET_STRATEGY",
    "RepresentationControlSpec",
    "RepresentationNodeSpec",
    "ascent_graph_artifact",
    "control_registry_artifact",
    "validate_ascent_graph",
    "validate_ascent_graph_artifact",
    "validate_control_registry",
    "validate_control_registry_artifact",
]
