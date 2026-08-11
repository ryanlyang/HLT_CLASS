"""Exact paired RSET/RREL graph over the factorized HCWDL U/D homotopy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_homotopy_graph import DOMAINS as HOMOTOPY_DOMAINS
from .hcwdl_homotopy_representation_contracts import (
    FIT_COUNT, GRAPH_CONTRACT, NODE_SPEC_CONTRACT, TARGET_BANK_COUNT,
)
from .hcwdl_ladder import TeacherSpec
from .hcwdl_representation_graph import RREL_STRATEGY, RSET_STRATEGY
from .training import LossConfiguration, derive_seed


GRAPH_LABEL: Final = "HCWDL_HOMOTOPY_REPRESENTATION_KD"
STRATEGIES: Final = (RSET_STRATEGY, RREL_STRATEGY)
TRACK_PREFIX: Final = {RSET_STRATEGY: "F_RSET", RREL_STRATEGY: "F_RREL"}
TRAINING_PASSES: Final = 60
VALIDATIONS: Final = 60


@dataclass(frozen=True)
class HomotopyRepresentationNodeSpec:
    node_id: str
    strategy: str
    track: str
    transition_index: int
    stage: str
    student_domain: str
    teacher: TeacherSpec
    seed_alias: str
    temperature: float
    target_bank_identity: str
    parent_counterpart: str
    initialization: str = "fresh"
    initialization_parent: None = None
    deployable: bool = False

    def payload(self) -> dict[str, object]:
        value = asdict(self)
        value["teacher"] = asdict(self.teacher)
        return {"contract": NODE_SPEC_CONTRACT, "schema_version": 1, **value}


def _path_rows() -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for index in range(1, 11):
        label = f"U{index * 10:03d}"
        rows.append((label, f"u{index * 10:03d}", "upper"))
    for index in range(1, 11):
        level = 100 - index * 10
        rows.append((f"D{level}", f"d{level}f", "down"))
    rows.append(("M1", "hlt", "born_again"))
    return tuple(rows)


PATH_ROWS: Final = _path_rows()


def _build_registry() -> dict[str, HomotopyRepresentationNodeSpec]:
    nodes: dict[str, HomotopyRepresentationNodeSpec] = {}
    for strategy in STRATEGIES:
        prefix = TRACK_PREFIX[strategy]
        predecessor = "TOFF"
        predecessor_domain = "toff"
        for index, (label, domain, stage) in enumerate(PATH_ROWS, start=1):
            node_id = f"{prefix}_{label}"
            temperature = 1.0 if stage == "born_again" else 2.0
            counterpart = "M1F" if label == "M1" else (
                f"{label}F" if label.startswith("D") else label
            )
            nodes[node_id] = HomotopyRepresentationNodeSpec(
                node_id=node_id, strategy=strategy, track="factorized_cold",
                transition_index=index, stage=stage, student_domain=domain,
                teacher=TeacherSpec(predecessor, predecessor_domain, "sole"),
                seed_alias=f"transition_{index:02d}", temperature=temperature,
                target_bank_identity=predecessor,
                parent_counterpart=counterpart,
                deployable=domain == "hlt",
            )
            predecessor, predecessor_domain = node_id, domain
    return nodes


NODE_REGISTRY: Final[Mapping[str, HomotopyRepresentationNodeSpec]] = MappingProxyType(
    _build_registry()
)
DOMAINS: Final = HOMOTOPY_DOMAINS


def ordered_nodes(strategy: str) -> tuple[HomotopyRepresentationNodeSpec, ...]:
    if strategy not in STRATEGIES:
        raise ValueError("unknown HCWDL-U-RKD strategy")
    return tuple(sorted(
        (node for node in NODE_REGISTRY.values() if node.strategy == strategy),
        key=lambda node: node.transition_index,
    ))


def target_bank_registry() -> dict[str, tuple[str, ...]]:
    consumers: dict[str, list[str]] = {}
    for node in NODE_REGISTRY.values():
        consumers.setdefault(node.target_bank_identity, []).append(node.node_id)
    return {key: tuple(sorted(value)) for key, value in sorted(consumers.items())}


def validate_graph(registry: Mapping[str, HomotopyRepresentationNodeSpec] = NODE_REGISTRY) -> str:
    if len(registry) != FIT_COUNT or len(set(registry)) != FIT_COUNT:
        raise ValueError("HCWDL-U-RKD graph must contain exactly 42 fits")
    for strategy in STRATEGIES:
        rows = ordered_nodes(strategy)
        if len(rows) != 21 or [row.transition_index for row in rows] != list(range(1, 22)):
            raise ValueError("HCWDL-U-RKD strategy transition registry differs")
        if rows[0].teacher != TeacherSpec("TOFF", "toff", "sole"):
            raise ValueError("first representation homotopy node must use native TOFF")
        for parent, child in zip(rows, rows[1:]):
            if child.teacher != TeacherSpec(parent.node_id, parent.student_domain, "sole"):
                raise ValueError("HCWDL-U-RKD immediate predecessor routing differs")
    if len(target_bank_registry()) != TARGET_BANK_COUNT:
        raise ValueError("HCWDL-U-RKD must register exactly 41 logical target banks")
    if target_bank_registry().get("TOFF") != (
        "F_RREL_U010", "F_RSET_U010",
    ):
        raise ValueError("shared TOFF target consumers differ")
    if any(
        node.initialization != "fresh" or node.initialization_parent is not None
        for node in registry.values()
    ):
        raise ValueError("every HCWDL-U-RKD node must be cold-started")
    if any(node.student_domain not in DOMAINS for node in registry.values()):
        raise ValueError("HCWDL-U-RKD graph contains an unknown view domain")
    return canonical_sha256(graph_payload(registry))


def graph_payload(
    registry: Mapping[str, HomotopyRepresentationNodeSpec] = NODE_REGISTRY,
) -> dict[str, object]:
    return {
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "fit_count": FIT_COUNT, "target_bank_count": TARGET_BANK_COUNT,
        "trained_u000": False, "warm_node_count": 0,
        "terminal_candidates": ["F_RSET_M1", "F_RREL_M1"],
        "nodes": [node.payload() for node in registry.values()],
    }


GRAPH_SHA256: Final = validate_graph()


def graph_artifact() -> dict[str, object]:
    return with_content_hash(graph_payload())


def resolved_base_loss(node_id: str) -> LossConfiguration:
    node = NODE_REGISTRY[node_id]
    # Target-bank logits are routed through the single hlt-teacher slot in the
    # representation engine. Temperature remains explicitly graph-bound.
    return LossConfiguration.for_mixture(
        arm=f"HCWDL_U_RKD_{node_id}", ce=0.25, hlt_kd=0.75,
        privileged_kd=0.0, hlt_temperature=node.temperature,
        privileged_temperature=node.temperature,
    )


def seed_for_node(replicate_seed: int, node_id: str, *, purpose: str) -> int:
    if node_id not in NODE_REGISTRY or not purpose:
        raise ValueError("HCWDL-U-RKD seed request differs")
    return derive_seed(
        int(replicate_seed),
        f"hcwdl_uj/{purpose}/{NODE_REGISTRY[node_id].seed_alias}",
    )


__all__ = [
    "DOMAINS", "GRAPH_LABEL", "GRAPH_SHA256", "HomotopyRepresentationNodeSpec",
    "NODE_REGISTRY", "PATH_ROWS", "STRATEGIES", "TRAINING_PASSES", "VALIDATIONS",
    "graph_artifact", "ordered_nodes", "resolved_base_loss", "seed_for_node",
    "target_bank_registry", "validate_graph",
]
