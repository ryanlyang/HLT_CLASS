"""Exact 80-fit graph and recipe overlay for the validation-only HCWDL-UJ study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256, require_sha256, with_content_hash

from .hcwdl_homotopy_contracts import (
    GRAPH_CONTRACT, NODE_SPEC_CONTRACT, RECIPE_CONTRACT, coordinate_payload,
)
from .hcwdl_ladder import TeacherSpec
from .training import LossConfiguration, derive_seed


GRAPH_LABEL: Final = "HCWDL_STRUCTURAL_FEATURE_HOMOTOPY"
TRAINING_PASSES: Final = 60
VALIDATIONS: Final = 60


@dataclass(frozen=True)
class HomotopyNodeSpec:
    node_id: str
    track: str
    stage: str
    student_domain: str
    initialization: str
    initialization_parent: str | None
    teachers: tuple[TeacherSpec, ...]
    loss_kind: str
    deployable: bool
    transition_index: int | None
    seed_alias: str
    temperature: float

    def payload(self) -> dict[str, object]:
        # ``dataclasses.asdict`` preserves tuples.  That is semantically fine
        # for canonical JSON hashing, but it is not stable across an actual
        # JSON publication/reload: ``teachers`` returns as a list.  Campaign
        # validation compares the immutable graph artifact to these payloads,
        # so the public contract representation must be JSON-native before it
        # is hashed or written.
        value = asdict(self)
        value["teachers"] = [asdict(teacher) for teacher in self.teachers]
        return {
            "contract": NODE_SPEC_CONTRACT,
            "schema_version": 1,
            **value,
        }


def _domain(input_key: str, s: float | None, f: float | None, *, deployable: bool = False):
    return {"input": input_key, "s": s, "f": f, "deployable": deployable}


def _build_domains() -> dict[str, dict[str, object]]:
    result = {
        "toff": _domain("toff", None, None),
        "p0": _domain("privileged", 0.0, 0.0),
        "d100": _domain("privileged", 1.0, 0.0),
        "hlt": _domain("hlt", 1.0, 1.0, deployable=True),
    }
    for index in range(1, 11):
        result[f"u{index * 10:03d}"] = _domain("privileged", index / 10, 0.0)
        result[f"d{100-index * 10}f"] = _domain("privileged", 1.0, index / 10,
                                                 deployable=index == 10)
    for index in range(1, 21):
        result[f"j{index * 5:03d}"] = _domain(
            "privileged" if index < 20 else "hlt", index / 20, index / 20,
            deployable=index == 20,
        )
    return result


DOMAINS: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
    {key: MappingProxyType(value) for key, value in _build_domains().items()}
)


def _build_registry() -> dict[str, HomotopyNodeSpec]:
    nodes: dict[str, HomotopyNodeSpec] = {}

    def add(
        node_id: str, *, track: str, stage: str, domain: str,
        teacher: tuple[str, str] | None, transition: int | None,
        seed_alias: str, temperature: float = 2.0, ce: bool = False,
    ) -> None:
        teachers = () if teacher is None else (TeacherSpec(teacher[0], teacher[1], "sole"),)
        nodes[node_id] = HomotopyNodeSpec(
            node_id=node_id, track=track, stage=stage, student_domain=domain,
            initialization="fresh", initialization_parent=None, teachers=teachers,
            loss_kind="ce" if ce else "ce_kd", deployable=domain == "hlt",
            transition_index=transition, seed_alias=seed_alias,
            temperature=float(temperature),
        )

    # Explicit paired/adapter/direct controls (seven fits).
    add("M0paired", track="control", stage="root", domain="hlt", teacher=None,
        transition=None, seed_alias="paired_m0_root", temperature=1.0, ce=True)
    add("P0CE", track="control", stage="adapter", domain="p0", teacher=None,
        transition=None, seed_alias="p0_pair_root", temperature=1.0, ce=True)
    add("P0KD", track="control", stage="adapter", domain="p0",
        teacher=("TOFF", "toff"), transition=None, seed_alias="p0_pair_root")
    add("U010P0KD", track="control", stage="adapter", domain="u010",
        teacher=("P0KD", "p0"), transition=1, seed_alias="transition_01")
    add("D100direct", track="control", stage="direct", domain="d100",
        teacher=("TOFF", "toff"), transition=10, seed_alias="transition_10")
    add("D0direct", track="control", stage="direct", domain="hlt",
        teacher=("TOFF", "toff"), transition=20, seed_alias="transition_20")
    add("M0self", track="control", stage="born_again", domain="hlt",
        teacher=("M0paired", "hlt"), transition=21, seed_alias="transition_21",
        temperature=1.0)

    previous, previous_domain = "TOFF", "toff"
    for index in range(1, 11):
        node_id, domain = f"U{index * 10:03d}", f"u{index * 10:03d}"
        add(node_id, track="factorized", stage="upper", domain=domain,
            teacher=(previous, previous_domain), transition=index,
            seed_alias=f"transition_{index:02d}")
        previous, previous_domain = node_id, domain
    for index in range(1, 11):
        level = 100 - index * 10
        node_id, domain = f"D{level}F", f"d{level}f"
        add(node_id, track="factorized", stage="down", domain=domain,
            teacher=(previous, previous_domain), transition=10 + index,
            seed_alias=f"transition_{10 + index:02d}")
        previous, previous_domain = node_id, domain
    add("M1F", track="factorized", stage="born_again", domain="hlt",
        teacher=("D0F", "hlt"), transition=21, seed_alias="transition_21",
        temperature=1.0)

    previous, previous_domain = "TOFF", "toff"
    for index in range(1, 21):
        node_id, domain = f"J{index * 5:03d}", f"j{index * 5:03d}"
        add(node_id, track="joint", stage="joint", domain=domain,
            teacher=(previous, previous_domain), transition=index,
            seed_alias=f"transition_{index:02d}")
        previous, previous_domain = node_id, domain
    add("M1J", track="joint", stage="born_again", domain="hlt",
        teacher=("J100", "hlt"), transition=21, seed_alias="transition_21",
        temperature=1.0)

    previous, previous_domain = "TOFF", "toff"
    for index in range(1, 11):
        node_id = f"S100_{index:02d}"
        add(node_id, track="stationary_d100", stage="stationary", domain="d100",
            teacher=(previous, previous_domain), transition=index,
            seed_alias=f"transition_{index:02d}")
        previous, previous_domain = node_id, "d100"

    previous, previous_domain = "TOFF", "toff"
    for index in range(1, 22):
        node_id = f"S0_{index:02d}"
        add(node_id, track="stationary_hlt", stage="stationary", domain="hlt",
            teacher=(previous, previous_domain), transition=index,
            seed_alias=f"transition_{index:02d}",
            temperature=1.0 if index == 21 else 2.0)
        previous, previous_domain = node_id, "hlt"
    return nodes


NODE_REGISTRY: Final[Mapping[str, HomotopyNodeSpec]] = MappingProxyType(_build_registry())


def validate_graph(
    registry: Mapping[str, HomotopyNodeSpec] = NODE_REGISTRY,
) -> str:
    if len(registry) != 80 or len(set(registry)) != 80:
        raise ValueError("HCWDL-UJ graph must contain exactly 80 new fits")
    expected_tracks = {
        "control": 7, "factorized": 21, "joint": 21,
        "stationary_d100": 10, "stationary_hlt": 21,
    }
    actual = {name: sum(node.track == name for node in registry.values()) for name in expected_tracks}
    if actual != expected_tracks:
        raise ValueError("HCWDL-UJ graph track counts differ")
    if any(node.initialization != "fresh" or node.initialization_parent is not None for node in registry.values()):
        raise ValueError("HCWDL-UJ nodes must all be cold-started")
    if any(node.student_domain not in DOMAINS for node in registry.values()):
        raise ValueError("HCWDL-UJ graph contains an unknown student domain")
    if NODE_REGISTRY["U010"].teachers[0] != TeacherSpec("TOFF", "toff", "sole"):
        raise ValueError("U010 must receive native TOFF targets")
    if NODE_REGISTRY["J005"].teachers[0] != TeacherSpec("TOFF", "toff", "sole"):
        raise ValueError("J005 must receive native TOFF targets")
    if NODE_REGISTRY["D90F"].teachers[0].node_id != "U100":
        raise ValueError("factorized upper/lower join differs")
    if NODE_REGISTRY["D0F"].student_domain != "d0f" or NODE_REGISTRY["J100"].student_domain != "j100":
        raise ValueError("HCWDL-UJ HLT endpoints differ")
    if any(node.temperature != (1.0 if node.node_id in {"M1F", "M1J", "M0self", "S0_21"} else 2.0)
           for node in registry.values() if node.loss_kind == "ce_kd"):
        raise ValueError("HCWDL-UJ temperature routing differs")
    payload = {
        "contract": GRAPH_CONTRACT,
        "schema_version": 1,
        "coordinate_sha256": coordinate_payload()["content_hash"],
        "fit_count": 80,
        "nodes": [node.payload() for node in registry.values()],
    }
    return canonical_sha256(payload)


GRAPH_SHA256: Final = validate_graph()


def resolved_loss(node_id: str) -> LossConfiguration:
    if node_id not in NODE_REGISTRY:
        raise ValueError("unknown HCWDL-UJ node")
    node = NODE_REGISTRY[node_id]
    if node.loss_kind == "ce":
        return LossConfiguration(
            arm=f"HCWDL_UJ_{node_id}_CE", ce=1.0, hlt_kd=0.0,
            privileged_kd=0.0, temperature=1.0,
            privileged_temperature=1.0,
        )
    teacher_domain = node.teachers[0].domain
    hlt_kd = 0.75 if teacher_domain == "hlt" else 0.0
    privileged_kd = 0.0 if teacher_domain == "hlt" else 0.75
    # S0_02..S0_20 are deliberately T2 despite having HLT-domain teachers.
    return LossConfiguration.for_mixture(
        arm=f"HCWDL_UJ_{node_id}_SINGLE", ce=0.25,
        hlt_kd=hlt_kd, privileged_kd=privileged_kd,
        hlt_temperature=node.temperature,
        privileged_temperature=node.temperature,
    )


def seed_for_node(replicate_seed: int, node_id: str, *, purpose: str) -> int:
    if node_id not in NODE_REGISTRY or not purpose:
        raise ValueError("HCWDL-UJ seed request differs")
    return derive_seed(
        int(replicate_seed), f"hcwdl_uj/{purpose}/{NODE_REGISTRY[node_id].seed_alias}",
    )


def build_recipe_overlay(*, parent_recipe_sha256: str) -> dict[str, Any]:
    parent_hash = require_sha256(parent_recipe_sha256, name="parent HCWDL recipe SHA-256")
    rows = []
    for node in NODE_REGISTRY.values():
        loss = resolved_loss(node.node_id)
        rows.append({
            "node_id": node.node_id,
            "student_domain": node.student_domain,
            "teacher": None if not node.teachers else asdict(node.teachers[0]),
            "loss": asdict(loss),
            "peak_learning_rate": 3.0e-4,
            "passes": TRAINING_PASSES,
            "validation_checks": VALIDATIONS,
            "seed_alias": node.seed_alias,
        })
    return with_content_hash({
        "contract": RECIPE_CONTRACT,
        "schema_version": 1,
        "parent_recipe_sha256": parent_hash,
        "graph_sha256": GRAPH_SHA256,
        "class_weighting_policy": "unweighted_per_jet_population_mean_v1",
        "class_weights": [1.0] * 15,
        "rows": rows,
    })


def validate_recipe_overlay(value: Mapping[str, Any], *, parent_recipe_sha256: str | None = None) -> str:
    from hlt_classification.data.cache_contracts import validate_content_hash
    digest = validate_content_hash(value, expected_contract=RECIPE_CONTRACT, expected_schema_version=1)
    parent = str(value.get("parent_recipe_sha256"))
    require_sha256(parent, name="parent HCWDL recipe SHA-256")
    if parent_recipe_sha256 is not None and parent != require_sha256(
        parent_recipe_sha256, name="expected parent recipe SHA-256",
    ):
        raise ValueError("HCWDL-UJ overlay parent recipe differs")
    if value != build_recipe_overlay(parent_recipe_sha256=parent):
        raise ValueError("HCWDL-UJ recipe overlay differs")
    return digest


__all__ = [
    "DOMAINS", "GRAPH_LABEL", "GRAPH_SHA256", "HomotopyNodeSpec",
    "NODE_REGISTRY", "TRAINING_PASSES", "VALIDATIONS", "build_recipe_overlay",
    "resolved_loss", "seed_for_node", "validate_graph", "validate_recipe_overlay",
]
