"""Immutable 25-fit graph for Strategy-B adjacent learned fusion handoff."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256
from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_adjacent_output_handoff_graph import (
    SEED_DOMAIN as OUTPUT_HANDOFF_SEED_DOMAIN,
)
from .hcwdl_adjacent_learned_handoff_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, RECIPE_CONTRACT, artifact,
)


COORDINATES: Final = MappingProxyType({
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D080": HomotopyCoordinate(1, 1, 1, 5),
    "D060": HomotopyCoordinate(1, 1, 2, 5),
    "D040": HomotopyCoordinate(1, 1, 3, 5),
    "D020": HomotopyCoordinate(1, 1, 4, 5),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
})
RUNG_ORDER: Final = ("D080", "D060", "D040", "D020", "D000")
PARENT_COORDINATE: Final = MappingProxyType(dict(zip(
    RUNG_ORDER, ("U100", "D080", "D060", "D040", "D020"), strict=True,
)))
WITHDRAWAL_ALPHA: Final = {
    "kind": "hold_cosine_zero_tail_v1", "hold_through_pass": 10,
    "decay_through_pass": 60, "zero_from_pass": 61,
}
WITHDRAWAL_LOSS: Final = {
    "zero_ce": .25, "zero_kd": .30, "privileged_ce": .15,
    "privileged_kd": .20, "logit_consistency": .05,
    "representation_consistency": .05, "temperature": 2.0,
    "representation_blocks": [2, 4, 6, 8],
}
TRAINING: Final = {
    "maximum_passes": 100, "minimum_passes": 60, "patience_passes": 15,
    "minimum_auc_delta": 5e-5, "effective_batch_size": 256,
    "optimizer": "AdamW", "peak_learning_rate": 3e-4,
    "weight_decay": .01, "warmup_passes": 3, "hold_through_pass": 45,
    "decay_through_pass": 60, "learning_rate_floor_fraction": .05,
    "schedule": "warmup_hold_cosine_floor_tail_v1",
    "forward_precision": "bfloat16", "restore_best_checkpoint": True,
}


@dataclass(frozen=True)
class LearnedNode:
    node_id: str
    role: str
    primary_coordinate: str
    context_coordinate: str | None
    teacher_distribution_id: str | None
    ce_weight: float
    kd_weight: float
    temperature: float
    seed_alias: str
    initialization: str = "fresh"
    input_protocol: str = "standard_hlt_v1"
    selection_route: str = "ordinary"

    @property
    def coordinate_name(self) -> str:
        return self.primary_coordinate

    @property
    def coordinate(self) -> HomotopyCoordinate:
        return COORDINATES[self.primary_coordinate]

    @property
    def auxiliary(self) -> str:
        return "none"

    @property
    def representation_carrier_id(self):
        return None

    @property
    def training_passes(self) -> int:
        return 100

    @property
    def batch_size(self) -> int:
        return 256

    @property
    def node_contract(self) -> str:
        return NODE_CONTRACT

    @property
    def deployable(self) -> bool:
        return self.primary_coordinate == "D000" and (
            self.input_protocol == "standard_hlt_v1"
            or self.selection_route == "alpha_zero"
        )

    def payload(self) -> dict[str, object]:
        return {
            "contract": NODE_CONTRACT, "node_id": self.node_id,
            "role": self.role, "track": "LEARNED_FUSION",
            "coordinate_name": self.primary_coordinate,
            "coordinate_exact": self.coordinate.payload(),
            "primary_coordinate": self.primary_coordinate,
            "context_coordinate": self.context_coordinate,
            "teacher_distribution_id": self.teacher_distribution_id,
            "distribution_teacher_id": self.teacher_distribution_id,
            "distribution_teacher_kind": (
                "none" if self.teacher_distribution_id is None else "probability_bank"
            ),
            "representation_carrier_id": None, "auxiliary": "none",
            "ce_weight": self.ce_weight, "kd_weight": self.kd_weight,
            "temperature": self.temperature, "seed_alias": self.seed_alias,
            "architecture_seed_domain": (
                None if self.input_protocol == "standard_hlt_v1"
                else self.seed_alias + "/fusion_context_architecture"
            ),
            "representation_seed_alias": None, "training_passes": 100,
            "validation_every_passes": 1, "batch_size": 256,
            "initialization": self.initialization,
            "input_protocol": self.input_protocol,
            "selection_route": self.selection_route,
            "deployable": self.deployable,
        }


def carrier_distribution(coordinate: str) -> str:
    return "SOURCE_U100" if coordinate == "U100" else f"LEARNED_T_{coordinate}"


def acquisition_distribution(coordinate: str) -> str:
    return f"LEARNED_Q_{coordinate}"


def _seed(coordinate: str) -> str:
    if coordinate == "D000":
        return f"{OUTPUT_HANDOFF_SEED_DOMAIN}/terminal/S1"
    return f"{OUTPUT_HANDOFF_SEED_DOMAIN}/rung/{coordinate}"


def _build_nodes() -> dict[str, LearnedNode]:
    nodes: dict[str, LearnedNode] = {}
    for coordinate in RUNG_ORDER:
        parent = PARENT_COORDINATE[coordinate]
        teacher = carrier_distribution(parent)
        seed = _seed(coordinate)
        nodes[f"LEARNED_DIRECT_{coordinate}"] = LearnedNode(
            f"LEARNED_DIRECT_{coordinate}", "direct_kd", coordinate, None,
            teacher, .25, .75, 2.0, seed,
        )
        nodes[f"LEARNED_ACQUIRE_{coordinate}"] = LearnedNode(
            f"LEARNED_ACQUIRE_{coordinate}", "fusion_acquisition", coordinate,
            parent, teacher, .25, .75, 2.0, seed,
            input_protocol="adjacent_fusion_v1", selection_route="alpha_one",
        )
        nodes[f"LEARNED_WITHDRAW_{coordinate}"] = LearnedNode(
            f"LEARNED_WITHDRAW_{coordinate}", "fusion_withdrawal", coordinate,
            parent, acquisition_distribution(coordinate), .45, .55, 2.0, seed,
            initialization="warm_selected_checkpoint",
            input_protocol="adjacent_withdrawal_v1", selection_route="alpha_zero",
        )
    for coordinate in ("D080", "D000"):
        seed = _seed(coordinate)
        nodes[f"FUSION_LOW_LOW_{coordinate}"] = LearnedNode(
            f"FUSION_LOW_LOW_{coordinate}", "low_low_ce", coordinate,
            coordinate, None, 1., 0., 1., seed,
            input_protocol="adjacent_fusion_v1", selection_route="alpha_one",
        )
        nodes[f"LOW_WARM_CONTINUE_{coordinate}"] = LearnedNode(
            f"LOW_WARM_CONTINUE_{coordinate}", "warm_continue_ce", coordinate,
            None, None, 1., 0., 1., seed,
            initialization="warm_selected_checkpoint",
        )
        nodes[f"LOW_PARAMETER_MATCHED_{coordinate}"] = LearnedNode(
            f"LOW_PARAMETER_MATCHED_{coordinate}", "parameter_matched_ce",
            coordinate, None, None, 1., 0., 1., seed,
        )
    terminal_seed = _seed("D000")
    nodes["CE_SINGLE_D000"] = LearnedNode(
        "CE_SINGLE_D000", "cold_single_ce", "D000", None, None, 1., 0., 1., terminal_seed,
    )
    nodes["STATIC_U100_D000"] = LearnedNode(
        "STATIC_U100_D000", "static_global_fusion_ce", "D000", "U100",
        None, 1., 0., 1., terminal_seed, input_protocol="adjacent_fusion_v1",
        selection_route="alpha_one",
    )
    nodes["DIRECT_VIEW_MORPH_U100_TO_D000"] = LearnedNode(
        "DIRECT_VIEW_MORPH_U100_TO_D000", "dynamic_view_morph_ce", "D000",
        "DYNAMIC_U100_TO_D000", None, 1., 0., 1., terminal_seed,
        input_protocol="adjacent_fusion_v1", selection_route="alpha_one",
    )
    nodes["DIRECT_VIEW_MORPH_WITHDRAW_D000"] = LearnedNode(
        "DIRECT_VIEW_MORPH_WITHDRAW_D000", "morph_withdrawal", "D000", "D000",
        "MORPH_Q_D000", .45, .55, 2., terminal_seed,
        initialization="warm_selected_checkpoint",
        input_protocol="adjacent_withdrawal_v1", selection_route="alpha_zero",
    )
    return nodes


NODE_REGISTRY: Final = MappingProxyType(_build_nodes())
FIT_ORDER: Final = tuple(NODE_REGISTRY)


def distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    result = []
    for node in NODE_REGISTRY.values():
        if node.teacher_distribution_id == distribution_id:
            result.append(node.node_id)
    return tuple(result)


def graph_payload() -> dict[str, object]:
    return artifact({
        "rung_order": list(RUNG_ORDER), "parent_coordinates": dict(PARENT_COORDINATE),
        "fit_order": list(FIT_ORDER),
        "nodes": [NODE_REGISTRY[name].payload() for name in FIT_ORDER],
        "fresh_fit_count": 25, "acquisition_count": 5,
        "withdrawal_count": 5, "direct_kd_count": 5,
        "cross_attention_blocks": [2, 4, 6, 8],
        "primary_branch_owns_classifier": True,
        "matched_primary_and_separate_context_seed_domains": True,
        "alpha_zero_skips_context_execution": True,
        "next_rung_is_cold_started": True, "next_rung_uses_logits_only": True,
        "morph_context_numerator_over_50": list(range(0, 51)),
        "morph_pass_1_is_explicit_U100": True,
        "morph_checkpoint_selection_minimum_pass": 51,
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def recipe_payload() -> dict[str, object]:
    return artifact({
        "training": dict(TRAINING), "withdrawal_alpha": dict(WITHDRAWAL_ALPHA),
        "withdrawal_loss": dict(WITHDRAWAL_LOSS),
        "acquisition_loss": {"ce_weight": .25, "kd_weight": .75, "temperature": 2.},
        "direct_loss": {"ce_weight": .25, "kd_weight": .75, "temperature": 2.},
        "morph_schedule": {
            "pass_1": "U100", "passes_2_through_51": "D(100-2*(pass-1))",
            "passes_52_through_100": "D000", "exact_denominator": 50,
            "checkpoint_selection_minimum_pass": 51,
        },
        "validation_selection_role": "V_checkpoint",
        "reporting_role": "V_report", "batch_size": 256,
        "durable_particle_views": False, "durable_hidden_states": False,
        "durable_probability_banks_only": True, "rolling_resume": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


def validate_graph() -> str:
    roles = [node.role for node in NODE_REGISTRY.values()]
    if (
        len(NODE_REGISTRY) != 25 or len(set(NODE_REGISTRY)) != 25
        or roles.count("fusion_acquisition") != 5
        or roles.count("fusion_withdrawal") != 5
        or roles.count("direct_kd") != 5
        or any(NODE_REGISTRY[f"LEARNED_WITHDRAW_{c}"].selection_route != "alpha_zero" for c in RUNG_ORDER)
    ):
        raise ValueError("adjacent learned-handoff graph differs")
    return GRAPH_SHA256


__all__ = [
    "COORDINATES", "FIT_ORDER", "GRAPH_SHA256", "LearnedNode", "NODE_REGISTRY",
    "PARENT_COORDINATE", "RUNG_ORDER", "TRAINING", "WITHDRAWAL_ALPHA",
    "WITHDRAWAL_LOSS", "acquisition_distribution", "carrier_distribution",
    "distribution_consumers", "graph_payload", "recipe_payload", "validate_graph",
]
