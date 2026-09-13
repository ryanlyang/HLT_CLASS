"""Immutable 54-fit graph for JetClass2 salience learned fusion handoff."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from types import MappingProxyType
from typing import Final

from .campaign import coordinate as registered_coordinate
from .salience_learned_contracts import artifact


SPINES: Final = MappingProxyType({
    "DIRECT": ("D000",),
    "COARSE": ("U050", "U100", "D066", "D033", "D000"),
    "DENSE": (
        "U033", "U066", "U100", "D080", "D060", "D040", "D020",
        "D000",
    ),
})
BRANCH_ORDER: Final = tuple(SPINES)
WITHDRAWAL_ALPHA: Final = {
    "kind": "hold_cosine_zero_tail_v1", "hold_through_pass": 10,
    "decay_through_pass": 60, "zero_from_pass": 61,
}
WITHDRAWAL_LOSS: Final = {
    "zero_ce": .25, "zero_kd": .30, "privileged_ce": .15,
    "privileged_kd": .20, "logit_consistency": .05,
    "representation_consistency": .05, "temperature": 2.,
    "representation_blocks": [2, 4, 6, 8],
}
TRAINING: Final = {
    "maximum_passes": 100, "minimum_passes": 60, "patience": 15,
    "patience_clock_start_pass": 60,
    "minimum_auc_delta": 5e-5, "batch_size": 256,
    "optimizer": "AdamW", "betas": [.9, .999], "eps": 1e-8,
    "weight_decay": .01, "peak_lr": 3e-4, "floor_lr": 1.5e-5,
    "warmup_passes": 3, "hold_through_pass": 45,
    "decay_through_pass": 60,
    "schedule": "warmup_hold_cosine_floor_tail_v1",
    "precision": "bf16_forward_fp32_loss", "restore_best": True,
    "rolling_resume": False,
}


def coordinate(name: str) -> tuple[Fraction, Fraction]:
    """Resolve registered rung and exact denominator-25 morph coordinates."""
    if name in {"U000", *(x for values in SPINES.values() for x in values)}:
        return registered_coordinate(name)
    if len(name) == 4 and name[0] in "UD" and name[1:].isdigit():
        amount = int(name[1:])
        if amount % 4 == 0 and 0 <= amount <= 100:
            fraction = Fraction(amount, 100)
            return (fraction, Fraction(0)) if name[0] == "U" else (
                Fraction(1), 1 - fraction,
            )
    raise ValueError(f"Unregistered learned-handoff coordinate: {name}")


def morph_context_for_pass(pass_number: int) -> tuple[str, tuple[Fraction, Fraction]]:
    if type(pass_number) is not int or not 1 <= pass_number <= 100:
        raise ValueError("U000-to-D000 morph pass differs")
    if pass_number <= 26:
        strength = 4 * (pass_number - 1)
        name = f"U{strength:03d}"
    elif pass_number <= 51:
        strength = 100 - 4 * (pass_number - 26)
        name = f"D{strength:03d}"
    else:
        name = "D000"
    return name, coordinate(name)


def learning_rate(pass_position: float) -> float:
    import math
    if not math.isfinite(pass_position) or not 0 < pass_position <= 100:
        raise ValueError("Learning-rate position differs")
    if pass_position <= 3:
        return 3e-4 * pass_position / 3
    if pass_position <= 45:
        return 3e-4
    if pass_position <= 60:
        return 1.5e-5 + (3e-4 - 1.5e-5) * .5 * (
            1 + math.cos(math.pi * (pass_position - 45) / 15)
        )
    return 1.5e-5


def alpha_for_pass(pass_position: float) -> float:
    import math
    if not math.isfinite(pass_position) or pass_position <= 0:
        raise ValueError("Withdrawal pass differs")
    if pass_position <= 10:
        return 1.
    if pass_position >= 61:
        return 0.
    progress = min(1., max(0., (pass_position - 10) / 50))
    return .5 * (1 + math.cos(math.pi * progress))


def _seed(alias: str, domain: str) -> int:
    if domain not in {"initialization", "sampler", "context_architecture"}:
        raise ValueError("Unknown learned-handoff seed domain")
    payload = f"JC2/SALIENCE/LFH/v1/{alias}/{domain}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True)
class Node:
    node_id: str
    role: str
    branch: str
    primary: str
    context: str | None
    teacher: str | None
    seed_alias: str
    initialization_parent: str | None = None
    selection_route: str = "ordinary"
    input_protocol: str = "single_view"

    def payload(self) -> dict:
        u, f = coordinate(self.primary)
        return artifact(
            "NODE_SPEC", node_id=self.node_id, role=self.role,
            branch=self.branch, primary_coordinate=self.primary,
            context_coordinate=self.context, teacher_distribution=self.teacher,
            initialization_parent=self.initialization_parent,
            initialization="fresh" if self.initialization_parent is None else "selected_checkpoint_fresh_optimizer",
            selection_route=self.selection_route,
            input_protocol=self.input_protocol,
            coordinate_exact={"u": [u.numerator, u.denominator],
                              "f": [f.numerator, f.denominator]},
            seed_alias=self.seed_alias,
            initialization_seed=_seed(self.seed_alias, "initialization"),
            sampler_seed=_seed(self.seed_alias, "sampler"),
            context_architecture_seed=(
                None if self.context is None else
                _seed(self.seed_alias, "context_architecture")
            ),
            batch_size=256, maximum_passes=100,
            deployable=self.primary == "D000" and self.selection_route in {
                "ordinary", "alpha_zero",
            }, final_test_accessed=False,
        )


def carrier(branch: str, coordinate_name: str) -> str:
    return "CARRIER_U000" if coordinate_name == "U000" else f"CARRIER_{branch}_{coordinate_name}"


def transition_id(kind: str, branch: str, parent: str, child: str) -> str:
    return f"{kind}_{branch}_{child}_from_{parent}"


def _main_nodes() -> list[Node]:
    rows = [
        Node("M0HLT", "reference_ce", "REFERENCE", "D000", None, None, "REFERENCE/M0HLT"),
        Node("U000", "reference_ce", "REFERENCE", "U000", None, None, "REFERENCE/U000"),
    ]
    for branch, path in SPINES.items():
        parent = "U000"
        for child in path:
            alias = f"{branch}/{child}_from_{parent}"
            source = carrier(branch, parent)
            direct = transition_id("DIRECT", branch, parent, child)
            acquire = transition_id("ACQUIRE", branch, parent, child)
            withdraw = transition_id("WITHDRAW", branch, parent, child)
            rows.extend((
                Node(direct, "direct_kd", branch, child, None, source, alias),
                Node(acquire, "fusion_acquisition", branch, child, parent, source,
                     alias, selection_route="alpha_one", input_protocol="paired_view"),
                Node(withdraw, "fusion_withdrawal", branch, child, parent,
                     f"ACQUISITION_{branch}_{child}_from_{parent}", alias,
                     initialization_parent=acquire, selection_route="alpha_zero",
                     input_protocol="paired_view"),
            ))
            parent = child
    return rows


def _control_nodes() -> list[Node]:
    first_parent, first_child = "U100", "D080"
    terminal_parent, terminal_child = "D020", "D000"
    result = []
    for parent, child in ((first_parent, first_child), (terminal_parent, terminal_child)):
        alias = f"DENSE/{child}_from_{parent}"
        direct = transition_id("DIRECT", "DENSE", parent, child)
        result.extend((
            Node(f"FUSION_LOW_LOW_{child}", "low_low_ce", "CONTROL", child,
                 child, None, alias, selection_route="alpha_one", input_protocol="paired_view"),
            Node(f"LOW_WARM_CONTINUE_{child}", "warm_continue_ce", "CONTROL",
                 child, None, None, alias, initialization_parent=direct),
            Node(f"LOW_PARAMETER_MATCHED_{child}", "parameter_matched_ce",
                 "CONTROL", child, None, None, alias),
        ))
    terminal_alias = "DENSE/D000_from_D020"
    result.extend((
        Node("CE_SINGLE_D000", "cold_single_ce", "CONTROL", "D000", None,
             None, terminal_alias),
        Node("STATIC_U000_D000", "static_global_fusion_ce", "CONTROL", "D000",
             "U000", None, terminal_alias, selection_route="alpha_one",
             input_protocol="paired_view"),
        Node("DIRECT_VIEW_MORPH_U000_TO_D000", "dynamic_view_morph_ce",
             "CONTROL", "D000", "DYNAMIC_U000_TO_D000", None, terminal_alias,
             selection_route="alpha_one", input_protocol="dynamic_paired_view"),
        Node("DIRECT_VIEW_MORPH_WITHDRAW_D000", "morph_withdrawal", "CONTROL",
             "D000", "D000", "MORPH_Q_D000", terminal_alias,
             initialization_parent="DIRECT_VIEW_MORPH_U000_TO_D000",
             selection_route="alpha_zero", input_protocol="paired_view"),
    ))
    return result


NODES: Final = tuple(_main_nodes() + _control_nodes())
NODE_REGISTRY: Final = MappingProxyType({node.node_id: node for node in NODES})
FIT_ORDER: Final = tuple(NODE_REGISTRY)


def graph_payload() -> dict:
    transitions = []
    for branch, path in SPINES.items():
        parent = "U000"
        for child in path:
            transitions.append({"branch": branch, "parent": parent, "child": child})
            parent = child
    return artifact(
        "GRAPH", branch_order=list(BRANCH_ORDER),
        spines={name: list(path) for name, path in SPINES.items()},
        transitions=transitions, fit_order=list(FIT_ORDER),
        nodes=[NODE_REGISTRY[name].payload() for name in FIT_ORDER],
        fresh_fit_count=54, reference_count=2, transition_count=14,
        direct_count=14, acquisition_count=14, withdrawal_count=14,
        control_count=10, controls_duplicated_by_spine=False,
        shared_control_attachment={"first": "DENSE:U100->D080",
                                   "terminal": "DENSE:D020->D000"},
        global_control_path="U000->D000",
        morph_schedule=[morph_context_for_pass(p)[0] for p in range(1, 101)],
        morph_selection_minimum_pass=51,
        primary_branch_owns_classifier=True,
        alpha_zero_skips_context=True, extracted_carriers_only=True,
        next_rung_cold_started=True, final_test_accessed=False,
    )


def recipe_payload() -> dict:
    return artifact(
        "RECIPE", training=dict(TRAINING),
        direct_and_acquisition_loss={
            "ce_weight": .25, "kd_weight": .75, "temperature": 2.,
            "kd_loss": "forward_KL_one_T_squared",
        },
        withdrawal_alpha=dict(WITHDRAWAL_ALPHA),
        withdrawal_loss=dict(WITHDRAWAL_LOSS),
        validation_partitions=["V_checkpoint", "V_diagnostic", "V_report"],
        selection_partition="V_checkpoint", reporting_partition="V_report",
        persistent_hlt=True, durable_particle_views=False,
        durable_hidden_states=False, durable_probability_banks_only=True,
        rolling_resume=False, final_test_accessed=False,
    )


def validate_graph() -> str:
    roles = [node.role for node in NODES]
    schedule = [morph_context_for_pass(p)[0] for p in range(1, 101)]
    if (
        len(NODES) != 54 or len(NODE_REGISTRY) != 54
        or roles.count("reference_ce") != 2
        or roles.count("direct_kd") != 14
        or roles.count("fusion_acquisition") != 14
        or roles.count("fusion_withdrawal") != 14
        or len([x for x in roles if x not in {
            "reference_ce", "direct_kd", "fusion_acquisition", "fusion_withdrawal",
        }]) != 10
        or schedule[0] != "U000" or schedule[25] != "U100"
        or schedule[26] != "D096" or schedule[50:] != ["D000"] * 50
        or TRAINING["patience_clock_start_pass"] != 60
    ):
        raise ValueError("Salience learned-handoff graph differs")
    return graph_payload()["content_hash"]


__all__ = [
    "BRANCH_ORDER", "FIT_ORDER", "NODES", "NODE_REGISTRY", "SPINES",
    "TRAINING", "WITHDRAWAL_ALPHA", "WITHDRAWAL_LOSS", "Node",
    "alpha_for_pass", "carrier", "coordinate", "graph_payload",
    "learning_rate", "morph_context_for_pass", "recipe_payload",
    "transition_id", "validate_graph",
]
