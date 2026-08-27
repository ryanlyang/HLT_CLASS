"""Predeclared full-data D033E-to-D000 optimization-budget screen."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_graph import NODE_REGISTRY, Tri60Node
from .hcwdl_tri60_d000_budget_screen_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, artifact, validate_artifact,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI60-D000-OPTIMIZATION-BUDGET-SCREEN"
SOURCE_NODE_ID: Final = "LOGIT_D000_from_D033E"
TEACHER_ID: Final = "LOGIT_D033E"
IMPORTED_CONTROL_ID: Final = "SOURCE_LOGIT_D000_FROM_D033E"
SEED_ALIAS: Final = NODE_REGISTRY[SOURCE_NODE_ID].seed_alias


def _constant_loss(passes: int) -> dict[str, Any]:
    return {
        "kind": "constant_v1", "ce_weight": .25, "kd_weight": .75,
    }


def _kd_first(passes: int, *, through: int, ce: float) -> dict[str, Any]:
    return {
        "kind": "piecewise_constant_v1",
        "segments": [
            {"through_pass": through, "ce_weight": ce, "kd_weight": 1 - ce},
            {"through_pass": passes, "ce_weight": .25, "kd_weight": .75},
        ],
    }


def _standard_lr(*, floor: float = .05) -> dict[str, Any]:
    return {
        "kind": "fractional_warmup_cosine_v1",
        "warmup_fraction": .05, "minimum_lr_fraction": floor,
    }


def _hold_lr(*, hold: int) -> dict[str, Any]:
    return {
        "kind": "warmup_hold_cosine_v1", "warmup_passes": 3,
        "hold_through_pass": hold, "minimum_lr_fraction": .05,
    }


@dataclass(frozen=True)
class BudgetCondition:
    condition_id: str
    axis: str
    passes: int
    peak_learning_rate: float
    learning_rate_schedule: Mapping[str, Any]
    loss_schedule: Mapping[str, Any]

    @property
    def node(self) -> Tri60Node:
        return Tri60Node(
            node_id=self.condition_id, track="LOGIT_BUDGET_SCREEN",
            coordinate_name="D000", distribution_teacher_id=TEACHER_ID,
            distribution_teacher_kind="probability_bank",
            representation_carrier_id=None, auxiliary="none",
            ce_weight=.25, kd_weight=.75, temperature=2.0,
            seed_alias=SEED_ALIAS, representation_seed_alias=None,
            training_passes=self.passes, batch_size=256,
            initialization="fresh", node_contract=NODE_CONTRACT,
        )

    def payload(self) -> dict[str, Any]:
        return {
            **self.node.payload(), "axis": self.axis,
            "peak_learning_rate": self.peak_learning_rate,
            "learning_rate_schedule": dict(self.learning_rate_schedule),
            "loss_schedule": dict(self.loss_schedule),
        }


def _condition(
    condition_id: str, *, axis: str, passes: int = 60,
    peak: float = 3e-4, hold: int | None = None,
    floor: float = .05, kd_first_through: int | None = None,
    kd_first_ce: float = .10,
) -> BudgetCondition:
    lr = _standard_lr(floor=floor) if hold is None else _hold_lr(hold=hold)
    loss = (
        _constant_loss(passes) if kd_first_through is None
        else _kd_first(passes, through=kd_first_through, ce=kd_first_ce)
    )
    return BudgetCondition(
        condition_id=condition_id, axis=axis, passes=passes,
        peak_learning_rate=peak, learning_rate_schedule=lr,
        loss_schedule=loss,
    )


CONDITIONS: Final = (
    _condition("P60_H20_LR3E4", axis="decay_shape", hold=20),
    _condition("P60_H30_LR3E4", axis="decay_shape", hold=30),
    _condition("P60_H40_LR3E4", axis="decay_shape", hold=40),
    _condition("P60_H45_LR3E4", axis="decay_shape", hold=45),
    _condition("P60_STD_F20_LR3E4", axis="decay_floor", floor=.20),
    _condition("P60_H30_LR1P5E4", axis="peak_lr", hold=30, peak=1.5e-4),
    _condition("P60_H30_LR2E4", axis="peak_lr", hold=30, peak=2e-4),
    _condition("P60_H30_LR4E4", axis="peak_lr", hold=30, peak=4e-4),
    _condition("P60_H30_LR5E4", axis="peak_lr", hold=30, peak=5e-4),
    _condition(
        "P60_H30_KD90_P05", axis="kd_first", hold=30,
        kd_first_through=5,
    ),
    _condition(
        "P60_H30_KD90_P10", axis="kd_first", hold=30,
        kd_first_through=10,
    ),
    _condition(
        "P60_H30_KD90_P20", axis="kd_first", hold=30,
        kd_first_through=20,
    ),
    _condition(
        "P60_H30_KD95_P10", axis="kd_first", hold=30,
        kd_first_through=10, kd_first_ce=.05,
    ),
    _condition("P90_STD_LR3E4", axis="horizon", passes=90),
    _condition("P90_H45_LR3E4", axis="horizon", passes=90, hold=45),
    _condition("P90_H60_LR3E4", axis="horizon", passes=90, hold=60),
    _condition(
        "P90_H45_KD90_P15", axis="horizon_kd_first", passes=90,
        hold=45, kd_first_through=15,
    ),
)
if len(CONDITIONS) != 17 or len({row.condition_id for row in CONDITIONS}) != 17:
    raise RuntimeError("TRI60 D000 budget-screen condition registry differs")

CONDITION_REGISTRY: Final[Mapping[str, BudgetCondition]] = MappingProxyType({
    row.condition_id: row for row in CONDITIONS
})
FIT_ORDER: Final = tuple(CONDITION_REGISTRY)


def graph_payload() -> dict[str, Any]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "source_node_id": SOURCE_NODE_ID, "teacher_id": TEACHER_ID,
        "imported_control_id": IMPORTED_CONTROL_ID,
        "condition_order": list(FIT_ORDER),
        "conditions": [row.payload() for row in CONDITIONS],
        "condition_count": 18, "fresh_fit_count": 17,
        "paired_seed_alias": SEED_ALIAS,
        "primary_budget_passes": 60,
        "compromise_budget_passes": 90,
        "temperature": 2.0, "batch_size": 256,
        "selection_policy": "macro_auc_ce_logr50_earliest_update_v1",
        "automatic_followup_selection": False,
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def validate_graph() -> str:
    value = graph_payload()
    digest = validate_artifact(value, contract=GRAPH_CONTRACT)
    source = NODE_REGISTRY[SOURCE_NODE_ID]
    if (
        digest != GRAPH_SHA256
        or tuple(CONDITION_REGISTRY) != FIT_ORDER
        or source.distribution_teacher_id != TEACHER_ID
        or source.ce_weight != .25 or source.kd_weight != .75
        or source.temperature != 2.0
    ):
        raise ValueError("TRI60 D000 budget-screen graph differs")
    if (
        any(row.node.coordinate_name != "D000" for row in CONDITIONS)
        or any(row.node.seed_alias != SEED_ALIAS for row in CONDITIONS)
        or canonical_sha256(value) != canonical_sha256(graph_payload())
    ):
        raise ValueError("TRI60 D000 budget-screen node registry differs")
    return digest


__all__ = [
    "CAMPAIGN_LABEL", "CONDITIONS", "CONDITION_REGISTRY", "FIT_ORDER",
    "GRAPH_SHA256", "IMPORTED_CONTROL_ID", "SEED_ALIAS", "SOURCE_NODE_ID",
    "TEACHER_ID", "BudgetCondition", "graph_payload", "validate_graph",
]
