"""Immutable twenty-condition LOGIT M1 compression screen."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_graph import NODE_REGISTRY as SOURCE_NODES, Tri60Node
from .hcwdl_tri60_m1_screen_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, artifact, validate_artifact,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI60-M1-COMPRESSION-SCREEN"
SEED_ALIAS: Final = SOURCE_NODES["M1_LOGIT"].seed_alias
IMPORTED_CONTROL_ID: Final = "SOURCE_M1_LOGIT"
TEACHER_ID: Final = "LOGIT_D000E"
WARM_SOURCE_ID: Final = "LOGIT_D000_from_D033E"


@dataclass(frozen=True)
class ScreenCondition:
    condition_id: str
    initialization: str
    ce_weight: float
    kd_weight: float
    temperature: float
    peak_learning_rate: float
    initialization_source: str | None
    loss_schedule: Mapping[str, Any]

    @property
    def node(self) -> Tri60Node:
        return Tri60Node(
            node_id=self.condition_id, track="M1_SCREEN",
            coordinate_name="D000", distribution_teacher_id=TEACHER_ID,
            distribution_teacher_kind="probability_bank",
            representation_carrier_id=None, auxiliary="none",
            ce_weight=self.ce_weight, kd_weight=self.kd_weight,
            temperature=self.temperature, seed_alias=SEED_ALIAS,
            representation_seed_alias=None, training_passes=60,
            batch_size=256, initialization=self.initialization,
            node_contract=NODE_CONTRACT,
        )

    def payload(self) -> dict[str, Any]:
        return {
            **self.node.payload(),
            "peak_learning_rate": self.peak_learning_rate,
            "initialization_source": self.initialization_source,
            "loss_schedule": dict(self.loss_schedule),
        }


def _constant(ce: float) -> dict[str, Any]:
    return {"kind": "constant_v1", "ce_weight": ce, "kd_weight": 1.0 - ce}


def _condition(
    *, initialization: str, ce: float, temperature: float, lr: float,
    source: str | None = None, suffix: str | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> ScreenCondition:
    init = {"fresh": "COLD", "warm_selected_checkpoint": "WARM",
            "polish_selected_checkpoint": "POLISH"}[initialization]
    loss = f"C{round(100 * ce):02d}P{round(100 * (1-ce)):02d}"
    lr_name = {3e-4: "LR3E4", 1e-4: "LR1E4", 5e-5: "LR5E5"}[lr]
    condition_id = suffix or f"{init}_{loss}_T{int(temperature)}_{lr_name}"
    return ScreenCondition(
        condition_id=condition_id, initialization=initialization,
        ce_weight=ce, kd_weight=1.0 - ce, temperature=temperature,
        peak_learning_rate=lr, initialization_source=source,
        loss_schedule=dict(schedule or _constant(ce)),
    )


def _conditions() -> tuple[ScreenCondition, ...]:
    rows = []
    # The cold C10P90/T1/3e-4 cell is imported as SOURCE_M1_LOGIT.
    for ce in (.10, .25):
        for temperature in (1.0, 2.0):
            for lr in (3e-4, 1e-4):
                if (ce, temperature, lr) != (.10, 1.0, 3e-4):
                    rows.append(_condition(
                        initialization="fresh", ce=ce,
                        temperature=temperature, lr=lr,
                    ))
    for ce in (.10, .25):
        for temperature in (1.0, 2.0):
            for lr in (1e-4, 5e-5):
                rows.append(_condition(
                    initialization="warm_selected_checkpoint", ce=ce,
                    temperature=temperature, lr=lr, source=WARM_SOURCE_ID,
                ))
    rows.extend((
        _condition(
            initialization="fresh", ce=.50, temperature=2.0, lr=1e-4,
        ),
        _condition(
            initialization="warm_selected_checkpoint", ce=.50,
            temperature=2.0, lr=5e-5, source=WARM_SOURCE_ID,
        ),
        _condition(
            initialization="fresh", ce=.10, temperature=2.0, lr=1e-4,
            suffix="COLD_RAMP_C75P25_TO_C10P90_T2_LR1E4",
            schedule={
                "kind": "linear_ce_to_kd_v1",
                "initial_ce_weight": .75, "initial_kd_weight": .25,
                "hold_through_pass": 5.0,
                "target_ce_weight": .10, "target_kd_weight": .90,
                "target_at_pass": 15.0,
            },
        ),
        _condition(
            initialization="polish_selected_checkpoint", ce=.25,
            temperature=2.0, lr=5e-5, source=IMPORTED_CONTROL_ID,
            suffix="POLISH_SOURCE_M1_C25P75_T2_LR5E5",
        ),
    ))
    if len(rows) != 19 or len({row.condition_id for row in rows}) != 19:
        raise RuntimeError("TRI60 M1 screen condition count differs")
    return tuple(rows)


CONDITIONS: Final = _conditions()
CONDITION_REGISTRY: Final[Mapping[str, ScreenCondition]] = MappingProxyType({
    row.condition_id: row for row in CONDITIONS
})
FIT_ORDER: Final = tuple(CONDITION_REGISTRY)


def graph_payload() -> dict[str, Any]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "imported_control_id": IMPORTED_CONTROL_ID,
        "teacher_id": TEACHER_ID, "warm_source_id": WARM_SOURCE_ID,
        "condition_order": list(FIT_ORDER),
        "conditions": [row.payload() for row in CONDITIONS],
        "condition_count": 20, "fresh_fit_count": 19,
        "paired_seed_alias": SEED_ALIAS,
        "teacher_temperature_semantics": (
            "T1=authenticated_probability;T2=softmax(log(authenticated_probability)/2)"
        ),
        "validation_selects_followup": False,
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def validate_graph() -> str:
    value = graph_payload()
    digest = validate_artifact(value, contract=GRAPH_CONTRACT)
    if (
        digest != GRAPH_SHA256
        or tuple(CONDITION_REGISTRY) != FIT_ORDER
        or any(row.node.coordinate_name != "D000" for row in CONDITIONS)
        or any(row.node.auxiliary != "none" for row in CONDITIONS)
        or any(row.node.seed_alias != SEED_ALIAS for row in CONDITIONS)
        or canonical_sha256(value) != canonical_sha256(graph_payload())
    ):
        raise ValueError("TRI60 M1 screen graph differs")
    return digest


__all__ = [
    "CAMPAIGN_LABEL", "CONDITIONS", "CONDITION_REGISTRY", "FIT_ORDER",
    "GRAPH_SHA256", "IMPORTED_CONTROL_ID", "SEED_ALIAS", "TEACHER_ID",
    "WARM_SOURCE_ID", "ScreenCondition", "graph_payload", "validate_graph",
]
