"""Exact matched floor-tail comparator for the existing P90 H45 fit."""

from __future__ import annotations

from typing import Any, Final

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_graph import Tri60Node
from .hcwdl_tri60_d000_budget_screen_graph import (
    CONDITION_REGISTRY as REFERENCE_CONDITIONS,
    SEED_ALIAS, TEACHER_ID,
)
from .hcwdl_tri60_d000_floor_tail_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, artifact, validate_artifact,
)


CAMPAIGN_LABEL: Final = "HCWDL-TRI60-D000-FLOOR-TAIL-CONFIRMATION"
REFERENCE_CONDITION_ID: Final = "P90_H45_LR3E4"
CONDITION_ID: Final = "P100_H45_D60_FLOOR_ES15"

LOSS_SCHEDULE: Final = {
    "kind": "constant_v1", "ce_weight": .25, "kd_weight": .75,
}
LR_SCHEDULE: Final = {
    "kind": "warmup_hold_cosine_floor_tail_v1",
    "warmup_passes": 3,
    "hold_through_pass": 45,
    "decay_through_pass": 60,
    "minimum_lr_fraction": .05,
}
EARLY_STOPPING: Final = {
    "kind": "macro_auc_patience_v1",
    "minimum_passes": 60,
    "patience_passes": 15,
    "minimum_auc_delta": 5.0e-5,
}

NODE: Final = Tri60Node(
    node_id=CONDITION_ID, track="LOGIT_FLOOR_TAIL_CONFIRMATION",
    coordinate_name="D000", distribution_teacher_id=TEACHER_ID,
    distribution_teacher_kind="probability_bank",
    representation_carrier_id=None, auxiliary="none",
    ce_weight=.25, kd_weight=.75, temperature=2.0,
    seed_alias=SEED_ALIAS, representation_seed_alias=None,
    training_passes=100, batch_size=256, initialization="fresh",
    node_contract=NODE_CONTRACT,
)


def reference_payload() -> dict[str, Any]:
    return REFERENCE_CONDITIONS[REFERENCE_CONDITION_ID].payload()


def graph_payload() -> dict[str, Any]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "condition_id": CONDITION_ID,
        "condition": NODE.payload(),
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "reference_condition": reference_payload(),
        "loss_schedule": dict(LOSS_SCHEDULE),
        "learning_rate_schedule": dict(LR_SCHEDULE),
        "early_stopping": dict(EARLY_STOPPING),
        "peak_learning_rate": 3.0e-4,
        "batch_size": 256,
        "temperature": 2.0,
        "single_gpu": True,
        "changed_variable": "registered_training_protocol_only_v1",
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def validate_graph() -> str:
    value = graph_payload()
    digest = validate_artifact(value, contract=GRAPH_CONTRACT)
    reference = REFERENCE_CONDITIONS[REFERENCE_CONDITION_ID].node
    matched = (
        "coordinate_name", "distribution_teacher_id",
        "distribution_teacher_kind", "representation_carrier_id",
        "auxiliary", "ce_weight", "kd_weight", "temperature", "seed_alias",
        "representation_seed_alias", "batch_size", "initialization",
    )
    if (
        digest != GRAPH_SHA256
        or any(getattr(NODE, name) != getattr(reference, name) for name in matched)
        or NODE.training_passes != 100
        or canonical_sha256(value) != canonical_sha256(graph_payload())
    ):
        raise ValueError("D000 floor-tail matched graph differs")
    return digest


__all__ = [
    "CAMPAIGN_LABEL", "CONDITION_ID", "EARLY_STOPPING", "GRAPH_SHA256",
    "LOSS_SCHEDULE", "LR_SCHEDULE", "NODE", "REFERENCE_CONDITION_ID",
    "graph_payload", "reference_payload", "validate_graph",
]
