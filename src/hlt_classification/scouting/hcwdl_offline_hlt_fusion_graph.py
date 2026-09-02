"""Immutable graph and recipes for the complete fusion-withdrawal study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_graph import COORDINATES
from .hcwdl_offline_hlt_fusion_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, RECIPE_CONTRACT, artifact,
)


ORACLE_NODES: Final = (
    "CONCAT_UNTAGGED", "CONCAT_TAGGED",
    "SYMMETRIC_FUSION_OO", "SYMMETRIC_FUSION_HH", "SYMMETRIC_FUSION_OH",
    "HLT_WARM_CONTINUE", "ANCHORED_FUSION_HH", "ANCHORED_FUSION_OH",
)
STUDY_C_NODES: Final = (
    "FUSION_DIRECT_KD_WARM", "FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP",
)
FIT_ORDER: Final = (*ORACLE_NODES, *STUDY_C_NODES)
TEACHER_NODE: Final = "ANCHORED_FUSION_OH"
TEACHER_DISTRIBUTION: Final = "ANCHORED_FUSION_OH_T2"


@dataclass(frozen=True)
class FusionNode:
    node_id: str
    track: str
    coordinate_name: str
    distribution_teacher_id: str | None
    distribution_teacher_kind: str
    representation_carrier_id: None
    auxiliary: str
    ce_weight: float
    kd_weight: float
    temperature: float
    seed_alias: str
    representation_seed_alias: None
    training_passes: int
    batch_size: int
    initialization: str
    node_contract: str
    deployable: bool

    def payload(self) -> dict[str, object]:
        return {
            "contract": self.node_contract, "node_id": self.node_id,
            "track": self.track, "coordinate_name": self.coordinate_name,
            "coordinate_exact": COORDINATES[self.coordinate_name].payload(),
            "distribution_teacher_id": self.distribution_teacher_id,
            "distribution_teacher_kind": self.distribution_teacher_kind,
            "representation_carrier_id": None, "auxiliary": self.auxiliary,
            "ce_weight": self.ce_weight, "kd_weight": self.kd_weight,
            "temperature": self.temperature, "seed_alias": self.seed_alias,
            "representation_seed_alias": None,
            "training_passes": self.training_passes,
            "validation_every_passes": 1, "batch_size": self.batch_size,
            "initialization": self.initialization,
            "deployable": self.deployable,
        }


def _node(
    node_id: str, *, seed_alias: str, passes: int,
    initialization: str = "fresh", kd: float = 0.0,
) -> FusionNode:
    deployable = node_id in {
        "HLT_WARM_CONTINUE", "FUSION_DIRECT_KD_WARM",
        "FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP",
    }
    return FusionNode(
        node_id=node_id, track="FUSION", coordinate_name="D000",
        distribution_teacher_id=(TEACHER_DISTRIBUTION if kd else None),
        distribution_teacher_kind=("single_model" if kd else "none"),
        representation_carrier_id=None, auxiliary="none",
        ce_weight=1.0 - kd, kd_weight=kd,
        temperature=(2.0 if kd else 1.0), seed_alias=seed_alias,
        representation_seed_alias=None, training_passes=passes,
        batch_size=256, initialization=initialization,
        node_contract=NODE_CONTRACT, deployable=deployable,
    )


NODE_REGISTRY: Final = {
    "CONCAT_UNTAGGED": _node(
        "CONCAT_UNTAGGED", seed_alias="fusion/oracle/concat", passes=60,
    ),
    "CONCAT_TAGGED": _node(
        "CONCAT_TAGGED", seed_alias="fusion/oracle/concat", passes=60,
    ),
    "SYMMETRIC_FUSION_OO": _node(
        "SYMMETRIC_FUSION_OO", seed_alias="fusion/oracle/symmetric", passes=60,
    ),
    "SYMMETRIC_FUSION_HH": _node(
        "SYMMETRIC_FUSION_HH", seed_alias="fusion/oracle/symmetric", passes=60,
    ),
    "SYMMETRIC_FUSION_OH": _node(
        "SYMMETRIC_FUSION_OH", seed_alias="fusion/oracle/symmetric", passes=60,
    ),
    "HLT_WARM_CONTINUE": _node(
        "HLT_WARM_CONTINUE", seed_alias="fusion/oracle/anchored", passes=60,
        initialization="warm_selected_checkpoint",
    ),
    "ANCHORED_FUSION_HH": _node(
        "ANCHORED_FUSION_HH", seed_alias="fusion/oracle/anchored", passes=60,
        initialization="warm_selected_checkpoint",
    ),
    "ANCHORED_FUSION_OH": _node(
        "ANCHORED_FUSION_OH", seed_alias="fusion/oracle/anchored", passes=60,
        initialization="warm_selected_checkpoint",
    ),
    "FUSION_DIRECT_KD_WARM": _node(
        "FUSION_DIRECT_KD_WARM", seed_alias="fusion/study_c/direct", passes=100,
        initialization="warm_selected_checkpoint", kd=.75,
    ),
    "FUSION_WITHDRAW_COS": _node(
        "FUSION_WITHDRAW_COS", seed_alias="fusion/study_c/cos", passes=100,
        initialization="warm_selected_checkpoint", kd=.75,
    ),
    "FUSION_WITHDRAW_STEP": _node(
        "FUSION_WITHDRAW_STEP", seed_alias="fusion/study_c/step", passes=100,
        initialization="warm_selected_checkpoint", kd=.75,
    ),
}


COSINE_ALPHA: Final = {
    "kind": "hold_cosine_zero_tail_v1", "hold_through_pass": 10,
    "decay_through_pass": 60, "zero_from_pass": 61,
}
STEP_ALPHA: Final = {
    "kind": "step_to_zero_v1", "hold_through_pass": 60,
    "zero_from_pass": 61,
}
WITHDRAWAL_LOSS: Final = {
    "zero_ce": .25, "zero_kd": .30,
    "privileged_ce": .15, "privileged_kd": .20,
    "logit_consistency": .05, "representation_consistency": .05,
    "temperature": 2.0, "representation_blocks": [2, 4, 6, 8],
}
TRAINING_60: Final = {
    "passes": 60, "effective_batch_size": 256, "optimizer": "AdamW",
    "peak_learning_rate": 3.0e-4, "weight_decay": .01,
    "warmup_fraction": .05, "learning_rate_floor_fraction": .05,
    "schedule": "linear_warmup_cosine_decay_v1",
    "forward_precision": "bfloat16", "restore_best_checkpoint": True,
    "performance_early_stopping": False,
}
TRAINING_100: Final = {
    "maximum_passes": 100, "minimum_passes": 60,
    "patience_passes": 15, "minimum_auc_delta": 1.0e-5,
    "effective_batch_size": 256, "optimizer": "AdamW",
    "peak_learning_rate": 3.0e-4, "weight_decay": .01,
    "warmup_passes": 3, "hold_through_pass": 45,
    "decay_through_pass": 60, "learning_rate_floor_fraction": .05,
    "schedule": "warmup_hold_cosine_floor_tail_v1",
    "forward_precision": "bfloat16", "restore_best_checkpoint": True,
    "checkpoint_selection_route": "alpha_zero_macro_auc_v1",
}


def recipe_payload() -> dict[str, object]:
    return artifact({
        "oracle_training": dict(TRAINING_60),
        "study_c_training": dict(TRAINING_100),
        "direct_kd_loss": {
            "kind": "constant_ce_kd_v1", "ce_weight": .25,
            "kd_weight": .75, "temperature": 2.0,
        },
        "withdrawal_loss": dict(WITHDRAWAL_LOSS),
        "alpha_schedules": {
            "FUSION_WITHDRAW_COS": dict(COSINE_ALPHA),
            "FUSION_WITHDRAW_STEP": dict(STEP_ALPHA),
        },
        "teacher_policy": "fixed_anchored_fusion_oh_not_score_selected_v1",
        "teacher_probability_roles": ["train", "validation"],
        "teacher_probability_dtype": "float32",
        "particle_and_hidden_state_storage": "ram_or_device_only_v1",
        "rolling_resume": False, "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


def graph_payload() -> dict[str, object]:
    return artifact({
        "fit_order": list(FIT_ORDER),
        "oracle_nodes": list(ORACLE_NODES),
        "study_c_nodes": list(STUDY_C_NODES),
        "nodes": [NODE_REGISTRY[name].payload() for name in FIT_ORDER],
        "teacher_node": TEACHER_NODE,
        "teacher_distribution": TEACHER_DISTRIBUTION,
        "study_c_runs_regardless_of_oracle_metrics": True,
        "raw_hlt_oracle_consumers": [
            "CONCAT_UNTAGGED", "CONCAT_TAGGED", "SYMMETRIC_FUSION_OO",
            "SYMMETRIC_FUSION_HH", "SYMMETRIC_FUSION_OH",
        ],
        "canonical_hlt_200_consumers": [
            "HLT_WARM_CONTINUE", "ANCHORED_FUSION_HH",
            "ANCHORED_FUSION_OH", *STUDY_C_NODES,
        ],
        "anchored_injection_blocks": [2, 4, 6, 8],
        "matching_indices_are_model_inputs": False,
        "fresh_fit_count": len(FIT_ORDER),
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def validate_graph() -> str:
    if (
        len(FIT_ORDER) != 11 or len(set(FIT_ORDER)) != 11
        or set(NODE_REGISTRY) != set(FIT_ORDER)
        or NODE_REGISTRY[TEACHER_NODE].kd_weight != 0
        or any(NODE_REGISTRY[name].training_passes != 100 for name in STUDY_C_NODES)
    ):
        raise ValueError("fusion-withdrawal graph differs")
    return GRAPH_SHA256


__all__ = [
    "COSINE_ALPHA", "FIT_ORDER", "FusionNode", "GRAPH_SHA256", "NODE_REGISTRY",
    "ORACLE_NODES", "STEP_ALPHA", "STUDY_C_NODES", "TEACHER_DISTRIBUTION",
    "TEACHER_NODE", "TRAINING_100", "TRAINING_60", "WITHDRAWAL_LOSS",
    "graph_payload", "recipe_payload", "validate_graph",
]
