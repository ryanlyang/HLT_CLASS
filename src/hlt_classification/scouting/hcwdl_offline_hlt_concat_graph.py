"""Frozen graph and recipe for one tagged offline+HLT concatenation fit."""

from __future__ import annotations

from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_graph import NODE_REGISTRY as TRI60_NODES, Tri60Node
from .hcwdl_offline_hlt_concat_contracts import (
    GRAPH_CONTRACT, NODE_CONTRACT, RECIPE_CONTRACT, artifact,
)


NODE_ID: Final = "CONCAT_TAGGED"
MODEL_INPUT_PROTOCOL: Final = "tagged_offline_hlt_concat_v2"


def node() -> Tri60Node:
    anchor = TRI60_NODES["U000"]
    return Tri60Node(
        node_id=NODE_ID, track="ORACLE", coordinate_name="U000",
        distribution_teacher_id=None, distribution_teacher_kind="none",
        representation_carrier_id=None, auxiliary="none",
        ce_weight=1.0, kd_weight=0.0, temperature=1.0,
        seed_alias=anchor.seed_alias, representation_seed_alias=None,
        training_passes=60, batch_size=256, initialization="fresh",
        node_contract=NODE_CONTRACT,
    )


_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT, "schema_version": 2,
    "campaign_label": "HCWDL-OFFLINE-HLT-TAGGED-CONCAT-PILOT",
    "node": node().payload(), "fit_order": [NODE_ID],
    "fresh_fit_count": 1, "input_sequence": "offline_then_hlt_v1",
    "duplicates_retained": True, "matching_indices_are_model_inputs": False,
    "content_source_embedding": (
        "learned_two_entry_128d_truncated_normal_std_0p02_"
        "added_after_numeric_embed_v1"
    ),
    "model_input_protocol": MODEL_INPUT_PROTOCOL,
    "ordinary_access_roles": ["train", "validation"],
    "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    value = artifact(
        {key: item for key, item in _GRAPH_BODY.items()
         if key not in {"contract", "schema_version"}},
        contract=GRAPH_CONTRACT,
    )
    if value["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("tagged concatenation graph hash differs")
    return value


def recipe_payload() -> dict[str, object]:
    return artifact({
        "node_id": NODE_ID,
        "training": {
            "passes": 60, "validation_every_passes": 1,
            "effective_batch_size": 256, "optimizer": "AdamW",
            "peak_learning_rate": 3.0e-4, "weight_decay": .01,
            "adam_betas": [.9, .999], "adam_epsilon": 1.0e-8,
            "warmup_fraction": .05,
            "schedule": "linear_warmup_cosine_decay_v1",
            "learning_rate_floor_fraction": .05,
            "forward_precision": "bfloat16",
            "performance_early_stopping": False,
            "checkpoint_selector": (
                "maximum_macro_auc_then_minimum_ce_then_maximum_logr50_"
                "then_earliest_update_v1"
            ),
        },
        "loss": {"ce_weight": 1.0, "kd_weight": 0.0, "temperature": 1.0},
        "view": {
            "capacity": 496, "order": "offline_then_hlt_v1",
            "ordinary_hlt_200_token_cap_applies": False,
            "all_raw_hlt_particles_retained": True,
            "zero_truncation_required": True, "deduplicate_matches": False,
            "content_source_codes": {"offline": 0, "hlt": 1, "padding": -1},
        },
        "model_input_protocol": MODEL_INPUT_PROTOCOL,
        "content_source_embedding_initialization": {
            "kind": "truncated_normal", "standard_deviation": 0.02,
        },
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


def validate_graph() -> str:
    if node().payload() != _GRAPH_BODY["node"]:
        raise RuntimeError("tagged concatenation node drifted")
    return graph_payload()["content_hash"]


__all__ = [
    "GRAPH_SHA256", "MODEL_INPUT_PROTOCOL", "NODE_ID", "graph_payload",
    "node", "recipe_payload", "validate_graph",
]
