"""Frozen 26-fit graph for performance-constrained adjacent-view handoff."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_adjacent_output_handoff_contracts import (
    GRAPH_CONTRACT, RECIPE_CONTRACT, SCHEMA_VERSION, artifact,
)


CAMPAIGN_LABEL: Final = "HCWDL-ADJACENT-OUTPUT-FUSION-HANDOFF"
SEED_DOMAIN: Final = f"{CAMPAIGN_LABEL}/v2"
COORDINATE_ORDER: Final = ("U100", "D080", "D060", "D040", "D020", "D000")
LOWER_COORDINATES: Final = COORDINATE_ORDER[1:]
TERMINAL_SEEDS: Final = ("S1", "S2", "S3", "S4", "S5")
ALPHA_GRID: Final = tuple((index, 40) for index in range(41))
FUSION_FAMILIES: Final = ("calibrated_centered_logit", "arithmetic_probability")

COORDINATES: Final = MappingProxyType({
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D080": HomotopyCoordinate(1, 1, 1, 5),
    "D060": HomotopyCoordinate(1, 1, 2, 5),
    "D040": HomotopyCoordinate(1, 1, 3, 5),
    "D020": HomotopyCoordinate(1, 1, 4, 5),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
})

LR_SCHEDULE: Final = MappingProxyType({
    "kind": "warmup_hold_cosine_floor_tail_v1", "warmup_passes": 3,
    "hold_through_pass": 45, "decay_through_pass": 60,
    "minimum_lr_fraction": .05,
})
EARLY_STOPPING: Final = MappingProxyType({
    "kind": "macro_auc_patience_v1", "minimum_passes": 60,
    "patience_passes": 15, "minimum_auc_delta": 5.0e-5,
})


@dataclass(frozen=True)
class HandoffNode:
    node_id: str
    coordinate_name: str
    teacher_distribution_id: str | None
    seed_alias: str
    role: str
    ce_weight: float = .25
    kd_weight: float = .75
    temperature: float = 2.0
    auxiliary: str = "none"
    representation_carrier_id: None = None
    representation_seed_alias: None = None
    track: str = "LOGIT"
    training_passes: int = 100
    batch_size: int = 256
    initialization: str = "fresh"

    @property
    def coordinate(self):
        return COORDINATES[self.coordinate_name]

    @property
    def distribution_teacher_id(self):
        return self.teacher_distribution_id

    @property
    def deployable(self) -> bool:
        return self.coordinate_name == "D000"

    def payload(self) -> dict[str, object]:
        return {
            "node_id": self.node_id, "coordinate_name": self.coordinate_name,
            "coordinate_exact": self.coordinate.payload(), "role": self.role,
            "distribution_teacher_id": self.teacher_distribution_id,
            "distribution_teacher_kind": (
                "none" if self.teacher_distribution_id is None else "probability_bank"
            ),
            "seed_alias": self.seed_alias, "track": self.track,
            "ce_weight": self.ce_weight, "kd_weight": self.kd_weight,
            "temperature": self.temperature, "auxiliary": self.auxiliary,
            "training_passes": self.training_passes, "batch_size": self.batch_size,
            "initialization": self.initialization, "deployable": self.deployable,
        }


def _build_nodes():
    nodes: dict[str, HandoffNode] = {}
    carrier = "SOURCE_U100"
    for coordinate in LOWER_COORDINATES[:-1]:
        seed = f"{SEED_DOMAIN}/rung/{coordinate}"
        direct = f"OUTPUT_DIRECT_{coordinate}"
        compression = f"OUTPUT_COMPRESSION_{coordinate}"
        nodes[direct] = HandoffNode(direct, coordinate, carrier, seed, "direct")
        nodes[compression] = HandoffNode(
            compression, coordinate, f"OUTPUT_MIX_{coordinate}", seed, "compression",
        )
        carrier = f"OUTPUT_T_{coordinate}"
    for seed_name in TERMINAL_SEEDS:
        alias = f"{SEED_DOMAIN}/terminal/{seed_name}"
        direct = f"OUTPUT_DIRECT_D000_{seed_name}"
        compression = f"OUTPUT_COMPRESSION_D000_{seed_name}"
        ce = f"CE_D000_{seed_name}"
        nodes[direct] = HandoffNode(direct, "D000", carrier, alias, "direct")
        nodes[compression] = HandoffNode(
            compression, "D000", f"OUTPUT_MIX_D000_{seed_name}", alias, "compression",
        )
        nodes[ce] = HandoffNode(
            ce, "D000", None, alias, "ce", ce_weight=1.0, kd_weight=0.0,
            temperature=1.0,
        )
    final_alias = f"{SEED_DOMAIN}/final/matched"
    for family, teacher in (
        ("FINAL_CE_SEED_D000", "CE_D000_E5"),
        ("FINAL_DIRECT_D000", "OUTPUT_DIRECT_D000_E5"),
        ("FINAL_HANDOFF_D000", "OUTPUT_COMPRESSION_D000_E5"),
    ):
        nodes[family] = HandoffNode(
            family, "D000", teacher, final_alias, "final_distiller",
            ce_weight=.10, kd_weight=.90, temperature=1.0,
        )
    return MappingProxyType(nodes)


NODE_REGISTRY: Final = _build_nodes()
FIT_ORDER: Final = tuple(NODE_REGISTRY)
DIRECT_NODES: Final = tuple(n for n in FIT_ORDER if NODE_REGISTRY[n].role == "direct")
COMPRESSION_NODES: Final = tuple(n for n in FIT_ORDER if NODE_REGISTRY[n].role == "compression")
CE_NODES: Final = tuple(n for n in FIT_ORDER if NODE_REGISTRY[n].role == "ce")
FINAL_NODES: Final = tuple(n for n in FIT_ORDER if NODE_REGISTRY[n].role == "final_distiller")


def node_distribution(node_id: str) -> str:
    node = NODE_REGISTRY[node_id]
    if node.role == "compression" and node.coordinate_name != "D000":
        return f"OUTPUT_T_{node.coordinate_name}"
    return node_id


SELECTION_IDS: Final = tuple(
    [f"OUTPUT_MIX_{coordinate}" for coordinate in LOWER_COORDINATES[:-1]]
    + [f"OUTPUT_MIX_D000_{seed}" for seed in TERMINAL_SEEDS]
)
ENSEMBLE_FAMILIES: Final = (
    "CE_D000", "OUTPUT_DIRECT_D000", "OUTPUT_COMPRESSION_D000",
)
ENSEMBLE_IDS: Final = tuple(
    f"{family}_E{count}" for family in ENSEMBLE_FAMILIES for count in range(1, 6)
)


def selection_components(selection_id: str) -> tuple[str, str]:
    if selection_id.startswith("OUTPUT_MIX_D000_"):
        seed = selection_id.rsplit("_", 1)[1]
        return "OUTPUT_T_D020", f"OUTPUT_DIRECT_D000_{seed}"
    coordinate = selection_id.removeprefix("OUTPUT_MIX_")
    index = LOWER_COORDINATES.index(coordinate)
    rich = "SOURCE_U100" if index == 0 else f"OUTPUT_T_{LOWER_COORDINATES[index - 1]}"
    return rich, f"OUTPUT_DIRECT_{coordinate}"


def ensemble_components(ensemble_id: str) -> tuple[str, ...]:
    for family in ENSEMBLE_FAMILIES:
        prefix = f"{family}_E"
        if ensemble_id.startswith(prefix):
            count = int(ensemble_id[len(prefix):])
            if count not in range(1, 6):
                break
            return tuple(f"{family}_S{index}" for index in range(1, count + 1))
    raise KeyError("unknown output-handoff ensemble")


def distribution_consumers(distribution_id: str) -> tuple[str, ...]:
    consumers = [
        node_id for node_id, node in NODE_REGISTRY.items()
        if node.teacher_distribution_id == distribution_id
    ]
    consumers.extend(
        f"select_{selection}" for selection in SELECTION_IDS
        if distribution_id in selection_components(selection)
    )
    consumers.extend(
        f"reduce_{ensemble}" for ensemble in ENSEMBLE_IDS
        if distribution_id in ensemble_components(ensemble)
    )
    return tuple(consumers)


def recipe_payload() -> dict[str, object]:
    return artifact({
        "campaign_label": CAMPAIGN_LABEL,
        "training": {
            "maximum_passes": 100, "minimum_passes": 60,
            "validation_every_passes": 1, "effective_batch_size": 256,
            "peak_learning_rate": 3e-4, "weight_decay": .01,
            "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
            "forward_precision": "bfloat16", "learning_rate_schedule": dict(LR_SCHEDULE),
            "early_stopping": dict(EARLY_STOPPING), "restore_best_checkpoint": True,
        },
        "bridge_and_compression_loss": {
            "kind": "constant_ce_kd_v1", "ce_weight": .25,
            "kd_weight": .75, "temperature": 2.0,
        },
        "terminal_ce_loss": {"ce_weight": 1.0, "kd_weight": 0.0},
        "final_distillation_loss": {
            "kind": "constant_ce_kd_v1", "ce_weight": .10,
            "kd_weight": .90, "temperature": 1.0,
        },
        "fusion": {
            "families": list(FUSION_FAMILIES), "alpha_grid": [list(x) for x in ALPHA_GRID],
            "temperature_bounds": [.25, 4.0], "temperature_fit_role": "V_blend",
            "log_probability_floor": 2.0 ** -126,
        },
        "noninferiority": {
            "auc_point_floor": -1e-4, "auc_lower_95_floor": -3e-4,
            "per_class_r50_ratio_floor": .95, "bootstrap_replicates": 2000,
            "bootstrap_method": "paired_class_stratified_gaussian_multiplier_auc_influence_bootstrap_v1",
            "same_multiplier_stream_for_every_candidate": True,
            "one_sided_confidence_level": .95, "selection_role": "V_blend",
        },
        "probability_storage": {
            "durable_temperature": 1.0,
            "consumer_temperature_derived_in_ram_at_identity_join": True,
            "duplicate_softened_train_bank": False,
        },
        "source_view": {
            "campaign_family": "fullcard_bottleneck_nonpersistent_v2",
            "node_id": "SP4_COARSE_U100_from_U050",
            "support_policy": "replace_source_with_target_v1",
        },
        "report_role": "V_report", "checkpoint_role": "V_checkpoint",
        "initialization": "cold_start_all_fits", "rolling_resume": False,
        "durable_particle_views": False, "durable_hidden_states": False,
        "final_test_accessed": False,
    }, contract=RECIPE_CONTRACT)


_GRAPH_BODY: Final = {
    "contract": GRAPH_CONTRACT, "schema_version": SCHEMA_VERSION,
    "campaign_label": CAMPAIGN_LABEL, "coordinate_order": list(COORDINATE_ORDER),
    "coordinates": {k: v.payload() for k, v in COORDINATES.items()},
    "nodes": [NODE_REGISTRY[n].payload() for n in FIT_ORDER],
    "fit_order": list(FIT_ORDER), "terminal_seeds": list(TERMINAL_SEEDS),
    "selection_ids": list(SELECTION_IDS), "ensemble_ids": list(ENSEMBLE_IDS),
    "fresh_fit_count": len(FIT_ORDER), "source_anchor": "SOURCE_U100",
    "strategy": "performance_constrained_output_fusion_handoff_v2",
    "source_campaign_family": "fullcard_bottleneck_nonpersistent_v2",
    "source_node_id": "SP4_COARSE_U100_from_U050",
    "source_support_policy": "replace_source_with_target_v1",
    "final_test_accessed": False,
}
GRAPH_SHA256: Final = canonical_sha256(_GRAPH_BODY)


def graph_payload() -> dict[str, object]:
    value = with_content_hash(_GRAPH_BODY)
    if value["content_hash"] != GRAPH_SHA256:
        raise RuntimeError("adjacent output-handoff graph hash differs")
    return value


def validate_graph() -> str:
    if len(FIT_ORDER) != 26:
        raise ValueError("adjacent output-handoff fresh-fit count differs")
    if len(DIRECT_NODES) != 9 or len(COMPRESSION_NODES) != 9 or len(CE_NODES) != 5 or len(FINAL_NODES) != 3:
        raise ValueError("adjacent output-handoff node family counts differ")
    for coordinate in LOWER_COORDINATES[:-1]:
        direct = NODE_REGISTRY[f"OUTPUT_DIRECT_{coordinate}"]
        compression = NODE_REGISTRY[f"OUTPUT_COMPRESSION_{coordinate}"]
        if direct.seed_alias != compression.seed_alias:
            raise ValueError("adjacent output-handoff rung seed pairing differs")
    for seed in TERMINAL_SEEDS:
        rows = [NODE_REGISTRY[f"{prefix}_D000_{seed}"] for prefix in (
            "OUTPUT_DIRECT", "OUTPUT_COMPRESSION", "CE",
        )]
        if len({row.seed_alias for row in rows}) != 1:
            raise ValueError("adjacent output-handoff terminal seed pairing differs")
    return GRAPH_SHA256


__all__ = [
    "ALPHA_GRID", "CAMPAIGN_LABEL", "CE_NODES", "COMPRESSION_NODES",
    "COORDINATES", "COORDINATE_ORDER", "DIRECT_NODES", "EARLY_STOPPING",
    "FINAL_NODES", "FIT_ORDER", "FUSION_FAMILIES", "GRAPH_SHA256",
    "LOWER_COORDINATES", "LR_SCHEDULE", "NODE_REGISTRY", "SEED_DOMAIN",
    "SELECTION_IDS", "ENSEMBLE_FAMILIES", "ENSEMBLE_IDS", "TERMINAL_SEEDS",
    "distribution_consumers", "ensemble_components", "node_distribution",
    "selection_components", "graph_payload", "recipe_payload", "validate_graph",
]
