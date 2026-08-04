"""Immutable configuration-driven PRAD experiment registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from hlt_classification.data.cache_contracts import canonical_sha256


@dataclass(frozen=True)
class PradExperiment:
    experiment_id: str
    role: str
    relation_module: bool
    attention_injection: bool
    oracle_bias: bool = False
    hard_class_loss: bool = True
    relation_bottleneck_loss: bool = False
    relation_bias_loss: bool = False
    semantic_loss: bool = False
    logit_kd: bool = False
    shuffle_relation_targets: bool = False
    gates_fixed_zero: bool = False
    context_depth: int = 2
    relation_dim: int = 16
    gate_structure: str = "layer_head"
    injection_depth: str = "after_context"
    retain_standard_pair_bias: bool = True
    relation_target: str = "teacher_relation_and_bias"
    initialization: str = "best_hlt_baseline"

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("PRAD experiment id is required")
        if self.role not in {
            "baseline",
            "offline_teacher",
            "oracle_diagnostic",
            "capacity_control",
            "reference_baseline",
            "scientific_candidate",
            "null_control",
        }:
            raise ValueError("PRAD experiment role differs")
        if self.attention_injection and not (
            self.relation_module or self.oracle_bias
        ):
            raise ValueError("PRAD attention injection has no relation source")
        if self.relation_module and not self.attention_injection and not self.gates_fixed_zero:
            raise ValueError("non-injected PRAD relation graphs must fix gates at zero")
        if self.oracle_bias and self.role != "oracle_diagnostic":
            raise ValueError("oracle bias is restricted to oracle diagnostics")
        if self.shuffle_relation_targets and not (
            self.relation_bottleneck_loss or self.relation_bias_loss
        ):
            raise ValueError("shuffled relation control has no relation target")
        if self.context_depth < 0 or self.relation_dim <= 0:
            raise ValueError("PRAD experiment dimensions are invalid")
        if self.gate_structure not in {"layer_head", "layer", "global"}:
            raise ValueError("PRAD gate structure differs")
        if self.injection_depth not in {
            "all",
            "after_context",
            "final_half",
        }:
            raise ValueError("PRAD injection depth differs")
        if self.relation_target not in {
            "none",
            "teacher_relation_and_bias",
            "teacher_bias",
            "teacher_relation",
            "semantic",
            "teacher_pair_embed",
        }:
            raise ValueError("PRAD relation target differs")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["graph_sha256"] = canonical_sha256(payload)
        return payload


CORE_EXPERIMENTS = {
    row.experiment_id: row
    for row in (
        PradExperiment("E0", "baseline", False, False, relation_target="none"),
        PradExperiment(
            "E1", "offline_teacher", True, True, semantic_loss=True,
            relation_target="semantic", initialization="from_scratch"
        ),
        PradExperiment(
            "E2", "oracle_diagnostic", False, True, oracle_bias=True,
            relation_target="teacher_bias"
        ),
        PradExperiment("E3", "capacity_control", True, True, relation_target="none"),
        PradExperiment("E4", "reference_baseline", False, False, logit_kd=True,
                       relation_target="none"),
        PradExperiment(
            "E5", "scientific_candidate", True, False, semantic_loss=True,
            gates_fixed_zero=True, relation_target="semantic"
        ),
        PradExperiment(
            "E6",
            "scientific_candidate",
            True,
            False,
            relation_bottleneck_loss=True,
            relation_bias_loss=True,
            semantic_loss=True,
            gates_fixed_zero=True,
            relation_target="teacher_relation_and_bias",
        ),
        PradExperiment(
            "E7", "scientific_candidate", True, True, semantic_loss=True,
            relation_target="semantic"
        ),
        PradExperiment(
            "E8",
            "scientific_candidate",
            True,
            True,
            relation_bottleneck_loss=True,
            relation_bias_loss=True,
            semantic_loss=True,
        ),
        PradExperiment(
            "E9",
            "scientific_candidate",
            True,
            True,
            relation_bottleneck_loss=True,
            relation_bias_loss=True,
            semantic_loss=True,
            logit_kd=True,
        ),
        PradExperiment(
            "E10",
            "null_control",
            True,
            True,
            relation_bottleneck_loss=True,
            relation_bias_loss=True,
            semantic_loss=True,
            shuffle_relation_targets=True,
        ),
    )
}


def experiment_variant(base_id: str, variant_id: str) -> PradExperiment:
    try:
        base = CORE_EXPERIMENTS[base_id]
    except KeyError as error:
        raise ValueError(f"unknown PRAD base experiment {base_id!r}") from error
    variants = {
        "V1": dict(relation_bottleneck_loss=False, relation_bias_loss=True,
                   relation_target="teacher_bias"),
        "V2": dict(relation_bottleneck_loss=True, relation_bias_loss=False,
                   relation_target="teacher_relation"),
        "V3": dict(context_depth=0, injection_depth="all"),
        "V4": dict(context_depth=4, injection_depth="after_context"),
        "V5": dict(
            relation_bottleneck_loss=False,
            relation_bias_loss=False,
            semantic_loss=True,
            logit_kd=False,
            relation_target="semantic",
        ),
        "V6": dict(relation_bottleneck_loss=False, relation_bias_loss=True,
                   relation_target="teacher_pair_embed", context_depth=0,
                   injection_depth="all"),
        "V7_8": dict(relation_dim=8),
        "V7_16": dict(relation_dim=16),
        "V7_32": dict(relation_dim=32),
        "V8_ALL": dict(context_depth=0, injection_depth="all"),
        "V8_AFTER2": dict(context_depth=2, injection_depth="after_context"),
        "V8_FINAL": dict(injection_depth="final_half"),
        "V9_LAYER_HEAD": dict(gate_structure="layer_head"),
        "V9_LAYER": dict(gate_structure="layer"),
        "V9_GLOBAL": dict(gate_structure="global"),
        "V10": dict(retain_standard_pair_bias=False),
    }
    try:
        overrides = variants[variant_id]
    except KeyError as error:
        raise ValueError(f"unknown PRAD variant {variant_id!r}") from error
    return replace(base, experiment_id=f"{base_id}_{variant_id}", **overrides)


def experiment_requires_teacher(experiment: PradExperiment) -> bool:
    """Return whether a graph is authorized to consume frozen-teacher output."""

    return bool(
        experiment.oracle_bias
        or experiment.relation_bottleneck_loss
        or experiment.relation_bias_loss
        or experiment.logit_kd
    )


def experiment_copies_teacher_relation_heads(experiment: PradExperiment) -> bool:
    """Restrict teacher parameter transfer to learned-relation students."""

    return bool(
        not experiment.oracle_bias
        and (
            experiment.relation_bottleneck_loss
            or experiment.relation_bias_loss
        )
    )


__all__ = [
    "CORE_EXPERIMENTS",
    "PradExperiment",
    "experiment_copies_teacher_relation_heads",
    "experiment_requires_teacher",
    "experiment_variant",
]
