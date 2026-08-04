"""Configuration-driven PRAD training semantics and staged parameter control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hlt_classification.models.prad_particle_transformer import (
    PradForwardOutput,
    PradParticleTransformer,
)

from .experiments import PradExperiment
from .losses import (
    relation_distillation_loss,
    semantic_pair_loss,
    temperature_kl_loss,
)

PRAD_TRAINING_CONTRACT = "hlt_classification_prad_training_v2"
PRAD_RELATION_SHUFFLE_ALGORITHM = (
    "within_realized_batch_identity_derangement_v1"
)
PRAD_STAGE_A_EPOCHS = 5
PRAD_STAGE_B_EPOCHS = 5
PRAD_STAGE_C_MAX_EPOCHS = 50
PRAD_CONFIRMATION_SEEDS = (11, 22, 33, 44, 55)


@dataclass(frozen=True)
class PradTrainingConfig:
    experiment: PradExperiment
    seed: int
    batch_size: int = 64
    stage_a_epochs: int = PRAD_STAGE_A_EPOCHS
    stage_b_epochs: int = PRAD_STAGE_B_EPOCHS
    stage_c_epochs: int = PRAD_STAGE_C_MAX_EPOCHS
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 1.0
    relation_lr_a_b: float = 3.0e-4
    pretrained_lr_b: float = 3.0e-5
    relation_lr_c: float = 1.0e-4
    pretrained_lr_c: float = 1.0e-5
    lambda_relation: float = 1.0
    lambda_semantic: float = 0.2
    lambda_kd: float = 0.5
    kd_temperature: float = 2.0
    amp_dtype: str = "bfloat16"
    realization_policy: str = "R_MULTI"
    checkpoint_interval_updates: int = 1000
    history_interval_updates: int = 100

    def __post_init__(self) -> None:
        if self.seed < 0 or self.batch_size <= 0:
            raise ValueError("PRAD training seed or batch size differs")
        if (self.stage_a_epochs, self.stage_b_epochs, self.stage_c_epochs) != (
            PRAD_STAGE_A_EPOCHS,
            PRAD_STAGE_B_EPOCHS,
            PRAD_STAGE_C_MAX_EPOCHS,
        ):
            raise ValueError("PRAD registered stage budgets differ")
        if self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("PRAD AMP dtype differs")
        if self.realization_policy not in {"R_FIXED", "R_MULTI"}:
            raise ValueError("PRAD realization policy differs")
        if self.checkpoint_interval_updates <= 0 or self.history_interval_updates <= 0:
            raise ValueError("PRAD checkpoint/history interval must be positive")
        if min(
            self.weight_decay,
            self.gradient_clip_norm,
            self.relation_lr_a_b,
            self.pretrained_lr_b,
            self.relation_lr_c,
            self.pretrained_lr_c,
            self.lambda_relation,
            self.lambda_semantic,
            self.lambda_kd,
        ) < 0:
            raise ValueError("PRAD training coefficients must be nonnegative")

    @property
    def total_epochs(self) -> int:
        return self.stage_a_epochs + self.stage_b_epochs + self.stage_c_epochs

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": PRAD_TRAINING_CONTRACT,
            "experiment": self.experiment.to_dict(),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "stage_a_epochs": self.stage_a_epochs,
            "stage_b_epochs": self.stage_b_epochs,
            "stage_c_epochs": self.stage_c_epochs,
            "fixed_budget_no_performance_cancellation": True,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "learning_rates": {
                "relation_a_b": self.relation_lr_a_b,
                "pretrained_b": self.pretrained_lr_b,
                "relation_c": self.relation_lr_c,
                "pretrained_c": self.pretrained_lr_c,
            },
            "loss_coefficients": {
                "hard": 1.0,
                "relation": self.lambda_relation,
                "semantic": self.lambda_semantic,
                "kd_max": self.lambda_kd,
            },
            "kd_temperature": self.kd_temperature,
            "amp_dtype": self.amp_dtype,
            "realization_policy": self.realization_policy,
            "relation_shuffle_algorithm": PRAD_RELATION_SHUFFLE_ALGORITHM,
            "checkpoint_interval_updates": self.checkpoint_interval_updates,
            "history_interval_updates": self.history_interval_updates,
        }


def initialize_student(
    student: PradParticleTransformer,
    *,
    baseline_state_dict: Mapping[str, torch.Tensor],
    frozen_teacher: PradParticleTransformer | None,
    copy_teacher_heads: bool,
) -> None:
    """Apply the registered baseline initialization and optional head copy."""

    student.load_baseline_state_dict(dict(baseline_state_dict))
    if copy_teacher_heads:
        if frozen_teacher is None:
            raise ValueError("teacher head copy requested without a teacher")
        assert_frozen_teacher_has_no_gradients(frozen_teacher)
        student.relation_to_bias.load_state_dict(
            frozen_teacher.relation_to_bias.state_dict(), strict=True
        )
        student.semantic_heads.load_state_dict(
            frozen_teacher.semantic_heads.state_dict(), strict=True
        )
    with torch.no_grad():
        student.gated_bias.raw_gates.zero_()


@dataclass(frozen=True)
class PairPayloadLayout:
    relation_dim: int
    attention_heads: int
    semantic_targets: int = 3

    @property
    def channels(self) -> int:
        return self.relation_dim + self.attention_heads + 2 * self.semantic_targets + 1


def pack_training_pair_payload(
    *,
    teacher_relation: torch.Tensor,
    teacher_bias: torch.Tensor,
    semantic_targets: torch.Tensor,
    semantic_valid: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[torch.Tensor, PairPayloadLayout]:
    if teacher_relation.ndim != 4 or teacher_bias.ndim != 4:
        raise ValueError("teacher pair outputs must be rank four")
    batch, particles, other, relation_dim = teacher_relation.shape
    if particles != other or teacher_bias.shape[0] != batch or teacher_bias.shape[2:] != (particles, particles):
        raise ValueError("teacher relation and bias shapes differ")
    semantic_count = semantic_targets.shape[-1]
    expected_semantic = (batch, particles, particles, semantic_count)
    if semantic_targets.shape != expected_semantic or semantic_valid.shape != expected_semantic:
        raise ValueError("semantic pair payload shapes differ")
    if pair_mask.shape != (batch, particles, particles):
        raise ValueError("pair supervision mask shape differs")
    layout = PairPayloadLayout(
        relation_dim=relation_dim,
        attention_heads=teacher_bias.shape[1],
        semantic_targets=semantic_count,
    )
    last_dimension = torch.cat(
        (
            teacher_relation.detach(),
            teacher_bias.detach().permute(0, 2, 3, 1),
            semantic_targets,
            semantic_valid.to(semantic_targets.dtype),
            pair_mask[..., None].to(semantic_targets.dtype),
        ),
        dim=-1,
    )
    return last_dimension.permute(0, 3, 1, 2).contiguous(), layout


def unpack_training_pair_payload(
    payload: torch.Tensor,
    layout: PairPayloadLayout,
) -> dict[str, torch.Tensor]:
    if payload.ndim != 4 or payload.shape[1] != layout.channels:
        raise ValueError("aligned PRAD pair payload shape differs")
    values = payload.permute(0, 2, 3, 1)
    cursor = 0
    relation = values[..., cursor : cursor + layout.relation_dim]
    cursor += layout.relation_dim
    bias = values[..., cursor : cursor + layout.attention_heads].permute(0, 3, 1, 2)
    cursor += layout.attention_heads
    semantic_targets = values[..., cursor : cursor + layout.semantic_targets]
    cursor += layout.semantic_targets
    semantic_valid = values[..., cursor : cursor + layout.semantic_targets].to(torch.bool)
    cursor += layout.semantic_targets
    pair_mask = values[..., cursor].to(torch.bool)
    return {
        "teacher_relation": relation,
        "teacher_bias": bias,
        "semantic_targets": semantic_targets,
        "semantic_valid": semantic_valid,
        "pair_mask": pair_mask,
    }


def stage_for_epoch(config: PradTrainingConfig, epoch: int) -> str:
    if epoch < 0 or epoch >= config.total_epochs:
        raise ValueError("PRAD epoch lies outside the fixed budget")
    pair_supervised = (
        config.experiment.relation_bottleneck_loss
        or config.experiment.relation_bias_loss
        or config.experiment.semantic_loss
    )
    if not pair_supervised:
        return "C"
    if epoch < config.stage_a_epochs:
        return "A"
    if epoch < config.stage_a_epochs + config.stage_b_epochs:
        return "B"
    return "C"


def kd_coefficient(config: PradTrainingConfig, epoch: int) -> float:
    if not config.experiment.logit_kd:
        return 0.0
    stage = stage_for_epoch(config, epoch)
    if stage == "A":
        return 0.0
    if stage == "B":
        stage_epoch = epoch - config.stage_a_epochs
        return config.lambda_kd * (stage_epoch + 1) / config.stage_b_epochs
    return config.lambda_kd


def _set_trainable(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def configure_student_stage(
    model: PradParticleTransformer,
    config: PradTrainingConfig,
    epoch: int,
) -> dict[str, Any]:
    """Apply the registered freeze schedule and return optimizer group intent."""

    stage = stage_for_epoch(config, epoch)
    experiment = config.experiment
    _set_trainable(model, False)
    if stage == "A":
        _set_trainable(model.relation, True)
    elif stage == "B":
        _set_trainable(model.relation, True)
        _set_trainable(model.relation_to_bias, True)
        _set_trainable(model.semantic_heads, True)
        if not experiment.gates_fixed_zero:
            _set_trainable(model.gated_bias, True)
        midpoint = model.num_layers // 2
        for block in model.baseline.mod.blocks[midpoint:]:
            _set_trainable(block, True)
        _set_trainable(model.baseline.mod.cls_blocks, True)
        _set_trainable(model.baseline.mod.norm, True)
        _set_trainable(model.baseline.mod.fc, True)
    else:
        _set_trainable(model, True)
        if experiment.gates_fixed_zero:
            _set_trainable(model.gated_bias, False)
    if experiment.gates_fixed_zero:
        with torch.no_grad():
            model.gated_bias.raw_gates.zero_()
    relation_ids = {
        id(parameter)
        for module in (
            model.relation,
            model.relation_to_bias,
            model.gated_bias,
            model.semantic_heads,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    relation_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in relation_ids
    ]
    pretrained_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in relation_ids
    ]
    relation_lr = (
        config.relation_lr_a_b if stage in {"A", "B"} else config.relation_lr_c
    )
    pretrained_lr = (
        config.pretrained_lr_b if stage == "B" else config.pretrained_lr_c
    )
    return {
        "stage": stage,
        "relation_parameters": relation_parameters,
        "pretrained_parameters": pretrained_parameters,
        "relation_learning_rate": relation_lr,
        "pretrained_learning_rate": pretrained_lr,
    }


def freeze_teacher(teacher: nn.Module) -> nn.Module:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher


def assert_frozen_teacher_has_no_gradients(teacher: nn.Module) -> None:
    invalid = [
        name
        for name, parameter in teacher.named_parameters()
        if parameter.requires_grad or parameter.grad is not None
    ]
    if invalid:
        raise RuntimeError(f"frozen teacher acquired gradients: {invalid}")


def deterministic_relation_shuffle(
    identity_keys: Sequence[str], *, seed: int
) -> np.ndarray:
    """Return a deterministic non-class-preserving derangement."""

    keys = tuple(str(key) for key in identity_keys)
    if len(keys) < 2 or len(keys) != len(set(keys)):
        raise ValueError("relation shuffle needs at least two unique identities")
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    for key in keys:
        digest.update(b"\0")
        digest.update(key.encode("utf-8"))
    rng = np.random.default_rng(int.from_bytes(digest.digest()[:8], "big"))
    permutation = rng.permutation(len(keys))
    fixed = np.flatnonzero(permutation == np.arange(len(keys)))
    if len(fixed) == 1:
        index = int(fixed[0])
        other = 0 if index != 0 else 1
        permutation[index], permutation[other] = (
            permutation[other],
            permutation[index],
        )
    elif len(fixed) > 1:
        permutation[fixed] = np.roll(permutation[fixed], 1)
    if np.any(permutation == np.arange(len(keys))):
        raise RuntimeError("relation shuffle derangement failed")
    return permutation.astype(np.int64)


def map_offline_pairs_to_hlt(
    offline_pair_values: torch.Tensor,
    hlt_to_offline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather `[B,Noff,Noff,C]` targets into HLT ordering."""

    if offline_pair_values.ndim != 4:
        raise ValueError("offline pair values must have shape [B,N,N,C]")
    batch, offline_particles, other, channels = offline_pair_values.shape
    if offline_particles != other or hlt_to_offline.ndim != 2:
        raise ValueError("offline pair matrix or match shape differs")
    if hlt_to_offline.shape[0] != batch:
        raise ValueError("offline pair and match batch sizes differ")
    mapping = hlt_to_offline.long()
    valid = (mapping >= 0) & (mapping < offline_particles)
    safe = mapping.clamp(0, max(offline_particles - 1, 0))
    batch_index = torch.arange(batch, device=safe.device)[:, None, None]
    left = safe[:, :, None].expand(-1, -1, safe.shape[1])
    right = safe[:, None, :].expand(-1, safe.shape[1], -1)
    gathered = offline_pair_values[batch_index, left, right]
    pair_mask = valid[:, :, None] & valid[:, None, :]
    diagonal = torch.eye(
        safe.shape[1], dtype=torch.bool, device=safe.device
    )[None]
    pair_mask &= ~diagonal
    return gathered.reshape(batch, safe.shape[1], safe.shape[1], channels), pair_mask


def semantic_targets_from_assignments(
    assignments: torch.Tensor,
    hlt_to_offline: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build K=2,3,4 same-branch labels for teacher or matched student."""

    if assignments.ndim != 3 or assignments.shape[1] != 3:
        raise ValueError("C/A assignments must have shape [B,3,N]")
    values = assignments.transpose(1, 2)
    same = values[:, :, None, :] == values[:, None, :, :]
    valid_particle = values >= 0
    valid = valid_particle[:, :, None, :] & valid_particle[:, None, :, :]
    diagonal = torch.eye(values.shape[1], dtype=torch.bool, device=values.device)[
        None, :, :, None
    ]
    valid &= ~diagonal
    targets = same & valid
    if hlt_to_offline is None:
        return targets.to(torch.float32), valid
    mapped_targets, pair_mask = map_offline_pairs_to_hlt(
        targets.to(torch.float32), hlt_to_offline
    )
    mapped_valid, _ = map_offline_pairs_to_hlt(
        valid.to(torch.float32), hlt_to_offline
    )
    final_valid = mapped_valid.to(torch.bool) & pair_mask[..., None]
    return mapped_targets, final_valid


@dataclass(frozen=True)
class StudentLossResult:
    total: torch.Tensor
    hard: torch.Tensor
    relation: torch.Tensor
    relation_bottleneck: torch.Tensor
    relation_bias: torch.Tensor
    semantic: torch.Tensor
    kd: torch.Tensor
    coefficients: Mapping[str, float]


def student_loss(
    *,
    output: PradForwardOutput,
    labels: torch.Tensor,
    experiment: PradExperiment,
    stage: str,
    semantic_targets: torch.Tensor | None = None,
    semantic_valid: torch.Tensor | None = None,
    semantic_positive_weights: torch.Tensor | None = None,
    teacher_relation: torch.Tensor | None = None,
    teacher_bias: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
    teacher_true_class_confidence: torch.Tensor | None = None,
    pair_mask: torch.Tensor | None = None,
    lambda_relation: float = 1.0,
    lambda_semantic: float = 0.2,
    lambda_kd: float = 0.5,
    kd_temperature: float = 2.0,
) -> StudentLossResult:
    """Assemble independently normalized losses for one registered graph."""

    if stage not in {"A", "B", "C"}:
        raise ValueError("unknown PRAD training stage")
    zero = output.logits.sum() * 0.0
    hard = (
        zero
        if stage == "A" or not experiment.hard_class_loss
        else F.cross_entropy(output.logits, labels)
    )
    relation = zero
    relation_bottleneck = zero
    relation_bias = zero
    use_relation = (
        experiment.relation_bottleneck_loss or experiment.relation_bias_loss
    )
    if use_relation:
        if any(
            value is None
            for value in (
                teacher_relation,
                teacher_bias,
                teacher_true_class_confidence,
                pair_mask,
            )
        ):
            raise ValueError("registered relation loss lacks teacher targets")
        relation_result = relation_distillation_loss(
            student_relation=output.relation,
            student_bias=output.privileged_bias,
            teacher_relation=teacher_relation,
            teacher_bias=teacher_bias,
            pair_mask=pair_mask,
            teacher_true_class_confidence=teacher_true_class_confidence,
            use_bottleneck=experiment.relation_bottleneck_loss,
            use_bias=experiment.relation_bias_loss,
        )
        relation = relation_result.total
        relation_bottleneck = relation_result.bottleneck
        relation_bias = relation_result.bias
    semantic = zero
    if experiment.semantic_loss:
        if semantic_targets is None or semantic_valid is None or semantic_positive_weights is None:
            raise ValueError("registered semantic loss lacks targets")
        semantic = semantic_pair_loss(
            output.semantic_logits,
            semantic_targets,
            semantic_valid,
            positive_weights=semantic_positive_weights,
        )
    kd = zero
    if experiment.logit_kd and stage != "A":
        if teacher_logits is None:
            raise ValueError("registered KD loss lacks teacher logits")
        kd = temperature_kl_loss(
            output.logits, teacher_logits, temperature=kd_temperature
        )
    coefficients = {
        "hard": 0.0 if stage == "A" else 1.0,
        "relation": lambda_relation,
        "semantic": lambda_semantic,
        "kd": 0.0 if stage == "A" else lambda_kd,
    }
    total = (
        coefficients["hard"] * hard
        + coefficients["relation"] * relation
        + coefficients["semantic"] * semantic
        + coefficients["kd"] * kd
    )
    if not torch.isfinite(total):
        raise FloatingPointError("PRAD total student loss is nonfinite")
    return StudentLossResult(
        total,
        hard,
        relation,
        relation_bottleneck,
        relation_bias,
        semantic,
        kd,
        coefficients,
    )


def teacher_loss(
    *,
    output: PradForwardOutput,
    labels: torch.Tensor,
    semantic_targets: torch.Tensor,
    semantic_valid: torch.Tensor,
    semantic_positive_weights: torch.Tensor,
    vertex_loss: torch.Tensor | None = None,
    vertex_coefficient: float = 0.0,
) -> dict[str, torch.Tensor]:
    if vertex_coefficient != 0.0:
        if vertex_loss is None:
            raise ValueError("nonzero vertex coefficient lacks vertex targets")
        raise ValueError("current PRAD source does not authorize vertex loss")
    class_loss = F.cross_entropy(output.logits, labels)
    tree_loss = semantic_pair_loss(
        output.semantic_logits,
        semantic_targets,
        semantic_valid,
        positive_weights=semantic_positive_weights,
    )
    total = class_loss + 0.2 * tree_loss
    return {
        "total": total,
        "class": class_loss,
        "tree": tree_loss,
        "vertex": total * 0.0,
    }


__all__ = [
    "PRAD_CONFIRMATION_SEEDS",
    "PRAD_RELATION_SHUFFLE_ALGORITHM",
    "PradTrainingConfig",
    "StudentLossResult",
    "assert_frozen_teacher_has_no_gradients",
    "configure_student_stage",
    "deterministic_relation_shuffle",
    "freeze_teacher",
    "initialize_student",
    "kd_coefficient",
    "map_offline_pairs_to_hlt",
    "pack_training_pair_payload",
    "semantic_targets_from_assignments",
    "stage_for_epoch",
    "student_loss",
    "teacher_loss",
    "unpack_training_pair_payload",
]
