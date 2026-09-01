"""Phase-I attention re-optimization semantics shared by four-spine workers.

Dense teacher surfaces are intentionally ordinary in-memory tensors.  This
module has no target publisher and no filesystem API.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    validate_content_hash,
    with_content_hash,
)


PARAMETER_REGISTRY_CONTRACT = (
    "HCWDL_DISTILLATION_GUIDED_ATTENTION_PARAMETER_REGISTRY/v1"
)
RECIPE_KIND = "stage0_attention_gram_joint_v1"


@dataclass(frozen=True)
class AttentionStageRecipe:
    stage0_passes: int = 60
    attention_passes: int = 15
    joint_passes: int = 25
    attention_learning_rate: float = 3.0e-5
    joint_learning_rate: float = 1.5e-5
    relational_weight: float = 0.25
    trust_weight: float = 0.01
    block_indices: tuple[int, ...] = tuple(range(8))

    @property
    def total_passes(self) -> int:
        return self.stage0_passes + self.attention_passes + self.joint_passes

    def payload(self) -> dict[str, Any]:
        return {
            "kind": RECIPE_KIND,
            "stage0": {
                "passes": self.stage0_passes,
                "trainable": "all_parameters",
                "objective": "constant_c25p75_t2_v1",
                "learning_rate": "warmup_hold_cosine_to_pass_60_v1",
                "restore_selected_checkpoint_before_stage_a": True,
            },
            "stage_a": {
                "passes": self.attention_passes,
                "trainable": "pair_embed_and_particle_self_attention_v1",
                "learning_rate": self.attention_learning_rate,
                "objective": "c25p75_plus_support_aligned_block_delta_gram_v1",
            },
            "stage_b": {
                "passes": self.joint_passes,
                "trainable": "all_parameters",
                "learning_rate": self.joint_learning_rate,
                "objective": "c25p75_plus_support_aligned_block_delta_gram_v1",
            },
            "relational_weight": self.relational_weight,
            "trust_weight": self.trust_weight,
            "block_indices": list(self.block_indices),
            "teacher_mode": "eval_no_grad_immediate_parent_v1",
            "alignment": "transported_visible_index_and_family_intersection_v1",
            "channel_comparison": "l2_normalized_token_gram_v1",
            "dense_target_storage": "batch_local_ram_or_device_only_v1",
            "final_test_accessed": False,
        }

    def validate(self) -> None:
        if (
            self.stage0_passes != 60
            or self.attention_passes != 15
            or self.joint_passes != 25
            or self.total_passes != 100
            or self.attention_learning_rate != 3.0e-5
            or self.joint_learning_rate != 1.5e-5
            or self.relational_weight != 0.25
            or self.trust_weight != 0.01
            or self.block_indices != tuple(range(8))
        ):
            raise ValueError("attention re-optimization recipe differs")


DEFAULT_ATTENTION_RECIPE = AttentionStageRecipe()


def normalize_attention_recipe(value: Mapping[str, Any]) -> AttentionStageRecipe:
    if dict(value) != DEFAULT_ATTENTION_RECIPE.payload():
        raise ValueError("attention re-optimization payload differs")
    DEFAULT_ATTENTION_RECIPE.validate()
    return DEFAULT_ATTENTION_RECIPE


def attention_stage(recipe: AttentionStageRecipe, completed_passes: int) -> str:
    recipe.validate()
    if not 0 <= completed_passes < recipe.total_passes:
        raise ValueError("attention stage pass index differs")
    if completed_passes < recipe.stage0_passes:
        return "stage0"
    if completed_passes < recipe.stage0_passes + recipe.attention_passes:
        return "stage_a"
    return "stage_b"


def _qualified_name(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def compile_attention_parameter_registry(model) -> dict[str, Any]:
    """Compile parameter membership through module ownership, never names."""

    import torch

    encoder = getattr(model, "mod", None)
    if encoder is None or not isinstance(getattr(encoder, "pair_embed", None), torch.nn.Module):
        raise TypeError("attention registry lacks canonical pair_embed")
    blocks = tuple(getattr(encoder, "blocks", ()))
    if len(blocks) != 8:
        raise TypeError("attention registry requires eight particle blocks")
    attention_attribute = None
    attention_modules = []
    for candidate in ("attn", "self_attn"):
        modules = tuple(getattr(block, candidate, None) for block in blocks)
        if all(isinstance(module, torch.nn.Module) for module in modules):
            attention_attribute = candidate
            attention_modules = list(modules)
            break
    if attention_attribute is None:
        raise TypeError("installed Weaver particle attention module is unrecognized")

    owned = [encoder.pair_embed, *attention_modules]
    owned_ids = {
        id(parameter)
        for module in owned
        for parameter in module.parameters(recurse=True)
    }
    rows = []
    all_ids = set()
    for name, parameter in model.named_parameters():
        identifier = id(parameter)
        if identifier in all_ids:
            raise RuntimeError("model parameter identity repeats")
        all_ids.add(identifier)
        if identifier in owned_ids:
            rows.append({
                "name": name,
                "shape": list(parameter.shape),
                "elements": int(parameter.numel()),
            })
    if not rows or {id(parameter) for parameter in model.parameters() if id(parameter) in owned_ids} != owned_ids:
        raise RuntimeError("attention parameter registry coverage differs")
    return with_content_hash({
        "contract": PARAMETER_REGISTRY_CONTRACT,
        "schema_version": 1,
        "model_class": _qualified_name(model),
        "encoder_class": _qualified_name(encoder),
        "pair_embed_class": _qualified_name(encoder.pair_embed),
        "particle_block_classes": [_qualified_name(block) for block in blocks],
        "attention_module_attribute": attention_attribute,
        "attention_module_classes": [_qualified_name(module) for module in attention_modules],
        "parameter_rows": rows,
        "parameter_count": len(rows),
        "parameter_elements": sum(row["elements"] for row in rows),
        "selection_policy": "module_ownership_pair_embed_plus_particle_attention_v1",
        "substring_parameter_selection": False,
        "final_test_accessed": False,
    })


def validate_attention_parameter_registry(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=PARAMETER_REGISTRY_CONTRACT,
        expected_schema_version=1,
    )
    rows = value.get("parameter_rows")
    if (
        not isinstance(rows, list)
        or not rows
        or value.get("parameter_count") != len(rows)
        or value.get("parameter_elements") != sum(int(row["elements"]) for row in rows)
        or value.get("attention_module_attribute") not in {"attn", "self_attn"}
        or value.get("selection_policy")
        != "module_ownership_pair_embed_plus_particle_attention_v1"
        or value.get("substring_parameter_selection") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("attention parameter registry differs")
    names = [row.get("name") for row in rows]
    if len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("attention parameter names differ")
    for row in rows:
        if set(row) != {"name", "shape", "elements"} or int(row["elements"]) <= 0:
            raise ValueError("attention parameter row differs")
    return digest


def configure_attention_stage(model, registry: Mapping[str, Any], stage: str):
    """Apply one frozen registry and return exactly the trainable parameters."""

    import torch

    validate_attention_parameter_registry(registry)
    if stage not in {"stage0", "stage_a", "stage_b"}:
        raise ValueError("unknown attention training stage")
    current = compile_attention_parameter_registry(model)
    if current != dict(registry):
        raise ValueError("runtime attention parameter registry drifted")
    allowed = {row["name"] for row in registry["parameter_rows"]}
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = stage != "stage_a" or name in allowed
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled:
            trainable.append(parameter)
    if not trainable or any(not isinstance(parameter, torch.nn.Parameter) for parameter in trainable):
        raise RuntimeError("attention stage has no trainable parameters")
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = set(dict(model.named_parameters())) if stage != "stage_a" else allowed
    if actual != expected:
        raise RuntimeError("attention stage trainable set differs")
    return trainable


def freeze_attention_teacher(teacher):
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher


def assert_frozen_attention_teacher(teacher) -> None:
    invalid = [
        name for name, parameter in teacher.named_parameters()
        if parameter.requires_grad or parameter.grad is not None
    ]
    if teacher.training or invalid:
        raise RuntimeError(f"attention teacher is not frozen: {invalid}")


def attention_parameter_snapshot(model, registry: Mapping[str, Any]) -> dict[str, Any]:
    validate_attention_parameter_registry(registry)
    names = {row["name"] for row in registry["parameter_rows"]}
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in names
    }
    if set(snapshot) != names:
        raise ValueError("attention trust-region snapshot coverage differs")
    return snapshot


def attention_trust_region(model, snapshot: Mapping[str, Any]):
    import torch

    current = dict(model.named_parameters())
    if not snapshot or set(snapshot) - set(current):
        raise ValueError("attention trust-region snapshot differs")
    numerator = None
    denominator = 0
    for name, reference in snapshot.items():
        parameter = current[name]
        if reference.shape != parameter.shape:
            raise ValueError("attention trust-region shape differs")
        value = (parameter.float() - reference.to(parameter.device).float()).square().sum()
        numerator = value if numerator is None else numerator + value
        denominator += parameter.numel()
    if numerator is None or denominator <= 0:
        raise ValueError("attention trust-region is empty")
    result = numerator / denominator
    if not torch.isfinite(result):
        raise FloatingPointError("attention trust-region is nonfinite")
    return result


def _aligned_teacher_indices(student, teacher):
    import torch

    student_mask = student.particle_mask.bool()
    teacher_mask = teacher.particle_mask.bool()
    same_id = student.visible_indices[:, :, None] == teacher.visible_indices[:, None, :]
    same_family = student.family_codes[:, :, None] == teacher.family_codes[:, None, :]
    matches = same_id & same_family & student_mask[:, :, None] & teacher_mask[:, None, :]
    counts = matches.sum(dim=-1)
    reverse_counts = matches.sum(dim=1)
    if bool((counts > 1).any()) or bool((reverse_counts > 1).any()):
        raise ValueError("attention token alignment is not one-to-one")
    aligned = counts == 1
    indices = matches.to(torch.int64).argmax(dim=-1)
    return indices, aligned


def support_aligned_block_delta_gram_loss(
    student,
    teacher,
    *,
    block_indices: tuple[int, ...] = tuple(range(8)),
):
    """Compare channel-rotation-tolerant block-update geometry on shared tokens."""

    import torch
    import torch.nn.functional as functional

    if (
        len(student.block_residual_deltas) != 8
        or len(teacher.block_residual_deltas) != 8
        or block_indices != tuple(range(8))
    ):
        raise ValueError("attention block-delta registry differs")
    indices, common = _aligned_teacher_indices(student, teacher)
    batch = indices.shape[0]
    batch_index = torch.arange(batch, device=indices.device)[:, None]
    diagonal = torch.eye(indices.shape[1], dtype=torch.bool, device=indices.device)[None]
    pair_mask = common[:, :, None] & common[:, None, :] & ~diagonal
    valid_pairs = pair_mask.sum()
    if int(valid_pairs.detach().cpu()) <= 0:
        raise ValueError("attention relational batch has no common token pairs")
    losses = []
    for block_index in block_indices:
        student_delta = functional.normalize(
            student.block_residual_deltas[block_index].float(), dim=-1, eps=1.0e-6,
        )
        teacher_delta = teacher.block_residual_deltas[block_index].detach().float()
        teacher_delta = teacher_delta[batch_index, indices]
        teacher_delta = functional.normalize(teacher_delta, dim=-1, eps=1.0e-6)
        student_gram = student_delta @ student_delta.transpose(1, 2)
        teacher_gram = teacher_delta @ teacher_delta.transpose(1, 2)
        losses.append((student_gram - teacher_gram).square()[pair_mask].mean())
    result = torch.stack(losses).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("attention relational loss is nonfinite")
    return result, {
        "common_tokens": common.sum(),
        "common_ordered_pairs": valid_pairs,
        "active_blocks": len(losses),
    }


def attention_kernel_context(stage: str, device) -> Any:
    """Use the reliable math SDPA backend for the mixed Stage-A freeze graph."""

    import torch

    if stage == "stage_a" and torch.device(device).type == "cuda":
        return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    return nullcontext()


__all__ = [
    "AttentionStageRecipe", "DEFAULT_ATTENTION_RECIPE",
    "PARAMETER_REGISTRY_CONTRACT", "RECIPE_KIND",
    "assert_frozen_attention_teacher", "attention_kernel_context",
    "attention_parameter_snapshot", "attention_stage",
    "attention_trust_region", "compile_attention_parameter_registry",
    "configure_attention_stage", "freeze_attention_teacher",
    "normalize_attention_recipe", "support_aligned_block_delta_gram_loss",
    "validate_attention_parameter_registry",
]
