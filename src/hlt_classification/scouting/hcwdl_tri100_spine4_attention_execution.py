"""Installed-Weaver acceptance for attention re-optimization workers."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .hcwdl_attention_reoptimization import (
    DEFAULT_ATTENTION_RECIPE, attention_kernel_context,
    compile_attention_parameter_registry,
    configure_attention_stage, freeze_attention_teacher,
    support_aligned_block_delta_gram_loss,
    validate_attention_parameter_registry,
)
from .hcwdl_homotopy import HomotopyCoordinate, PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_homotopy_stream import iterate_unified_balanced_batches
from .hcwdl_tri100_spine4_attention_contracts import (
    EXECUTION_ACCEPTANCE_CONTRACT, PARAMETER_LOCK_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_attention_graph import EXECUTION
from .hcwdl_tri100_spine4_bottleneck_execution import (
    run_execution_acceptance as run_base_acceptance,
    validate_execution_acceptance as validate_base_acceptance,
)
from .hcwdl_tri100_spine4_persistent_support import validate_support_audit
from .hcwdl_unified_balanced_runner import _load_common
from .training import derive_seed


def _batch(spec: Mapping[str, Any], coordinate: HomotopyCoordinate):
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    split, _, _, selections, assignments, balanced = _load_common(foundation)
    return next(iterate_unified_balanced_batches(
        split, data_root=foundation["data_root"], role="validation",
        assignment_store=assignments["validation"],
        coupling_store=balanced["validation"],
        row_selection=selections["validation"], coordinate=coordinate,
        repair_seed=derive_seed(
            int(spec["replicate_seed"]), "tri60/repair/shared_v1",
        ),
        batch_size=2, workers=1, source_index=0,
        include_training_metadata=True,
        support_policy=PERSISTENT_HLT_SUPPORT_POLICY,
    ))


def _tensors(batch, device):
    import torch

    view = batch["privileged"]
    features = torch.as_tensor(view.features, dtype=torch.float32, device=device)
    vectors = torch.as_tensor(view.vectors, dtype=torch.float32, device=device)
    mask = torch.as_tensor(view.mask, dtype=torch.bool, device=device)
    if mask.ndim == 2:
        mask = mask[:, None]
    visible = torch.as_tensor(view.visible_indices, dtype=torch.long, device=device)
    family = torch.as_tensor(view.family_codes, dtype=torch.int8, device=device)
    labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
    return features, vectors, mask, visible, family, labels


def run_attention_execution_acceptance(
    *, spec: Mapping[str, Any], source_commit: str, device: str = "cuda",
    require_production: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    base = run_base_acceptance(
        spec=spec, source_commit=source_commit, device=device,
        require_production=require_production,
    )
    base_hash = validate_base_acceptance(base, spec=spec)
    target = torch.device(device)
    torch.manual_seed(derive_seed(int(spec["replicate_seed"]), "attention/preflight/student"))
    student = build_scouting_particle_transformer().to(target)
    torch.manual_seed(derive_seed(int(spec["replicate_seed"]), "attention/preflight/teacher"))
    teacher = freeze_attention_teacher(
        build_scouting_particle_transformer().to(target)
    )
    registry = compile_attention_parameter_registry(student)
    registry_hash = validate_attention_parameter_registry(registry)
    parameter_lock = artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "graph": spec["parents"]["graph"],
            "recipe": spec["parents"]["recipe"],
            "base_execution_acceptance": base_hash,
        },
        "registry": registry,
        "registry_sha256": registry_hash,
        "installed_weaver_compiled": True,
        "final_test_accessed": False,
    }, contract=PARAMETER_LOCK_CONTRACT)
    configure_attention_stage(student, registry, "stage_a")
    parent = _batch(spec, HomotopyCoordinate(0, 1, 0, 1))
    child = _batch(spec, HomotopyCoordinate(1, 2, 0, 1))
    if not np.array_equal(parent["identity_digests"], child["identity_digests"]):
        raise ValueError("attention acceptance parent/student identities differ")
    sf, sv, sm, si, sc, labels = _tensors(child, target)
    tf, tv, tm, ti, tc, teacher_labels = _tensors(parent, target)
    if not torch.equal(labels, teacher_labels):
        raise ValueError("attention acceptance labels differ")
    with attention_kernel_context("stage_a", target):
        with torch.autocast(
            device_type=target.type,
            dtype=torch.bfloat16,
            enabled=target.type == "cuda",
        ):
            student_surfaces = student.forward_attention_reoptimization_surfaces(
                sf, sv, sm, si, sc,
            )
            with torch.no_grad():
                teacher_surfaces = teacher.forward_attention_reoptimization_surfaces(
                    tf, tv, tm, ti, tc,
                )
    relational, diagnostics = support_aligned_block_delta_gram_loss(
        student_surfaces, teacher_surfaces,
    )
    ce = torch.nn.functional.cross_entropy(student_surfaces.logits.float(), labels)
    loss = ce + DEFAULT_ATTENTION_RECIPE.relational_weight * relational
    with attention_kernel_context("stage_a", target):
        loss.backward()
    registered = {row["name"] for row in registry["parameter_rows"]}
    gradient_names = {
        name for name, parameter in student.named_parameters()
        if parameter.grad is not None
    }
    if gradient_names != registered:
        raise RuntimeError("attention acceptance gradient registry differs")
    gradient_norm = torch.stack([
        parameter.grad.detach().float().norm()
        for parameter in student.parameters() if parameter.grad is not None
    ]).norm()
    observed = {
        "ce": float(ce.detach().cpu()),
        "relational": float(relational.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "common_tokens": int(diagnostics["common_tokens"].detach().cpu()),
        "common_ordered_pairs": int(
            diagnostics["common_ordered_pairs"].detach().cpu()
        ),
    }
    if (
        any(not math.isfinite(value) for name, value in observed.items()
            if name in {"ce", "relational", "gradient_norm"})
        or observed["gradient_norm"] <= 0
        or observed["common_ordered_pairs"] <= 0
    ):
        raise RuntimeError("attention acceptance result is invalid")
    configure_attention_stage(student, registry, "stage_b")
    if not all(parameter.requires_grad for parameter in student.parameters()):
        raise RuntimeError("attention acceptance Stage B registry differs")
    acceptance = artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "graph": spec["parents"]["graph"],
            "recipe": spec["parents"]["recipe"],
            "support_audit": validate_support_audit(
                load_json(spec["artifact_paths"]["support_audit"]), spec=spec,
            ),
            "base_execution_acceptance": base_hash,
            "parameter_lock": parameter_lock["content_hash"],
        },
        "source_commit": source_commit,
        "execution": dict(EXECUTION),
        "base_execution_acceptance": base,
        "observed": observed,
        "stage_a_gradients_exactly_registered": True,
        "stage_b_all_parameters_trainable": True,
        "teacher_eval_no_grad": True,
        "dense_targets_published": False,
        "genuine_tigris_single_gh200_worker": base[
            "genuine_tigris_single_gh200_worker"
        ],
        "passed": True,
        "final_test_accessed": False,
    }, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    return parameter_lock, acceptance


def validate_parameter_lock(value: Mapping[str, Any], *, spec: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=PARAMETER_LOCK_CONTRACT)
    registry = value.get("registry")
    if (
        not isinstance(registry, Mapping)
        or value.get("registry_sha256")
        != validate_attention_parameter_registry(registry)
        or value.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or value.get("parents", {}).get("graph") != spec["parents"]["graph"]
        or value.get("parents", {}).get("recipe") != spec["parents"]["recipe"]
        or value.get("installed_weaver_compiled") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("attention parameter lock differs")
    return digest


def validate_attention_execution_acceptance(
    value: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    parameter = load_json(spec["artifact_paths"]["parameter_lock"])
    parameter_hash = validate_parameter_lock(parameter, spec=spec)
    base = value.get("base_execution_acceptance")
    if not isinstance(base, Mapping):
        raise ValueError("attention base acceptance is absent")
    base_hash = validate_base_acceptance(base, spec=spec)
    observed = value.get("observed", {})
    if (
        value.get("parents") != {
            "campaign_spec": spec["content_hash"],
            "graph": spec["parents"]["graph"],
            "recipe": spec["parents"]["recipe"],
            "support_audit": validate_support_audit(
                load_json(spec["artifact_paths"]["support_audit"]), spec=spec,
            ),
            "base_execution_acceptance": base_hash,
            "parameter_lock": parameter_hash,
        }
        or value.get("execution") != dict(EXECUTION)
        or value.get("stage_a_gradients_exactly_registered") is not True
        or value.get("stage_b_all_parameters_trainable") is not True
        or value.get("teacher_eval_no_grad") is not True
        or value.get("dense_targets_published") is not False
        or value.get("genuine_tigris_single_gh200_worker") is not True
        or value.get("passed") is not True
        or value.get("final_test_accessed") is not False
        or int(observed.get("common_ordered_pairs", 0)) <= 0
        or any(
            not math.isfinite(float(observed.get(name, float("nan"))))
            for name in ("ce", "relational", "gradient_norm")
        )
    ):
        raise ValueError("attention execution acceptance differs")
    return digest


__all__ = [
    "run_attention_execution_acceptance", "validate_attention_execution_acceptance",
    "validate_parameter_lock",
]
