"""Frozen Weaver Particle Transformer baseline and FP32 parity attestation."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import importlib
from typing import Any

import torch
from torch import nn


PARTICLE_TRANSFORMER_CONTRACT = "hlt_classification_weaver_part_v2"
PARTICLE_TRANSFORMER_SCHEMA_VERSION = 2
FP32_ABSOLUTE_TOLERANCE = 1.0e-6
FP32_RELATIVE_TOLERANCE = 1.0e-5

_CANONICAL_CONFIG: dict[str, Any] = {
    "input_dim": 17,
    "num_classes": 10,
    "pair_input_dim": 4,
    "use_pre_activation_pair": False,
    "embed_dims": [128, 512, 128],
    "pair_embed_dims": [64, 64, 64],
    "num_heads": 8,
    "num_layers": 8,
    "block_params": None,
    "num_cls_layers": 2,
    "cls_block_params": {
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "activation_dropout": 0.0,
    },
    "fc_params": [],
    "activation": "gelu",
    "trim": True,
    "for_inference": False,
}


def canonical_particle_transformer_config() -> dict[str, Any]:
    """Return a defensive copy of the immutable scientific baseline."""

    return deepcopy(_CANONICAL_CONFIG)


def load_weaver_particle_transformer_class() -> type[nn.Module]:
    """Load Weaver lazily so data-only tools do not require PyTorch/Weaver."""

    try:
        module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    except ImportError as error:
        raise ImportError(
            "Weaver is required for ParticleTransformer models; activate the "
            "atlas_kd_tigris environment (with PYTHONNOUSERSITE=1 on Tigris)"
        ) from error
    model_class = getattr(module, "ParticleTransformer", None)
    if model_class is None:
        raise ImportError("installed Weaver module has no ParticleTransformer class")
    return model_class


class CanonicalParticleTransformer(nn.Module):
    """Thin, state-dictionary-transparent adapter around Weaver."""

    def __init__(self) -> None:
        super().__init__()
        model_class = load_weaver_particle_transformer_class()
        self.mod = model_class(**canonical_particle_transformer_config())

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}


def build_particle_transformer() -> CanonicalParticleTransformer:
    """Build a freshly initialized canonical baseline."""

    return CanonicalParticleTransformer()


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.detach() - right.detach()).abs().max().cpu())


def _allclose(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        torch.allclose(
            left,
            right,
            atol=FP32_ABSOLUTE_TOLERANCE,
            rtol=FP32_RELATIVE_TOLERANCE,
        )
    )


def validate_weaver_fp32_parity(
    *,
    device: str = "cpu",
    seed: int = 20260730,
    batch_size: int = 4,
    particles: int = 12,
) -> dict[str, Any]:
    """Compare the adapter against the installed Weaver implementation.

    The direct model and wrapped model are independent instances with identical
    state.  The test is authoritative only in FP32 with autocast disabled.
    """

    if batch_size < 2 or particles < 4:
        raise ValueError("parity requires batch_size >= 2 and particles >= 4")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("parity device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requested but CUDA is unavailable")

    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_class = load_weaver_particle_transformer_class()
    direct = model_class(**canonical_particle_transformer_config()).to(target)
    wrapped = build_particle_transformer().to(target)
    wrapped.mod.load_state_dict(direct.state_dict(), strict=True)
    direct.eval()
    wrapped.eval()

    direct_state = direct.state_dict()
    wrapped_state = wrapped.state_dict()
    expected_wrapped_keys = {f"mod.{key}" for key in direct_state}
    state_keys_exact = set(wrapped_state) == expected_wrapped_keys
    state_values_exact = state_keys_exact and all(
        torch.equal(value, wrapped_state[f"mod.{key}"])
        for key, value in direct_state.items()
    )

    generator = torch.Generator(device=target)
    generator.manual_seed(seed + 1)
    feature_base = torch.randn(
        batch_size, 17, particles, generator=generator, device=target
    )
    vector_base = torch.randn(
        batch_size, 4, particles, generator=generator, device=target
    )
    vector_base[:, 3, :] = vector_base[:, :3, :].square().sum(1).add(1.0).sqrt()
    points = torch.randn(
        batch_size,
        2,
        particles,
        generator=generator,
        device=target,
        requires_grad=True,
    )
    mask = torch.ones(batch_size, 1, particles, dtype=torch.bool, device=target)
    # Preserve padding coverage without creating a masked-masked pair of
    # identical zero four-vectors.  Weaver's Lorentz pair features have an
    # undefined input derivative for that artificial zero/zero pair even
    # though deployable inputs never differentiate their four-vectors.
    mask[0, :, -1] = False
    mask[1, :, -2] = False
    mask_before = mask.clone()

    direct_features = feature_base.detach().clone().requires_grad_(True)
    direct_vectors = vector_base.detach().clone().requires_grad_(True)
    wrapped_features = feature_base.detach().clone().requires_grad_(True)
    wrapped_vectors = vector_base.detach().clone().requires_grad_(True)

    autocast_context = (
        torch.autocast(device_type=target.type, enabled=False)
        if hasattr(torch, "autocast")
        else nullcontext()
    )
    with autocast_context:
        direct_logits = direct(
            direct_features, v=direct_vectors, mask=mask
        )
        wrapped_logits = wrapped(
            points, wrapped_features, wrapped_vectors, mask
        )
        loss_weights = torch.linspace(
            0.25,
            1.25,
            wrapped_logits.numel(),
            dtype=torch.float32,
            device=target,
        ).reshape_as(wrapped_logits)
        direct_loss = (direct_logits * loss_weights).sum()
        wrapped_loss = (wrapped_logits * loss_weights).sum()

    direct_loss.backward()
    wrapped_loss.backward()

    direct_parameter_grads = {
        name: parameter.grad for name, parameter in direct.named_parameters()
    }
    wrapped_parameter_grads = {
        name.removeprefix("mod."): parameter.grad
        for name, parameter in wrapped.named_parameters()
    }
    parameter_names_exact = (
        set(direct_parameter_grads) == set(wrapped_parameter_grads)
    )
    missing_parameter_grads = sorted(
        name
        for name, gradient in direct_parameter_grads.items()
        if gradient is None or wrapped_parameter_grads.get(name) is None
    )
    parameter_gradient_maximum = 0.0
    parameter_gradients_close = parameter_names_exact and not missing_parameter_grads
    if parameter_gradients_close:
        for name, left_gradient in direct_parameter_grads.items():
            right_gradient = wrapped_parameter_grads[name]
            assert left_gradient is not None and right_gradient is not None
            parameter_gradient_maximum = max(
                parameter_gradient_maximum,
                _maximum_difference(left_gradient, right_gradient),
            )
            parameter_gradients_close &= _allclose(
                left_gradient, right_gradient
            )

    logits_close = _allclose(direct_logits, wrapped_logits)
    feature_gradients_close = (
        direct_features.grad is not None
        and wrapped_features.grad is not None
        and _allclose(direct_features.grad, wrapped_features.grad)
    )
    vector_gradients_close = (
        direct_vectors.grad is not None
        and wrapped_vectors.grad is not None
        and _allclose(direct_vectors.grad, wrapped_vectors.grad)
    )
    mask_exact = torch.equal(mask, mask_before)
    fp32_exact = (
        direct_logits.dtype == torch.float32
        and wrapped_logits.dtype == torch.float32
        and direct_features.grad is not None
        and direct_features.grad.dtype == torch.float32
        and wrapped_features.grad is not None
        and wrapped_features.grad.dtype == torch.float32
    )
    points_ignored = points.grad is None
    required_gradients = (
        direct_features.grad,
        wrapped_features.grad,
        direct_vectors.grad,
        wrapped_vectors.grad,
        *direct_parameter_grads.values(),
        *wrapped_parameter_grads.values(),
    )
    required_tensors_finite = bool(
        torch.isfinite(direct_logits).all()
        and torch.isfinite(wrapped_logits).all()
        and all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in required_gradients
        )
    )

    checks = {
        "logits_close": logits_close,
        "feature_gradients_close": feature_gradients_close,
        "lorentz_vector_gradients_close": vector_gradients_close,
        "parameter_names_exact": parameter_names_exact,
        "parameter_gradients_close": parameter_gradients_close,
        "state_dictionary_keys_exact": state_keys_exact,
        "state_dictionary_values_exact": state_values_exact,
        "mask_exact": mask_exact,
        "points_ignored": points_ignored,
        "fp32_outputs_and_gradients": fp32_exact,
        "required_outputs_and_gradients_finite": required_tensors_finite,
        "mixed_precision_disabled": True,
        "trim_enabled": canonical_particle_transformer_config()["trim"] is True,
    }
    report = {
        "contract": PARTICLE_TRANSFORMER_CONTRACT,
        "schema_version": PARTICLE_TRANSFORMER_SCHEMA_VERSION,
        "authoritative_path": "installed_weaver_fp32",
        "device": str(target),
        "seed": seed,
        "batch_size": batch_size,
        "particles": particles,
        "masked_particles_per_row": (~mask[:, 0]).sum(dim=1).cpu().tolist(),
        "torch_version": torch.__version__,
        "weaver_module": model_class.__module__,
        "config": canonical_particle_transformer_config(),
        "absolute_tolerance": FP32_ABSOLUTE_TOLERANCE,
        "relative_tolerance": FP32_RELATIVE_TOLERANCE,
        "maximum_absolute_errors": {
            "logits": _maximum_difference(direct_logits, wrapped_logits),
            "feature_gradients": _maximum_difference(
                direct_features.grad, wrapped_features.grad
            ),
            "lorentz_vector_gradients": _maximum_difference(
                direct_vectors.grad, wrapped_vectors.grad
            ),
            "parameter_gradients": parameter_gradient_maximum,
        },
        "missing_parameter_gradients": missing_parameter_grads,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report


def validate_weaver_bf16_finiteness(
    *,
    device: str = "cuda",
    seed: int = 20260730,
    batch_size: int = 4,
    particles: int = 12,
) -> dict[str, Any]:
    """Exercise BF16 forward/backward as a non-authoritative path check."""

    if batch_size < 1 or particles < 1:
        raise ValueError("batch_size and particles must be positive")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("BF16 device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA BF16 requested but CUDA is unavailable")
    if target.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_particle_transformer().to(target)
    model.train()
    features = torch.randn(
        batch_size, 17, particles, device=target, requires_grad=True
    )
    vectors = torch.randn(
        batch_size, 4, particles, device=target, requires_grad=True
    )
    points = torch.randn(batch_size, 2, particles, device=target)
    mask = torch.ones(batch_size, 1, particles, dtype=torch.bool, device=target)
    if particles > 1:
        mask[0, :, -1] = False
    mask_before = mask.clone()

    with torch.autocast(
        device_type=target.type, dtype=torch.bfloat16, enabled=True
    ):
        logits = model(points, features, vectors, mask)
        loss = logits.square().mean()
    loss.backward()

    parameter_gradients = [
        parameter.grad for parameter in model.parameters()
    ]
    checks = {
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "feature_gradients_finite": (
            features.grad is not None
            and bool(torch.isfinite(features.grad).all())
        ),
        "lorentz_vector_gradients_finite": (
            vectors.grad is not None
            and bool(torch.isfinite(vectors.grad).all())
        ),
        "parameter_gradients_present": all(
            gradient is not None for gradient in parameter_gradients
        ),
        "parameter_gradients_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in parameter_gradients
        ),
        "mask_exact": torch.equal(mask, mask_before),
        "autocast_bf16_requested": True,
    }
    return {
        "contract": PARTICLE_TRANSFORMER_CONTRACT,
        "schema_version": PARTICLE_TRANSFORMER_SCHEMA_VERSION,
        "path": "non_authoritative_bf16_finiteness",
        "device": str(target),
        "seed": seed,
        "batch_size": batch_size,
        "particles": particles,
        "torch_version": torch.__version__,
        "config": canonical_particle_transformer_config(),
        "logits_dtype": str(logits.dtype),
        "checks": checks,
        "passed": all(checks.values()),
    }
