"""Canonical 21-input, 15-output Scouting Particle Transformer."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import importlib
from typing import Any

import torch
from torch import nn
from dataclasses import dataclass

SCOUTING_PARTICLE_TRANSFORMER_CONTRACT = "hlt_classification_scouting_part_v1"
FP32_ATOL = 1.0e-6
FP32_RTOL = 1.0e-5

_CONFIG: dict[str, Any] = {
    "input_dim": 21,
    "num_classes": 15,
    "pair_input_dim": 4,
    "use_pre_activation_pair": False,
    "embed_dims": [128, 512, 128],
    "pair_embed_dims": [64, 64, 64],
    "num_heads": 8,
    "num_layers": 8,
    "block_params": None,
    "num_cls_layers": 2,
    "cls_block_params": {
        "dropout": 0.0, "attn_dropout": 0.0, "activation_dropout": 0.0,
    },
    "fc_params": [],
    "activation": "gelu",
    "trim": True,
    "for_inference": False,
}


def scouting_particle_transformer_config() -> dict[str, Any]:
    return deepcopy(_CONFIG)


def _weaver_class() -> type[nn.Module]:
    try:
        module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    except ImportError as error:
        raise ImportError(
            "Weaver is required; use the locked atlas_kd_tigris environment"
        ) from error
    model_class = getattr(module, "ParticleTransformer", None)
    if model_class is None:
        raise ImportError("installed Weaver has no ParticleTransformer class")
    return model_class


class ScoutingParticleTransformer(nn.Module):
    """State-transparent HLT-only adapter; no offline or match argument exists."""

    def __init__(self) -> None:
        super().__init__()
        self.mod = _weaver_class()(**scouting_particle_transformer_config())

    def forward(
        self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 21:
            raise ValueError("Scouting features must be [batch,21,particles]")
        if vectors.shape != (features.shape[0], 4, features.shape[2]):
            raise ValueError("Scouting vectors must be [batch,4,particles]")
        if mask.shape != (features.shape[0], 1, features.shape[2]):
            raise ValueError("Scouting mask must be [batch,1,particles]")
        return self.mod(features, v=vectors, mask=mask)

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}

    def forward_representations(
        self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor,
    ) -> "ScoutingRepresentationOutput":
        """Run the installed Weaver blocks while exposing predeclared late states."""
        mod = self.mod
        required = ("trimmer", "embed", "pair_embed", "blocks", "_forward_aggregator", "fc", "block_ids_with_attn_mask")
        if any(not hasattr(mod, name) for name in required):
            raise TypeError("installed Weaver lacks representation-KD surfaces")
        features, vectors, mask, _ = mod.trimmer(features, vectors, mask, None)
        particle_mask = mask.squeeze(1); padding_mask = ~particle_mask
        hidden = mod.embed(features).masked_fill(~mask.transpose(1, 2), 0)
        pair = mod.pair_embed(vectors, uu=None, mask=mask)
        policy = mod.block_ids_with_attn_mask
        captures: list[torch.Tensor] = []
        capture_ids = {max(0, len(mod.blocks) - 3), len(mod.blocks) - 1}
        for index, block in enumerate(mod.blocks):
            enabled = policy[index] if isinstance(policy, (list, tuple)) and len(policy) == len(mod.blocks) and all(isinstance(item, bool) for item in policy) else index in policy
            hidden = block(hidden, x_cls=None, padding_mask=padding_mask, attn_mask=pair if enabled else None)
            if index in capture_ids: captures.append(hidden)
        pooled = mod._forward_aggregator(hidden, padding_mask)
        logits = mod.fc(pooled)
        mean = (hidden * particle_mask[..., None]).sum(1) / particle_mask.sum(1, keepdim=True).clamp_min(1)
        return ScoutingRepresentationOutput(
            logits=logits, class_token=pooled, pooled_particles=mean,
            late_particles=hidden, late_depths=tuple(captures),
            pair_geometry=pair, particle_mask=particle_mask,
        )


@dataclass(frozen=True)
class ScoutingRepresentationOutput:
    logits: torch.Tensor
    class_token: torch.Tensor
    pooled_particles: torch.Tensor
    late_particles: torch.Tensor
    late_depths: tuple[torch.Tensor, ...]
    pair_geometry: torch.Tensor
    particle_mask: torch.Tensor


class RepresentationScoutingParticleTransformer(nn.Module):
    """Canonical graph plus matched-capacity learned projections for R1--R5."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in {"R1", "R2", "R3", "R4", "R5"}:
            raise ValueError("representation wrapper requires R1--R5")
        self.arm = arm; self.baseline = ScoutingParticleTransformer()
        count = 2 if arm == "R5" else 1
        self.projections = nn.ModuleList(nn.Linear(128, 128, bias=False) for _ in range(count))
        for projection in self.projections:
            nn.init.eye_(projection.weight)

    def forward(self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.baseline(features, vectors, mask)

    def forward_representations(self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor):
        output = self.baseline.forward_representations(features, vectors, mask)
        if self.arm == "R1": target = self.projections[0](output.class_token)
        elif self.arm == "R2": target = self.projections[0](output.pooled_particles)
        elif self.arm == "R3": target = self.projections[0](output.late_particles)
        elif self.arm == "R4":
            projected = self.projections[0](output.late_particles)
            target = torch.matmul(
                torch.nn.functional.normalize(projected, dim=-1),
                torch.nn.functional.normalize(projected, dim=-1).transpose(1, 2),
            )
        else:
            target = tuple(
                projection(value) for projection, value in zip(self.projections, output.late_depths, strict=True)
            )
        return output.logits, target, output.particle_mask


def build_representation_scouting_particle_transformer(arm: str) -> RepresentationScoutingParticleTransformer:
    return RepresentationScoutingParticleTransformer(arm)


def build_scouting_particle_transformer() -> ScoutingParticleTransformer:
    return ScoutingParticleTransformer()


class NativeOfflineParticleTransformer(nn.Module):
    """Nondeployable TOFF/v1 diagnostic with separate charged/neutral encoders."""

    def __init__(self) -> None:
        super().__init__()
        model_class = _weaver_class()
        common = scouting_particle_transformer_config()
        common["num_classes"] = 128
        charged = dict(common); charged["input_dim"] = 19
        neutral = dict(common); neutral["input_dim"] = 7
        self.charged_encoder = model_class(**charged)
        self.neutral_encoder = model_class(**neutral)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 15),
        )

    def forward(
        self, charged_features: torch.Tensor, charged_vectors: torch.Tensor,
        charged_mask: torch.Tensor, neutral_features: torch.Tensor,
        neutral_vectors: torch.Tensor, neutral_mask: torch.Tensor,
    ) -> torch.Tensor:
        charged = self.charged_encoder(
            charged_features, v=charged_vectors, mask=charged_mask,
        )
        neutral = self.neutral_encoder(
            neutral_features, v=neutral_vectors, mask=neutral_mask,
        )
        return self.classifier(torch.cat((charged, neutral), dim=-1))


def build_native_offline_particle_transformer() -> NativeOfflineParticleTransformer:
    return NativeOfflineParticleTransformer()


def validate_scouting_weaver_fp32_parity(
    *, device: str = "cpu", seed: int = 20260805, batch_size: int = 3,
    particles: int = 12,
) -> dict[str, object]:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requested but unavailable")
    torch.manual_seed(seed)
    direct = _weaver_class()(**scouting_particle_transformer_config()).to(target)
    wrapped = build_scouting_particle_transformer().to(target)
    wrapped.mod.load_state_dict(direct.state_dict(), strict=True)
    direct.eval(); wrapped.eval()
    generator = torch.Generator(device=target).manual_seed(seed + 1)
    feature_base = torch.randn(batch_size, 21, particles, generator=generator, device=target)
    vector_base = torch.randn(batch_size, 4, particles, generator=generator, device=target)
    vector_base[:, 3] = vector_base[:, :3].square().sum(1).add(1).sqrt()
    mask = torch.ones(batch_size, 1, particles, dtype=torch.bool, device=target)
    mask[0, :, -2:] = False
    mask_before = mask.clone()
    direct_features = feature_base.clone().requires_grad_(True)
    wrapped_features = feature_base.clone().requires_grad_(True)
    direct_vectors = vector_base.clone().requires_grad_(True)
    wrapped_vectors = vector_base.clone().requires_grad_(True)
    context = torch.autocast(device_type=target.type, enabled=False) if hasattr(torch, "autocast") else nullcontext()
    with context:
        expected = direct(direct_features, v=direct_vectors, mask=mask)
        actual = wrapped(wrapped_features, wrapped_vectors, mask)
        weights = torch.linspace(.25, 1.25, expected.numel(), device=target).reshape_as(expected)
        expected_loss = (expected * weights).sum(); actual_loss = (actual * weights).sum()
    expected_loss.backward(); actual_loss.backward()
    maximum = float((expected - actual).abs().max().cpu())
    feature_gradient_maximum = float((direct_features.grad - wrapped_features.grad).abs().max().cpu())
    vector_gradient_finite_topology = bool(torch.equal(torch.isfinite(direct_vectors.grad), torch.isfinite(wrapped_vectors.grad)))
    parameter_gradients = all(
        left.grad is not None and right.grad is not None
        and torch.allclose(left.grad, right.grad, atol=FP32_ATOL, rtol=FP32_RTOL)
        for left, right in zip(direct.parameters(), wrapped.mod.parameters(), strict=True)
    )
    wrapped.mod.trimmer.enabled = False
    wrapped.eval()
    with torch.inference_mode():
        public_rep_logits = wrapped(feature_base, vector_base, mask)
        manual_rep_logits = wrapped.forward_representations(feature_base, vector_base, mask).logits
    representation_forward_parity = bool(torch.allclose(
        public_rep_logits, manual_rep_logits, atol=FP32_ATOL, rtol=FP32_RTOL,
    ))
    passed = expected.shape == (batch_size, 15) and torch.allclose(
        expected, actual, atol=FP32_ATOL, rtol=FP32_RTOL,
    ) and feature_gradient_maximum <= FP32_ATOL and vector_gradient_finite_topology and parameter_gradients and torch.equal(mask, mask_before) and representation_forward_parity
    return {
        "contract": SCOUTING_PARTICLE_TRANSFORMER_CONTRACT,
        "passed": bool(passed), "maximum_absolute_difference": maximum,
        "absolute_tolerance": FP32_ATOL, "relative_tolerance": FP32_RTOL,
        "device": str(target), "dtype": "float32",
        "feature_gradient_maximum_absolute_difference": feature_gradient_maximum,
        "vector_gradient_finite_topology_exact": vector_gradient_finite_topology,
        "parameter_gradients_close": parameter_gradients,
        "mask_unchanged": bool(torch.equal(mask, mask_before)),
        "representation_forward_parity": representation_forward_parity,
    }


__all__ = [
    "SCOUTING_PARTICLE_TRANSFORMER_CONTRACT", "ScoutingParticleTransformer",
    "NativeOfflineParticleTransformer", "build_native_offline_particle_transformer",
    "RepresentationScoutingParticleTransformer", "ScoutingRepresentationOutput",
    "build_representation_scouting_particle_transformer",
    "build_scouting_particle_transformer", "scouting_particle_transformer_config",
    "validate_scouting_weaver_fp32_parity",
]
