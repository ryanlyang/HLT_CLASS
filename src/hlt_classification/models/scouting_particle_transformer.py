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
HCWDL_SCOUTING_SURFACE_CONTRACT = "HCWDL_SCOUTING_SURFACES/v1"
HCWDL_NATIVE_OFFLINE_SURFACE_CONTRACT = "HCWDL_NATIVE_OFFLINE_SURFACES/v1"
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


@dataclass(frozen=True)
class HCWDLScoutingSurfaces:
    """Single-forward HLT/Shell-Exact surfaces used only during RKD training."""

    logits: torch.Tensor
    particle_block_2: torch.Tensor
    jet_penultimate: torch.Tensor
    particle_mask: torch.Tensor
    vectors: torch.Tensor
    visible_indices: torch.Tensor
    family_codes: torch.Tensor


@dataclass(frozen=True)
class HCWDLNativeOfflineSurfaces:
    """Single-forward native-offline surfaces with separate latent families."""

    logits: torch.Tensor
    charged_particle_block_2: torch.Tensor
    neutral_particle_block_2: torch.Tensor
    offline_jet_penultimate: torch.Tensor
    charged_mask: torch.Tensor
    neutral_mask: torch.Tensor
    charged_vectors: torch.Tensor
    neutral_vectors: torch.Tensor
    charged_visible_indices: torch.Tensor
    neutral_visible_indices: torch.Tensor


@dataclass(frozen=True)
class _WeaverHCWDLSurfaces:
    output: torch.Tensor
    particle_block_2: torch.Tensor
    penultimate: torch.Tensor
    particle_mask: torch.Tensor
    vectors: torch.Tensor
    visible_indices: torch.Tensor
    family_codes: torch.Tensor | None


def _validate_surface_inputs(
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    visible_indices: torch.Tensor,
    family_codes: torch.Tensor | None,
    *,
    input_dim: int,
) -> None:
    batch, _, particles = features.shape if features.ndim == 3 else (0, 0, 0)
    if features.ndim != 3 or features.shape[1] != input_dim:
        raise ValueError(f"HCWDL features must be [batch,{input_dim},particles]")
    if vectors.shape != (batch, 4, particles):
        raise ValueError("HCWDL vectors must be [batch,4,particles]")
    if mask.shape != (batch, 1, particles) or mask.dtype != torch.bool:
        raise ValueError("HCWDL mask must be boolean [batch,1,particles]")
    if visible_indices.shape != (batch, particles) or visible_indices.dtype not in {
        torch.int16, torch.int32, torch.int64,
    }:
        raise ValueError("HCWDL visible indices must be integer [batch,particles]")
    visible = mask[:, 0]
    if bool((visible_indices[visible] < 0).any()):
        raise ValueError("visible HCWDL token IDs must be nonnegative")
    if bool((visible_indices[~visible] != -1).any()):
        raise ValueError("padded HCWDL token IDs must equal -1")
    for row in range(batch):
        ids = visible_indices[row, visible[row]]
        if ids.numel() != torch.unique(ids).numel():
            raise ValueError("visible HCWDL token IDs must be unique within a jet")
    if family_codes is not None:
        if family_codes.shape != (batch, particles) or family_codes.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise ValueError("HCWDL family codes must be integer [batch,particles]")
        if bool((family_codes[visible] < -128).any()) or bool((family_codes[visible] > 127).any()):
            raise ValueError("visible HCWDL family codes exceed int8")
        if bool((family_codes[~visible] != -1).any()):
            raise ValueError("padded HCWDL family codes must equal -1")


def _attention_mask_blocks(mod: nn.Module) -> tuple[int, ...]:
    policy = mod.block_ids_with_attn_mask
    count = len(mod.blocks)
    if (
        isinstance(policy, (list, tuple))
        and len(policy) == count
        and all(isinstance(item, bool) for item in policy)
    ):
        return tuple(index for index, enabled in enumerate(policy) if enabled)
    try:
        return tuple(index for index in range(count) if index in policy)
    except TypeError as error:
        raise TypeError("installed Weaver attention-mask policy differs") from error


def _forward_hcwdl_weaver_surfaces(
    mod: nn.Module,
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    visible_indices: torch.Tensor,
    family_codes: torch.Tensor | None,
    *,
    input_dim: int,
    output_dim: int,
) -> _WeaverHCWDLSurfaces:
    """Execute one authenticated Weaver path and retain the block-two state.

    Integer bookkeeping is appended only while Weaver's trimmer jointly
    permutes/truncates the particle axis.  It is removed before ``embed`` and
    therefore has no parameter or logit path.
    """

    required = (
        "trimmer", "embed", "pair_embed", "blocks", "_forward_aggregator",
        "fc", "block_ids_with_attn_mask",
    )
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise TypeError(f"installed Weaver lacks HCWDL surfaces: {missing}")
    if len(mod.blocks) != 8:
        raise TypeError("HCWDL surface contract requires eight particle blocks")
    _validate_surface_inputs(
        features, vectors, mask, visible_indices, family_codes,
        input_dim=input_dim,
    )
    metadata = [visible_indices[:, None].to(features.dtype)]
    if family_codes is not None:
        metadata.append(family_codes[:, None].to(features.dtype))
    combined = torch.cat((features, *metadata), dim=1)
    combined, vectors, mask, extra = mod.trimmer(combined, vectors, mask, None)
    if extra is not None:
        raise TypeError("installed Weaver trimmer returned unexpected pair payload")
    model_features = combined[:, :input_dim]
    transported = combined[:, input_dim:]
    expected_metadata = 2 if family_codes is not None else 1
    if transported.shape[1] != expected_metadata:
        raise TypeError("installed Weaver trimmer changed metadata channels")
    rounded = transported.float().round()
    if not torch.equal(transported.float(), rounded):
        raise RuntimeError("HCWDL trimmer corrupted integer metadata")
    transported_ids = rounded[:, 0].to(torch.int64)
    particle_mask = mask.squeeze(1)
    transported_ids = transported_ids.masked_fill(~particle_mask, -1)
    transported_family = None
    if family_codes is not None:
        transported_family = rounded[:, 1].to(torch.int8).masked_fill(
            ~particle_mask, -1,
        )
    padding_mask = ~particle_mask
    hidden = mod.embed(model_features)
    expected_hidden = (features.shape[0], model_features.shape[2], 128)
    if hidden.shape != expected_hidden:
        raise TypeError(
            "installed Weaver embedding layout/width differs from HCWDL: "
            f"{tuple(hidden.shape)} != {expected_hidden}"
        )
    hidden = hidden.masked_fill(~mask.transpose(1, 2), 0)
    pair = mod.pair_embed(vectors, uu=None, mask=mask)
    if pair.ndim != 4 or pair.shape[0] != features.shape[0] or pair.shape[1] != 8 or pair.shape[2:] != (
        model_features.shape[2], model_features.shape[2],
    ):
        raise TypeError("installed Weaver pair-bias layout differs from HCWDL")
    attention_blocks = _attention_mask_blocks(mod)
    block_two = None
    for index, block in enumerate(mod.blocks):
        hidden = block(
            hidden,
            x_cls=None,
            padding_mask=padding_mask,
            attn_mask=pair if index in attention_blocks else None,
        )
        if hidden.shape != expected_hidden:
            raise TypeError("installed Weaver particle-block layout differs")
        if index == 1:
            block_two = hidden
    if block_two is None:
        raise RuntimeError("HCWDL particle block two was not captured")
    penultimate = mod._forward_aggregator(hidden, padding_mask)
    if penultimate.shape != (features.shape[0], 128):
        raise TypeError("installed Weaver aggregator width differs from HCWDL")
    output = mod.fc(penultimate)
    if output.shape != (features.shape[0], output_dim):
        raise TypeError("installed Weaver classifier output differs from HCWDL")
    return _WeaverHCWDLSurfaces(
        output=output,
        particle_block_2=block_two,
        penultimate=penultimate,
        particle_mask=particle_mask,
        vectors=vectors,
        visible_indices=transported_ids,
        family_codes=transported_family,
    )


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

    def forward_hcwdl_surfaces(
        self,
        features: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        visible_indices: torch.Tensor,
        family_codes: torch.Tensor,
    ) -> HCWDLScoutingSurfaces:
        """Expose the registered RKD taps without changing public inference."""

        output = _forward_hcwdl_weaver_surfaces(
            self.mod,
            features,
            vectors,
            mask,
            visible_indices,
            family_codes,
            input_dim=21,
            output_dim=15,
        )
        assert output.family_codes is not None
        return HCWDLScoutingSurfaces(
            logits=output.output,
            particle_block_2=output.particle_block_2,
            jet_penultimate=output.penultimate,
            particle_mask=output.particle_mask,
            vectors=output.vectors,
            visible_indices=output.visible_indices,
            family_codes=output.family_codes,
        )

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
    """Canonical graph plus matched-capacity projections for declared KD surfaces."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in {"R1", "R2", "R3", "R4_PAIR", "R4_GRAM", "R5"}:
            raise ValueError("unknown representation wrapper arm")
        self.arm = arm; self.baseline = ScoutingParticleTransformer()
        count = 2 if arm == "R5" else 0 if arm == "R4_PAIR" else 1
        self.projections = nn.ModuleList(nn.Linear(128, 128, bias=False) for _ in range(count))
        for projection in self.projections: nn.init.eye_(projection.weight)
        self.pair_projection = nn.Conv2d(8, 8, 1, bias=False) if arm == "R4_PAIR" else None
        if self.pair_projection is not None:
            nn.init.eye_(self.pair_projection.weight[:, :, 0, 0])

    def forward(self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.baseline(features, vectors, mask)

    def no_weight_decay(self) -> set[str]:
        return {f"baseline.{name}" for name in self.baseline.no_weight_decay()}

    def forward_representations(self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor):
        output = self.baseline.forward_representations(features, vectors, mask)
        if self.arm == "R1": target = self.projections[0](output.class_token)
        elif self.arm == "R2": target = self.projections[0](output.pooled_particles)
        elif self.arm == "R3": target = self.projections[0](output.late_particles)
        elif self.arm == "R4_PAIR":
            if output.pair_geometry.ndim != 4 or output.pair_geometry.shape[1] != 8:
                raise ValueError("installed Weaver pair geometry is not [batch,8,N,N]")
            target = self.pair_projection(output.pair_geometry)
        elif self.arm == "R4_GRAM":
            projected = self.projections[0](output.late_particles)
            with torch.autocast(device_type=projected.device.type, enabled=False):
                normalized = torch.nn.functional.normalize(projected.float(), dim=-1)
                target = torch.matmul(normalized, normalized.transpose(1, 2))
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

    def forward_hcwdl_surfaces(
        self,
        charged_features: torch.Tensor,
        charged_vectors: torch.Tensor,
        charged_mask: torch.Tensor,
        neutral_features: torch.Tensor,
        neutral_vectors: torch.Tensor,
        neutral_mask: torch.Tensor,
        charged_visible_indices: torch.Tensor,
        neutral_visible_indices: torch.Tensor,
    ) -> HCWDLNativeOfflineSurfaces:
        """Expose TOFF's two latent token spaces in one top-level forward."""

        charged = _forward_hcwdl_weaver_surfaces(
            self.charged_encoder,
            charged_features,
            charged_vectors,
            charged_mask,
            charged_visible_indices,
            None,
            input_dim=19,
            output_dim=128,
        )
        neutral = _forward_hcwdl_weaver_surfaces(
            self.neutral_encoder,
            neutral_features,
            neutral_vectors,
            neutral_mask,
            neutral_visible_indices,
            None,
            input_dim=7,
            output_dim=128,
        )
        if (
            not isinstance(self.classifier, nn.Sequential)
            or len(self.classifier) != 4
            or not isinstance(self.classifier[0], nn.LayerNorm)
            or not isinstance(self.classifier[1], nn.Linear)
            or not isinstance(self.classifier[2], nn.GELU)
            or not isinstance(self.classifier[3], nn.Linear)
        ):
            raise TypeError("TOFF classifier topology differs from HCWDL")
        merged = torch.cat((charged.output, neutral.output), dim=-1)
        penultimate = self.classifier[2](
            self.classifier[1](self.classifier[0](merged))
        )
        if penultimate.shape != (charged_features.shape[0], 128):
            raise TypeError("TOFF penultimate representation width differs")
        logits = self.classifier[3](penultimate)
        if logits.shape != (charged_features.shape[0], 15):
            raise TypeError("TOFF classifier output differs")
        return HCWDLNativeOfflineSurfaces(
            logits=logits,
            charged_particle_block_2=charged.particle_block_2,
            neutral_particle_block_2=neutral.particle_block_2,
            offline_jet_penultimate=penultimate,
            charged_mask=charged.particle_mask,
            neutral_mask=neutral.particle_mask,
            charged_vectors=charged.vectors,
            neutral_vectors=neutral.vectors,
            charged_visible_indices=charged.visible_indices,
            neutral_visible_indices=neutral.visible_indices,
        )


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
    "HCWDL_NATIVE_OFFLINE_SURFACE_CONTRACT", "HCWDL_SCOUTING_SURFACE_CONTRACT",
    "HCWDLNativeOfflineSurfaces", "HCWDLScoutingSurfaces",
    "SCOUTING_PARTICLE_TRANSFORMER_CONTRACT", "ScoutingParticleTransformer",
    "NativeOfflineParticleTransformer", "build_native_offline_particle_transformer",
    "RepresentationScoutingParticleTransformer", "ScoutingRepresentationOutput",
    "build_representation_scouting_particle_transformer",
    "build_scouting_particle_transformer", "scouting_particle_transformer_config",
    "validate_scouting_weaver_fp32_parity",
]
