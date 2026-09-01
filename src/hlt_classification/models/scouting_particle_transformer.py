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
SPLIT_SCOUTING_PARTICLE_TRANSFORMER_CONTRACT = (
    "hlt_classification_split_scouting_part_v1"
)
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
class HCWDLAttentionReoptimizationSurfaces:
    """Training-only particle-block deltas with authenticated token identity."""

    logits: torch.Tensor
    block_residual_deltas: tuple[torch.Tensor, ...]
    particle_mask: torch.Tensor
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
    block_residual_deltas: tuple[torch.Tensor, ...]


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
    capture_block_residual_deltas: bool = False,
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
    block_residual_deltas: list[torch.Tensor] = []
    for index, block in enumerate(mod.blocks):
        before = hidden
        hidden = block(
            hidden,
            x_cls=None,
            padding_mask=padding_mask,
            attn_mask=pair if index in attention_blocks else None,
        )
        if hidden.shape != expected_hidden:
            raise TypeError("installed Weaver particle-block layout differs")
        if capture_block_residual_deltas:
            block_residual_deltas.append(
                (hidden - before).masked_fill(~particle_mask[..., None], 0)
            )
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
        block_residual_deltas=tuple(block_residual_deltas),
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

    def forward_attention_reoptimization_surfaces(
        self,
        features: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        visible_indices: torch.Tensor,
        family_codes: torch.Tensor,
    ) -> HCWDLAttentionReoptimizationSurfaces:
        """Expose complete particle-block residual updates for training only.

        The ordinary ``forward`` path is untouched.  These are deliberately
        named complete-block deltas: Weaver does not expose a stable public
        hook for a pre-output-projection attention message.
        """

        output = _forward_hcwdl_weaver_surfaces(
            self.mod,
            features,
            vectors,
            mask,
            visible_indices,
            family_codes,
            input_dim=21,
            output_dim=15,
            capture_block_residual_deltas=True,
        )
        if output.family_codes is None or len(output.block_residual_deltas) != 8:
            raise RuntimeError("HCWDL attention-reoptimization surfaces differ")
        return HCWDLAttentionReoptimizationSurfaces(
            logits=output.output,
            block_residual_deltas=output.block_residual_deltas,
            particle_mask=output.particle_mask,
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


def _compact_partition(
    features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor,
    *, charged: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stably compact one exhaustive token partition without dropping tokens.

    The canonical 21-channel schema stores the mutually exclusive charged
    identity flags in channels 2--4.  Unknown/unclassified and neutral tokens
    are assigned to the noncharged stream.  Both streams retain the original
    per-stream token order and remain padded to the original capacity.
    """

    active = mask[:, 0].bool()
    charged_identity = features[:, 2:5].amax(dim=1) > 0.5
    selected = active & (charged_identity if charged else ~charged_identity)
    # Sorting the binary complement is a stable, vectorized compaction:
    # selected tokens first, preserving their canonical order.
    order = torch.argsort((~selected).to(torch.int8), dim=1, stable=True)
    feature_index = order[:, None, :].expand(-1, features.shape[1], -1)
    vector_index = order[:, None, :].expand(-1, vectors.shape[1], -1)
    compact_features = torch.gather(features, 2, feature_index)
    compact_vectors = torch.gather(vectors, 2, vector_index)
    lengths = selected.sum(dim=1)
    positions = torch.arange(features.shape[2], device=features.device)[None, :]
    compact_mask = (positions < lengths[:, None])[:, None, :]
    compact_features = compact_features.masked_fill(~compact_mask, 0)
    compact_vectors = compact_vectors.masked_fill(~compact_mask, 0)
    return compact_features, compact_vectors, compact_mask


class SplitScoutingParticleTransformer(nn.Module):
    """Common-schema charged/noncharged two-stream architecture control.

    Unlike canonical TOFF this adapter consumes exactly the same unified
    21-channel particle view as :class:`ScoutingParticleTransformer`.  It is
    therefore suitable for a clean input-by-architecture factorial; it is not
    represented as the native 19/7 TOFF adapter.
    """

    def __init__(self) -> None:
        super().__init__()
        model_class = _weaver_class()
        common = scouting_particle_transformer_config()
        common["num_classes"] = 128
        # Retain the native TOFF topology: two full eight-block encoders and
        # the same 256->128->15 fusion shape. The 21-channel common schema is
        # the deliberate difference. Exact parameter counts are reported
        # because this architecture is necessarily larger than the unified arm.
        self.charged_encoder = model_class(**dict(common))
        self.noncharged_encoder = model_class(**dict(common))
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 15),
        )

    def forward(
        self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 21:
            raise ValueError("split Scouting features must be [batch,21,particles]")
        if vectors.shape != (features.shape[0], 4, features.shape[2]):
            raise ValueError("split Scouting vectors must be [batch,4,particles]")
        if mask.shape != (features.shape[0], 1, features.shape[2]):
            raise ValueError("split Scouting mask must be [batch,1,particles]")
        charged = _compact_partition(features, vectors, mask, charged=True)
        noncharged = _compact_partition(features, vectors, mask, charged=False)
        charged_embedding = self._encode_nonempty(self.charged_encoder, charged)
        noncharged_embedding = self._encode_nonempty(
            self.noncharged_encoder, noncharged,
        )
        return self.classifier(torch.cat((charged_embedding, noncharged_embedding), dim=-1))

    @staticmethod
    def _encode_nonempty(encoder: nn.Module, partition) -> torch.Tensor:
        """Encode only nonempty rows; absence is the exact zero embedding."""

        features, vectors, mask = partition
        rows = mask[:, 0].any(dim=1)
        if rows.any():
            encoded = encoder(features[rows], v=vectors[rows], mask=mask[rows])
            # Weaver follows the active autocast policy and therefore returns
            # BF16 during Tigris training even though the cached inputs are
            # FP32.  Allocate from ``encoded`` so index_copy is dtype exact;
            # allocating from ``features`` makes the scatter fail under BF16.
            indices = torch.nonzero(rows, as_tuple=False).flatten()
            output = encoded.new_zeros((features.shape[0], encoded.shape[1]))
            return output.index_copy(0, indices, encoded)
        return features.new_zeros((features.shape[0], 128))

    def no_weight_decay(self) -> set[str]:
        return {
            "charged_encoder.cls_token", "noncharged_encoder.cls_token",
        }


def build_split_scouting_particle_transformer() -> SplitScoutingParticleTransformer:
    return SplitScoutingParticleTransformer()


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


def validate_native_offline_weaver_fp32_parity(
    *, device: str = "cpu", seed: int = 20260811, batch_size: int = 3,
    charged_particles: int = 12, neutral_particles: int = 9,
) -> dict[str, object]:
    """Validate the two-stream TOFF factory against direct Weaver modules.

    The comparison covers logits, feature gradients, vector-gradient finite
    topology, parameter gradients, masks, and the exact charged/neutral
    Weaver configurations in FP32.  It is an installed-runtime acceptance
    check; it does not replace the native teacher's checkpoint lineage.
    """

    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if charged_particles < 2 or neutral_particles < 2:
        raise ValueError("native parity requires at least two particles per stream")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requested but unavailable")
    torch.manual_seed(seed)
    model_class = _weaver_class()
    common = scouting_particle_transformer_config()
    common["num_classes"] = 128
    charged_config = dict(common); charged_config["input_dim"] = 19
    neutral_config = dict(common); neutral_config["input_dim"] = 7
    direct_charged = model_class(**charged_config).to(target)
    direct_neutral = model_class(**neutral_config).to(target)
    direct_classifier = nn.Sequential(
        nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 15),
    ).to(target)
    wrapped = build_native_offline_particle_transformer().to(target)
    wrapped.charged_encoder.load_state_dict(direct_charged.state_dict(), strict=True)
    wrapped.neutral_encoder.load_state_dict(direct_neutral.state_dict(), strict=True)
    wrapped.classifier.load_state_dict(direct_classifier.state_dict(), strict=True)
    direct_charged.eval(); direct_neutral.eval(); direct_classifier.eval(); wrapped.eval()

    generator = torch.Generator(device=target).manual_seed(seed + 1)
    charged_feature_base = torch.randn(
        batch_size, 19, charged_particles, generator=generator, device=target,
    )
    neutral_feature_base = torch.randn(
        batch_size, 7, neutral_particles, generator=generator, device=target,
    )
    charged_vector_base = torch.randn(
        batch_size, 4, charged_particles, generator=generator, device=target,
    )
    neutral_vector_base = torch.randn(
        batch_size, 4, neutral_particles, generator=generator, device=target,
    )
    charged_vector_base[:, 3] = charged_vector_base[:, :3].square().sum(1).add(1).sqrt()
    neutral_vector_base[:, 3] = neutral_vector_base[:, :3].square().sum(1).add(1).sqrt()
    charged_mask = torch.ones(
        batch_size, 1, charged_particles, dtype=torch.bool, device=target,
    )
    neutral_mask = torch.ones(
        batch_size, 1, neutral_particles, dtype=torch.bool, device=target,
    )
    charged_mask[0, :, -2:] = False
    neutral_mask[1, :, -1:] = False
    charged_mask_before = charged_mask.clone(); neutral_mask_before = neutral_mask.clone()

    direct_cf = charged_feature_base.clone().requires_grad_(True)
    wrapped_cf = charged_feature_base.clone().requires_grad_(True)
    direct_nf = neutral_feature_base.clone().requires_grad_(True)
    wrapped_nf = neutral_feature_base.clone().requires_grad_(True)
    direct_cv = charged_vector_base.clone().requires_grad_(True)
    wrapped_cv = charged_vector_base.clone().requires_grad_(True)
    direct_nv = neutral_vector_base.clone().requires_grad_(True)
    wrapped_nv = neutral_vector_base.clone().requires_grad_(True)
    context = (
        torch.autocast(device_type=target.type, enabled=False)
        if hasattr(torch, "autocast") else nullcontext()
    )
    with context:
        direct_logits = direct_classifier(torch.cat((
            direct_charged(direct_cf, v=direct_cv, mask=charged_mask),
            direct_neutral(direct_nf, v=direct_nv, mask=neutral_mask),
        ), dim=-1))
        wrapped_logits = wrapped(
            wrapped_cf, wrapped_cv, charged_mask,
            wrapped_nf, wrapped_nv, neutral_mask,
        )
        weights = torch.linspace(
            .25, 1.25, direct_logits.numel(), device=target,
        ).reshape_as(direct_logits)
        (direct_logits * weights).sum().backward()
        (wrapped_logits * weights).sum().backward()

    feature_gradients_close = bool(
        torch.allclose(direct_cf.grad, wrapped_cf.grad, atol=FP32_ATOL, rtol=FP32_RTOL)
        and torch.allclose(direct_nf.grad, wrapped_nf.grad, atol=FP32_ATOL, rtol=FP32_RTOL)
    )
    vector_gradient_topology_exact = bool(
        torch.equal(torch.isfinite(direct_cv.grad), torch.isfinite(wrapped_cv.grad))
        and torch.equal(torch.isfinite(direct_nv.grad), torch.isfinite(wrapped_nv.grad))
    )
    direct_parameters = list(direct_charged.parameters()) + list(
        direct_neutral.parameters()
    ) + list(direct_classifier.parameters())
    wrapped_parameters = list(wrapped.charged_encoder.parameters()) + list(
        wrapped.neutral_encoder.parameters()
    ) + list(wrapped.classifier.parameters())
    parameter_gradients_close = all(
        left.grad is not None and right.grad is not None
        and torch.allclose(left.grad, right.grad, atol=FP32_ATOL, rtol=FP32_RTOL)
        for left, right in zip(direct_parameters, wrapped_parameters, strict=True)
    )
    logits_close = bool(torch.allclose(
        direct_logits, wrapped_logits, atol=FP32_ATOL, rtol=FP32_RTOL,
    ))
    masks_unchanged = bool(
        torch.equal(charged_mask, charged_mask_before)
        and torch.equal(neutral_mask, neutral_mask_before)
    )
    return {
        "contract": "hlt_classification_native_offline_part_parity_v1",
        "passed": bool(
            direct_logits.shape == (batch_size, 15) and logits_close
            and feature_gradients_close and vector_gradient_topology_exact
            and parameter_gradients_close and masks_unchanged
        ),
        "device": str(target), "dtype": "float32",
        "maximum_absolute_difference": float(
            (direct_logits - wrapped_logits).abs().max().detach().cpu()
        ),
        "absolute_tolerance": FP32_ATOL, "relative_tolerance": FP32_RTOL,
        "charged_config": charged_config, "neutral_config": neutral_config,
        "feature_gradients_close": feature_gradients_close,
        "vector_gradient_finite_topology_exact": vector_gradient_topology_exact,
        "parameter_gradients_close": parameter_gradients_close,
        "masks_unchanged": masks_unchanged,
    }


__all__ = [
    "HCWDL_NATIVE_OFFLINE_SURFACE_CONTRACT", "HCWDL_SCOUTING_SURFACE_CONTRACT",
    "HCWDLNativeOfflineSurfaces", "HCWDLScoutingSurfaces",
    "SCOUTING_PARTICLE_TRANSFORMER_CONTRACT",
    "SPLIT_SCOUTING_PARTICLE_TRANSFORMER_CONTRACT",
    "ScoutingParticleTransformer", "SplitScoutingParticleTransformer",
    "NativeOfflineParticleTransformer", "build_native_offline_particle_transformer",
    "RepresentationScoutingParticleTransformer", "ScoutingRepresentationOutput",
    "build_representation_scouting_particle_transformer",
    "build_scouting_particle_transformer", "build_split_scouting_particle_transformer",
    "scouting_particle_transformer_config",
    "validate_native_offline_weaver_fp32_parity",
    "validate_scouting_weaver_fp32_parity",
]
