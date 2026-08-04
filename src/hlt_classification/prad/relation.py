"""Contextual symmetric relation prediction and gated attention bias."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _particle_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 2:
        raise ValueError("particle mask must have shape [B,N] or [B,1,N]")
    return mask.to(dtype=torch.bool)


class ContextualPairRelation(nn.Module):
    """Build the registered symmetric PRAD relation bottleneck."""

    def __init__(
        self,
        *,
        context_dim: int,
        scalar_dim: int,
        categorical_cardinalities: Sequence[int] = (),
        categorical_embedding_dim: int = 8,
        standard_pair_dim: int = 4,
        relation_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or scalar_dim < 0 or relation_dim <= 0:
            raise ValueError("PRAD relation dimensions are invalid")
        self.context_dim = int(context_dim)
        self.scalar_dim = int(scalar_dim)
        self.relation_dim = int(relation_dim)
        self.standard_pair_dim = int(standard_pair_dim)
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(int(cardinality), categorical_embedding_dim)
            for cardinality in categorical_cardinalities
        )
        particle_extra = scalar_dim + len(self.category_embeddings) * int(
            categorical_embedding_dim
        )
        input_dim = 3 * context_dim + standard_pair_dim + 3 * particle_extra
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, relation_dim),
            nn.LayerNorm(relation_dim),
        )

    def forward(
        self,
        context: torch.Tensor,
        standard_pair: torch.Tensor,
        particle_mask: torch.Tensor,
        *,
        scalar_features: torch.Tensor | None = None,
        categorical_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [B,N,context_dim]")
        batch, particles, _ = context.shape
        if standard_pair.shape != (
            batch,
            particles,
            particles,
            self.standard_pair_dim,
        ):
            raise ValueError("standard pair feature shape differs")
        mask = _particle_mask(particle_mask)
        if mask.shape != (batch, particles):
            raise ValueError("relation particle mask shape differs")
        if self.scalar_dim:
            if scalar_features is None or scalar_features.shape != (
                batch,
                particles,
                self.scalar_dim,
            ):
                raise ValueError("relation scalar feature shape differs")
            extras = [scalar_features]
        else:
            extras = []
        if self.category_embeddings:
            if categorical_features is None or categorical_features.shape != (
                batch,
                particles,
                len(self.category_embeddings),
            ):
                raise ValueError("relation categorical feature shape differs")
            extras.extend(
                embedding(categorical_features[..., index].long())
                for index, embedding in enumerate(self.category_embeddings)
            )
        left = context[:, :, None, :]
        right = context[:, None, :, :]
        features = [left + right, (left - right).abs(), left * right]
        if extras:
            particle = torch.cat(extras, dim=-1)
            extra_left = particle[:, :, None, :]
            extra_right = particle[:, None, :, :]
            features.extend(
                [
                    extra_left + extra_right,
                    (extra_left - extra_right).abs(),
                    extra_left * extra_right,
                ]
            )
        # Standard-four quantities are scientifically symmetric. Enforce that
        # property at this boundary so a malformed directional caller cannot
        # silently make the registered relation directional.
        features.append(0.5 * (standard_pair + standard_pair.transpose(1, 2)))
        relation = self.mlp(torch.cat(features, dim=-1))
        relation = 0.5 * (relation + relation.transpose(1, 2))
        pair_mask = mask[:, :, None] & mask[:, None, :]
        return relation * pair_mask[..., None].to(relation.dtype)


class RelationBiasProjector(nn.Module):
    """Project, bound, center, and mask a relation matrix."""

    def __init__(self, relation_dim: int, attention_heads: int) -> None:
        super().__init__()
        if relation_dim <= 0 or attention_heads <= 0:
            raise ValueError("relation projection dimensions are invalid")
        self.relation_dim = int(relation_dim)
        self.attention_heads = int(attention_heads)
        self.projection = nn.Linear(relation_dim, attention_heads)

    def forward(
        self, relation: torch.Tensor, particle_mask: torch.Tensor
    ) -> torch.Tensor:
        if relation.ndim != 4 or relation.shape[-1] != self.relation_dim:
            raise ValueError("relation must have shape [B,N,N,R]")
        batch, query_count, key_count, _ = relation.shape
        if query_count != key_count:
            raise ValueError("relation matrix must be square")
        mask = _particle_mask(particle_mask)
        if mask.shape != (batch, query_count):
            raise ValueError("relation bias mask shape differs")
        raw = 3.0 * torch.tanh(self.projection(relation))
        key_mask = mask[:, None, :, None].to(raw.dtype)
        key_count_tensor = key_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (raw * key_mask).sum(dim=2, keepdim=True) / key_count_tensor
        centered = raw - mean
        pair_mask = (mask[:, :, None] & mask[:, None, :])[..., None]
        centered = centered * pair_mask.to(centered.dtype)
        return centered.permute(0, 3, 1, 2).contiguous()


class GatedRelationBias(nn.Module):
    """Combine the ordinary and privileged biases for later ParT blocks."""

    def __init__(
        self,
        injection_layers: int,
        attention_heads: int,
        *,
        structure: str = "layer_head",
    ) -> None:
        super().__init__()
        if injection_layers <= 0 or attention_heads <= 0:
            raise ValueError("PRAD gate dimensions are invalid")
        if structure not in {"layer_head", "layer", "global"}:
            raise ValueError("PRAD gate structure differs")
        self.injection_layers = int(injection_layers)
        self.attention_heads = int(attention_heads)
        self.structure = structure
        shape = {
            "layer_head": (injection_layers, attention_heads),
            "layer": (injection_layers, 1),
            "global": (1, 1),
        }[structure]
        self.raw_gates = nn.Parameter(torch.zeros(shape))

    @property
    def gates(self) -> torch.Tensor:
        gates = torch.tanh(self.raw_gates)
        if self.structure == "global":
            gates = gates.expand(self.injection_layers, 1)
        return gates.expand(self.injection_layers, self.attention_heads)

    def forward(
        self,
        standard_bias: torch.Tensor,
        privileged_bias: torch.Tensor,
        *,
        injection_layer: int,
        remove_standard_bias: bool = False,
    ) -> torch.Tensor:
        if standard_bias.shape != privileged_bias.shape:
            raise ValueError("standard and privileged bias shapes differ")
        if injection_layer not in range(self.injection_layers):
            raise IndexError("PRAD injection layer lies outside gate table")
        gate = self.gates[injection_layer][None, :, None, None]
        base = torch.zeros_like(standard_bias) if remove_standard_bias else standard_bias
        return base + gate.to(privileged_bias.dtype) * privileged_bias


__all__ = [
    "ContextualPairRelation",
    "GatedRelationBias",
    "RelationBiasProjector",
]
