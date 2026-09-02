"""Particle Transformer for the privileged tagged offline+HLT oracle."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from .scouting_particle_transformer import (
    _attention_mask_blocks,
    _weaver_class,
    scouting_particle_transformer_config,
)


TAGGED_CONCAT_MODEL_CONTRACT: Final = (
    "HCWDL_OFFLINE_HLT_TAGGED_CONCAT_PARTICLE_TRANSFORMER/v1"
)
OFFLINE_CONTENT: Final = 0
HLT_CONTENT: Final = 1


class TaggedConcatParticleTransformer(nn.Module):
    """One canonical ParT over ``offline then HLT`` with a domain embedding.

    Source codes are integer training/oracle metadata.  They are jointly
    transported by Weaver's trimmer, removed before numerical embedding, and
    used only to select one of two learned 128-dimensional content vectors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mod = _weaver_class()(**scouting_particle_transformer_config())
        self.content_source_embedding = nn.Embedding(2, 128)
        nn.init.trunc_normal_(self.content_source_embedding.weight, std=0.02)

    def no_weight_decay(self) -> set[str]:
        # Preserve the canonical optimizer rule exactly: only the class token
        # is exempt. The new source embedding follows ordinary weight decay.
        return {"mod.cls_token"}

    def forward(
        self,
        features: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        content_source_codes: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, particles = (
            features.shape if features.ndim == 3 else (0, 0, 0)
        )
        if channels != 21:
            raise ValueError("tagged concatenation features must be [batch,21,tokens]")
        if vectors.shape != (batch, 4, particles):
            raise ValueError("tagged concatenation vectors differ")
        if mask.shape != (batch, 1, particles) or mask.dtype != torch.bool:
            raise ValueError("tagged concatenation mask differs")
        if content_source_codes.shape != (batch, particles):
            raise ValueError("tagged concatenation source codes differ")
        visible = mask[:, 0]
        codes = content_source_codes.to(torch.int64)
        if bool(((codes[visible] < 0) | (codes[visible] > 1)).any()):
            raise ValueError("visible concatenation source code differs")
        if bool((codes[~visible] != -1).any()):
            raise ValueError("padded concatenation source code differs")

        combined = torch.cat((features, codes[:, None].to(features.dtype)), dim=1)
        combined, vectors, mask, extra = self.mod.trimmer(
            combined, vectors, mask, None,
        )
        if extra is not None or combined.shape[1] != 22:
            raise TypeError("installed Weaver trimmer changed tagged metadata")
        transported = combined[:, 21].float()
        if not torch.equal(transported, transported.round()):
            raise RuntimeError("Weaver trimmer corrupted source codes")
        transported = transported.round().to(torch.int64)
        visible = mask[:, 0]
        transported = transported.masked_fill(~visible, -1)
        if bool(((transported[visible] < 0) | (transported[visible] > 1)).any()):
            raise RuntimeError("trimmed visible source codes differ")

        model_features = combined[:, :21]
        hidden = self.mod.embed(model_features)
        expected = (batch, model_features.shape[2], 128)
        if hidden.shape != expected:
            raise TypeError("installed Weaver embedding layout differs")
        source_hidden = self.content_source_embedding(transported.clamp_min(0))
        hidden = (hidden + source_hidden).masked_fill(~visible[..., None], 0)

        padding_mask = ~visible
        pair = self.mod.pair_embed(vectors, uu=None, mask=mask)
        if pair.shape != (batch, 8, model_features.shape[2], model_features.shape[2]):
            raise TypeError("installed Weaver pair-bias layout differs")
        attention_blocks = _attention_mask_blocks(self.mod)
        for index, block in enumerate(self.mod.blocks):
            hidden = block(
                hidden,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=pair if index in attention_blocks else None,
            )
        penultimate = self.mod._forward_aggregator(hidden, padding_mask)
        output = self.mod.fc(penultimate)
        if output.shape != (batch, 15):
            raise TypeError("tagged concatenation classifier output differs")
        return output


def build_tagged_concat_particle_transformer() -> TaggedConcatParticleTransformer:
    return TaggedConcatParticleTransformer()


__all__ = [
    "HLT_CONTENT", "OFFLINE_CONTENT", "TAGGED_CONCAT_MODEL_CONTRACT",
    "TaggedConcatParticleTransformer", "build_tagged_concat_particle_transformer",
]
