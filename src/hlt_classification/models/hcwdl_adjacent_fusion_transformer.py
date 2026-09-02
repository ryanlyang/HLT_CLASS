"""One-way adjacent-view fusion with an exactly extractable primary ParT."""

from __future__ import annotations

from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
    AnchoredFusionOutput, AnchoredFusionParticleTransformer, WithdrawalOutput,
)
from hlt_classification.models.scouting_particle_transformer import (
    _weaver_class, scouting_particle_transformer_config,
)
from torch import nn


class AdjacentFusionParticleTransformer(AnchoredFusionParticleTransformer):
    """Treat source code 1 as lower primary and code 0 as richer context.

    The inherited implementation has the required zero-initialized one-way
    injections after blocks 2/4/6/8.  Names referring to HLT in that reusable
    implementation denote the future-deployable *primary* branch here.
    """

    def __init__(self, *, context_initialization_seed: int | None = None) -> None:
        super().__init__(
            "O", context_initialization_seed=context_initialization_seed,
        )
        # The context branch supplies particle-block states only.  Its class
        # token, class-attention stack, normalization, and classifier are not
        # on the forward path and are frozen so that the primary branch is the
        # sole trainable owner of the prediction head.
        for name, parameter in self.context_mod.named_parameters():
            parameter.requires_grad_(name.startswith((
                "embed.", "pair_embed.", "blocks.",
            )))
        # This auxiliary module contributes only standard-four pair geometry.
        # Freezing its unused token blocks/head keeps the declared trainable
        # capacity honest and prevents optimizer-only parameter inflation.
        for name, parameter in self.cross_pair_mod.named_parameters():
            parameter.requires_grad_(name.startswith("pair_embed."))

    def extract_primary(self):
        return self.extract_hlt()


class ParameterMatchedSingleViewParticleTransformer(nn.Module):
    """Frozen wider single-view capacity control for the fusion topology."""

    def __init__(self) -> None:
        super().__init__()
        config = scouting_particle_transformer_config()
        config["embed_dims"] = [192, 768, 192]
        config["pair_embed_dims"] = [96, 96, 96]
        config["num_heads"] = 8
        self.mod = _weaver_class()(**config)

    def forward(self, features, vectors, mask):
        return self.mod(features, v=vectors, mask=mask)

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}


__all__ = [
    "AdjacentFusionParticleTransformer", "AnchoredFusionOutput",
    "ParameterMatchedSingleViewParticleTransformer", "WithdrawalOutput",
]
