"""Native 21-feature/15-class fusion with a separate-view caller interface."""
from __future__ import annotations

import torch

from hlt_classification.models.hcwdl_adjacent_fusion_transformer import AdjacentFusionParticleTransformer
from hlt_classification.models.scouting_particle_transformer import ScoutingParticleTransformer


class CMSFusion(AdjacentFusionParticleTransformer):
    def _prepare_injection_bias(self, pair_bias, context_padding):
        # MHA normally adds the same padding mask separately at each of four
        # injections. Merge it once instead. Out-of-place addition also lets
        # the slice's full (primary+context)^2 backing tensor be released after
        # the caller rebinds it. Do not detach: all injections train pair_embed.
        # Compute the full Weaver pair embedding first: cropping its inputs
        # would change its BatchNorm population and therefore the model.
        padding = torch.zeros(
            context_padding.shape, dtype=pair_bias.dtype,
            device=pair_bias.device,
        ).masked_fill(context_padding, float("-inf"))
        return (pair_bias + padding[:, None, None, :]).contiguous(), None

    @staticmethod
    def _tag(primary, context=None):
        features, vectors, mask = primary
        if features.shape[1] != 21 or mask.shape[2] > 200:
            raise ValueError("Native CMS primary shape differs")
        codes = torch.where(mask[:, 0], 1, -1).to(torch.int8)
        if context is None:
            return features, vectors, mask, codes
        cf, cv, cm = context
        cc = torch.where(cm[:, 0], 0, -1).to(torch.int8)
        return (torch.cat((cf, features), -1), torch.cat((cv, vectors), -1),
                torch.cat((cm, mask), -1), torch.cat((cc, codes), -1))

    def forward_fused(self, features, vectors, mask, *context, alpha=1.):
        # Do not even inspect context tensors at zero; inherited zero path
        # also returns before either context encoder or cross attention.
        return super().forward_fused(
            *self._tag((features, vectors, mask), context if alpha else None), alpha=alpha,
        )

    def forward_withdrawal(self, features, vectors, mask, *context, alpha=1.):
        if alpha == 0.:
            from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import WithdrawalOutput
            zero = self.forward_fused(features, vectors, mask, alpha=0.)
            return WithdrawalOutput(zero, zero, 0.)
        return super().forward_withdrawal(*self._tag((features, vectors, mask), context), alpha=alpha)

    def forward(self, features, vectors, mask, *context):
        return self.forward_fused(features, vectors, mask, *context, alpha=1.).logits


def build_model(node):
    # All initial models are cold. Only the withdrawal worker later loads its
    # selected acquisition state. Preserve the caller's seed stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(node["initialization_seed"])
        if node["context_coordinate"] is None:
            return ScoutingParticleTransformer()
        return CMSFusion(context_initialization_seed=node["context_architecture_seed"])
