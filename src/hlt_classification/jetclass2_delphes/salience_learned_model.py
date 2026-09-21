"""Asymmetric JetClass2 two-view fusion with exact primary extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from hlt_classification.models.particle_transformer import (
    load_weaver_particle_transformer_class,
)
from hlt_classification.models.scouting_particle_transformer import (
    _attention_mask_blocks,
)
from .model import DelphesParticleTransformer, model_config


INJECTION_BLOCKS: Final = (2, 4, 6, 8)


def _validate(features, vectors, mask) -> None:
    if features.ndim != 3:
        raise ValueError("Fusion features differ")
    batch, channels, particles = features.shape
    if channels != 17 or vectors.shape != (batch, 4, particles):
        raise ValueError("Fusion particle tensors differ")
    if mask.shape != (batch, 1, particles) or mask.dtype != torch.bool:
        raise ValueError("Fusion mask differs")
    if bool((~torch.isfinite(features)).any()) or bool((~torch.isfinite(vectors)).any()):
        raise ValueError("Nonfinite fusion input")


def _trim_embed(mod, features, vectors, mask):
    _validate(features, vectors, mask)
    features, vectors, mask, extra = mod.trimmer(features, vectors, mask, None)
    if extra is not None:
        raise TypeError("Installed Weaver trimmer returned pair payload")
    hidden = mod.embed(features).masked_fill(~mask.transpose(1, 2), 0)
    pair = mod.pair_embed(vectors, uu=None, mask=mask)
    return hidden, vectors, mask, pair


def _run_blocks(mod, hidden, mask, pair, *, captures=()):
    padding = ~mask[:, 0]
    attention = set(_attention_mask_blocks(mod))
    states = []
    for index, block in enumerate(mod.blocks, start=1):
        hidden = block(
            hidden, x_cls=None, padding_mask=padding,
            attn_mask=pair if index - 1 in attention else None,
        )
        if index in captures:
            states.append(hidden)
    return hidden, tuple(states)


@dataclass(frozen=True)
class FusionOutput:
    logits: torch.Tensor
    primary_states: tuple[torch.Tensor, ...]
    primary_mask: torch.Tensor

    # The shared withdrawal objective uses historical offline/HLT names.
    # Here they mean the lower/primary branch, including on U-side arrows;
    # they do not imply native-HLT content at intermediate coordinates.
    # Return the original tensors so masking and autograd remain unchanged.
    @property
    def hlt_states(self) -> tuple[torch.Tensor, ...]:
        return self.primary_states

    @property
    def hlt_mask(self) -> torch.Tensor:
        return self.primary_mask


@dataclass(frozen=True)
class WithdrawalOutput:
    zero: FusionOutput
    privileged: FusionOutput
    alpha: float


class _CrossInjection(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.value_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0., batch_first=True,
        )
        self.residual_projection = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, primary, context, *, context_padding, pair_bias, alpha):
        batch, primary_tokens, _ = primary.shape
        context_tokens = context.shape[1]
        bias = pair_bias.reshape(
            batch * self.num_heads, primary_tokens, context_tokens,
        )
        message, _ = self.attention(
            self.query_norm(primary), self.key_norm(context),
            self.value_norm(context), key_padding_mask=context_padding,
            attn_mask=bias, need_weights=False,
        )
        return primary + float(alpha) * torch.sigmoid(self.gate_logit) * (
            self.residual_projection(message)
        )


class DelphesAdjacentFusionParticleTransformer(nn.Module):
    """Child-primary ParT receiving one-way residuals from a parent view."""

    def __init__(self, *, context_initialization_seed: int | None = None) -> None:
        super().__init__()
        config = model_config()
        cls = load_weaver_particle_transformer_class()
        self.primary_mod = cls(**config)

        def initialize_context() -> None:
            self.context_mod = cls(**config)
            self.cross_pair_mod = cls(**config)
            embed_dim = int(config["embed_dims"][-1])
            heads = int(config["num_heads"])
            self.injections = nn.ModuleList(
                _CrossInjection(embed_dim, heads) for _ in INJECTION_BLOCKS
            )

        if context_initialization_seed is None:
            initialize_context()
        else:
            if type(context_initialization_seed) is not int or context_initialization_seed < 0:
                raise ValueError("Context architecture seed differs")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(context_initialization_seed)
                initialize_context()
        for name, parameter in self.context_mod.named_parameters():
            parameter.requires_grad_(name.startswith(("embed.", "pair_embed.", "blocks.")))
        for name, parameter in self.cross_pair_mod.named_parameters():
            parameter.requires_grad_(name.startswith("pair_embed."))

    def no_weight_decay(self) -> set[str]:
        return {"primary_mod.cls_token", "context_mod.cls_token", "cross_pair_mod.cls_token"}

    def extract_primary(self) -> DelphesParticleTransformer:
        result = DelphesParticleTransformer()
        loaded = result.mod.load_state_dict(self.primary_mod.state_dict(), strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError("Extracted primary state differs")
        return result

    def _context(self, features, vectors, mask):
        hidden, vectors, mask, pair = _trim_embed(
            self.context_mod, features, vectors, mask,
        )
        hidden, states = _run_blocks(
            self.context_mod, hidden, mask, pair, captures=INJECTION_BLOCKS,
        )
        del hidden
        return states, vectors, mask

    def _prepare_injection_bias(self, pair_bias, context_padding):
        """Default preserves historical execution; new adapters may compact it."""
        return pair_bias, context_padding

    def _paths(
        self, primary_features, primary_vectors, primary_mask,
        context_features, context_vectors, context_mask, *, alpha: float,
        include_zero: bool,
    ):
        if not 0. < float(alpha) <= 1.:
            raise ValueError("Nonzero fusion alpha differs")
        context_states, context_vectors, context_mask = self._context(
            context_features, context_vectors, context_mask,
        )
        initial, primary_vectors, primary_mask, primary_pair = _trim_embed(
            self.primary_mod, primary_features, primary_vectors, primary_mask,
        )
        combined_vectors = torch.cat((primary_vectors, context_vectors), dim=2)
        combined_mask = torch.cat((primary_mask, context_mask), dim=2)
        cross_pair = self.cross_pair_mod.pair_embed(
            combined_vectors, uu=None, mask=combined_mask,
        )[:, :, :initial.shape[1], initial.shape[1]:]
        cross_pair, context_padding = self._prepare_injection_bias(
            cross_pair, ~context_mask[:, 0],
        )
        padding = ~primary_mask[:, 0]
        attention = set(_attention_mask_blocks(self.primary_mod))
        privileged = initial
        zero = initial.clone() if include_zero else None
        privileged_states, zero_states = [], []
        injection = 0
        for block_index, block in enumerate(self.primary_mod.blocks, start=1):
            block_mask = primary_pair if block_index - 1 in attention else None
            privileged = block(
                privileged, x_cls=None, padding_mask=padding,
                attn_mask=block_mask,
            )
            if include_zero:
                zero = block(
                    zero, x_cls=None, padding_mask=padding,
                    attn_mask=block_mask,
                )
            if block_index in INJECTION_BLOCKS:
                if include_zero:
                    zero_states.append(zero)
                privileged = self.injections[injection](
                    privileged, context_states[injection],
                    context_padding=context_padding,
                    pair_bias=cross_pair, alpha=float(alpha),
                )
                privileged_states.append(privileged)
                injection += 1
        privileged_logits = self.primary_mod.fc(
            self.primary_mod._forward_aggregator(privileged, padding),
        )
        privileged_output = FusionOutput(
            privileged_logits, tuple(privileged_states), primary_mask[:, 0],
        )
        if not include_zero:
            return None, privileged_output
        zero_logits = self.primary_mod.fc(
            self.primary_mod._forward_aggregator(zero, padding),
        )
        return FusionOutput(zero_logits, tuple(zero_states), primary_mask[:, 0]), privileged_output

    def forward_fused(
        self, primary_features, primary_vectors, primary_mask,
        context_features=None, context_vectors=None, context_mask=None, *, alpha=1.,
    ) -> FusionOutput:
        _validate(primary_features, primary_vectors, primary_mask)
        if not 0. <= float(alpha) <= 1.:
            raise ValueError("Fusion alpha differs")
        if float(alpha) == 0.:
            logits = self.primary_mod(
                primary_features, v=primary_vectors, mask=primary_mask,
            )
            return FusionOutput(logits, (), primary_mask[:, 0])
        if context_features is None or context_vectors is None or context_mask is None:
            raise ValueError("Nonzero fusion requires context")
        return self._paths(
            primary_features, primary_vectors, primary_mask,
            context_features, context_vectors, context_mask,
            alpha=float(alpha), include_zero=False,
        )[1]

    def forward_withdrawal(
        self, primary_features, primary_vectors, primary_mask,
        context_features=None, context_vectors=None, context_mask=None, *, alpha=1.,
    ) -> WithdrawalOutput:
        _validate(primary_features, primary_vectors, primary_mask)
        if not 0. <= float(alpha) <= 1.:
            raise ValueError("Withdrawal alpha differs")
        if float(alpha) == 0.:
            zero = self.forward_fused(
                primary_features, primary_vectors, primary_mask, alpha=0.,
            )
            return WithdrawalOutput(zero, zero, 0.)
        zero, privileged = self._paths(
            primary_features, primary_vectors, primary_mask,
            context_features, context_vectors, context_mask,
            alpha=float(alpha), include_zero=True,
        )
        return WithdrawalOutput(zero, privileged, float(alpha))

    def forward(
        self, primary_features, primary_vectors, primary_mask,
        context_features, context_vectors, context_mask,
    ):
        return self.forward_fused(
            primary_features, primary_vectors, primary_mask,
            context_features, context_vectors, context_mask, alpha=1.,
        ).logits


class DelphesParameterMatchedSingleViewParticleTransformer(nn.Module):
    """Frozen wider single-view capacity control."""

    def __init__(self) -> None:
        super().__init__()
        config = model_config()
        config.update(embed_dims=[192, 768, 192], pair_embed_dims=[96, 96, 96], num_heads=8)
        self.mod = load_weaver_particle_transformer_class()(**config)

    def forward(self, features, vectors, mask):
        return self.mod(features, v=vectors, mask=mask)

    def no_weight_decay(self):
        return {"mod.cls_token"}


__all__ = [
    "DelphesAdjacentFusionParticleTransformer",
    "DelphesParameterMatchedSingleViewParticleTransformer", "FusionOutput",
    "INJECTION_BLOCKS", "WithdrawalOutput",
]
