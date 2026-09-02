"""Privileged offline/HLT fusion models with an exactly removable context path.

The models in this module are training-time or oracle architectures.  The
only deployable object they can produce is the ordinary 21-input HLT Particle
Transformer returned by :meth:`AnchoredFusionParticleTransformer.extract_hlt`.
No matching index is accepted by any forward method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch
from torch import nn

from .hcwdl_tagged_concat_transformer import TaggedConcatParticleTransformer
from .scouting_particle_transformer import (
    ScoutingParticleTransformer, _attention_mask_blocks, _weaver_class,
    scouting_particle_transformer_config,
)


FUSION_MODEL_CONTRACT: Final = "HCWDL_OFFLINE_HLT_FUSION_TRANSFORMER/v1"
OFFLINE_CONTENT: Final = 0
HLT_CONTENT: Final = 1
HLT_DEPLOYABLE_CAPACITY: Final = 200
INJECTION_BLOCKS: Final = (2, 4, 6, 8)


@dataclass(frozen=True)
class AnchoredFusionOutput:
    """Logits and declared HLT states from one anchored forward."""

    logits: torch.Tensor
    hlt_states: tuple[torch.Tensor, ...]
    hlt_mask: torch.Tensor


@dataclass(frozen=True)
class WithdrawalOutput:
    """Paired deployable and privileged routes used by Study C."""

    zero: AnchoredFusionOutput
    privileged: AnchoredFusionOutput
    alpha: float


def _validate_tagged(
    features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor,
    sources: torch.Tensor,
) -> None:
    batch, channels, particles = (
        features.shape if features.ndim == 3 else (0, 0, 0)
    )
    if channels != 21 or vectors.shape != (batch, 4, particles):
        raise ValueError("fusion particle tensors differ")
    if mask.shape != (batch, 1, particles) or mask.dtype != torch.bool:
        raise ValueError("fusion mask differs")
    if sources.shape != (batch, particles):
        raise ValueError("fusion content-source codes differ")
    active = mask[:, 0]
    codes = sources.to(torch.int64)
    if bool(((codes[active] < 0) | (codes[active] > 1)).any()):
        raise ValueError("visible fusion content code differs")
    if bool((codes[~active] != -1).any()):
        raise ValueError("padded fusion content code differs")


def _content_view(
    features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor,
    sources: torch.Tensor, *, code: int, capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stably compact one content family and crop only padded columns."""

    selected = mask[:, 0] & (sources.to(torch.int64) == code)
    order = torch.argsort((~selected).to(torch.int8), dim=1, stable=True)
    features = torch.gather(
        features, 2, order[:, None].expand(-1, features.shape[1], -1),
    )
    vectors = torch.gather(
        vectors, 2, order[:, None].expand(-1, vectors.shape[1], -1),
    )
    lengths = selected.sum(1)
    if capacity is not None:
        lengths = lengths.clamp_max(capacity)
    width = max(1, int(lengths.max().item()))
    if capacity is not None:
        width = min(width, capacity)
    features = features[:, :, :width]
    vectors = vectors[:, :, :width]
    positions = torch.arange(width, device=features.device)[None]
    view_mask = (positions < lengths[:, None])[:, None]
    return (
        features.masked_fill(~view_mask, 0),
        vectors.masked_fill(~view_mask, 0),
        view_mask,
    )


def _trim_embed(mod, features, vectors, mask, embedding=None):
    features, vectors, mask, extra = mod.trimmer(features, vectors, mask, None)
    if extra is not None:
        raise TypeError("installed Weaver trimmer returned pair payload")
    hidden = mod.embed(features)
    if embedding is not None:
        if embedding.ndim != 1 or embedding.shape[0] != hidden.shape[-1]:
            raise ValueError("fusion content/role embedding differs")
        hidden = hidden + embedding.view(1, 1, -1)
    hidden = hidden.masked_fill(~mask.transpose(1, 2), 0)
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


class UntaggedConcatParticleTransformer(nn.Module):
    """Canonical ParT over O+H that deliberately ignores source identity."""

    def __init__(self) -> None:
        super().__init__()
        self.model = ScoutingParticleTransformer()

    def forward(self, features, vectors, mask, content_source_codes):
        _validate_tagged(features, vectors, mask, content_source_codes)
        return self.model(features, vectors, mask)

    def no_weight_decay(self) -> set[str]:
        return {f"model.{name}" for name in self.model.no_weight_decay()}


class SymmetricFusionParticleTransformer(nn.Module):
    """Two local two-block branches followed by six shared particle blocks."""

    def __init__(self, arm: Literal["OO", "HH", "OH"]) -> None:
        super().__init__()
        if arm not in {"OO", "HH", "OH"}:
            raise ValueError("symmetric fusion arm differs")
        self.arm = arm
        config = scouting_particle_transformer_config()
        local_config = dict(config, num_layers=2, num_cls_layers=1)
        shared_config = dict(config, num_layers=6)
        cls = _weaver_class()
        self.left = cls(**local_config)
        self.right = cls(**local_config)
        self.shared = cls(**shared_config)
        self.branch_role_embedding = nn.Embedding(2, 128)
        self.content_embedding = nn.Embedding(2, 128)
        nn.init.trunc_normal_(self.branch_role_embedding.weight, std=.02)
        nn.init.trunc_normal_(self.content_embedding.weight, std=.02)

    def no_weight_decay(self) -> set[str]:
        return {
            "left.cls_token", "right.cls_token", "shared.cls_token",
        }

    def _branch(self, mod, view, *, role: int, content: int):
        features, vectors, mask = view
        token_embedding = (
            self.branch_role_embedding.weight[role]
            + self.content_embedding.weight[content]
        )
        hidden, vectors, mask, pair = _trim_embed(
            mod, features, vectors, mask, token_embedding,
        )
        hidden, _ = _run_blocks(mod, hidden, mask, pair)
        return hidden, vectors, mask

    def forward(self, features, vectors, mask, content_source_codes):
        _validate_tagged(features, vectors, mask, content_source_codes)
        offline = _content_view(
            features, vectors, mask, content_source_codes, code=OFFLINE_CONTENT,
        )
        hlt = _content_view(
            features, vectors, mask, content_source_codes, code=HLT_CONTENT,
        )
        if self.arm == "OO":
            left_view, right_view, contents = offline, offline, (0, 0)
        elif self.arm == "HH":
            left_view, right_view, contents = hlt, hlt, (1, 1)
        else:
            left_view, right_view, contents = offline, hlt, (0, 1)
        left = self._branch(self.left, left_view, role=0, content=contents[0])
        right = self._branch(self.right, right_view, role=1, content=contents[1])
        hidden = torch.cat((left[0], right[0]), dim=1)
        joined_vectors = torch.cat((left[1], right[1]), dim=2)
        joined_mask = torch.cat((left[2], right[2]), dim=2)
        pair = self.shared.pair_embed(joined_vectors, uu=None, mask=joined_mask)
        hidden, _ = _run_blocks(self.shared, hidden, joined_mask, pair)
        pooled = self.shared._forward_aggregator(hidden, ~joined_mask[:, 0])
        logits = self.shared.fc(pooled)
        if logits.shape != (features.shape[0], 15):
            raise TypeError("symmetric fusion classifier output differs")
        return logits


class _CrossInjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(128)
        self.key_norm = nn.LayerNorm(128)
        self.value_norm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True,
        )
        self.residual_projection = nn.Linear(128, 128)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, hlt, context, *, context_padding, pair_bias, alpha: float):
        batch, hlt_tokens, _ = hlt.shape
        context_tokens = context.shape[1]
        bias = pair_bias.reshape(batch * 8, hlt_tokens, context_tokens)
        padding = torch.zeros(
            context_padding.shape, dtype=bias.dtype,
            device=context_padding.device,
        ).masked_fill(context_padding, float("-inf"))
        message, _ = self.attention(
            self.query_norm(hlt), self.key_norm(context),
            self.value_norm(context), key_padding_mask=padding,
            attn_mask=bias, need_weights=False,
        )
        return hlt + float(alpha) * torch.sigmoid(self.gate_logit) * (
            self.residual_projection(message)
        )


class AnchoredFusionParticleTransformer(nn.Module):
    """Canonical HLT ParT with one-way, exactly removable context residuals."""

    def __init__(self, context_domain: Literal["O", "H"] = "O") -> None:
        super().__init__()
        if context_domain not in {"O", "H"}:
            raise ValueError("anchored fusion context domain differs")
        self.context_domain = context_domain
        cls = _weaver_class()
        self.hlt_mod = cls(**scouting_particle_transformer_config())
        self.context_mod = cls(**scouting_particle_transformer_config())
        self.cross_pair_mod = cls(**scouting_particle_transformer_config())
        self.context_content_embedding = nn.Embedding(2, 128)
        nn.init.trunc_normal_(self.context_content_embedding.weight, std=.02)
        self.injections = nn.ModuleList(_CrossInjection() for _ in INJECTION_BLOCKS)

    def no_weight_decay(self) -> set[str]:
        return {
            "hlt_mod.cls_token", "context_mod.cls_token",
            "cross_pair_mod.cls_token",
        }

    def load_hlt_state(self, state: dict[str, torch.Tensor]) -> None:
        projected = {
            name.removeprefix("mod."): value
            for name, value in state.items() if name.startswith("mod.")
        }
        result = self.hlt_mod.load_state_dict(projected, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("anchored HLT warm state differs")

    def load_context_state(self, state: dict[str, torch.Tensor]) -> None:
        projected = {
            name.removeprefix("mod."): value
            for name, value in state.items() if name.startswith("mod.")
        }
        result = self.context_mod.load_state_dict(projected, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("anchored context warm state differs")

    def extract_hlt(self) -> ScoutingParticleTransformer:
        model = ScoutingParticleTransformer()
        result = model.mod.load_state_dict(self.hlt_mod.state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("anchored deployable state projection differs")
        return model

    def _split(self, features, vectors, mask, sources):
        hlt = _content_view(
            features, vectors, mask, sources, code=HLT_CONTENT,
            capacity=HLT_DEPLOYABLE_CAPACITY,
        )
        context_code = OFFLINE_CONTENT if self.context_domain == "O" else HLT_CONTENT
        context = _content_view(
            features, vectors, mask, sources, code=context_code,
            capacity=(None if context_code == OFFLINE_CONTENT else HLT_DEPLOYABLE_CAPACITY),
        )
        return hlt, context, context_code

    def forward_hlt(self, features, vectors, mask) -> torch.Tensor:
        return self.hlt_mod(features, v=vectors, mask=mask)

    def forward_fused(
        self, features, vectors, mask, content_source_codes, *, alpha: float,
    ) -> AnchoredFusionOutput:
        _validate_tagged(features, vectors, mask, content_source_codes)
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("anchored fusion alpha differs")
        hlt = _content_view(
            features, vectors, mask, content_source_codes, code=HLT_CONTENT,
            capacity=HLT_DEPLOYABLE_CAPACITY,
        )
        if float(alpha) == 0.0:
            # This branch is intentionally before every offline/context op.
            logits = self.hlt_mod(hlt[0], v=hlt[1], mask=hlt[2])
            return AnchoredFusionOutput(logits, (), hlt[2][:, 0])
        _, context, context_code = self._split(
            features, vectors, mask, content_source_codes,
        )

        context_embedding = self.context_content_embedding.weight[context_code]
        context_hidden, context_vectors, context_mask, context_pair = _trim_embed(
            self.context_mod, *context, context_embedding,
        )
        context_hidden, context_states = _run_blocks(
            self.context_mod, context_hidden, context_mask, context_pair,
            captures=INJECTION_BLOCKS,
        )
        del context_hidden

        hlt_hidden, hlt_vectors, hlt_mask, hlt_pair = _trim_embed(
            self.hlt_mod, *hlt,
        )
        combined_vectors = torch.cat((hlt_vectors, context_vectors), dim=2)
        combined_mask = torch.cat((hlt_mask, context_mask), dim=2)
        cross_pair = self.cross_pair_mod.pair_embed(
            combined_vectors, uu=None, mask=combined_mask,
        )[:, :, :hlt_hidden.shape[1], hlt_hidden.shape[1]:]
        hlt_padding = ~hlt_mask[:, 0]
        hlt_attention = set(_attention_mask_blocks(self.hlt_mod))
        captures = []
        injection_index = 0
        for block_index, block in enumerate(self.hlt_mod.blocks, start=1):
            hlt_hidden = block(
                hlt_hidden, x_cls=None, padding_mask=hlt_padding,
                attn_mask=hlt_pair if block_index - 1 in hlt_attention else None,
            )
            if block_index in INJECTION_BLOCKS:
                hlt_hidden = self.injections[injection_index](
                    hlt_hidden, context_states[injection_index],
                    context_padding=~context_mask[:, 0],
                    pair_bias=cross_pair, alpha=float(alpha),
                )
                captures.append(hlt_hidden)
                injection_index += 1
        pooled = self.hlt_mod._forward_aggregator(hlt_hidden, hlt_padding)
        logits = self.hlt_mod.fc(pooled)
        return AnchoredFusionOutput(logits, tuple(captures), hlt_mask[:, 0])

    def forward(self, features, vectors, mask, content_source_codes):
        # Oracle calls are privileged; validation of withdrawal models uses
        # ``forward_zero`` explicitly in the training worker.
        return self.forward_fused(
            features, vectors, mask, content_source_codes, alpha=1.0,
        ).logits

    def forward_zero(self, features, vectors, mask, content_source_codes):
        return self.forward_fused(
            features, vectors, mask, content_source_codes, alpha=0.0,
        )

    def forward_withdrawal(
        self, features, vectors, mask, content_source_codes, *, alpha: float,
    ) -> WithdrawalOutput:
        _validate_tagged(features, vectors, mask, content_source_codes)
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("anchored withdrawal alpha differs")
        if float(alpha) == 0.0:
            zero = self.forward_fused(
                features, vectors, mask, content_source_codes, alpha=0.0,
            )
            return WithdrawalOutput(zero=zero, privileged=zero, alpha=0.0)
        hlt, context, context_code = self._split(
            features, vectors, mask, content_source_codes,
        )

        context_embedding = self.context_content_embedding.weight[context_code]
        context_hidden, context_vectors, context_mask, context_pair = _trim_embed(
            self.context_mod, *context, context_embedding,
        )
        context_hidden, context_states = _run_blocks(
            self.context_mod, context_hidden, context_mask, context_pair,
            captures=INJECTION_BLOCKS,
        )
        del context_hidden
        initial, hlt_vectors, hlt_mask, hlt_pair = _trim_embed(
            self.hlt_mod, *hlt,
        )
        combined_vectors = torch.cat((hlt_vectors, context_vectors), dim=2)
        combined_mask = torch.cat((hlt_mask, context_mask), dim=2)
        cross_pair = self.cross_pair_mod.pair_embed(
            combined_vectors, uu=None, mask=combined_mask,
        )[:, :, :initial.shape[1], initial.shape[1]:]
        padding = ~hlt_mask[:, 0]
        attention = set(_attention_mask_blocks(self.hlt_mod))
        zero_hidden = initial
        privileged_hidden = initial.clone()
        zero_states = []
        privileged_states = []
        injection_index = 0
        for block_index, block in enumerate(self.hlt_mod.blocks, start=1):
            block_mask = hlt_pair if block_index - 1 in attention else None
            zero_hidden = block(
                zero_hidden, x_cls=None, padding_mask=padding,
                attn_mask=block_mask,
            )
            privileged_hidden = block(
                privileged_hidden, x_cls=None, padding_mask=padding,
                attn_mask=block_mask,
            )
            if block_index in INJECTION_BLOCKS:
                zero_states.append(zero_hidden)
                privileged_hidden = self.injections[injection_index](
                    privileged_hidden, context_states[injection_index],
                    context_padding=~context_mask[:, 0], pair_bias=cross_pair,
                    alpha=float(alpha),
                )
                privileged_states.append(privileged_hidden)
                injection_index += 1
        zero_logits = self.hlt_mod.fc(
            self.hlt_mod._forward_aggregator(zero_hidden, padding),
        )
        privileged_logits = self.hlt_mod.fc(
            self.hlt_mod._forward_aggregator(privileged_hidden, padding),
        )
        return WithdrawalOutput(
            zero=AnchoredFusionOutput(
                zero_logits, tuple(zero_states), hlt_mask[:, 0],
            ),
            privileged=AnchoredFusionOutput(
                privileged_logits, tuple(privileged_states), hlt_mask[:, 0],
            ),
            alpha=float(alpha),
        )


def build_untagged_concat_particle_transformer():
    return UntaggedConcatParticleTransformer()


def build_tagged_concat_fusion_control():
    return TaggedConcatParticleTransformer()


def build_symmetric_fusion_particle_transformer(arm: str):
    return SymmetricFusionParticleTransformer(arm)  # type: ignore[arg-type]


def build_anchored_fusion_particle_transformer(context_domain: str = "O"):
    return AnchoredFusionParticleTransformer(context_domain)  # type: ignore[arg-type]


__all__ = [
    "AnchoredFusionOutput", "AnchoredFusionParticleTransformer",
    "FUSION_MODEL_CONTRACT", "HLT_DEPLOYABLE_CAPACITY", "INJECTION_BLOCKS",
    "SymmetricFusionParticleTransformer", "UntaggedConcatParticleTransformer",
    "WithdrawalOutput", "build_anchored_fusion_particle_transformer",
    "build_symmetric_fusion_particle_transformer",
    "build_tagged_concat_fusion_control",
    "build_untagged_concat_particle_transformer",
]
