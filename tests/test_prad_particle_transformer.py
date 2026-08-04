from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from hlt_classification.models.prad_particle_transformer import (
    PradParticleTransformer,
    standard_four_pair_features,
    validate_prad_runtime,
)


class _FakeTrimmer(nn.Module):
    def forward(self, x, v=None, mask=None, uu=None):
        # A fixed reversal proves that pair payloads follow the particle order.
        permutation = torch.arange(x.shape[-1] - 1, -1, -1, device=x.device)
        x = x[..., permutation]
        v = v[..., permutation]
        mask = mask[..., permutation]
        if uu is not None:
            uu = uu[..., permutation, :][..., permutation]
        return x, v, mask, uu


class _FakeEmbed(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(17, width)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features.transpose(1, 2))


class _FakePairEmbed(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        self.head_scales = nn.Parameter(torch.linspace(0.1, 0.8, heads))

    def forward(self, vectors, uu=None, mask=None):
        del uu
        base = vectors[:, 3, :, None] + vectors[:, 3, None, :]
        pair = base[:, None] * self.head_scales[None, :, None, None]
        valid = mask[:, :, :, None] & mask[:, :, None, :]
        return pair * valid


class _FakeBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.embed_dim = width
        self.num_heads = heads
        self.projection = nn.Linear(width, width)

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        del x_cls
        update = self.projection(x)
        if attn_mask is not None:
            update = update + attn_mask.mean(dim=(1, 3))[..., None]
        return (x + update).masked_fill(padding_mask[..., None], 0)


class _FakeWeaver(nn.Module):
    def __init__(self, width: int = 12, heads: int = 3) -> None:
        super().__init__()
        self.embed = _FakeEmbed(width)
        self.pair_embed = _FakePairEmbed(heads)
        self.blocks = nn.ModuleList(_FakeBlock(width, heads) for _ in range(8))
        self.cls_blocks = nn.ModuleList()
        self.norm = nn.LayerNorm(width)
        self.fc = nn.Linear(width, 10)
        self.trimmer = _FakeTrimmer()
        # Weaver exposes one boolean per block, not a set of block indices.
        self.block_ids_with_attn_mask = [True] * 8

    def _forward_aggregator(self, hidden, padding_mask):
        valid = (~padding_mask)[..., None].to(hidden.dtype)
        return self.norm((hidden * valid).sum(1) / valid.sum(1).clamp_min(1))

    def forward(self, features, v=None, mask=None):
        features, v, mask, _ = self.trimmer(features, v, mask, None)
        hidden = self.embed(features).masked_fill(~mask.transpose(1, 2), 0)
        padding_mask = ~mask.squeeze(1)
        bias = self.pair_embed(v, mask=mask)
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                padding_mask=padding_mask,
                attn_mask=bias if self.block_ids_with_attn_mask[index] else None,
            )
        return self.fc(self._forward_aggregator(hidden, padding_mask))


class _FakeBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = _FakeWeaver()

    def forward(self, points, features, vectors, mask):
        del points
        return self.mod(features, v=vectors, mask=mask)

    def no_weight_decay(self):
        return {"mod.cls_token"}


def _inputs():
    torch.manual_seed(11)
    points = torch.randn(2, 2, 6)
    features = torch.randn(2, 17, 6)
    features[:, 6:11] = 0
    features[:, 6] = 1
    vectors = torch.randn(2, 4, 6)
    vectors[:, 3] = vectors[:, :3].square().sum(1).add(1).sqrt()
    mask = torch.tensor(
        [[[True, True, True, True, False, False]], [[True] * 6]]
    )
    return points, features, vectors, mask


def test_standard_four_pair_features_are_symmetric_and_finite() -> None:
    _, _, vectors, _ = _inputs()
    pair = standard_four_pair_features(vectors)
    assert pair.shape == (2, 6, 6, 4)
    assert torch.isfinite(pair).all()
    assert torch.equal(pair, pair.transpose(1, 2))


def test_zero_gates_reproduce_frozen_baseline_logits_exactly() -> None:
    baseline = _FakeBaseline().eval()
    model = PradParticleTransformer(
        baseline=baseline, relation_dropout=0.0
    ).eval()
    inputs = _inputs()
    with torch.no_grad():
        expected = baseline(*inputs)
        actual = model(*inputs)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(model.gated_bias.raw_gates) == 0
    assert model.attention_mask_blocks == tuple(range(8))


def test_weaver_boolean_attention_mask_policy_is_preserved_by_block() -> None:
    baseline = _FakeBaseline().eval()
    baseline.mod.block_ids_with_attn_mask = [
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    model = PradParticleTransformer(
        baseline=baseline, relation_dropout=0.0
    ).eval()
    with torch.no_grad():
        expected = baseline(*_inputs())
        actual = model(*_inputs())
    assert torch.equal(actual, expected)
    assert model.attention_mask_blocks == (0, 2, 5, 7)


def test_deployable_forward_is_hlt_only_and_gate_receives_gradient() -> None:
    model = PradParticleTransformer(
        baseline=_FakeBaseline(), relation_dropout=0.0
    )
    assert list(inspect.signature(model.forward).parameters) == [
        "points",
        "features",
        "lorentz_vectors",
        "mask",
    ]
    output = model.forward_with_relations(*_inputs())
    output.logits.square().mean().backward()
    assert model.gated_bias.raw_gates.grad is not None
    assert torch.isfinite(model.gated_bias.raw_gates.grad).all()


def test_auxiliary_only_deployment_prunes_relation_computation() -> None:
    baseline = _FakeBaseline().eval()
    model = PradParticleTransformer(
        baseline=baseline,
        relation_dropout=0.0,
        deploy_relation_attention=False,
    ).eval()
    with torch.no_grad():
        for parameter in model.relation.parameters():
            parameter.fill_(100.0)
        expected = baseline(*_inputs())
        actual = model(*_inputs())
    assert torch.equal(actual, expected)
    assert model.deployable_parameter_count() == sum(
        parameter.numel() for parameter in baseline.parameters()
    )


def test_training_payload_and_oracle_follow_the_weaver_trimmer() -> None:
    model = PradParticleTransformer(
        baseline=_FakeBaseline(), relation_dropout=0.0
    ).eval()
    inputs = _inputs()
    particles = inputs[1].shape[-1]
    payload = torch.arange(
        particles * particles, dtype=torch.float32
    ).reshape(1, 1, particles, particles).repeat(2, 1, 1, 1)
    expected = payload.flip((-2, -1))
    training = model.forward_training(*inputs, pair_payload=payload)
    assert torch.equal(training.aligned_pair_payload, expected)

    oracle = payload.repeat(1, model.attention_heads, 1, 1)
    oracle_output = model.forward_oracle(
        *inputs, offline_teacher_bias=oracle
    )
    assert torch.equal(oracle_output.aligned_pair_payload, oracle.flip((-2, -1)))
    with pytest.raises(ValueError, match="shape differs"):
        model.forward_oracle(*inputs, offline_teacher_bias=payload)


def test_prad_runtime_attestation_exercises_training_and_deployment_paths() -> None:
    report = validate_prad_runtime(
        seed=17,
        batch_size=2,
        particles=6,
        baseline_factory=_FakeBaseline,
    )
    assert report["passed"]
    assert report["maximum_zero_gate_logit_error"] < 1.0e-6
    assert report["relation_gradient_norm"] > 0.0
    assert report["gate_gradient_norm"] > 0.0
    assert report["contract"] == "hlt_classification_prad_runtime_validation_v2"
    assert report["schema_version"] == 2
    assert report["checks"]["stage_a_logits_excluded_from_backward"]
    assert report["checks"]["stage_a_relation_gradient_finite_nonzero"]
    assert report["stage_a_relation_gradient_norm"] > 0.0
