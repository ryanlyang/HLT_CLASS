"""Allocation-only CMS fusion optimization: values, gradients and updates."""
from __future__ import annotations

import copy
import types
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.cms_salience_learned import contracts, training
from hlt_classification.cms_salience_learned.model import CMSFusion, build_model
from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
    AnchoredFusionParticleTransformer,
)
from test_hcwdl_offline_hlt_fusion import _FakeWeaver, _Block


class _DifferentiablePair(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 8, 1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, vectors, uu=None, mask=None):
        differences = vectors[:, :, :, None] - vectors[:, :, None, :]
        return self.bn(self.projection(differences))


class _StochasticBlock(_Block):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.1)

    def forward(self, hidden, **kwargs):
        return self.dropout(super().forward(hidden, **kwargs))


class _StatefulWeaver(_FakeWeaver):
    def __init__(self, **config):
        super().__init__(**config)
        self.pair_embed = _DifferentiablePair()
        self.blocks = nn.ModuleList(_StochasticBlock() for _ in self.blocks)


@pytest.fixture
def stateful_weaver(monkeypatch):
    from hlt_classification.models import hcwdl_offline_hlt_fusion_transformer as fusion
    from hlt_classification.models import scouting_particle_transformer as scouting

    monkeypatch.setattr(fusion, "_weaver_class", lambda: _StatefulWeaver)
    monkeypatch.setattr(scouting, "_weaver_class", lambda: _StatefulWeaver)


def _raw_batch():
    rng = np.random.default_rng(719)

    def view(width):
        features = rng.normal(size=(3, 21, width)).astype(np.float32)
        vectors = rng.normal(size=(3, 4, width)).astype(np.float32)
        vectors[:, 3] = np.sqrt(np.square(vectors[:, :3]).sum(1) + 1.)
        mask = np.ones((3, 1, width), bool)
        mask[1, :, -2:] = False
        mask[2, :, 1:] = False
        features *= mask
        vectors *= mask
        return dict(features=features, vectors=vectors, mask=mask)

    return dict(primary=view(7), context=view(5), labels=np.arange(3))


def _models(role):
    node = contracts.node("temporary_memory", role, "D080", "U100", "teacher")
    optimized = build_model(node)
    # Nonzero projections exercise context/pair gradients, not just zero-init
    # primary equivalence. The production initialization itself is unchanged.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(319)
        for injection in optimized.injections:
            nn.init.normal_(injection.residual_projection.weight, std=.02)
    legacy = copy.deepcopy(optimized)
    legacy._prepare_injection_bias = types.MethodType(
        AnchoredFusionParticleTransformer._prepare_injection_bias, legacy,
    )
    assert optimized.state_dict().keys() == legacy.state_dict().keys()
    return node, optimized, legacy


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_bias_releases_square_storage_and_preserves_gradient(dtype):
    source = torch.randn(2, 8, 12, 12, dtype=dtype, requires_grad=True)
    square = source * 2
    square_ref = weakref.ref(square)
    bias = square[:, :, :7, 7:]
    padding = torch.tensor([[False] * 5, [False, False, True, True, True]])
    expected = bias.detach().clone().masked_fill(padding[:, None, None], -torch.inf)
    bias, prepared_padding = CMSFusion._prepare_injection_bias(None, bias, padding)
    del square
    assert square_ref() is None
    assert prepared_padding is None and bias.is_contiguous()
    assert bias.untyped_storage().nbytes() == bias.numel() * bias.element_size()
    torch.testing.assert_close(bias, expected, rtol=0, atol=0)
    # As in attention, padded context keys contribute zero to the gradient.
    bias.masked_fill(padding[:, None, None], 0).sum().backward()
    expected_grad = torch.zeros_like(source)
    expected_grad[:, :, :7, 7:] = (~padding[:, None, None]).to(dtype) * 2
    torch.testing.assert_close(source.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("route", ["forward_fused", "forward_withdrawal"])
def test_four_injections_share_one_compact_attention_mask(stateful_weaver, monkeypatch, route):
    torch.set_num_threads(1)
    _, optimized, legacy = _models("fusion_withdrawal")
    raw = _raw_batch()
    args = (*training.tensors(raw["primary"], "cpu"), *training.tensors(raw["context"], "cpu"))
    attention = torch.nn.functional.scaled_dot_product_attention
    masks = []

    def spy(q, k, v, attn_mask=None, *args, **kwargs):
        masks.append(attn_mask)
        return attention(q, k, v, attn_mask, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    results = []
    for model in (legacy, optimized):
        model.eval()
        masks.clear()
        with torch.no_grad():
            results.append(getattr(model, route)(*args, alpha=.5))
        assert len(masks) == 4
        stores = {mask.untyped_storage().data_ptr() for mask in masks}
        assert len(stores) == (4 if model is legacy else 1)
        assert all(mask.shape == (3, 8, 7, 5) for mask in masks)
        if model is optimized:
            assert masks[0].untyped_storage().nbytes() == 3 * 8 * 7 * 5 * 4
    for left, right in zip(_output_tensors(results[0]), _output_tensors(results[1])):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def _output_tensors(output):
    if hasattr(output, "zero"):
        return _output_tensors(output.zero) + _output_tensors(output.privileged)
    return (output.logits, *output.hlt_states, output.hlt_mask)


def _assert_training_parity(role, device="cpu"):
    torch.set_num_threads(1)
    node, optimized, legacy = _models(role)
    models = (legacy.to(device), optimized.to(device))
    optimizers = tuple(training.optimizer_for(model) for model in models)
    raw = _raw_batch()
    teacher = torch.full((3, 15), 1 / 15, device=device)
    # Four nonzero updates exercise Weaver's trimmer warmup/counter as well as
    # the shared gradient accumulation, then verify the exact zero endpoint.
    alphas = (1., .5, 1., .5, 0.) if role == "fusion_withdrawal" else (1.,) * 5
    for step, alpha in enumerate(alphas):
        batch = dict(raw["primary"], labels=raw["labels"]) if alpha == 0 else raw
        results = []
        for model, optimizer in zip(models, optimizers):
            model.train()
            torch.manual_seed(541 + step)
            if device == "cuda":
                torch.cuda.manual_seed_all(541 + step)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = training.batch_loss(
                model, batch, node=node, device=device, teacher=teacher, alpha=alpha,
            )
            loss.backward()
            gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
            assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
            rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
            optimizer.step()
            results.append((terms, gradients, rng))
        a, b = results
        assert a[0].keys() == b[0].keys() and a[1].keys() == b[1].keys()
        assert torch.equal(a[2], b[2])
        # Different floating reduction order is permitted; no stop-gradient,
        # extra stochastic pass, changed BN population or missing term is.
        rtol, atol = (2e-2, 3e-4) if device == "cuda" else (1e-4, 2e-6)
        for group in (0, 1):
            for name in a[group]:
                torch.testing.assert_close(a[group][name], b[group][name], rtol=rtol, atol=atol, msg=name)
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(value, optimized.state_dict()[name], rtol=rtol, atol=atol, msg=name)
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            assert left.keys() == right.keys()
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=rtol, atol=atol, msg=name)
        if alpha:
            assert any(name.startswith("cross_pair_mod.pair_embed.") and torch.count_nonzero(g)
                       for name, g in a[1].items())
    optimized.eval()
    legacy.eval()
    args = training.tensors(raw["primary"], device)
    with torch.no_grad():
        for alpha in (1., .5, 0.):
            context = () if alpha == 0 else training.tensors(raw["context"], device)
            for left, right in zip(_output_tensors(legacy.forward_fused(*args, *context, alpha=alpha)),
                                   _output_tensors(optimized.forward_fused(*args, *context, alpha=alpha))):
                torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
        zero = optimized.forward_fused(*args, alpha=0.).logits
        extracted = optimized.extract_primary().to(device).eval()(*args)
        torch.testing.assert_close(zero, extracted, rtol=0, atol=0)


@pytest.mark.parametrize("role", ["fusion_acquisition", "fusion_withdrawal"])
def test_temporary_memory_optimization_preserves_training(stateful_weaver, role):
    _assert_training_parity(role)


@pytest.mark.parametrize("role", ["fusion_acquisition", "fusion_withdrawal"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_installed_weaver_temporary_memory_parity(role, device):
    pytest.importorskip("weaver")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for BF16 execution parity")
    _assert_training_parity(role, device)
