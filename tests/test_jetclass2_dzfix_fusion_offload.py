"""Storage-only parity; synthetic CUDA tests are not real Weaver acceptance."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.jetclass2_delphes import dzfix_fusion_model as fusion
from test_jetclass2_dzfix_fusion_chain import fake_native, make_cache


class PairBN(nn.Module):
    """Trainable masked full-pair population with actual BatchNorm buffers."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv1d(4, 16, 1), nn.BatchNorm1d(16),
                                   nn.GELU(), nn.Conv1d(16, 8, 1), nn.BatchNorm1d(8))

    def forward(self, vectors, uu=None, mask=None):
        batch, _, length = vectors.shape
        pairs = vectors[:, :, :, None] - vectors[:, :, None, :]
        valid = (mask[:, :, :, None] & mask[:, :, None, :]).expand(-1, 4, -1, -1)
        selected = pairs.permute(1, 0, 2, 3)[valid.permute(1, 0, 2, 3)].reshape(1, 4, -1)
        embedded = self.embed(selected)[0].T
        result = vectors.new_zeros(batch, length, length, 8)
        result[(mask[:, 0, :, None] & mask[:, 0, None, :])] = embedded.to(result.dtype)
        return result.permute(0, 3, 1, 2)


@pytest.fixture
def pair_native(fake_native, monkeypatch):
    """Give the existing test double pair gradients, BN and stochastic blocks."""
    from hlt_classification.jetclass2_delphes import salience_learned_model
    native = salience_learned_model.load_weaver_particle_transformer_class()

    class Block(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.dropout = nn.Dropout(.1)

        def forward(self, hidden, *, x_cls=None, padding_mask=None, attn_mask=None):
            result = self.base(hidden, x_cls=x_cls, padding_mask=padding_mask, attn_mask=attn_mask)
            if attn_mask is not None:
                result = result + .01 * attn_mask.mean((1, 3))[..., None]
            return self.dropout(result)

    class WithBN(native):
        def __init__(self, **config):
            super().__init__(**config)
            self.pair_embed = PairBN()
            self.blocks = nn.ModuleList(Block(block) for block in self.blocks)

    monkeypatch.setattr(salience_learned_model, "load_weaver_particle_transformer_class", lambda: WithBN)


def inputs():
    primary = make_cache("train").batch(np.arange(4))
    for key in ("features", "vectors", "mask"):
        primary[key] = primary[key][..., :4].copy()
    momentum = np.random.default_rng(83).normal(size=(4, 3, 4)).astype(np.float32)
    energy = np.sqrt(np.sum(momentum**2, axis=1, keepdims=True) + .25)
    primary["vectors"] = np.concatenate((momentum, energy), axis=1)
    context = deepcopy(primary)
    # Distinct values, unequal view lengths and real masking, not just all-valid
    # identical-jet pairs. Keep Lorentz vectors physical for optional Weaver.
    primary["mask"][0, :, -1] = False
    context["features"] = context["features"][..., :3].copy() + .1
    context["vectors"] = context["vectors"][..., :3].copy() * 1.1
    context["mask"] = context["mask"][..., :3].copy()
    context["mask"][1, :, -1] = False
    return primary, context


def test_cpu_training_storage_parity_and_checkpoint_transparency(pair_native):
    report = fusion.native_offload_parity(*inputs(), device="cpu")
    assert report["steps"] == 3 and report["passed"]
    assert all(row["calls"] == 0 for row in report["offload_stats"].values())
    model = fusion.DzfixFusionParticleTransformer()
    original = fusion.DelphesAdjacentFusionParticleTransformer()
    assert model.state_dict().keys() == original.state_dict().keys()
    assert not any("offload" in name for name in model.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("bf16", [False, True])
def test_cuda_lossless_pinned_storage_training_updates(pair_native, monkeypatch, bf16):
    if bf16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    original = torch.autograd.graph.save_on_cpu
    packed, restored = [], []

    def storage(*, pin_memory):
        assert pin_memory is True
        manager = original(pin_memory=pin_memory)
        native_pack, native_unpack = manager.pack_hook, manager.unpack_hook

        def pack(value):
            result = native_pack(value)
            assert result[1].device.type == "cpu"
            # cuDNN may save an empty reserve-space tensor, which has no storage.
            assert value.numel() == 0 or result[1].is_pinned()
            assert result[1].dtype == value.dtype and result[1].shape == value.shape
            packed.append(value.numel() * value.element_size())
            return result

        def unpack(value):
            result = native_unpack(value)
            assert result.device == value[0]
            restored.append(result.numel() * result.element_size())
            return result

        manager.pack_hook, manager.unpack_hook = pack, unpack
        return manager

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", storage)
    report = fusion.native_offload_parity(*inputs(), device="cuda", bf16=bf16)
    assert report["passed"] and sum(packed) > 0 and sum(restored) > 0
    fusion.validate_offload_stats(report["offload_stats"], calls=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_eval_and_no_grad_bypass_offload_and_error_restores_hooks(pair_native, monkeypatch):
    model = fusion.DzfixFusionParticleTransformer().cuda()
    raw, _ = inputs()
    values = tuple(torch.from_numpy(raw[k]).cuda() for k in ("features", "vectors", "mask"))
    native = torch.autograd.graph.save_on_cpu

    def forbidden(**kwargs):
        pytest.fail("Inference must not offload saved tensors")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", forbidden)
    model.eval().forward_fused(*values, *values)
    model.train()
    with torch.no_grad():
        model.forward_fused(*values, *values)
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", native)
    calls = []

    def broken(*args, **kwargs):
        raise RuntimeError("injected pair failure")

    with torch.autograd.graph.saved_tensors_hooks(lambda x: calls.append(x.shape) or x.detach(), lambda x: x):
        monkeypatch.setattr(model.cross_pair_mod.pair_embed, "forward", broken)
        with pytest.raises(RuntimeError, match="injected"):
            model._pair_embedding(model.cross_pair_mod, values[1], values[2])
        torch.ones(3, device="cuda", requires_grad=True).square().sum().backward()
    assert calls  # Outer hooks restored even when the inner computation failed.


def test_training_parity_detects_changed_batchnorm_state(pair_native, monkeypatch):
    original = fusion.DzfixFusionParticleTransformer._pair_embedding

    def changed(self, mod, vectors, mask):
        result = original(self, mod, vectors, mask)
        if self.pair_offload:
            mod.pair_embed.embed[1].num_batches_tracked.add_(1)
        return result

    monkeypatch.setattr(fusion.DzfixFusionParticleTransformer, "_pair_embedding", changed)
    with pytest.raises(AssertionError):
        fusion.native_offload_parity(*inputs(), device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_saved_pair_tensor_offload_reduces_synthetic_cuda_peak(pair_native):
    # Isolated pair path, not a claim about a real Weaver/A100 production fit.
    model = fusion.DzfixFusionParticleTransformer().cuda().train()
    vectors = torch.randn(8, 4, 160, device="cuda")
    mask = torch.ones(8, 1, 160, device="cuda", dtype=torch.bool)
    peaks = []
    for offload in (False, True):
        model.pair_offload = offload
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = model._pair_embedding(model.cross_pair_mod, vectors, mask)
        result.float().square().mean().backward()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
        del result
    assert peaks[1] < peaks[0], peaks
    print(f"Synthetic pair CUDA peaks: native={peaks[0]}, offloaded={peaks[1]}")


def test_installed_weaver_training_parity():
    pytest.importorskip("weaver")
    torch.set_num_threads(1)
    assert fusion.native_offload_parity(*inputs(), device="cpu")["passed"]
    if torch.cuda.is_available():
        for bf16 in (False, True):
            assert fusion.native_offload_parity(*inputs(), device="cuda", bf16=bf16)["passed"]
