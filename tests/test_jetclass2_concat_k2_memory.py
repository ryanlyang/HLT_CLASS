"""Storage-only K2 tests; local CUDA doubles are not SPORC acceptance."""
from copy import deepcopy
import os

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.contracts import artifact as base_artifact
from hlt_classification.jetclass2_delphes import (
    concat_k2_model as memory, concat_k2_campaign as campaign,
    concat_k2_runtime as runtime, model as native,
)
from test_jetclass2_concat_k2 import spec_at
from test_jetclass2_dzfix_fusion_chain import fake_native, make_cache, rehash
# Reuse the committed synthetic BN pair population, not any fusion runtime.
from test_jetclass2_dzfix_fusion_offload import PairBN, inputs


@pytest.fixture(autouse=True)
def parity_cublas_config(monkeypatch):
    # The real worker sets this before Python starts. Tests should likewise be
    # launched with this environment setting before any CUDA handles exist.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture
def pair_native(fake_native, monkeypatch):
    base = native.load_weaver_particle_transformer_class()

    class WithPairGradients(base):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.pair_embed = PairBN()
            self.dropout = nn.Dropout(.1)

        def forward(self, features, v, mask):
            hidden = self.embed(features)
            pair = self.pair_embed(v, mask=mask)
            hidden = hidden + .01 * pair.mean((1, 3))[..., None]
            return self.fc(self.dropout(self._forward_aggregator(hidden, ~mask[:, 0])))

    monkeypatch.setattr(native, "load_weaver_particle_transformer_class", lambda: WithPairGradients)


def stats():
    return dict(calls=3, saved_cuda_tensors=30, saved_cuda_bytes=1000, restored_cuda_tensors=30)


def successful_probe():
    return dict(status="COMPLETED", step_seconds=[.1, .1, .1], validation_seconds=.1,
        storage_stats=stats(), peak_cuda_bytes=500, peak_reserved_cuda_bytes=600,
        elapsed_seconds=.4, steady_train_jets_per_second=2560)


def memory_evidence(spec):
    """Explicit test-only fabrication for isolated gate validation tests."""
    reports = [dict(node_id=name, precision=precision, steps=3, checks=memory.PARITY_CHECKS,
        passed=True, device_type="cuda", storage_stats=stats(),
        tolerance=memory.PARITY_TOLERANCES[precision], parity_backend=memory.PARITY_BACKEND,
        pair_storage=memory.PAIR_STORAGE)
        for name in ("CONCAT_K2_D100", "CONCAT_K2_D000") for precision in ("fp32", "bf16")]
    early = [campaign.artifact("EARLY_PARITY", campaign_sha256=spec["content_hash"],
        node_id=r["node_id"], sample=dict(role="train", rows=4, file_indices=[0]*4,
            identities=[f"{i:064x}" for i in range(4)], final_test_accessed=False),
        native_parity=base_artifact("WEAVER_PARITY", model=spec["model"], device="cuda", passed=True,
            forward_and_feature_and_parameter_gradients=True, final_test_accessed=False),
        storage_parity={k:v for k,v in r.items() if k!="node_id"},
        acceptance_only=True, final_test_accessed=False) for r in reports]
    probes = [campaign.artifact("BATCH_PROBE", **successful_probe(),
        campaign_sha256=spec["content_hash"], node_id="ACCEPTANCE_"+name,
        batch_size=size, train_rows=size, validation_rows=size, device_type="cuda",
        capacity=spec["foundation"]["inputs"]["capacity"], pair_storage=memory.PAIR_STORAGE,
        policy=memory.BATCH_PROBE_POLICY, acceptance_only=True, final_test_accessed=False)
        for name in runtime.PROBE_CASES for size in (128,)]
    return dict(pair_storage=memory.PAIR_STORAGE, batch_probe_policy=memory.BATCH_PROBE_POLICY,
        storage_parity_reports=reports, batch_probes=probes, early_parity_reports=early)


def test_cpu_optimizer_parity_and_checkpoint_namespace(pair_native):
    result = memory.storage_parity(inputs()[0], device="cpu")
    assert result["passed"] and result["steps"] == 3
    assert result["storage_stats"]["calls"] == 0
    model = memory.K2ParticleTransformer()
    reference = native.DelphesParticleTransformer()
    assert model.state_dict().keys() == reference.state_dict().keys()
    reference.load_state_dict(model.state_dict(), strict=True)
    assert model.no_weight_decay() == reference.no_weight_decay()
    assert isinstance(runtime.new_model(campaign.nodes()[0]), memory.K2ParticleTransformer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("bf16", [False, True])
def test_cuda_actual_pinned_storage_and_three_update_parity(pair_native, monkeypatch, bf16):
    if bf16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    pack, unpack = memory._pack_pair_tensor, memory._unpack_pair_tensor
    counts = [0, 0]
    def save(tensor):
        saved = pack(tensor)
        assert saved[1].device.type == "cpu" and saved[1].dtype == tensor.dtype
        assert saved[2:] == (tuple(tensor.shape), tuple(tensor.stride()))
        assert tensor.numel() == 0 or saved[1].is_pinned()
        counts[0] += 1
        return saved

    def restore(saved):
        restored = unpack(saved)
        assert restored.device == saved[0]
        assert (tuple(restored.shape), tuple(restored.stride())) == saved[2:]
        counts[1] += 1
        return restored

    monkeypatch.setattr(memory, "_pack_pair_tensor", save)
    monkeypatch.setattr(memory, "_unpack_pair_tensor", restore)
    result = memory.storage_parity(inputs()[0], device="cuda", bf16=bf16)
    assert result["passed"] and min(counts) > 0
    memory.validate_storage_stats(result["storage_stats"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_eval_no_grad_no_offload_and_batchnorm_not_recomputed(pair_native, monkeypatch):
    model = memory.K2ParticleTransformer().cuda()
    raw = inputs()[0]
    values = [torch.from_numpy(raw[k]).cuda() for k in ("features", "vectors", "mask")]
    model.eval()(*values)
    with torch.no_grad():
        model.train()(*values)
    assert model.pair_storage_stats()["calls"] == 0
    before = [b.num_batches_tracked.clone() for b in model.modules() if isinstance(b, nn.BatchNorm1d)]
    model(*values).float().square().mean().backward()
    after = [b.num_batches_tracked for b in model.modules() if isinstance(b, nn.BatchNorm1d)]
    assert all(int(b-a) == 1 for a, b in zip(before, after))
    memory.validate_storage_stats(model.pair_storage_stats(), calls=1)

    def broken(*a, **kw):
        raise ValueError("injected pair failure")
    monkeypatch.setattr(model, "_native_pair_forward", broken)
    with pytest.raises(ValueError, match="injected"):
        model(*values)
    unchanged = model.pair_storage_stats()
    x = torch.ones(2, device="cuda", requires_grad=True)
    x.square().sum().backward()
    assert model.pair_storage_stats() == unchanged  # no leaked global hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_isolated_pair_storage_reduces_cuda_peak(pair_native):
    model = memory.K2ParticleTransformer().cuda().train()
    vectors = torch.randn(8, 4, 160, device="cuda")
    mask = torch.ones(8, 1, 160, device="cuda", dtype=torch.bool)
    peaks = []
    for enabled in (False, True):
        model.pair_storage = enabled
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        result = model.mod.pair_embed(vectors, mask=mask)
        result.float().square().mean().backward()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
        del result
    assert peaks[1] < peaks[0], peaks
    print(f"Synthetic K2 pair peaks: native={peaks[0]}, offloaded={peaks[1]}")


def test_training_parity_detects_bn_state_change(pair_native, monkeypatch):
    original = memory.K2ParticleTransformer._pair_forward
    def changed(self, *a, **kw):
        result = original(self, *a, **kw)
        self.mod.pair_embed.embed[1].num_batches_tracked.add_(1)
        return result
    monkeypatch.setattr(memory.K2ParticleTransformer, "_pair_forward", changed)
    with pytest.raises(AssertionError): memory.storage_parity(inputs()[0], device="cpu")


@pytest.mark.parametrize("failure", [128, None])
def test_registered_128_probe_keeps_results_and_never_attempts_256(tmp_path, monkeypatch, failure):
    spec = spec_at(tmp_path); node = dict(campaign.nodes()[3], node_id="ACCEPTANCE_CONCAT_K2_D100")
    seen = []
    def attempt(s, n, t, v, d, size):
        seen.append(size)
        if size == failure:
            return dict(status="CUDA_OOM", peak_cuda_bytes=900, error="injected")
        return dict(**successful_probe(), train_rows=size, validation_rows=size)
    monkeypatch.setattr(runtime, "_probe_attempt", attempt)
    if failure:
        with pytest.raises(MemoryError, match="registered batch 128 CUDA OOM"):
            runtime.batch_probes(spec, tmp_path, node, None, None, "cuda")
    else:
        assert len(runtime.batch_probes(spec, tmp_path, node, None, None, "cuda")) == 1
    assert seen == [128]
    records = [load_json(p) for p in sorted(tmp_path.glob("batch_probe_*.json"))]
    assert len(records) == len(seen)
    assert records[-1]["status"] == ("CUDA_OOM" if failure else "COMPLETED")
    assert spec["training"]["batch_size"] == 128
    assert not (tmp_path / "acceptance.json").exists()


def test_probe_catches_only_cuda_oom_and_releases_attempt_frame(tmp_path, monkeypatch):
    import weakref
    refs = []
    class Allocation: pass
    def fail(*a):
        local = Allocation(); refs.append(weakref.ref(local))
        raise torch.OutOfMemoryError("simulated OOM")
    monkeypatch.setattr(runtime, "_exercise_probe", fail)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 900)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 950)
    result = runtime._probe_attempt(None, None, None, None, "cuda", 128)
    assert result["status"] == "CUDA_OOM" and refs[0]() is None
    def invalid(*a): raise ValueError("not an OOM")
    monkeypatch.setattr(runtime, "_exercise_probe", invalid)
    with pytest.raises(ValueError, match="not an OOM"):
        runtime._probe_attempt(None, None, None, None, "cuda", 128)


def test_real_probe_requires_distinct_rows_and_uses_full_batch(tmp_path, pair_native, monkeypatch):
    from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
    small = make_cache("train")
    block = small.blocks[0]
    count = 300
    indices = np.arange(count) % len(small)
    ids = np.asarray([np.frombuffer(i.to_bytes(32, "big"), np.uint8) for i in range(count)])
    large = RamCache([RamBlock(0, np.arange(count+1)*4,
        block.features.reshape(len(small), 4, 17)[indices].reshape(-1, 17),
        block.vectors.reshape(len(small), 4, 4)[indices].reshape(-1, 4),
        ids, np.arange(count)%11)], role="train", foundation_sha256="f"*64, coordinate_name="D100")
    spec = spec_at(tmp_path); node = campaign.nodes()[3]; seen = []
    original = runtime._stress
    def stress(model, raw, *args):
        seen.append(len(raw["labels"]))
        return original(model, raw, *args)
    monkeypatch.setattr(runtime, "_stress", stress)
    for size in (128, 256):
        result = runtime._exercise_probe(spec, node, large, large, "cpu", size)
        assert result["train_rows"] == result["validation_rows"] == size
        assert len(result["step_seconds"]) == 3 and result["steady_train_jets_per_second"] > 0
    assert seen == [128, 256]
    with pytest.raises(ValueError, match="distinct real rows"):
        runtime._exercise_probe(spec, node, small, small, "cpu", 128)


@pytest.mark.parametrize("fault", ["missing", "order", "cpu", "no_restore", "batch", "capacity", "nan", "parent", "oom", "parity", "tolerance", "backend"])
def test_memory_acceptance_rejects_incomplete_or_noop_evidence(tmp_path, fault):
    spec = spec_at(tmp_path)
    evidence = dict(**memory_evidence(spec), capacity=32, peak_cuda_bytes=800)
    runtime.validate_memory_evidence(spec, evidence)
    bad = deepcopy(evidence)
    row = bad["batch_probes"][0]
    if fault == "missing": bad.pop("batch_probes")
    elif fault == "order": bad["batch_probes"].reverse()
    elif fault == "cpu": row["device_type"] = "cpu"
    elif fault == "no_restore": row["storage_stats"]["restored_cuda_tensors"] = 0
    elif fault == "batch": row["train_rows"] = 64
    elif fault == "capacity": row["capacity"] = 16
    elif fault == "nan": row["step_seconds"][1] = -1.
    elif fault == "parent": row["campaign_sha256"] = "f"*64
    elif fault == "oom": row["status"] = "CUDA_OOM"
    elif fault == "tolerance": bad["storage_parity_reports"][0]["tolerance"] = dict(rtol=.1,atol=.1)
    elif fault == "backend": bad["storage_parity_reports"][0]["parity_backend"] = {}
    else: bad["storage_parity_reports"][0]["checks"] = ["logits"]
    if fault not in {"missing", "order"}:
        bad["batch_probes"][0] = rehash(row)
    with pytest.raises(ValueError): runtime.validate_memory_evidence(spec, bad)


@pytest.mark.parametrize("length", [4, 96, 192])
@pytest.mark.parametrize("duplicates", [False, True])
def test_installed_weaver_optional_training_parity(length, duplicates):
    pytest.importorskip("weaver")
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(83)
    p = rng.normal(size=(4, 3, length)).astype(np.float32)
    raw = dict(features=rng.normal(size=(4,17,length)).astype(np.float32),
        vectors=np.concatenate((p,np.sqrt((p*p).sum(1,keepdims=True)+.25)),axis=1),
        mask=np.ones((4,1,length),bool), labels=np.arange(4,dtype=np.int64))
    raw["mask"][0,:,-max(1,length//8):] = False
    if duplicates:
        for key in ("features", "vectors", "mask"):
            raw[key] = np.repeat(raw[key], 3, axis=-1)
    for bf16 in ([False, True] if device == "cuda" and torch.cuda.is_bf16_supported() else [False]):
        assert memory.storage_parity(raw, device=device, bf16=bf16)["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "gapped", "expanded", "overlap", "empty", "scalar"])
def test_saved_tensor_restores_bitwise_values_dtype_and_strides(layout):
    x = torch.arange(80, device="cuda", dtype=torch.float32).reshape(8,10)
    value = {"contiguous":x, "transposed":x.T.unsqueeze(0), "gapped":x[1::2,2::3],
             "expanded":x[1:2,:].expand(3,-1), "overlap":x.as_strided((4,3),(1,1),5),
             "empty":x[:0], "scalar":x[1,2]}[layout]
    saved = memory._pack_pair_tensor(value)
    assert saved[1].device.type == "cpu" and not saved[1].requires_grad
    assert value.numel() == 0 or saved[1].is_pinned()
    restored = memory._unpack_pair_tensor(saved)
    torch.testing.assert_close(restored, value, rtol=0, atol=0)
    assert restored.shape == value.shape and restored.stride() == value.stride()
    assert restored.dtype == value.dtype and restored.storage_offset() == 0


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_parity_restores_backend_flags_even_on_failure(fail, precision):
    def flags():
        return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
            torch.get_float32_matmul_precision())
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    before = flags()
    try:
        try:
            with memory.parity_backend("cpu"):
                assert flags() == (True,False,False,True,False,False,"highest")
                if fail:
                    raise RuntimeError("injected")
        except RuntimeError:
            assert fail
        assert flags() == before
    finally:
        torch.set_float32_matmul_precision(original_precision)


def test_parity_requires_cublas_configuration_without_late_environment_mutation(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="before starting Python"):
        with memory.parity_backend("cuda"):
            pytest.fail("Must fail before touching CUDA")
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


@pytest.mark.parametrize("fault", ["missing", "hash", "tolerance", "backend", "storage", "test", "identity", "files", "native"])
def test_early_gate_requires_exact_sample_policy_and_native_evidence(tmp_path, fault):
    spec = spec_at(tmp_path)
    evidence = dict(**memory_evidence(spec), capacity=32, peak_cuda_bytes=800)
    runtime.validate_memory_evidence(spec,evidence)
    row = evidence["early_parity_reports"][0]
    if fault == "missing": evidence.pop("early_parity_reports")
    elif fault == "tolerance": row["storage_parity"]["tolerance"] = dict(rtol=.1,atol=.1)
    elif fault == "backend": row["storage_parity"]["parity_backend"] = {}
    elif fault == "storage": row["storage_parity"]["pair_storage"] = {}
    elif fault == "test": row["sample"]["role"] = "final_test"
    elif fault == "identity": row["sample"]["identities"][0] = row["sample"]["identities"][1]
    elif fault == "files": row["sample"]["file_indices"] = [1]*4
    elif fault == "native": row["native_parity"] = rehash(row["native_parity"],device="cpu")
    else: row["content_hash"] = "f"*64
    if fault not in {"missing", "hash"}:
        evidence["early_parity_reports"][0] = rehash(row)
    with pytest.raises(ValueError): runtime.validate_memory_evidence(spec,evidence)
