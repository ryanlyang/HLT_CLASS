"""TP recomputation preserves native math/state while bounding live activations."""
import numpy as np
import pytest
import torch
from hlt_classification.luka_embedding_bridge_hydra import activation, distributed as d, banks, registry
from hlt_classification.luka_subclass_hydra.benchmark_training import parity_snapshot, compare
from hlt_classification.luka_subclass_ddp_hydra.resources import PROBE_TOLERANCES
from test_luka_embedding_bridge_hydra import synthetic_dataset, small


@pytest.fixture(autouse=True)
def math_settings():
    threads = torch.get_num_threads()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    yield
    torch.set_num_threads(threads)
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags


def test_replay_context_restores_references_versions_and_exception():
    module = torch.nn.BatchNorm1d(3).train()
    before = {k: (v, v.clone(), v._version) for k, v in module.named_buffers()}
    context = activation.ReplayBuffers(module)
    for fail in (False, True, False):
        try:
            with context:
                module(torch.randn(7, 3))
                if fail: raise RuntimeError('injected')
        except RuntimeError as exc:
            assert str(exc) == 'injected'
        for k, v in module.named_buffers():
            ref, value, version = before[k]
            assert v is ref and v._version == version and torch.equal(v, value)


def test_block_checkpoint_rejects_unexpected_mutable_buffers(monkeypatch):
    from hlt_classification.luka_embedding_bridge_hydra.model import Model
    small(monkeypatch);m=Model('C0')
    m.mod.blocks[0].register_buffer('unexpected',torch.zeros(1))
    with pytest.raises(ValueError,match='mutable'):
        activation.install_blocks(m.mod)


@pytest.mark.parametrize('batch', range(6))
def test_checkpoint_update_and_repeated_grad_diagnostics(monkeypatch, batch):
    small(monkeypatch)
    ds = synthetic_dataset(); ctx = d.Context(0, 1, torch.device('cpu'))
    target = banks.Targets(ds.spec, 'TP', ds)
    snapshots = []; rngs = []; diags = []
    for enabled, diagnose in ((False, False), (True, False), (True, True)):
        m, w, o = d.build(ds.spec, 'TP', ctx)
        m.checkpoint_encoding = enabled
        values, diag = d.step(m, w, o, 'TP', ds, target, np.arange(11), 3, batch, 3., ctx, diagnose=diagnose)
        snapshots.append(parity_snapshot(m, o, values['total']))
        rngs.append(torch.get_rng_state()); diags.append(diag)
    for current, rng in zip(snapshots[1:], rngs[1:]):
        result = compare(snapshots[0], current, PROBE_TOLERANCES)
        assert result['passed'], result
        assert torch.equal(rngs[0], rng)
    assert diags[-1]['ce_norm'] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_full_depth_checkpoint_memory_native_reload_and_rng():
    from hlt_classification.luka_embedding_bridge_hydra.model import Model, Native
    from test_luka_embedding_bridge_hydra import physical
    from hlt_classification.luka_embedding_bridge_hydra import regions, losses
    ctx = d.Context(0, 1, torch.device('cuda', 0))
    # A long full jet and eight equally long independently encoded crops: a
    # conservative activation stress fixture, not a scientific region target.
    raw = regions.padded([regions.encode(physical(80)) for _ in range(12)], np.arange(12)%11)
    x = d.arrays(raw, ctx); y = torch.as_tensor(raw['labels'], device=ctx.device)
    valid = torch.ones(12, 8, dtype=torch.bool, device=ctx.device)
    outputs = []; gradients = []; buffers = []; rngs = []; peaks = []
    for enabled in (False, True):
        torch.manual_seed(23); m = Model('TP').to(ctx.device).train()
        m.checkpoint_encoding = enabled
        assert len(m.mod.blocks) == 8 and len(m.mod.cls_blocks) == 2
        d.release_memory(ctx); torch.cuda.reset_peak_memory_stats(ctx.device)
        out = m(**x, crops=[x]*8, valid=valid)
        total, _ = losses.objective('TP', out, y, dict(valid=valid), 3.)
        total.backward(); ctx.sync()
        peaks.append(torch.cuda.max_memory_allocated(ctx.device))
        outputs.append({k:v.detach().cpu() for k,v in out.items()})
        gradients.append(torch.cat([p.grad.flatten().cpu() for p in m.parameters()]))
        buffers.append({k:v.cpu().clone() for k,v in m.named_buffers()})
        rngs.append((torch.get_rng_state(), torch.cuda.get_rng_state(ctx.device)))
        if enabled:
            native = Native().to(ctx.device).eval(); native.load_state_dict(m.native_state(), strict=True)
            m.eval()
            with torch.no_grad():
                torch.testing.assert_close(m(**x, auxiliary=False), native(**x), rtol=0, atol=0)
            del native
        del m, out, total
        d.release_memory(ctx)
    assert peaks[1] < .45*peaks[0], peaks
    for k in outputs[0]:torch.testing.assert_close(outputs[0][k], outputs[1][k], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-5, atol=2e-6)
    for k in buffers[0]:assert torch.equal(buffers[0][k], buffers[1][k]), k
    assert all(torch.equal(a,b) for a,b in zip(*rngs))
    print('TP full-depth CUDA activation peaks (ordinary/checkpoint):', peaks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_checkpoint_long_support_per_rank_batch64():
    from hlt_classification.luka_embedding_bridge_hydra.model import Model
    from test_luka_embedding_bridge_hydra import physical
    from hlt_classification.luka_embedding_bridge_hydra import regions,losses
    ctx=d.Context(0,1,torch.device('cuda',0));torch.manual_seed(23)
    m=Model('TP').to(ctx.device).train()
    raw=regions.padded([regions.encode(physical(298)) for _ in range(64)],np.arange(64)%11)
    x=d.arrays(raw,ctx);y=torch.as_tensor(raw['labels'],device=ctx.device)
    valid=torch.ones(64,8,dtype=torch.bool,device=ctx.device)
    # Every independently supplied crop has full298 support. Actual partitions
    # contain fewer particles; this fixture stresses padding and saved graphs.
    crops=[{k:v.clone() for k,v in x.items()} for _ in range(8)]
    d.release_memory(ctx);torch.cuda.reset_peak_memory_stats(ctx.device)
    out=m(**x,crops=crops,valid=valid)
    total,_=losses.objective('TP',out,y,dict(valid=valid),3.)
    total.backward();ctx.sync()
    peak=torch.cuda.max_memory_allocated(ctx.device)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert peak<.88*10.9*2**30,peak
    print('TP checkpoint batch64 / full298 / eight full298 crops peak:',peak)
    del m,out,total,crops,x;d.release_memory(ctx)


@pytest.mark.parametrize('bad', ['rng','gradient','buffers','policy','condition','rank','stress_rows','stress_gpu','stress_time'])
def test_admission_rejects_resealed_incomplete_checkpoint_evidence(tmp_path,bad):
    from test_luka_embedding_bridge_hydra import execution_spec, resource_diagnostic
    from hlt_classification.data.cache_contracts import with_content_hash
    from hlt_classification.luka_embedding_bridge_hydra import resources
    spec=execution_spec(tmp_path);value=resource_diagnostic(spec)
    checkpoint=value['tp_checkpoint'];row=checkpoint['checks'][0]['ranks'][1];stress=value['tp_stress']
    if bad=='rng':row['rng_equal']=False
    if bad=='gradient':row['gradient_relative_l2']=1.
    if bad=='buffers':row['buffer_abs']=1.
    if bad=='policy':checkpoint['policy']['preserve_rng_state']=False
    if bad=='condition':checkpoint['checks'][0]['condition']=['HLT','whole']
    if bad=='rank':checkpoint['checks'][0]['ranks'].pop()
    if bad=='stress_rows':stress['cases'][0]['rows']=64
    if bad=='stress_gpu':stress['memory']['ranks'][0]['gpu_peak_bytes']=11*2**30
    if bad=='stress_time':stress['cases'][0]['seconds']=-1.
    for part in ('tp_checkpoint','tp_stress'):
        value[part].pop('content_hash');value[part]=with_content_hash(value[part])
    value.pop('content_hash');value=with_content_hash(value)
    with pytest.raises((ValueError,MemoryError)):resources.accept(spec,value)


def test_tp_checks_precede_numerical_replay_and_miniatures(monkeypatch,tmp_path):
    from hlt_classification.luka_embedding_bridge_hydra import resources
    from types import SimpleNamespace
    events=[]
    monkeypatch.setattr(banks,'Targets',lambda *a:object())
    monkeypatch.setattr(resources.qualification,'tp_checkpoint',lambda *a:events.append('parity'))
    def stop(*a):events.append('stress');raise MemoryError('early')
    monkeypatch.setattr(resources,'tp_stress',stop)
    monkeypatch.setattr(banks,'prepare_u100',lambda *a:pytest.fail('Late work ran before early stress'))
    with pytest.raises(MemoryError,match='early'):
        resources.measure(dict(root=str(tmp_path)),None,SimpleNamespace(world=2,device=torch.device('cuda')),None,1.)
    assert events==['parity','stress']


def _two_cuda_checkpoint(rank,root,init_method):
    from datetime import timedelta
    from pathlib import Path
    from hlt_classification.luka_embedding_bridge_hydra import qualification,contracts
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.distributed.init_process_group('nccl',rank=rank,world_size=2,init_method=init_method,timeout=timedelta(minutes=5))
    try:
        ds=synthetic_dataset();ds.spec.update(root=root,tp_execution_policy=contracts.tp_execution_policy())
        ctx=d.Context(rank,2,torch.device('cuda',rank))
        ctx.main(lambda:(Path(root)/'setup').mkdir())
        qualification.tp_checkpoint(ds.spec,ds,banks.Targets(ds.spec,'TP',ds),ctx)
    finally:torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<2 or not torch.distributed.is_nccl_available(),reason='Two CUDA GPUs with NCCL required')
def test_two_gpu_checkpoint_syncbn_ddp_parity(tmp_path):
    torch.multiprocessing.spawn(_two_cuda_checkpoint,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=2,join=True)
