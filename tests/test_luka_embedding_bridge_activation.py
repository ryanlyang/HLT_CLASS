"""TP recomputation preserves native math/state while bounding live activations."""
import numpy as np
import pytest
import torch
from hlt_classification.luka_embedding_bridge_hydra import activation, distributed as d, banks, registry, execution, campaign
from hlt_classification.luka_subclass_hydra.benchmark_training import parity_snapshot, compare
from hlt_classification.luka_subclass_ddp_hydra.resources import PROBE_TOLERANCES
from test_luka_embedding_bridge_hydra import synthetic_dataset, small


@pytest.fixture(autouse=True)
def math_settings(monkeypatch):
    threads = torch.get_num_threads()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    deterministic=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1)
    execution.configure_math(campaign.registration(),'setup')
    yield
    torch.set_num_threads(threads)
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags
    torch.use_deterministic_algorithms(deterministic[0],warn_only=deterministic[1])


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
def test_full_depth_long_crop_optimizer_repeat_and_checkpoint():
    """Long, variable crops exercise SDPA backward splitting missed by tiny jets.

    Without deterministic algorithms, even ordinary-versus-ordinary failed
    parameter parity locally (~1.31e-5), despite identical loss and RNG. Test
    complete AdamW updates, including nearly cancelling normalization biases.
    """
    from hlt_classification.luka_embedding_bridge_hydra.model import Model
    from test_luka_embedding_bridge_hydra import physical
    from hlt_classification.luka_embedding_bridge_hydra import regions,losses
    ctx=d.Context(0,1,torch.device('cuda',0));particles=[]
    for i in range(15):
        a=physical(60+i*3)
        a[:,6:10]*=np.random.default_rng(i).uniform(.5,2,(len(a),4))
        particles.append(a)
    labels=np.arange(15)%11
    raw=regions.padded([regions.encode(a) for a in particles],labels)
    snapshots=[];rngs=[]
    for scale in registry.SCALES:
        crops,valid=regions.crops(particles,particles,labels,scale)
        args=d.arrays(raw,ctx);args['crops']=[d.arrays(x,ctx) for x in crops]
        args['valid']=torch.as_tensor(valid,device=ctx.device)
        y=torch.as_tensor(labels,device=ctx.device)
        snapshots.clear();rngs.clear()
        for enabled,diagnose in ((False,False),(False,False),(True,False),(True,True)):
            torch.manual_seed(23);m=Model('TP').to(ctx.device).train();m.checkpoint_encoding=enabled
            opt=d.optimizer(m)
            for group in opt.param_groups:group['lr']=d.learning_rate(3.)
            target=dict(valid=args['valid'])
            if diagnose:d.probe(m,'TP',args,y,target,3.,1.,ctx)
            out=m(**args);total,_=losses.objective('TP',out,y,target,3.)
            total.backward();opt.step()
            snapshots.append(parity_snapshot(m,opt,float(total)))
            rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state(ctx.device)))
            del m,opt,out,total;d.release_memory(ctx)
        for observed,rng in zip(snapshots[1:],rngs[1:]):
            result=compare(snapshots[0],observed,PROBE_TOLERANCES)
            assert result['passed'],(scale,result)
            assert all(torch.equal(a,b) for a,b in zip(rngs[0],rng))


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


@pytest.mark.parametrize('bad', ['rng','gradient','buffers','policy','condition','rank','stress_rows','stress_gpu','stress_time',
                               'repeat','numerical_settings','math_policy'])
def test_admission_rejects_resealed_incomplete_checkpoint_evidence(tmp_path,bad):
    from test_luka_embedding_bridge_hydra import execution_spec, resource_diagnostic
    from hlt_classification.data.cache_contracts import with_content_hash
    from hlt_classification.luka_embedding_bridge_hydra import resources
    spec=execution_spec(tmp_path);value=resource_diagnostic(spec)
    checkpoint=value['tp_checkpoint'];row=checkpoint['checks'][0]['ranks'][0];stress=value['tp_stress']
    if bad=='rng':row['rng_equal']=False
    if bad=='gradient':row['gradient_relative_l2']=1.
    if bad=='buffers':row['buffer_abs']=1.
    if bad=='policy':checkpoint['policy']['preserve_rng_state']=False
    if bad=='condition':checkpoint['checks'][0]['condition']=['HLT','whole']
    if bad=='rank':checkpoint['checks'][0]['ranks'].pop()
    if bad=='repeat':row['ordinary_repeat']['parameters_relative_l2']=1.
    if bad=='numerical_settings':checkpoint['numerical_settings']['deterministic_algorithms']=False
    if bad=='math_policy':checkpoint['execution_math_policy']['a6000_1_fp32']['warn_only']=True
    if bad=='stress_rows':stress['cases'][0]['rows']=64
    if bad=='stress_gpu':stress['memory']['ranks'][0]['gpu_peak_bytes']=value['devices'][0]['total_memory_bytes']
    if bad=='stress_time':stress['cases'][0]['seconds']=-1.
    for part in ('tp_checkpoint','tp_stress'):
        value[part].pop('content_hash');value[part]=with_content_hash(value[part])
    value.pop('content_hash');value=with_content_hash(value)
    with pytest.raises((ValueError,MemoryError)):resources.accept(spec,value)


def test_tp_checks_precede_numerical_replay_and_miniatures(monkeypatch,tmp_path):
    from hlt_classification.luka_embedding_bridge_hydra import resources
    from types import SimpleNamespace
    events=[]
    monkeypatch.setattr(resources.qualified,'profiles',lambda *a:dict.fromkeys(resources.qualified.REUSED))
    monkeypatch.setattr(resources.qualified,'profile_devices',lambda *a:[])
    monkeypatch.setattr(resources.regional_inputs,'qualify',lambda *a:events.append('frontend'))
    monkeypatch.setattr(banks,'Targets',lambda *a:object())
    monkeypatch.setattr(resources.qualification,'tp_checkpoint',lambda *a:events.append('parity'))
    def stop(*a):events.append('stress');raise MemoryError('early')
    monkeypatch.setattr(resources,'tp_stress',stop)
    monkeypatch.setattr(banks,'prepare_u100',lambda *a:pytest.fail('Late work ran before early stress'))
    with pytest.raises(MemoryError,match='early'):
        resources.measure(dict(campaign.registration(),root=str(tmp_path)),None,SimpleNamespace(world=1,device=torch.device('cuda'),main=lambda fn:fn()),None,1.)
    assert events==['frontend','parity','stress']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_single_gpu_production_checkpoint_qualification(tmp_path):
    from hlt_classification.luka_embedding_bridge_hydra import qualification,contracts
    from hlt_classification.data.cache_contracts import load_json
    torch.cuda.set_device(0)
    execution.configure_math(campaign.registration(),'setup')
    ds=synthetic_dataset();ds.spec.update(campaign.registration(),root=str(tmp_path))
    (tmp_path/'setup').mkdir();ctx=d.Context(0,1,torch.device('cuda',0))
    value=qualification.tp_checkpoint(ds.spec,ds,banks.Targets(ds.spec,'TP',ds),ctx)
    assert value==load_json(tmp_path/'setup/tp_checkpoint.json') and value['world_size']==1
    assert len(value['checks'])==6 and all(len(row['ranks'])==1 for row in value['checks'])
    assert not torch.distributed.is_initialized()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_single_gpu_global128_and_tails_use_full_normalization_population():
    ds=synthetic_dataset(train_rows=139);s=ds.spec;s.update(campaign.registration())
    ctx=d.Context(0,1,torch.device('cuda',0));target=banks.Targets(s,'TP',ds)
    m,w,opt=d.build(s,'TP',ctx)
    assert m is w and not any(isinstance(x,torch.nn.SyncBatchNorm) for x in m.modules())
    observed=[]
    bn=next(x for x in m.modules() if isinstance(x,torch.nn.BatchNorm1d))
    hook=bn.register_forward_pre_hook(lambda module,args:observed.append(len(args[0])))
    try:
        for count in (128,11,98):
            for batch in range(6):
                observed.clear();ix=np.arange(count)
                args,y,target_batch,weight=d.prepare('TP',ds,target,ix,3,batch,ctx)
                assert len(y)==count and weight==1. and all(len(crop['mask'])==count for crop in args['crops'])
                values,_=d.step(m,w,opt,'TP',ds,target,ix,3,batch,3.,ctx)
                assert observed and set(observed)=={count}
                assert np.isfinite(list(values.values())).all()
    finally:
        hook.remove();del m,w,opt;d.release_memory(ctx)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_single_rank_scientific_tp_fit_export_and_consumer(tmp_path):
    from hlt_classification.luka_embedding_bridge_hydra import training
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    ds=synthetic_dataset();s=ds.spec;counts=s['counts'];s.update(campaign.registration(),counts=counts,root=str(tmp_path))
    ctx=d.Context(0,1,torch.device('cuda',0));target=banks.Targets(s,'TP',ds)
    for options in (dict(epochs=2),dict(train_indices=np.arange(22))):
        with pytest.raises(PermissionError):training.fit(tmp_path/('bad_'+next(iter(options))),s,'TP',ds,target,ctx,**options)
    m,report=training.fit(tmp_path/'fit',s,'TP',ds,target,ctx)
    assert report['epochs']==30 and report['world_size']==1 and report['native_reload']
    assert training.validate_fit(tmp_path/'fit',s,'TP')[0]==report
    checkpoint=file_ref(tmp_path/'fit/selected_full.pt')
    banks.export(tmp_path/'bank',s,'TP',m,ds,ctx,checkpoint)
    parent=banks.Bank(tmp_path/'bank',s,'TP',ds.view('train','D000'),checkpoint)
    consumer=banks.Targets(s,'P7',ds,parent_bank=parent)
    student,wrapped,opt=d.build(s,'P7',ctx)
    values,_=d.step(student,wrapped,opt,'P7',ds,consumer,np.arange(15),3,0,3.,ctx)
    assert np.isfinite(list(values.values())).all()
    del m,student,wrapped,opt;d.release_memory(ctx)
