"""Privileged routing experiment: semantics, real adapters, artifacts and queue."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_luka_attention import cache, tiny_config, parent_nodes
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_gating import contracts as c, model as m, banks, training, worker, campaign


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def gate_fixture(root):
    return c.artifact('GATE',root=str(root),project_dir='/project',source_commit='a'*40,
        **campaign.registration(dict(nodes=parent_nodes())))


def test_registered_graph_and_matched_design():
    nodes=c.nodes(parent_nodes()); tasks=c.tasks(nodes)
    assert len(nodes)==13 and len(tasks)==23
    assert c.recipe()['base']['batch_size']==128 and c.recipe()['epochs']==30
    for a,b,d in zip(nodes[:3],nodes[3:6],nodes[6:9]):
        assert a['initialization_seed']==b['initialization_seed']==d['initialization_seed']
        assert a['sampler_seed']==b['sampler_seed']==d['sampler_seed']
        assert a['context_coordinate'] is b['context_coordinate'] is None
        assert d['context_coordinate'] in ('U100','D066','D033')
        assert a['warm_start']==a['teacher'] and b['warm_start']==b['teacher']
    for n in nodes[9:12]:
        assert n['warm_start']==n['teacher']=='PRIV_D000'
        assert n['gate_mode']=='self' and n['context_coordinate'] is None
    assert nodes[-1]['teacher']=='SELF_D000'
    assert [n['freeze_epochs'] for n in nodes[9:]]==[0,0,3,0]
    assert [n['gate_weight'] for n in nodes[9:]]==[0.,.1,.1,0.]
    seen=set()
    for t in tasks:
        assert set(t['dependencies'])<=seen
        assert t['task_id'] not in seen
        seen.add(t['task_id'])
    assert 'ORACLE' in c.input_label(nodes[8])
    assert nodes[8]['deployable'] is False
    assert all(n['deployable'] for n in nodes[9:])
    assert all(c.input_label(n).startswith('HLT ONLY') for n in nodes[9:])
    assert c.recipe()['independent_test_available'] is False


def test_log_gate_bounds_mask_transport_centering_and_gradients():
    a=torch.randn(2,4,9,requires_grad=True); b=torch.randn(2,4,9,requires_grad=True)
    mask=torch.tensor([[True,True,False,False],[True]*4])
    loss=m.centered_gate_loss(a,b,mask)
    perm=torch.tensor([1,0,3,2])
    torch.testing.assert_close(loss,m.centered_gate_loss(a[:,perm],b[:,perm],mask[:,perm]))
    assert m.centered_gate_loss(a,a,mask).item()==0
    assert m.centered_gate_loss(a[:,:1],b[:,:1],mask[:,:1]).item()==pytest.approx(0)
    loss.backward()
    assert a.grad is not None and a.grad.abs().sum()>0 and b.grad is None
    assert a.grad[0,2:].abs().sum()==0
    bad=b.detach().clone(); bad[0,2:]=float('nan')
    assert torch.isfinite(m.centered_gate_loss(a,bad,mask))
    r=m.log_gate(a,mask)
    assert r.abs().max()<=np.log(4) and r[0,2:].abs().sum()==0 and r[0,:,2:].abs().sum()==0
    scores=torch.randn(2,4,4).masked_fill(~mask[:,None,:],float('-inf'))
    weighted=scores.softmax(-1)*r.exp(); weighted/=weighted.sum(-1,keepdim=True)
    torch.testing.assert_close(weighted,(scores+r).softmax(-1))
    with pytest.raises(ValueError): m.log_gate(a,torch.zeros_like(mask))
    with pytest.raises(ValueError): m.log_gate(a[...,:8],mask)
    bad[0,0]=float('nan')
    with pytest.raises(ValueError): m.log_gate(bad,mask)


def test_identity_and_only_registered_attention_entry(monkeypatch):
    tiny_config(monkeypatch)
    native=DelphesParticleTransformer().eval(); model=m.GatedParT('privileged').eval()
    model.mod.load_state_dict(native.mod.state_dict())
    raw=cache().batch(np.arange(4)); inputs={k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')}
    extra=dict(context_features=inputs['features'].clone(),context_mask=inputs['mask'])
    with torch.no_grad():
        torch.testing.assert_close(native(**inputs),model(**inputs,**extra),rtol=0,atol=0)
        assert m.log_gate(model.factors,inputs['mask']).abs().sum()==0
    # The first modified block's QKV and native pair embedding must not consume context.
    with torch.no_grad(): model.gate.net[-1].weight[4:].normal_(0,.2)
    captured=[]
    handle=model.mod.blocks[6].attn.register_forward_pre_hook(
        lambda module,args,kw:captured.append(tuple(x.detach().clone() for x in args[:3])),with_kwargs=True)
    pair=[]
    pair_handle=model.mod.pair_embed.register_forward_hook(lambda module,args,out:pair.append(out.detach().clone()))
    changed=extra['context_features'].clone(); changed[:,0]+=3
    with torch.no_grad():
        a=model(**inputs,**extra); first=model.factors.clone()
        b=model(**inputs,**dict(extra,context_features=changed)); second=model.factors.clone()
    handle.remove(); pair_handle.remove()
    torch.testing.assert_close(captured[0],captured[1],rtol=0,atol=0)
    torch.testing.assert_close(pair[0],pair[1],rtol=0,atol=0)
    assert not torch.allclose(m.log_gate(first,inputs['mask']),m.log_gate(second,inputs['mask']))
    assert not torch.allclose(a,b)
    endpoint=m.GatedParT('self').eval(); endpoint.load_state_dict(model.state_dict())
    with pytest.raises(PermissionError): endpoint(**inputs,**extra)
    with pytest.raises(ValueError): model(**inputs)
    with pytest.raises(PermissionError): m.GatedParT('none')(**inputs,**extra)
    endpoint(**inputs).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in endpoint.gate.parameters())


def test_warm_start_and_freeze_buffers(monkeypatch):
    tiny_config(monkeypatch)
    node=c.nodes(parent_nodes())[6]
    native=DelphesParticleTransformer(); old=deepcopy(native.state_dict())
    model=m.new_model(node,old)
    for k in old: assert torch.equal(old[k],model.state_dict()[k])
    copied=m.new_model(c.nodes(parent_nodes())[9],model.state_dict())
    assert all(torch.equal(v,copied.state_dict()[k]) for k,v in model.state_dict().items())
    bad=dict(old); bad.pop(next(iter(bad)))
    with pytest.raises(ValueError): m.new_model(node,bad)
    training.epoch_mode(copied,True)
    frozen=deepcopy(copied.mod.state_dict()); raw=cache().batch(np.arange(4))
    opt=torch.optim.AdamW(copied.parameters(),lr=.01)
    out=banks.forward(copied,raw,None,'cpu'); out.square().sum().backward(); opt.step()
    assert all(torch.equal(v,copied.mod.state_dict()[k]) for k,v in frozen.items())
    assert all(p.grad is None for p in copied.mod.parameters())
    training.epoch_mode(copied,False); opt.zero_grad(set_to_none=True)
    banks.forward(copied,raw,None,'cpu').square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in copied.mod.parameters())


def test_execution_policy_is_not_a_scientific_recipe_change():
    assert c.execution_policy()['backend']=='SDPBackend.MATH'
    assert c.execution_policy()['otherwise']=='native_dispatch'
    assert c.recipe()['schema_version']==1
    assert c.artifact('PREFLIGHT')['schema_version']==2
    old=with_content_hash(dict(c.artifact('PREFLIGHT'),schema_version=1,
                              contract='LUKA_PRIVILEGED_GATE_PREFLIGHT/v1'))
    with pytest.raises(ValueError): c.validate(old,'PREFLIGHT')


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA frozen-backbone regression')
def test_standard_cuda_frozen_backend_gradients_and_unfreeze(bf16):
    from hlt_classification.luka_gating.preflight import frozen_backend_check
    # Non-aligned lengths and padding, actual standard ParT, fixed non-grad inputs.
    rng=np.random.default_rng(16)
    vectors=rng.normal(size=(4,4,17)).astype(np.float32)
    vectors[:,3]=np.sqrt((vectors[:,:3]**2).sum(1)+1)
    mask=np.arange(17)[None,None,:]<np.array([17,9,3,1])[:,None,None]
    raw=dict(features=rng.normal(size=(4,17,17)).astype(np.float32),vectors=vectors,mask=mask)
    result=frozen_backend_check(raw,'cuda',bf16=bf16)
    assert result['math_calls']==1 and result['joint_training_restored']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA backend context restoration')
def test_scoped_math_restored_after_exception_and_not_used_joint(monkeypatch):
    tiny_config(monkeypatch)
    model=m.GatedParT('self').cuda()
    training.epoch_mode(model,True)
    attention=model.mod.blocks[6].attn
    def flags():
        return tuple(getattr(torch.backends.cuda,n+'_sdp_enabled')() for n in ('math','flash','mem_efficient','cudnn'))
    before=flags()
    original=attention.forward
    def fail(*args,**kwargs):
        assert flags()==(True,False,False,False)
        raise RuntimeError('injected attention failure')
    monkeypatch.setattr(attention,'forward',fail)
    raw=cache().batch(np.arange(4))
    with pytest.raises(RuntimeError,match='injected attention'):
        banks.forward(model,raw,None,'cuda')
    assert flags()==before and model._backend_contexts=={} and model._correction is None
    monkeypatch.setattr(attention,'forward',original)
    banks.forward(model,raw,None,'cuda').float().square().mean().backward()
    assert model._math_calls==1 and flags()==before
    model.zero_grad(set_to_none=True); training.epoch_mode(model,False)
    banks.forward(model,raw,None,'cuda').float().square().mean().backward()
    assert model._math_calls==0 and flags()==before
    model.eval()
    with torch.inference_mode(): banks.forward(model,raw,None,'cuda')
    assert model._math_calls==0 and flags()==before


@pytest.mark.parametrize('batch_first',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='legacy Torch MHA CUDA regression')
def test_frozen_gate_native_torch_mha_adapter(batch_first,monkeypatch):
    tiny_config(monkeypatch)
    model=m.GatedParT('self').cuda()
    attention=torch.nn.MultiheadAttention(8,2,dropout=0.,batch_first=batch_first).cuda().eval()
    for p in attention.parameters(): p.requires_grad_(False)
    attention.register_forward_pre_hook(model._inject,with_kwargs=True)
    attention.register_forward_hook(model._restore_backend,with_kwargs=True,always_call=True)
    model._correction=torch.randn(2,9,9,device='cuda',requires_grad=True)
    x=torch.randn((2,9,8) if batch_first else (9,2,8),device='cuda')
    out,_=attention(x,x,x,attn_mask=torch.zeros(4,9,9,device='cuda'),need_weights=False)
    out.square().mean().backward()
    assert model._math_calls==1 and not model._backend_contexts
    assert torch.isfinite(model._correction.grad).all() and model._correction.grad.abs().sum()>0
    assert all(p.grad is None for p in attention.parameters())


def test_stress_exercises_frozen_and_joint_backbone(monkeypatch):
    from hlt_classification.luka_gating import preflight as p
    tiny_config(monkeypatch)
    node=next(n for n in c.nodes(parent_nodes()) if n['node_id']=='REPLACE_STAGED')
    for frozen in (True,False):
        result=p.stress(cache(coordinate='D000'),None,node,'cpu',backward=True,freeze=frozen)
        assert result['mode']==('frozen' if frozen else 'joint')
        assert result['frozen_backbone_unchanged']==(True if frozen else None)
    with pytest.raises(ValueError,match='Unregistered frozen'):
        p.stress(cache(),None,node,'cpu',backward=False,freeze=True)
    with pytest.raises(ValueError,match='Unregistered frozen'):
        p.stress(cache(),None,c.nodes(parent_nodes())[0],'cpu',backward=True,freeze=True)


def test_ragged_factor_bank_roundtrip_and_corruption(tmp_path,monkeypatch):
    tiny_config(monkeypatch)
    node=c.nodes(parent_nodes())[8]; teacher=m.new_model(node)
    with torch.no_grad(): teacher.gate.net[-1].weight[4:].normal_(0,.2)
    tr=cache(coordinate='D000',rows=259); ctx=cache(coordinate='D033',rows=259)
    parents=dict(checkpoint='c'*64,report='d'*64); root=tmp_path/'bank'
    manifest=banks.export(root,teacher,tr,ctx,parents=parents,node=node,device='cpu')
    loaded=banks.Bank(root,tr,parents=parents,teacher='PRIV_D000',require_factors=True)
    ix=np.array([258,1,129,1]); factors=loaded.factor_batch(ix,16)
    assert factors.shape==(4,16,9) and np.array_equal(factors[1],factors[3])
    assert sum(s['stop']-s['start'] for s in manifest['shards'])==259
    with torch.no_grad():
        banks.forward(teacher,tr.batch(ix),ctx.batch(ix),'cpu')
        expected=teacher.factors.numpy()
    np.testing.assert_allclose(factors[:,:expected.shape[1]],expected,atol=1e-6)
    with pytest.raises(PermissionError): banks.Bank(root,cache('validation','D000',259),parents=parents,teacher='PRIV_D000')
    with pytest.raises(ValueError): loaded.factor_batch(ix,1)
    for field,value in [('coordinate','D033'),('context_coordinate',None),('temperature',1.),('shards',[]),('factors',False)]:
        (root/'manifest.json').write_text(json.dumps(with_content_hash(dict(manifest,**{field:value}))))
        with pytest.raises(ValueError): banks.Bank(root,tr,parents=parents,teacher='PRIV_D000')
    (root/'manifest.json').write_text(json.dumps(manifest))
    shuffled=cache(coordinate='D033',rows=259); shuffled.identities=shuffled.identities[::-1].copy()
    with pytest.raises(ValueError): banks.paired(tr,shuffled)
    with pytest.raises(ValueError): banks.forward(teacher,tr.batch(ix),ctx.batch(ix[::-1]),'cpu')
    with (root/manifest['shards'][0]['payload']['path']).open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError): banks.Bank(root,tr,parents=parents,teacher='PRIV_D000')


def mocked_population(tmp_path,monkeypatch,device):
    tiny_config(monkeypatch)
    root=tmp_path/'science'; root.mkdir()
    parent=tmp_path/'parent'; parent.mkdir()
    anchor=m.new_model(c.anchor_node(c.nodes(parent_nodes()))).to(device)
    save_state(parent/'selected.pt',anchor.state_dict())
    def view(role,co):
        data=cache(role,co)
        for block in data.blocks:
            block.features[:,11]+={'U100':1.,'D066':.66,'D033':.33,'D000':0.}[co]
        return data
    val=view('validation','U100')
    metrics=evaluate_probabilities(val.labels,banks.predict(anchor,val,device=device))
    write_immutable_json(parent/'report.json',dict(validation=metrics))
    write_immutable_json(parent/'diagnostics.json',dict(original_validation=metrics))
    write_immutable_json(parent/'receipt.json',dict(complete=True))
    refs=dict(checkpoint=file_ref(parent/'selected.pt'),report=file_ref(parent/'report.json'),
              diagnostics=file_ref(parent/'diagnostics.json'),receipt=file_ref(parent/'receipt.json'))
    gate=gate_fixture(tmp_path)
    gate.update(counts=dict(train=22,validation=22),available_quota_gib=20,
        anchor=dict(evidence=dict(environment={},prepared='a'*64,prepared_root=str(parent),
            models={n:refs for n in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100')})))
    spec=c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='a'*40)
    monkeypatch.setattr(campaign,'check_spec',lambda s:(gate,{}))
    monkeypatch.setattr(worker,'execution',lambda *a,**kw:'123')
    monkeypatch.setattr(worker,'memory_snapshot',lambda t:dict(peak_rss_bytes=1,gpu_peak_bytes=1,rss_bytes=1))
    monkeypatch.setattr(worker,'data',lambda g,r,co:view(r,co))
    meta=dict(summaries={r:{co:dict(maximum=4) for co in ('D066','D033','D000')} for r in ('train','validation')})
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda p:(None,meta,{},{}))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda inv,s,ids,labels:np.arange(len(labels)))
    return gate,spec


@pytest.mark.skipif(not torch.cuda.is_available(),reason='local CUDA optional; remote acceptance still required')
def test_actual_installed_standard_part_cuda_parity(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from hlt_classification.luka_gating.preflight import parity
    raw=cache().batch(np.arange(4))
    for bf16 in (False,True):
        assert parity(raw,'cuda',bf16=bf16)['identity_updates']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real local GPU miniature')
def test_all_23_science_workers_30_epochs_and_aggregate(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    gate,spec=mocked_population(tmp_path,monkeypatch,'cuda')
    for task in gate['tasks']:
        result=worker.run(spec,task['task_id'])
        assert result==worker.completed(spec,gate,task['task_id'])
    root=Path(spec['root']); rows=load_json(root/'aggregate/results.json')['rows']
    assert len(rows)==17 and all(r['validation'] is not None for r in rows)
    for n in gate['nodes']:
        r=load_json(root/('train_'+n['node_id'])/'training_report.json')
        assert r['passes']==30 and r['scientific_fit'] is True
        assert sum(h['backbone_frozen'] for h in r['validation_history'])==n['freeze_epochs']
        assert r['parents']['warm_checkpoint'] and r['warm_start_policy']==c.recipe()['warm_start']
        if n['gate_weight']:
            assert max(h['losses']['gate'] for h in r['validation_history'])>0
    assert worker.run(spec,'aggregate')==worker.completed(spec,gate,'aggregate')
    with (root/'train_PRIV_D033/selected.pt').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError): worker.completed(spec,gate,'train_REPLACE_GATE')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real local GPU miniature')
def test_full_production_preflight_graph_mock_site(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from hlt_classification.luka_gating import preflight as p
    gate,_=mocked_population(tmp_path,monkeypatch,'cuda')
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(p,'installed_environment',lambda:{})
    monkeypatch.setattr(p,'memory_snapshot',lambda t:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=t.cuda.max_memory_allocated()))
    result=p.run(gate); p.validate_report(gate,result)
    assert len(result['measurements'])==22 and len(result['stress'])==27
    assert result['schema_version']==2 and len(result['frozen_checks'])==2
    assert result['stress'][0]['mode']=='frozen'
    assert result['technical_checks_passed'] and not result['scientific_fit']
    for field,value in [('scientific_fit',True),('technical_checks_passed',False),('stress',result['stress'][:-1]),
                        ('measurements',result['measurements'][:-1]),('parities',result['parities'][:1]),
                        ('storage',dict(result['storage'],required_bytes=1)),
                        ('attention_execution',{}),('frozen_checks',[]),
                        ('stress',result['stress'][1:]),
                        ('frozen_checks',[dict(p,math_calls=0) for p in result['frozen_checks']])]:
        with pytest.raises(ValueError): p.validate_report(gate,with_content_hash(dict(result,**{field:value})))
    with pytest.raises(FileExistsError): p.run(gate)
    path=Path(gate['root'])/'preflight/train_REPLACE_GATE/selected.pt'
    with path.open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError): p.validate_report(gate,result)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA preflight failure ordering')
@pytest.mark.parametrize('failure',['smoke','maximum_support'])
def test_frozen_failure_stops_before_expensive_graph(tmp_path,monkeypatch,failure):
    from hlt_classification.luka_gating import preflight as p
    gate,_=mocked_population(tmp_path,monkeypatch,'cuda')
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(p,'installed_environment',lambda:{})
    monkeypatch.setattr(p,'memory_snapshot',lambda t:dict(peak_rss_bytes=1,rss_bytes=1,gpu_peak_bytes=1))
    monkeypatch.setattr(p,'parity',lambda *a,**k:{})
    calls=[]
    monkeypatch.setattr(worker,'execute_task',lambda *a,**k:calls.append(a))
    def fail(*args,**kwargs): raise RuntimeError('frozen check failed')
    monkeypatch.setattr(p,'frozen_backend_check',fail if failure=='smoke' else lambda *a,**k:{})
    if failure=='maximum_support': monkeypatch.setattr(p,'stress',fail)
    with pytest.raises(RuntimeError,match='frozen check failed'): p.run(gate)
    assert not calls and not (Path(gate['root'])/'preflight/receipt.json').exists()


def test_budgets_use_joint_estimate_not_frozen_speed(tmp_path):
    from hlt_classification.luka_gating import preflight
    measured=[dict(kind='train',load_seconds=20,epoch_seconds=100,other_seconds=5,frozen_backbone=False),
              dict(kind='train',load_seconds=20,epoch_seconds=1,other_seconds=5,frozen_backbone=True),
              dict(kind='export',load_seconds=20,export_seconds=10)]
    assert preflight.budgets(measured)==(89,15)
    for v in (float('nan'),-1,10000):
        bad=deepcopy(measured); bad[0]['epoch_seconds']=v
        with pytest.raises(ValueError): preflight.budgets(bad)
    # An arbitrarily poor measured AUC is never an admission criterion.
    assert preflight.budgets([dict(x,auc=0.) for x in measured])==(89,15)
    with pytest.raises(OSError): campaign.storage(dict(gate_fixture(tmp_path),available_quota_gib=1),[1,2,3])


def test_exact_submission_no_retry_and_dag(tmp_path,monkeypatch):
    from hlt_classification.luka_gating import submission as s
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=gate_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(s,'check_site',lambda p:None)
    calls=[]
    monkeypatch.setattr(exact.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(stdout='12345\n')))
    dry=s.submit(gate)
    assert not calls and dry['jobs']==1
    assert '--time=08:00:00' in dry['commands'][0]['command']
    for h,a in [('wrong',campaign.AUTHORIZATION),(dry['content_hash'],'wrong')]:
        with pytest.raises(PermissionError): s.submit(gate,execute=True,plan_hash=h,authorization=a)
    s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)
    assert s.accepted_jobs(gate)==dict(preflight='12345') and len(calls)==1
    s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)
    assert len(calls)==1
    spec=c.artifact('SPEC',root=str(tmp_path/'science'),project_dir='/project',train_minutes=90,export_minutes=15)
    plan=s.render(spec,gate)
    assert plan['jobs']==23
    for row in plan['commands']:
        if row['dependencies']: assert any(a.startswith('--dependency=afterok:') for a in row['command'])
        assert '--partition=debug' in row['command'] and '--no-requeue' in row['command']
    with pytest.raises(ValueError): s.render(dict(spec,train_minutes=1441),gate)
    other=tmp_path/'other'; other.mkdir(); gate=gate_fixture(other); dry=s.submit(gate)
    def fail(*args,**kwargs): raise RuntimeError('Ambiguous scheduler response')
    monkeypatch.setattr(exact.subprocess,'run',fail)
    for error in (RuntimeError,FileExistsError):
        with pytest.raises(error): s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)


def test_source_compatibility_and_namespace_isolation(tmp_path,monkeypatch):
    from hlt_classification.luka_gating import source
    old,new=tmp_path/'old',tmp_path/'new'; frozen='src/hlt_classification/frozen.py'
    before={frozen}; after=before|source.CONCURRENT|source.ATTENTION|source.ADDITIONS
    for root,names in ((old,before),(new,after)):
        for name in names:
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'pinned\n')
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(source,'_files',lambda root:before if root==old else after)
    monkeypatch.setattr(source.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=b'pinned\n'))
    parent=dict(project_dir=str(old),source_snapshot={})
    source.compatible(parent,new)
    (new/frozen).write_bytes(b'different\n')
    with pytest.raises(ValueError,match='Frozen scientific'): source.compatible(parent,new)
    (new/frozen).write_bytes(b'pinned\n')
    name=sorted(source.ATTENTION)[0]; (new/name).write_bytes(b'changed\n')
    with pytest.raises(ValueError,match='Pinned existing'): source.compatible(parent,new)
    (new/name).write_bytes(b'pinned\n'); after.add('src/unregistered.py')
    with pytest.raises(ValueError,match='Unregistered'): source.compatible(parent,new)


def test_guarded_root_quota_and_failed_preflight_no_science(tmp_path,monkeypatch):
    from hlt_classification.luka_gating import source,preflight,submission
    project=tmp_path/'project'; prepared=tmp_path/'prepared'; old=tmp_path/'old/science'
    anchor=c.artifact('ANCHOR',evidence=dict(prepared_root=str(prepared),raw_container=str(tmp_path/'raw')))
    parent=dict(root=str(old),project_dir=str(tmp_path/'oldproject'),nodes=parent_nodes())
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='b'*64))
    monkeypatch.setattr(source,'authenticate',lambda *a:(anchor,parent))
    for root in (tmp_path,old/'run',project/'run',prepared/'run'):
        with pytest.raises(PermissionError): campaign.create(root,'parent',project,'a'*40,available_quota_gib=20)
    with pytest.raises(ValueError): campaign.create(tmp_path/'run','parent',project,'a'*40,available_quota_gib=0)
    gate=campaign.create(tmp_path/'run','parent',project,'a'*40,available_quota_gib=20)
    with pytest.raises(FileExistsError): campaign.create(tmp_path/'run','parent',project,'a'*40,available_quota_gib=20)
    path=Path(gate['root'])/'preflight'; path.mkdir(); write_immutable_json(path/'receipt.json',dict(job_id='1122'))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(preflight,'validate_report',lambda g,r:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='wrong'))
    with pytest.raises(ValueError,match='exact journal'): campaign.evidence(gate)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='1122'))
    def failed(job): raise ValueError('accounting FAILED')
    monkeypatch.setattr(campaign,'preflight_accounting',failed)
    with pytest.raises(ValueError,match='accounting FAILED'): campaign.science(gate)
    assert not (Path(gate['root'])/'science').exists()


def test_failed_final_authentication_does_not_publish_receipt(tmp_path,monkeypatch):
    gate,spec=mocked_population(tmp_path,monkeypatch,'cpu')
    def fail(): raise ValueError('Source changed during export')
    with pytest.raises(ValueError,match='Source changed'):
        worker.execute_task(spec,gate,'export_ANCHOR',device='cpu',before_publish=fail)
    assert not (Path(spec['root'])/'export_ANCHOR/receipt.json').exists()


def test_cli_recovery_is_already_percent():
    text=Path('scripts/luka_gating.py').read_text()
    assert "f'{rec:.1f}'" in text and '100*rec' not in text
    # CLI and source dependencies remain separate from Fresh_check.
    for path in Path('src/hlt_classification/luka_gating').glob('*.py'):
        assert 'import Fresh_check' not in path.read_text()


def test_queue_workers_and_runbook_shell_parse():
    import re
    import shutil
    import subprocess
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():
        found=shutil.which('bash')
        if not found: pytest.skip('Bash syntax check requires bash')
        bash=Path(found)
    for path in ('scripts/queue_luka_gating.sh','sbatch/run_luka_gating.sh'):
        subprocess.run([str(bash),'-n',path],check=True,capture_output=True,text=True)
    blocks=re.findall(r'```bash\n(.*?)```',Path('docs/LUKA_PRIVILEGED_GATE_RUNBOOK.md').read_text(),re.S)
    assert len(blocks)>=4
    for block in blocks:
        subprocess.run([str(bash),'-n'],input=block,check=True,capture_output=True,text=True)
