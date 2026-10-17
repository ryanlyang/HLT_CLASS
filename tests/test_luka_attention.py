from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_attention import model as m, banks, contracts as c, training
from hlt_classification.luka_fullsim.regular_campaign import nodes as parent_nodes
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def cache(role='train', coordinate='D066', rows=22):
    rng = np.random.default_rng(41)
    lengths = np.arange(rows)%4+1
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    f = rng.normal(size=(offsets[-1], 17)).astype(np.float32)
    p = rng.normal(size=(offsets[-1], 3)).astype(np.float32)
    v = np.column_stack((p, np.sqrt((p*p).sum(-1)+1))).astype(np.float32)
    ids = np.zeros((rows, 32), np.uint8)
    ids[:, :2] = np.array([[i%256, i//256] for i in range(rows)], np.uint8)
    return RamCache([RamBlock(0, offsets, f, v, ids, np.arange(rows, dtype=np.int64)%11)],
                    role=role, foundation_sha256='a'*64, coordinate_name=coordinate)


def tiny_config(monkeypatch):
    from hlt_classification.jetclass2_delphes import model
    config = model.model_config()
    config.update(embed_dims=[8], pair_embed_dims=[4], num_heads=2, num_layers=8,
                  num_cls_layers=1, fc_params=[])
    monkeypatch.setattr(model, 'model_config', lambda: deepcopy(config))


def test_loss_padding_transport_heads_and_gradient():
    p = torch.rand(2, 4, 4, requires_grad=True)
    q = torch.rand(2, 4, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True]*4])
    l = m.attention_kl(p, q, mask)
    perm = torch.tensor([1, 0, 3, 2])
    assert torch.allclose(l, m.attention_kl(p[:, perm][:, :, perm], q[:, perm][:, :, perm], mask[:, perm]))
    assert not torch.allclose(l, m.attention_kl(p, q[:, perm][:, :, perm], mask))
    l.backward()
    assert p.grad is not None and q.grad is None
    assert p.grad[0, 2:].abs().sum() == 0
    assert m.attention_kl(p, p, mask).item() == pytest.approx(0, abs=1e-7)
    single = torch.ones(1, 1, 1)
    assert m.attention_kl(single, single, torch.ones(1, 1, dtype=torch.bool)) == 0
    bad = q.detach().clone(); bad[0, 2:] = float('nan'); bad[0, :, 2:] = float('nan')
    assert torch.isfinite(m.attention_kl(p, bad, mask))
    for x in (torch.zeros_like(q), torch.full_like(q, float('nan')), -q):
        with pytest.raises(ValueError): m.attention_kl(p, x, mask)


@pytest.mark.parametrize('batch_first', [True, False])
def test_surface_matches_torch_native_mha(batch_first):
    a = nn.MultiheadAttention(8, 2, dropout=0, batch_first=batch_first)
    x = torch.randn(2, 4, 8) if batch_first else torch.randn(4, 2, 8)
    bias = torch.randn(4, 4, 4, requires_grad=True)
    pad = torch.tensor([[False, False, True, True], [False]*4])
    kwargs = dict(attn_mask=bias, key_padding_mask=pad)
    expected = a(x, x, x, **kwargs, need_weights=True, average_attn_weights=True)[1]
    result = m.attention_surface(a, (x,x,x), kwargs)
    torch.testing.assert_close(result, expected, atol=2e-7, rtol=2e-6)
    result.square().sum().backward()
    assert bias.grad is not None and bias.grad.abs().sum() > 0


def test_native_model_zero_loss_parity_and_aux_gradients(monkeypatch):
    tiny_config(monkeypatch)
    native, instrument = DelphesParticleTransformer(), m.AttentionParT()
    instrument.load_state_dict(native.state_dict(), strict=True)
    instrument.capture_attention = True
    batch = cache().batch(np.arange(4))
    kwargs = {k:torch.from_numpy(batch[k]) for k in ('features','vectors','mask')}
    outputs, grads, states = [], [], []
    for model in (native, instrument):
        model.train(); torch.manual_seed(101)
        x = dict(kwargs, features=kwargs['features'].clone().requires_grad_())
        out = model(**x)
        out.square().mean().backward()
        outputs.append(out.detach()); grads.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
        states.append(deepcopy(model.state_dict()))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert grads[0].keys() == grads[1].keys()
    for k in grads[0]: torch.testing.assert_close(grads[0][k], grads[1][k], rtol=0, atol=0)
    for k in states[0]: torch.testing.assert_close(states[0][k], states[1][k], rtol=0, atol=0)
    instrument.zero_grad(set_to_none=True)
    instrument(**kwargs)
    a = instrument.take_attention()
    m.attention_kl(a, torch.rand_like(a), kwargs['mask']).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in instrument.named_parameters() if 'in_proj' in n)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in instrument.named_parameters() if 'pair_embed' in n)


def test_bank_roundtrip_order_and_corruption(tmp_path, monkeypatch):
    tiny_config(monkeypatch)
    monkeypatch.setattr(banks,'TARGET_SHARD_BYTES',1024)
    teacher = m.new_model(c.nodes(parent_nodes())[0])
    root = tmp_path/'bank'
    parent = dict(checkpoint='c'*64, report='d'*64)
    original = cache(coordinate='U100', rows=259)
    manifest = banks.export(root, teacher, original, parents=parent, teacher='ANCHOR', attention=True, device='cpu')
    student = cache(rows=259)
    loaded = banks.Bank(root, student, parents=parent, teacher='ANCHOR', require_attention=True)
    ix = np.array([258, 1, 129, 1])
    a = loaded.attention_batch(ix, 16)
    assert a.shape == (4,16,16) and np.array_equal(a[1],a[3])
    assert len(manifest['shards']) == 3
    with pytest.raises(PermissionError): banks.Bank(root, cache('validation'), parents=parent, teacher='ANCHOR', require_attention=True)
    with pytest.raises(ValueError): banks.Bank(root, student, parents={'checkpoint':'x'}, teacher='ANCHOR', require_attention=True)
    student.identities = student.identities[::-1].copy()
    with pytest.raises(ValueError): banks.Bank(root, student, parents=parent, teacher='ANCHOR', require_attention=True)
    path = root/manifest['shards'][0]['payload']['path']
    with path.open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError): banks.Bank(root, original, parents=parent, teacher='ANCHOR', require_attention=True)


def test_full_two_arm_miniature(tmp_path, monkeypatch):
    tiny_config(monkeypatch)
    models = c.nodes(parent_nodes())
    assert len(c.tasks(models)) == 12 and len(models) == 6
    for a,b in zip(models[:3], models[3:]):
        assert a['initialization_seed']==b['initialization_seed'] and a['sampler_seed']==b['sampler_seed']
    parents = dict(checkpoint='c'*64, report='d'*64)
    anchor = m.new_model(models[0])
    banks.export(tmp_path/'ANCHOR', anchor, cache(coordinate='U100'), parents=parents,
                 teacher='ANCHOR', attention=True, device='cpu')
    for node in models:
        tr, val = cache(coordinate=node['coordinate']), cache('validation', node['coordinate'])
        bank = banks.Bank(tmp_path/node['teacher'], tr, parents=parents, teacher=node['teacher'], require_attention=node['branch']=='ATTN')
        from hlt_classification.luka_attention.worker import fit
        report,model = fit(tmp_path/('fit_'+node['node_id']), {}, node, tr, val, bank, device='cpu', technical=True)
        state=model.state_dict()
        assert report['passes']==1 and report['scientific_fit'] is False and report['selected_weights_restored']
        assert state.keys()==DelphesParticleTransformer().state_dict().keys()
        if node['coordinate']!='D000':
            banks.export(tmp_path/node['node_id'], model, tr, parents=parents, teacher=node['node_id'],
                         attention=node['branch']=='ATTN', device='cpu')


def test_production_parity_and_stress_functions_on_cpu(monkeypatch):
    tiny_config(monkeypatch)
    from hlt_classification.luka_attention import preflight
    value=preflight.parity(cache().batch(np.arange(4)), 'cpu')
    assert value['native_attention_surface'] and value['optimizer_updates']
    node=c.nodes(parent_nodes())[3]
    assert preflight.stress(cache(),node,'cpu',backward=True)['maximum']==4
    assert preflight.stress(cache('validation'),node,'cpu',backward=False)['backward'] is False


@pytest.mark.parametrize('arm_index',[0,3])
def test_actual_fixed30_scientific_fit(tmp_path,monkeypatch,arm_index):
    tiny_config(monkeypatch)
    node=c.nodes(parent_nodes())[arm_index]
    parents=dict(checkpoint='c'*64,report='d'*64)
    banks.export(tmp_path/'bank',m.new_model(node),cache(coordinate='U100'),parents=parents,
                 teacher='ANCHOR',attention=True,device='cpu')
    tr,val=cache(),cache('validation')
    bank=banks.Bank(tmp_path/'bank',tr,parents=parents,teacher='ANCHOR',require_attention=True)
    report,_=training.train(m.new_model(node),tr,val,bank=bank,node=node,device='cpu')
    assert report['passes']==30 and report['scientific_fit']
    assert report['validation_history'][-1]['learning_rate']==pytest.approx(1.5e-5)
    assert report['validation_history'][-1]['update']==30
    from hlt_classification.luka_attention.worker import report_valid
    report_valid(report,node,bank.manifest['content_hash'],tr.foundation_sha256,len(val))
    bad=with_content_hash(dict(report,scientific_fit=False))
    with pytest.raises(ValueError): report_valid(bad,node,bank.manifest['content_hash'],tr.foundation_sha256,len(val))


def gate_fixture(tmp_path):
    from hlt_classification.luka_attention import campaign
    return c.artifact('GATE',root=str(tmp_path),project_dir='/project',source_commit='a'*40,
                       **campaign.registration(dict(nodes=parent_nodes())))


def test_exact_dag_and_live_claims(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_attention import campaign,submission as s
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=gate_fixture(tmp_path)
    plan=s.render(gate)
    assert len(plan['commands'])==1 and '--partition=debug' in plan['commands'][0]['command']
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(s,'check_site',lambda p:None)
    calls=[]
    monkeypatch.setattr(exact.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(stdout='12345\n')))
    dry=s.submit(gate)
    assert not calls
    for h,a in [('wrong',campaign.AUTHORIZATION),(dry['content_hash'],'wrong')]:
        with pytest.raises(PermissionError): s.submit(gate,execute=True,plan_hash=h,authorization=a)
    s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)
    assert len(calls)==1 and s.accepted_jobs(gate)=={'preflight':'12345'}
    s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)
    assert len(calls)==1
    spec=c.artifact('SPEC',root=str(tmp_path/'science'),project_dir='/project',train_minutes=90,export_minutes=15)
    science=s.render(spec,gate)
    assert science['jobs']==12
    jobs={r['task_id']:r for r in science['commands']}
    for arm in ('LOGIT','ATTN'):
        assert jobs['train_'+arm+'_D066']['dependencies']==['export_ANCHOR']
        assert jobs['train_'+arm+'_D033']['dependencies']==['export_'+arm+'_D066']
        assert jobs['train_'+arm+'_D000']['dependencies']==['export_'+arm+'_D033']
    with pytest.raises(ValueError): s.render(dict(spec,train_minutes=1441),gate)


def test_resource_rejection_not_scientific_quality(tmp_path):
    from hlt_classification.luka_attention import preflight,campaign
    measured=[dict(load_seconds=20,epoch_seconds=100,readback_seconds=10,export_seconds=25)]
    assert preflight.budgets(measured)==(89,15)
    for bad in (float('nan'),-1):
        with pytest.raises((ValueError,OverflowError)): preflight.budgets([dict(measured[0],epoch_seconds=bad)])
    with pytest.raises(ValueError): preflight.budgets([dict(measured[0],epoch_seconds=10000)])
    gate=dict(gate_fixture(tmp_path),available_quota_gib=1)
    with pytest.raises(OSError): campaign.storage(gate,np.array([10,20]))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='local CUDA optional, remote A100 acceptance still required')
def test_actual_installed_standard_part_cuda_parity(monkeypatch):
    # Standard architecture, actual installed Weaver. This is NOT full-support A100 evidence.
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from hlt_classification.luka_attention.preflight import parity
    raw=cache().batch(np.arange(4))
    for bf16 in (False,True):
        assert parity(raw,'cuda',bf16=bf16)['optimizer_updates']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='local GPU integration')
def test_all_twelve_science_workers_and_aggregate(tmp_path,monkeypatch):
    tiny_config(monkeypatch)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from hlt_classification.luka_attention import worker,campaign
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    root=tmp_path/'science'; root.mkdir()
    parent=tmp_path/'parent'; parent.mkdir()
    anchor=m.new_model(c.nodes(parent_nodes())[0]).to('cuda')
    save_state(parent/'selected.pt',anchor.state_dict())
    metrics=evaluate_probabilities(cache('validation').labels,predict(anchor,cache('validation'),device='cuda',batch_size=128))
    write_immutable_json(parent/'report.json',dict(validation=metrics))
    write_immutable_json(parent/'diagnostics.json',dict(original_validation=metrics))
    write_immutable_json(parent/'receipt.json',dict(complete=True))
    refs=dict(checkpoint=file_ref(parent/'selected.pt'),report=file_ref(parent/'report.json'),
              diagnostics=file_ref(parent/'diagnostics.json'),receipt=file_ref(parent/'receipt.json'))
    gate=gate_fixture(tmp_path)
    gate.update(counts=dict(train=22,validation=22),anchor=dict(evidence=dict(prepared='a'*64,
        prepared_root=str(parent),models={n:refs for n in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100')})))
    spec=c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='a'*40)
    monkeypatch.setattr(campaign,'check_spec',lambda s:(gate,{}))
    monkeypatch.setattr(worker,'execution',lambda *a,**kw:'123')
    monkeypatch.setattr(worker,'memory_snapshot',lambda t:dict(peak_rss_bytes=1,gpu_peak_bytes=1,rss_bytes=1))
    monkeypatch.setattr(worker,'data',lambda g,r,co:cache(r,co))
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda p:(None,None,{},{}))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda inv,s,ids,labels:np.arange(len(labels)))
    for task in gate['tasks']:
        result=worker.run(spec,task['task_id'])
        assert result['task_id']==task['task_id']
        assert worker.completed(spec,gate,task['task_id'])==result
    report=load_json(root/'aggregate/results.json')
    assert len(report['rows'])==10 and report['primary_contrast']=='ATTN_D000 minus LOGIT_D000'
    assert set(report['contrasts'])=={'validation','original_validation'}
    assert report['independent_test_available'] is False
    for n in gate['nodes']:
        r=load_json(root/('train_'+n['node_id'])/'training_report.json')
        assert r['passes']==30 and r['scientific_fit'] is True
    assert worker.run(spec,'aggregate')==worker.completed(spec,gate,'aggregate')
    path=root/'train_ATTN_D033/selected.pt'
    with path.open('ab') as f: f.write(b'broken')
    with pytest.raises(ValueError): worker.completed(spec,gate,'train_ATTN_D000')


def test_interrupted_submission_never_automatically_retries(tmp_path,monkeypatch):
    from hlt_classification.luka_attention import campaign,submission as s
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=gate_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(s,'check_site',lambda p:None)
    dry=s.submit(gate)
    calls=[]
    def fail(*args,**kwargs):
        calls.append(args)
        raise RuntimeError('Ambiguous scheduler reply')
    monkeypatch.setattr(exact.subprocess,'run',fail)
    for error in (RuntimeError,FileExistsError):
        with pytest.raises(error):
            s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)
    assert len(calls)==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='local GPU integration')
def test_production_preflight_graph_with_mock_site(tmp_path,monkeypatch):
    tiny_config(monkeypatch)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from hlt_classification.luka_attention import preflight as p,worker,campaign
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    anchor=tmp_path/'anchor'; anchor.mkdir()
    save_state(anchor/'selected.pt',m.new_model(c.nodes(parent_nodes())[0]).state_dict())
    write_immutable_json(anchor/'report.json',dict(scientific=True))
    gate=gate_fixture(tmp_path)
    gate.update(counts=dict(train=22,validation=22),available_quota_gib=20,
        anchor=dict(evidence=dict(environment={},prepared='a'*64,prepared_root=str(anchor),
            models=dict(COARSE_U100=dict(checkpoint=file_ref(anchor/'selected.pt'),report=file_ref(anchor/'report.json'))))))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(worker,'execution',lambda *a,**kw:'123')
    monkeypatch.setattr(worker,'data',lambda g,r,co:cache(r,co))
    monkeypatch.setattr(p,'installed_environment',lambda:{})
    monkeypatch.setattr(p,'memory_snapshot',lambda t:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=t.cuda.max_memory_allocated()))
    meta=dict(summaries={r:{co:dict(maximum=4) for co in ('D066','D033','D000')} for r in ('train','validation')})
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda root:(None,meta,{},{}))
    result=p.run(gate)
    p.validate_report(gate,result)
    assert result['technical_checks_passed'] and len(result['measurements'])==6
    assert result['scientific_fit'] is False and result['train_minutes']<=1440
    for field,value in [('technical_checks_passed',False),('scientific_fit',True),('stress',result['stress'][:-1]),
                        ('storage',dict(result['storage'],required_bytes=1)),
                        ('validation_alignment',dict(result['validation_alignment'],prepared='b'*64))]:
        with pytest.raises(ValueError): p.validate_report(gate,with_content_hash(dict(result,**{field:value})))
    with pytest.raises(FileExistsError): p.run(gate)


def test_source_compatibility_preserves_science(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_attention import source
    old,new=tmp_path/'old',tmp_path/'new'
    frozen='src/hlt_classification/frozen.py'
    before={frozen}|(source.CONCURRENT-{'src/hlt_classification/luka_fullstat/recovery.py'})
    after=before|source.CONCURRENT|source.ADDITIONS
    for root,names in ((old,before),(new,after)):
        for name in names:
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'concurrent\n' if name in source.CONCURRENT else b'frozen\n')
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(source,'_files',lambda root:before if root==old else after)
    monkeypatch.setattr(source.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=b'concurrent\n'))
    parent=dict(project_dir=str(old),source_snapshot={})
    assert frozen in source.compatible(parent,new)
    (new/frozen).write_bytes(b'modified scientific code\n')
    with pytest.raises(ValueError,match='Frozen scientific source'): source.compatible(parent,new)
    (new/frozen).write_bytes(b'frozen\n')
    concurrent=sorted(source.CONCURRENT)[0]
    (new/concurrent).write_bytes(b'unregistered concurrent changes\n')
    with pytest.raises(ValueError,match='Concurrent FullStat'): source.compatible(parent,new)
    (new/concurrent).write_bytes(b'concurrent\n')
    after.add('src/unregistered.py')
    with pytest.raises(ValueError,match='Unregistered source'): source.compatible(parent,new)


def test_gate_paths_quota_and_source_failure_preserve_parent(tmp_path,monkeypatch):
    from hlt_classification.luka_attention import campaign,source
    prepared,project,parent_root=tmp_path/'prepared',tmp_path/'project',tmp_path/'parent/science'
    evidence=dict(prepared_root=str(prepared),raw_container=str(tmp_path/'raw'))
    anchor=c.artifact('ANCHOR',evidence=evidence)
    parent=dict(root=str(parent_root),project_dir=str(tmp_path/'oldproject'),nodes=parent_nodes())
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='b'*64))
    monkeypatch.setattr(source,'authenticate',lambda *a:(anchor,parent))
    for root in (project/'run',prepared/'run',parent_root/'run',tmp_path):
        with pytest.raises(PermissionError): campaign.create(root,'parent',project,'a'*40,available_quota_gib=20)
    with pytest.raises(ValueError): campaign.create(tmp_path/'run','parent',project,'a'*40,available_quota_gib=0)
    root=tmp_path/'run'
    gate=campaign.create(root,'parent',project,'a'*40,available_quota_gib=20)
    assert load_json(root/'gate_spec.json')==gate
    with pytest.raises(FileExistsError): campaign.create(root,'parent',project,'a'*40,available_quota_gib=20)
    assert not prepared.exists() and not parent_root.exists()


def test_gate_requires_successful_exact_accounting(tmp_path,monkeypatch):
    from hlt_classification.luka_attention import campaign,preflight,submission
    gate=gate_fixture(tmp_path)
    (tmp_path/'preflight').mkdir()
    report=dict(job_id='112233')
    write_immutable_json(tmp_path/'preflight/receipt.json',report)
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(preflight,'validate_report',lambda g,r:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='wrong_job'))
    with pytest.raises(ValueError,match='exact journal'): campaign.evidence(gate)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='112233'))
    def failed(job): raise ValueError('Preflight accounting FAILED')
    monkeypatch.setattr(campaign,'preflight_accounting',failed)
    with pytest.raises(ValueError,match='accounting FAILED'): campaign.science(gate)
    assert not (tmp_path/'science').exists()


def test_control_matches_original_fixed30_training_kernel(tmp_path,monkeypatch):
    tiny_config(monkeypatch)
    from hlt_classification.jetclass2_delphes.runner import train_kernel
    node=c.nodes(parent_nodes())[0]
    parents=dict(checkpoint='c'*64,report='d'*64)
    banks.export(tmp_path/'bank',m.new_model(node),cache(coordinate='U100'),parents=parents,
                 teacher='ANCHOR',attention=True,device='cpu')
    tr,val=cache(rows=22),cache('validation',rows=22)
    bank=banks.Bank(tmp_path/'bank',tr,parents=parents,teacher='ANCHOR',require_attention=True)
    torch.manual_seed(node['initialization_seed'])
    native=DelphesParticleTransformer()
    expected,state=train_kernel(native,tr,val,node=node,device='cpu',teacher_probabilities=bank.probabilities,
        teacher_identities=bank.identities,acceptance_passes=1,training_recipe=c.recipe()['base'])
    actual,produced=training.train(m.new_model(node),tr,val,node=node,device='cpu',bank=bank,technical=True)
    assert actual['validation']==expected['validation']
    for key in state:
        torch.testing.assert_close(state[key],produced[key],rtol=0,atol=0)


def test_bank_rejects_resigned_semantic_changes_and_partial_publication(tmp_path,monkeypatch):
    tiny_config(monkeypatch)
    import json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.data.cache_contracts import load_npz_arrays,deterministic_npz_bytes
    parent=dict(checkpoint='c'*64,report='d'*64)
    root=tmp_path/'bank'
    original=cache(coordinate='U100')
    bank=banks.export(root,m.new_model(c.nodes(parent_nodes())[0]),original,
        parents=parent,teacher='ANCHOR',attention=True,device='cpu')
    path=root/'manifest.json'
    def check(): return banks.Bank(root,cache(),parents=parent,teacher='ANCHOR',require_attention=True)
    for field,value in [('surface',dict(bank['surface'],block_index=6)),('temperature',1.),
                        ('coordinate','D033'),('teacher','ATTN_D066'),('shards',[])]:
        path.write_text(json.dumps(with_content_hash(dict(bank,**{field:value}))))
        with pytest.raises(ValueError): check()
    path.write_text(json.dumps(bank))
    pop_path=root/bank['population']['path']; original_bytes=pop_path.read_bytes()
    pop=load_npz_arrays(pop_path)
    for name in ('labels','counts','identities'):
        changed={k:v.copy() for k,v in pop.items()}
        changed[name].flat[0]+=1
        pop_path.write_bytes(deterministic_npz_bytes(changed))
        path.write_text(json.dumps(with_content_hash(dict(bank,population=file_ref(pop_path,root=root)))))
        with pytest.raises(ValueError): check()
        pop_path.write_bytes(original_bytes)
    path.unlink()
    with pytest.raises(FileNotFoundError): check()
    with pytest.raises(FileExistsError):
        banks.export(root,m.new_model(c.nodes(parent_nodes())[0]),original,
            parents=parent,teacher='ANCHOR',attention=True,device='cpu')
