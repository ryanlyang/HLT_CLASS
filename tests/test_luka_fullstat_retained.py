"""Retained deployment, matched no-target control and FullStat20 production fits."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat_readout import fixture as readout_fixture, fixture_spec as readout_spec
from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_fullstat_readout import contracts as rc, worker as rw
from hlt_classification.luka_fullstat_retained import contracts as c, campaign, model, training as t, worker as w, source, submission


def reuse(u):
    proof=rc.artifact('STARTUP',teacher_replay_hash=rc.teacher_replay()['content_hash'],
        teacher_readout_parity=True,installed_weaver_parity=True,finite_connected_gradients=True)
    return c.artifact('REUSE',source_commit=source.DONOR,counts=dict(c.COUNTS),u100=u,
        startup=proof,readout_reference=u['reference'],protected_roots=[])


def fixture_spec(root):
    parent=readout_spec(root); r=reuse(parent['reuse'])
    return c.artifact('SPEC',parents=dict(source='c'*64,reuse=r['content_hash']),
        root=str(root),project_dir='/project',source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),
        reuse=r,available_quota_gib=9,**campaign.registration(r))


def fixture(tmp_path,device='cpu'):
    old,tr,val,u,bank,_=readout_fixture(tmp_path,device)
    previous=deepcopy(old['reuse'])
    for role in previous['summaries'].values():
        for summary in role.values(): summary.setdefault('resident_bytes',1024)
    for name in ('anchor','teacher_refs','bank_manifest'): previous[name]=old[name]
    previous['reference']=old['references']['U100_W5']
    previous=with_content_hash(previous); r=reuse(previous)
    root=tmp_path/'luka_fullstat_retained_fixture';root.mkdir()
    s=c.artifact('SPEC',parents=dict(source='c'*64,reuse=r['content_hash']),root=str(root),
        project_dir='/project',source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),
        reuse=r,available_quota_gib=9,**campaign.registration(r))
    s=with_content_hash(dict(s,counts=dict(c.COUNTS,train=len(tr),validation=len(val))))
    return s,tr,val,u,bank


def test_registered_science_two_ordered_matched_models():
    s=fixture_spec('/fixture');r=c.recipe()
    assert s['counts']==dict(train=3000000,validation=500000,final_test=1500000,unused=105325)
    assert (r['ce_weight'],r['embedding_weight'],r['epochs'],r['batch_size'])==(1,5,20,128)
    assert not r['teacher_readout'] and not r['logit_kd'] and not r['relational'] and not r['warm_start']
    assert s['scientific_fits']==2 and s['preflight_jobs']==s['setup_jobs']==0
    assert s['control_order']=='afterany:train_retained'
    assert s['nodes']['train_control']['auxiliary_weight']==0
    assert s['nodes']['train_retained']['auxiliary_weight']==5
    assert all(n['retained_predictor'] and n['coordinate']=='D000' for n in s['nodes'].values())
    assert s['final_test_sealed'] and not s['final_test_accessed']
    assert s['teacher_replay']==rc.teacher_replay()


@pytest.mark.parametrize('bad',['teacher','count','fused','memory','site','parent','startup'])
def test_wrong_population_teacher_acceptance_or_resources_rejected(bad):
    s=fixture_spec('/fixture');r=deepcopy(s['reuse']);u=r['u100']
    if bad=='teacher':u['teacher_node']['node_id']='UE100'
    if bad=='count':u['counts']['train']=299234
    if bad=='fused':u['teacher_node']['context_coordinate']='D000'
    if bad=='memory':u['summaries']['train']['U100']['resident_bytes']=180000*1024**2
    if bad=='site':u['execution_site']['name']='sporc'
    if bad=='parent':r['source_commit']='a'*40
    if bad=='startup':r['startup']=with_content_hash(dict(r['startup'],teacher_readout_parity=False))
    r['u100']=with_content_hash(u)
    with pytest.raises((ValueError,MemoryError)):campaign.registration(with_content_hash(r))


@pytest.mark.parametrize('bad',['epochs','weight','coordinate','order','quota','resource','flag','contract'])
def test_registered_spec_cannot_silently_change_science(tmp_path,bad):
    s=fixture_spec(tmp_path/'luka_fullstat_retained_fixture')
    if bad=='epochs':s['recipe']['epochs']=30
    if bad=='weight':s['nodes']['train_retained']['auxiliary_weight']=10.
    if bad=='coordinate':s['nodes']['train_control']['coordinate']='U100'
    if bad=='order':s['control_order']='afterok:train_retained'
    if bad=='quota':s['available_quota_gib']=8
    if bad=='resource':s['minutes']['train_control']=720
    if bad=='flag':s['final_test_accessed']=True
    if bad=='contract':s['contract']='LUKA_FULLSTAT_READOUT_SPEC/v2'
    with pytest.raises((ValueError,PermissionError)):campaign.validate_spec(with_content_hash(s))


def test_matched_initialization_and_objective_gradient_paths(tmp_path,monkeypatch):
    tiny(monkeypatch);s,tr,_,_,bank=fixture(tmp_path)
    a=model.new_model(s['nodes']['train_retained'],'train_retained')
    b=model.new_model(s['nodes']['train_control'],'train_control')
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert not any(p.requires_grad for p in a.mod.fc.parameters())
    ids=np.arange(len(tr));raw=tr.batch(ids)
    loss,parts=model.batch_loss(a,raw,bank,ids,'train_retained','cpu')
    torch.testing.assert_close(loss,parts['ce']+5*parts['U100'],rtol=0,atol=0)
    grads=torch.autograd.grad(parts['ce'],tuple(a.predictors.parameters()),retain_graph=True)
    assert all(torch.count_nonzero(g)==0 for g in grads)  # Initially W_z is zero.
    combined=torch.autograd.grad(parts['ce'],tuple(a.combined.parameters()),retain_graph=True)
    loss.backward();t.finite_gradients(a)
    assert any(torch.any(p.grad!=0) for p in a.predictors.parameters())
    for p,g in zip(a.combined.parameters(),combined):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    opt=t.optimizer(a,c.recipe()['base']);opt.step();opt.zero_grad(set_to_none=True)
    _,parts=model.batch_loss(a,raw,bank,ids,'train_retained','cpu')
    grads=torch.autograd.grad(parts['ce'],tuple(a.predictors.parameters()))
    assert any(torch.count_nonzero(g)>0 for g in grads)
    ce,parts=model.batch_loss(b,raw,None,ids,'train_control','cpu')
    assert set(parts)=={'ce'} and ce is parts['ce']
    with pytest.raises(PermissionError):model.batch_loss(b,raw,bank,ids,'train_control','cpu')
    # Strict full graph deployment cannot silently fall back to a native checkpoint.
    state=t.cpu_state(a.state_dict());d=model.load_deployed(s['nodes']['train_retained'],'train_retained',state,'cpu')
    assert d.retained and not d.capture
    a.eval();a.capture=False
    with torch.inference_mode():torch.testing.assert_close(_forward(a,raw,'cpu',bf16=False),_forward(d,raw,'cpu',bf16=False),rtol=0,atol=0)
    with pytest.raises(RuntimeError):
        model.load_deployed(s['nodes']['train_retained'],'train_retained',{k:v for k,v in state.items() if k.startswith('mod.')},'cpu')


@pytest.mark.parametrize('task',c.TASKS)
def test_all20_epochs_run_despite_poor_finite_metrics(tmp_path,monkeypatch,task):
    tiny(monkeypatch);s,tr,val,u,bank=fixture(tmp_path)
    original=t.evaluate_probabilities;calls=[];snapshots=[]
    def poor(y,p):
        calls.append(1);return dict(original(y,p),macro_ovr_auc=.4 if len(calls)==1 else .1)
    monkeypatch.setattr(t,'evaluate_probabilities',poor)
    report,state,full=t.train(tr,val,bank=bank if task=='train_retained' else None,node=s['nodes'][task],
        task=task,parents={},device='cpu',save_epoch=lambda e,row,state:snapshots.append((e,row,state)))
    assert len(calls)==len(snapshots)==20 and full['epoch']==report['selected_pass']==1
    assert full['optimizer'] and 'cpu_rng' in full and 'cuda_rng' in full
    assert any(k.startswith('predictors.U100.') for k in state) and 'combined.weight' in state
    t.validate_report(report,s,task,{})
    bad=deepcopy(report);bad['validation_history'][1]['weighted_auxiliary_loss']+=1
    with pytest.raises(ValueError):t.validate_report(with_content_hash(bad),s,task,{})


def worker_fixture(tmp_path,monkeypatch,device):
    s,tr,val,u,bank=fixture(tmp_path,device)
    meta=(dict(content_hash=tr.foundation_sha256,summaries=s['reuse']['u100']['summaries']),None,None)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(w,'execution',lambda spec,task,*a:dict(train_retained='123',train_control='124')[task])
    monkeypatch.setattr(w,'metadata',lambda _:meta)
    loader=lambda s,r,c,**kw:u if c=='U100' else tr if r=='train' else val
    monkeypatch.setattr(w,'data',loader);monkeypatch.setattr(rw,'data',loader)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:dict(train_retained='123',train_control='124'))
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(w,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated())) if device=='cuda' else 1))
    return s,tr,val,u,bank


def check_worker_results(s,task):
    folder=Path(s['root'])/task
    assert len(list((folder/'epochs').glob('*/state.pt')))==20
    r=load_json(folder/'training_report.json');assert r['passes']==20 and r['task_id']==task
    state=torch.load(folder/'selected.pt',weights_only=True)
    assert 'combined.weight' in state and any(k.startswith('predictors.U100.') for k in state)
    cost=load_json(folder/'inference_cost.json');measurements=cost['measurements']
    assert measurements['retained']['parameters']>measurements['native_reference']['parameters']
    assert measurements['retained']['state_bytes']>measurements['native_reference']['state_bytes']
    assert all((v['cuda_peak_increment_bytes'] is None)==(cost['device']=='cpu') for v in measurements.values())
    assert cost['memory_before_measurement']['gpu_peak_bytes']>0
    result=w.results(s)
    assert len(result['rows'])==(5 if task=='train_retained' else 6)
    assert len(result['pending'])==(1 if task=='train_retained' else 0)
    assert result['historical_U100_reference'] and not result['final_test_accessed']
    return result


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_production_workers_both_arms(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable; not Oscar certification')
    if device=='cpu':
        tiny(monkeypatch);monkeypatch.setattr(w,'installed_parity',lambda *a,**kw:None)
    s,*_=worker_fixture(tmp_path,monkeypatch,device)
    w.run(s,'train_retained',device=device);check_worker_results(s,'train_retained')
    # CE-only worker must neither load bank/teacher nor read retained outputs.
    monkeypatch.setattr(w.banks,'Bank',lambda *a,**kw:pytest.fail('CE control loaded bank'))
    monkeypatch.setattr(rw,'teacher_parity',lambda *a,**kw:pytest.fail('CE control forwarded teacher'))
    original=w.data
    def hlt_only(s,r,coordinate,**kw):
        assert coordinate=='D000';return original(s,r,coordinate,**kw)
    monkeypatch.setattr(w,'data',hlt_only)
    w.run(s,'train_control',device=device);check_worker_results(s,'train_control')
    folder=Path(s['root'])/'train_retained';receipt=load_json(folder/'task_receipt.json')
    (folder/'task_receipt.json').write_text(json.dumps(with_content_hash(dict(receipt,job_id='999'))))
    with pytest.raises(ValueError):w.completed(s,'train_retained')


def test_control_runs_even_when_retained_has_no_receipt(tmp_path,monkeypatch):
    tiny(monkeypatch);monkeypatch.setattr(w,'installed_parity',lambda *a,**kw:None)
    s,*_=worker_fixture(tmp_path,monkeypatch,'cpu')
    monkeypatch.setattr(w.banks,'Bank',lambda *a,**kw:pytest.fail('No bank access'))
    assert w.run(s,'train_control',device='cpu')['task_id']=='train_control'
    result=w.results(s)
    assert result['pending']==[dict(task='train_retained',status='no_completed_receipt')]
    assert len(result['rows'])==5


@pytest.mark.parametrize('role,coord',[('final_test','D000'),('unused','D000'),('validation','U100'),('train','OFFLINE')])
def test_forbidden_roles(role,coord,monkeypatch):
    from hlt_classification.luka_fullstat_u100_embedding import worker as old
    monkeypatch.setattr(old,'metadata',lambda *_:pytest.fail('Forbidden data accessed'))
    with pytest.raises(PermissionError):w.data({},role,coord)


def test_timing_memory_and_quota_fail_before_science(tmp_path,monkeypatch):
    s=fixture_spec(tmp_path/'new')
    assert w.timing(s,'train_retained',.05,.02,300)['conservative_minutes']<1440
    with pytest.raises(ValueError):w.timing(s,'train_control',1.,.02,300)
    with pytest.raises(ValueError):w.timing(s,'train_retained',float('nan'),.02,300)
    with pytest.raises(MemoryError):w.check_memory(s,dict(gpu_peak_bytes=48*2**30,peak_rss_bytes=2**30))
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=8*2**30))
    with pytest.raises(OSError):campaign.storage(s)
