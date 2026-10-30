"""FullStat readout: matched kernel, frozen gradients, provenance and production miniature."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat_u100_embedding import fixture as old_fixture
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_u100_embedding import banks, training as old_training
from hlt_classification.luka_fullstat_readout import contracts as c, campaign, model, training as t, worker as w, source, submission


def fixture_spec(root):
    from test_luka_fullstat_meta_embedding import fixture_spec as parent_fixture
    old=parent_fixture(root)
    reuse=c.artifact('REUSE',**{k:v for k,v in old['reuse'].items() if k not in ('content_hash','contract','schema_version','parents',*c.FLAGS)})
    reuse['bank_hash']='b'*64
    reuse['summaries']={r:{co:dict(maximum=4,resident_bytes=1024) for co in cs}
                       for r,cs in [('train',('D000','U100')),('validation',('D000',))]}
    reuse['anchor']['evidence']['models']={n:dict(report={},predictions={}) for n in ('M0HLT','OFFLINE')}
    reuse=with_content_hash(reuse)
    return c.artifact('SPEC',parents=dict(source='c'*64,reuse=reuse['content_hash']),
        root=str(root),project_dir='/project',source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),
        reuse=reuse,available_quota_gib=4,**campaign.registration(reuse))


def fixture(tmp_path,device='cpu'):
    (tmp_path/'historical').mkdir()
    old,tr,val,u=old_fixture(tmp_path/'historical',device=device)
    s=fixture_spec(tmp_path/'luka_fullstat_readout_fixture')
    s['root']=str(tmp_path/'luka_fullstat_readout_fixture'); Path(s['root']).mkdir()
    s['counts']=dict(c.COUNTS,train=len(tr),validation=len(val))
    s['node']=c.node(old['node']); s['anchor']=old['anchor']; s['teacher_refs']=old['teacher_refs']
    s['bank_manifest']=file_ref(Path(old['root'])/'targets/manifest.json')
    s['reuse']['bank_hash']=load_json(s['bank_manifest']['path'])['content_hash']
    s['reuse']['summaries']=old['reuse']['summaries']
    s['reuse']['gpu']=old['reuse']['gpu']
    s['references']={n:old['anchor']['evidence']['models'][n] for n in ('M0HLT','OFFLINE')}
    # Fixture historical U100-w5 results share rows and use a real native checkpoint/report.
    s['references']['U100_W5']=s['references']['M0HLT']
    s=with_content_hash(s)
    bank=banks.Bank(Path(s['bank_manifest']['path']).parent,tr,refs=s['teacher_refs'])
    readout=model.make_readout(bank,Path(s['bank_manifest']['path']).parent,s['teacher_refs'],device)
    return s,tr,val,u,bank,readout


def test_registered_science_one_fresh_model():
    s=fixture_spec('/fixture'); r=c.recipe()
    assert s['counts']==dict(train=3000000,validation=500000,final_test=1500000,unused=105325)
    assert (r['ce_weight'],r['embedding_weight'],r['readout_ce_weight'],r['ramp_passes'])==(1,5,1,3)
    assert r['epochs']==20 and r['batch_size']==128 and not r['warm_start'] and not r['logit_kd']
    assert s['scientific_fits']==1 and s['preflight_jobs']==s['setup_jobs']==0 and s['historical_control']
    assert s['node']['teacher'] is None and s['node']['coordinate']=='D000'
    assert s['final_test_sealed'] and not s['final_test_accessed']


@pytest.mark.parametrize('bad',['teacher','count','fused','memory','site','parent'])
def test_reuse_rejects_wrong_population_teacher_or_resources(bad):
    s=fixture_spec('/fixture'); r=deepcopy(s['reuse'])
    if bad=='teacher': r['teacher_node']['node_id']='UE100'
    if bad=='count': r['counts']['train']=299234
    if bad=='fused': r['teacher_node']['context_coordinate']='D000'
    if bad=='memory': r['summaries']['train']['U100']['resident_bytes']=180000*1024**2
    if bad=='site': r['execution_site']['name']='sporc'
    if bad=='parent': r['source_commit']='a'*40
    with pytest.raises((ValueError,MemoryError)): campaign.registration(with_content_hash(r))


def test_formula_ramp_gradient_and_zero_coefficient_parity():
    torch.manual_seed(7)
    readout=model.FrozenReadout(torch.randn(128),torch.rand(128)+.1,torch.randn(11,128),torch.randn(11))
    logits=torch.randn(11,11,requires_grad=True); z=torch.randn(11,128,requires_grad=True)
    target=torch.randn(11,128,requires_grad=True); y=torch.arange(11)
    for position,weight in [(0,0),(1,1/3),(3,1),(20,1)]:
        loss,ce,aux,rc,a=model.objective(logits,y,z,target,readout,pass_coordinate=position)
        assert a==weight
        torch.testing.assert_close(loss,ce+5*aux+weight*rc,rtol=0,atol=0)
        grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
        torch.testing.assert_close(grad,torch.autograd.grad(ce,logits,retain_graph=True)[0],rtol=0,atol=0)
    assert not list(readout.parameters())
    assert torch.any(torch.autograd.grad(rc,z,retain_graph=True)[0]!=0)
    assert torch.autograd.grad(loss,target,allow_unused=True)[0] is None
    for bad in [-1,21,float('nan')]:
        with pytest.raises(ValueError): model.objective(logits,y,z,target,readout,pass_coordinate=bad)


def test_bank_classifier_join_parity_and_corruption(tmp_path,monkeypatch):
    tiny(monkeypatch); s,tr,val,u,bank,readout=fixture(tmp_path)
    monkeypatch.setattr(w,'data',lambda *a,**kw:u)
    evidence=w.teacher_parity(s,bank,readout,None,'cpu')
    assert evidence['teacher_readout_parity'] and evidence['teacher_train_maximum']==4
    folder=Path(s['root'])/w.TASK; folder.mkdir()
    w.save_readout(s,bank,'cpu'); w.validate_readout(s)
    state=torch.load(folder/'readout.pt',weights_only=True); state['bias'].add_(1)
    torch.save(state,folder/'readout.pt')
    rec=load_json(folder/'readout.json'); rec['payload']=file_ref(folder/'readout.pt')
    (folder/'readout.json').write_text(json.dumps(with_content_hash(rec)))
    with pytest.raises(ValueError,match='classifier/normalization'): w.validate_readout(s)
    wrong=deepcopy(s['teacher_refs']); wrong['checkpoint']['sha256']='0'*64
    with pytest.raises(ValueError): model.make_readout(bank,Path(s['bank_manifest']['path']).parent,wrong,'cpu')


def test_full20_zero_readout_matches_historical_training(tmp_path,monkeypatch):
    tiny(monkeypatch); s,tr,val,u,bank,readout=fixture(tmp_path)
    original=t.batch_loss
    monkeypatch.setattr(t,'batch_loss',lambda m,raw,b,ix,r,d,p:original(m,raw,b,ix,r,d,0.))
    with parity_backend('cpu'):
        old,state=old_training.train(tr,val,bank=bank,node=s['node'],device='cpu')
        new,actual,full=t.train(tr,val,bank=bank,node=s['node'],readout=readout,parents={},device='cpu')
    assert old['validation']==new['validation'] and old['selected_pass']==new['selected_pass']
    assert all(torch.equal(v,actual[k]) for k,v in state.items())
    assert [r['validation'] for r in old['validation_history']]==[r['validation'] for r in new['validation_history']]
    assert any(k.startswith('predictor.') for k in full['student'])
    assert not any(k.startswith('predictor.') for k in actual)


def test_all_epochs_run_despite_poor_finite_metrics(tmp_path,monkeypatch):
    tiny(monkeypatch); s,tr,val,u,bank,readout=fixture(tmp_path)
    original=t.evaluate_probabilities; calls=[]; snapshots=[]
    def poor(y,p):
        calls.append(1); return dict(original(y,p),macro_ovr_auc=.4 if len(calls)==1 else .1)
    monkeypatch.setattr(t,'evaluate_probabilities',poor)
    report,native,full=t.train(tr,val,bank=bank,node=s['node'],readout=readout,parents={},device='cpu',
        save_epoch=lambda e,row,state:snapshots.append((e,row,state)))
    assert len(calls)==len(snapshots)==20 and full['epoch']==report['selected_pass']==1
    assert full['optimizer'] and 'cpu_rng' in full and 'cuda_rng' in full
    t.validate_report(report,s,{})
    bad=deepcopy(report); bad['validation_history'][2]['readout_weight_end']=.5
    with pytest.raises(ValueError): t.validate_report(with_content_hash(bad),s,{})


@pytest.mark.parametrize('role,coord',[('final_test','D000'),('unused','D000'),('validation','U100'),('train','OFFLINE')])
def test_forbidden_roles(role,coord,monkeypatch):
    from hlt_classification.luka_fullstat_u100_embedding import worker as old
    monkeypatch.setattr(old,'metadata',lambda *_:pytest.fail('Forbidden data accessed'))
    with pytest.raises(PermissionError): w.data({},role,coord)


def test_timing_memory_and_quota_fail_before_science(tmp_path,monkeypatch):
    s=fixture_spec(tmp_path/'new')
    assert w.timing(s,.05,.02,300)['conservative_minutes']<1440
    with pytest.raises(ValueError): w.timing(s,1.,.02,300)
    with pytest.raises(ValueError): w.timing(s,float('nan'),.02,300)
    with pytest.raises(MemoryError): w.check_memory(s,dict(gpu_peak_bytes=48*2**30,peak_rss_bytes=2**30))
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=3*2**30))
    with pytest.raises(OSError): campaign.storage(s)


def worker_fixture(tmp_path,monkeypatch,device):
    s,tr,val,u,bank,readout=fixture(tmp_path,device)
    meta=(dict(content_hash=tr.foundation_sha256,summaries=s['reuse']['summaries']),None,None)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(w,'execution',lambda *a:'123')
    monkeypatch.setattr(w,'metadata',lambda _:meta)
    monkeypatch.setattr(w,'data',lambda s,r,c,**kw:u if c=='U100' else tr if r=='train' else val)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(train_readout='123'))
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(w,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated())) if device=='cuda' else 1))
    return s,tr,val,u,bank,readout


def check_worker_result(s):
    result=w.results(s); assert len(result['rows'])==4 and result['historical_control']
    folder=Path(s['root'])/w.TASK
    assert len(list((folder/'epochs').glob('*/state.pt')))==20
    assert load_json(folder/'training_report.json')['passes']==20
    assert not result['final_test_accessed'] and result['final_test_sealed']
    assert not any(k.startswith('predictor.') for k in torch.load(folder/'selected.pt',weights_only=True))
    before=load_json(folder/'task_receipt.json'); bad=with_content_hash(dict(before,job_id='999'))
    (folder/'task_receipt.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError): w.completed(s)


def test_complete_production_worker_cpu(tmp_path,monkeypatch):
    tiny(monkeypatch); s,*_=worker_fixture(tmp_path,monkeypatch,'cpu')
    # Reduced CPU fixture; authoritative full-size installed parity runs in CUDA test.
    monkeypatch.setattr(w,'installed_parity',lambda *a,**kw:None)
    assert w.run(s,w.TASK,device='cpu')['task_id']==w.TASK
    check_worker_result(s)


def test_real_installed_cuda_complete_worker(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable; not Oscar certification')
    s,*_=worker_fixture(tmp_path,monkeypatch,'cuda')
    assert w.run(s,w.TASK,device='cuda')['task_id']==w.TASK
    check_worker_result(s)
