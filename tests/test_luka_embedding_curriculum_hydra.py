"""Curriculum clocks, rollback, role separation, native parity and durable queueing."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import subprocess
import numpy as np
import pytest
import torch
from test_luka_attention import cache,parent_nodes
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT,head_input
from hlt_classification.luka_u100_embedding.training import objective as baseline_objective
from hlt_classification.luka_embedding_curriculum_hydra import contracts as c,training as t,campaign,inputs,resources,submission,worker,reporting,site
from hlt_classification.scouting import hcwdl_exact_dag_submission as dag

PRODUCTION_CONFIG=deepcopy(native.model_config())


@pytest.fixture(autouse=True)
def compact_native(monkeypatch):
    before=torch.get_num_threads(); torch.set_num_threads(1)
    config=native.model_config(); config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=8,num_layers=2,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    yield
    torch.set_num_threads(before)


def fixtures(root,rows=22):
    tr=cache(coordinate='D000',rows=rows); va=cache(role='validation',coordinate='D000',rows=rows)
    rng=np.random.default_rng(17)
    def bank(source,role,h): return SimpleNamespace(identities=source.identities,labels=source.labels,
        targets=rng.normal(0,.2,(len(source),128)).astype(np.float32),role=role,manifest=dict(content_hash=h))
    tb=bank(tr,'train','b'*64); vb=bank(va,'validation','c'*64)
    spec=dict(campaign.registration(),content_hash='a'*64,root=str(root),project_dir='/reviewed/source',
        imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)),embedding_hash='b'*64,
        anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    return tr,va,tb,vb,spec


def test_registry_twenty_unique_and_exact_priorities():
    assert len(c.TASKS)==20 and len(c.ORDER)==19 and 'E00' not in c.ORDER
    assert c.ORDER[:8]==('E01','E06','E12','E19','E03','E15','E17','E18')
    assert len({json.dumps(c.setting(x),sort_keys=True) for x in c.TASKS})==20
    assert c.recipe()['epochs']==30 and c.recipe()['batch_size']==128
    assert c.recipe()['baseline']['auxiliary_coefficient']==5
    assert campaign.registration()['fit_max_minutes']==2880


@pytest.mark.parametrize('task',c.TASKS)
def test_continuous_clocks_endpoints_and_finite(task):
    cfg=c.setting(task)
    for position in np.linspace(.001,30,101):
        ce,emb=c.weights(task,float(position)); lr=c.learning_rate(task,float(position))
        assert 0<=ce<=2 and 0<=emb<=5 and 0<lr<=3e-4
        if position<=cfg['pretrain']: assert ce==cfg['pre_ce'] and emb==5
    assert c.learning_rate(task,30)==pytest.approx(1.5e-5)
    assert c.weights(task,30)==(cfg['ce_max'],cfg['embedding_end'])
    for v in (-1,float('nan'),float('inf'),31):
        with pytest.raises(ValueError): c.weights(task,v)
        with pytest.raises(ValueError): c.learning_rate(task,v)


def test_boundary_losses_and_orthogonal_optimizer_lr_choices():
    assert c.weights('E01',10)==(0,5) and c.weights('E01',12.5)==(.5,5) and c.weights('E01',15)==(1,5)
    assert c.weights('E06',10.00001)==(1,5)
    assert c.weights('E10',12)==(.1,5) and c.weights('E10',12.001)==(.3,5) and c.weights('E10',15.001)==(1,5)
    assert c.weights('E15',22.5)==(1,3) and c.weights('E16',25)==(1,0)
    assert c.weights('E17',10)==(.05,5) and c.weights('E17',12.5)==pytest.approx((.525,5))
    assert {(c.setting(x)['reset_optimizer'],c.setting(x)['restart_lr']) for x in ('E01','E12','E13','E14')}=={(True,True),(True,False),(False,True),(False,False)}


def equal_tree(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    else: assert a==b


def test_baseline_step_parity_and_hlt_boundary(tmp_path):
    tr,_,tb,_,s=fixtures(tmp_path); model=t.new_model(s,'cpu'); other=t.new_model(s,'cpu')
    opt=t.optimizer(model); oo=t.optimizer(other); raw=tr.batch(np.arange(len(tr)))
    rng=torch.get_rng_state().clone()
    terms=t.step(model,opt,raw,tb.targets,'E00',1.,'cpu')
    torch.set_rng_state(rng); other.train(); other.capture=True; oo.zero_grad(set_to_none=True)
    for g in oo.param_groups: g['lr']=c.learning_rate('E00',1.)
    z=t.forward(other,raw,'cpu'); total,ce,emb=baseline_objective(z,torch.from_numpy(raw['labels']),other.prediction,torch.from_numpy(tb.targets),coefficient=5.)
    total.backward(); oo.step()
    equal_tree(model.state_dict(),other.state_dict()); equal_tree(opt.state_dict(),oo.state_dict())
    np.testing.assert_allclose(terms,[ce.detach(),emb.detach(),ce.detach(),5*emb.detach(),total.detach()])
    with pytest.raises(PermissionError): t.forward(model,dict(raw,u_features=raw['features']),'cpu')


def test_zero_terms_leave_classifier_and_predictor_untouched(tmp_path):
    tr,_,tb,_,s=fixtures(tmp_path); m=t.new_model(s,'cpu'); opt=t.optimizer(m); raw=tr.batch(np.arange(len(tr)))
    head=deepcopy(m.mod.fc.state_dict()); original=deepcopy(m.predictor.state_dict())
    t.step(m,opt,raw,tb.targets,'E01',1.,'cpu')
    equal_tree(head,m.mod.fc.state_dict()); assert all(p.grad is None and p not in opt.state for p in m.mod.fc.parameters())
    assert any(not torch.equal(v,original[k]) for k,v in m.predictor.state_dict().items())
    held=deepcopy(m.predictor.state_dict()); t.step(m,opt,raw,tb.targets,'E16',27.,'cpu')
    equal_tree(held,m.predictor.state_dict()); assert all(p.grad is None for p in m.predictor.parameters())


@pytest.mark.parametrize('task',['E01','E12','E13','E14'])
def test_transition_restores_full_selected_state_and_rng(tmp_path,task):
    tr,_,tb,_,s=fixtures(tmp_path); m=t.new_model(s,'cpu'); opt=t.optimizer(m); raw=tr.batch(np.arange(len(tr)))
    t.step(m,opt,raw,tb.targets,task,1.,'cpu'); best=t.snapshot(m,opt,1,.2,'cpu')
    t.step(m,opt,raw,tb.targets,task,2.,'cpu'); torch.rand(50)
    opt=t.transition(m,opt,best,task,'cpu')
    equal_tree(m.state_dict(),best['model']); equal_tree(torch.get_rng_state(),best['cpu_rng'])
    if c.setting(task)['reset_optimizer']: assert len(opt.state)==0
    else: equal_tree(opt.state_dict(),best['optimizer'])
    assert c.learning_rate(task,11)==(c.learning_rate('E01',11) if c.setting(task)['restart_lr'] else c.learning_rate('E00',11))


def test_validation_evaluation_no_updates_or_rng_and_roles(tmp_path):
    tr,va,tb,vb,s=fixtures(tmp_path); m=t.new_model(s,'cpu'); before=deepcopy(m.state_dict()); rng=torch.get_rng_state().clone()
    metrics,loss=t.evaluate(m,va,vb,'cpu')
    assert metrics['rows']==len(va) and loss>=0
    equal_tree(before,m.state_dict()); equal_tree(rng,torch.get_rng_state()); assert all(p.grad is None for p in m.parameters())
    with pytest.raises(PermissionError): t.evaluate(m,tr,tb,'cpu')
    with pytest.raises(PermissionError): t.join(tr,vb,'train')
    vb.targets[0,0]=float('nan')
    with pytest.raises(PermissionError): t.evaluate(m,va,vb,'cpu')


@pytest.mark.parametrize('task',c.TASKS)
def test_complete_thirty_epoch_scientific_recipe_and_selection(tmp_path,task):
    tr,va,tb,vb,s=fixtures(tmp_path)
    r=t.fit(tmp_path/task,s,task,tr,va,tb,vb,device='cpu')
    assert len(r['validation_history'])==30 and r['epochs']==30
    history=r['validation_history']; cfg=c.setting(task)
    eligible=[h for h in history if h['phase']=='classification']
    assert r['selected_pass'] in [h['pass_number'] for h in eligible]
    if cfg['switch']:
        transition=load_json(tmp_path/task/'transition.json'); pre=history[:cfg['switch']]
        expected=pre[-1] if cfg['selection']=='last' else min(pre,key=lambda x:(x['embedding_validation_loss'],x['pass_number']))
        assert transition['selected_epoch']==expected['pass_number']
        state=torch.load(tmp_path/task/'transition.pt',weights_only=True)
        assert state['epoch']==expected['pass_number'] and 'predictor.0.weight' in state['model']
    native_state=torch.load(tmp_path/task/'selected.pt',weights_only=True)
    assert not any(k.startswith('predictor.') for k in native_state)
    r['selected_pass']=0
    # Altering a report without the full hash chain cannot be accepted.
    (tmp_path/task/'training_report.json').write_text(json.dumps(r))
    with pytest.raises(ValueError): t.validate_fit(tmp_path/task,s,task,val_bank_hash=vb.manifest['content_hash'])


def test_teacher_validation_extract_frozen_train_normalization(tmp_path):
    _,va,_,_,s=fixtures(tmp_path); u=cache(role='validation',coordinate='U100',rows=len(va))
    model=native.DelphesParticleTransformer().eval().requires_grad_(False)
    mean=np.arange(128,dtype=np.float64)/128; scale=np.ones(128,np.float64)*2
    actual=inputs.extract(model,u,va,mean,scale,'cpu')
    raw=u.batch(np.arange(len(u)))
    with torch.no_grad(): _,z=head_input(model,**{k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')})
    np.testing.assert_array_equal(actual,((z.numpy().astype(np.float64)-mean)/scale).astype(np.float32))
    assert not np.allclose(actual.mean(0),0.) and all(p.grad is None for p in model.parameters())
    model.train()
    with pytest.raises(ValueError): inputs.extract(model,u,va,mean,scale,'cpu')
    u.role='train'
    with pytest.raises(PermissionError): inputs.extract(model,u,va,mean,scale,'cpu')


def test_submission_order_and_no_fit_dependencies(tmp_path):
    *_,s=fixtures(tmp_path); p=submission.plan(s,'bootstrap')
    assert [r['task_id'] for r in p['commands']]==['setup','release']
    assert '--dependency=afterok:${JOB_setup}' in p['commands'][1]['command']
    assert '--mem=4000M' in p['commands'][1]['command']
    a=dict(content_hash='f'*64,minutes={t:120 for t in c.ORDER})
    p=submission.plan(s,'science',a)
    assert [r['task_id'] for r in p['commands']]==list(c.ORDER)
    for r in p['commands']:
        assert r['dependencies']==[] and not any(x.startswith(('--dependency=','--hold','--array')) for x in r['command'])
        assert '--gres=gpu:nvidia_rtx_a6000:1' in r['command'] and '--mem=32000M' in r['command']
    a['minutes']['E01']=2881
    with pytest.raises(ValueError): submission.plan(s,'science',a)


def test_dry_run_claim_and_idempotent_submission(tmp_path,monkeypatch):
    *_,s=fixtures(tmp_path); monkeypatch.setattr(submission,'scheduler_review',lambda p:None)
    p=submission.plan(s,'bootstrap'); submission.submit_stage(s,p,execute=False)
    assert not (tmp_path/'submission_bootstrap/live_claim').exists()
    (tmp_path/'submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous'): submission.submit_stage(s,p,execute=True)


def test_failed_setup_never_releases_science(tmp_path,monkeypatch):
    *_,s=fixtures(tmp_path); monkeypatch.setattr(campaign,'check',lambda s:None)
    p=submission.plan(s,'bootstrap'); write_immutable_json(tmp_path/'submission_bootstrap/live_claim/intent.json',
        c.artifact('SUBMIT_CLAIM',parents=dict(plan=p['content_hash']),authorization=c.AUTHORIZATION))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'setup':'100','release':'101'})
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    monkeypatch.setattr(submission,'submit_stage',lambda *a,**k:pytest.fail('science submitted after failure'))
    with pytest.raises(RuntimeError,match='setup failed'): submission.finish(s,'101')
    with pytest.raises(PermissionError,match='Authorize'): submission.launch(s,execute=True,plan_hash='bad',authorization='bad')


def test_release_full_measured_plan_and_missing_acceptance(tmp_path,monkeypatch):
    *_,s=fixtures(tmp_path); monkeypatch.setattr(campaign,'check',lambda s:None)
    p=submission.plan(s,'bootstrap'); write_immutable_json(tmp_path/'submission_bootstrap/live_claim/intent.json',
        c.artifact('SUBMIT_CLAIM',parents=dict(plan=p['content_hash']),authorization=c.AUTHORIZATION))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'setup':'100','release':'101'})
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:dict(content_hash='f'*64,minutes={t:123 for t in c.ORDER}))
    calls=[]
    monkeypatch.setattr(submission,'submit_stage',lambda s,p,execute:calls.append((p,execute)) or {'jobs':19})
    assert submission.finish(s,'101')=={'jobs':19} and [v for p,v in calls]==[False,True]
    assert len(calls[-1][0]['commands'])==19
    monkeypatch.setattr(worker,'completed_setup',lambda s:(_ for _ in ()).throw(ValueError('bad acceptance')))
    calls.clear()
    with pytest.raises(ValueError): submission.finish(s,'101')
    assert not calls


def test_warmed_production_miniature_projects_all_phases_without_quality_gate(tmp_path):
    tr,va,tb,vb,s=fixtures(tmp_path,130)
    warm=t.fit(tmp_path/'warm',s,'E16',tr,va,tb,vb,device='cpu',technical=True)
    mini=t.fit(tmp_path/'mini',s,'E16',tr,va,tb,vb,device='cpu',technical=True)
    s['counts']=dict(train=299234,validation=74895)
    gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)
    r=resources.record(s,'E16',mini,warm,dict(gpu_peak_bytes=2**30,peak_rss_bytes=4*2**30),gpu,40.)
    assert r['passed'] and r['quality_gate'] is False and r['minutes']>=30
    assert [h['pass_number'] for h in mini['validation_history']]==[10,11,30]
    assert all(h['batch_sizes']==[128,2] for h in mini['validation_history'])
    assert mini['validation_history'][-1]['weighted_embedding']==0
    changed=deepcopy(mini); changed['validation_history'][0]['batch_seconds']=[20.,.01]
    fail=resources.record(s,'E16',changed,warm,dict(gpu_peak_bytes=2**30,peak_rss_bytes=4*2**30),gpu,40.)
    assert not fail['passed'] and fail['violations'][0]['resource']=='minutes'
    changed=deepcopy(mini); changed['validation_history'][0]['batch_sizes']=[128,1]
    with pytest.raises(ValueError): resources.record(s,'E16',changed,warm,r['memory'],gpu,40.)


def test_validation_bank_authentication_and_read_only_roles(tmp_path,monkeypatch):
    _,val,_,_,s=fixtures(tmp_path); s['teacher_context']=dict(encoder_hash='e'*64,delta_hash='d'*64)
    s['imported']['teacher']=dict(content_hash='f'*64)
    mean=np.arange(128,dtype=np.float64)/128; scale=np.ones(128,np.float64)
    monkeypatch.setattr(inputs,'moments',lambda spec:(mean,scale))
    folder=tmp_path/'setup'; folder.mkdir()
    path=folder/'validation_embeddings.npz'
    a=dict(identities=val.identities,labels=val.labels,targets=np.zeros((len(val),128),np.float32),mean=mean,scale=scale)
    atomic_publish_bytes(path,deterministic_npz_bytes(a))
    m=c.artifact('VALIDATION_BANK',parents=inputs.parents(s),role='validation',usage='evaluation_only',coordinate='U100',
        rows=len(val),width=128,precision='FP32',batch_size=128,statistics='original_TRAIN_frozen',payload=file_ref(path,root=folder))
    write_immutable_json(folder/'validation_embeddings.json',m)
    result=inputs.validation_bank(s,val); assert result.role=='validation'
    val.role='train'
    with pytest.raises(PermissionError): inputs.validation_bank(s,val)
    val.role='validation'; path.write_bytes(b'corrupted')
    with pytest.raises(ValueError): inputs.validation_bank(s,val)


def test_receipt_exact_job_and_changed_output_rejected(tmp_path,monkeypatch):
    *_,s=fixtures(tmp_path); folder=tmp_path/'E01'; folder.mkdir(); (folder/'output.txt').write_text('evidence')
    r=c.artifact('TASK',parents=dict(spec=s['content_hash']),task='E01',job_id='101',outputs=[file_ref(folder/'output.txt',root=folder)])
    write_immutable_json(folder/'receipt.json',r)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'E01':'101'})
    assert worker.receipt(s,'E01')==r
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'E01':'102'})
    with pytest.raises(ValueError): worker.receipt(s,'E01')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'E01':'101'}); (folder/'output.txt').write_text('changed')
    with pytest.raises(ValueError): worker.receipt(s,'E01')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('task',['E01','E12','E16'])
def test_full_native_cuda_transition_and_export(tmp_path,monkeypatch,task):
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(PRODUCTION_CONFIG))
    tr,va,tb,vb,s=fixtures(tmp_path,130)
    report=t.fit(tmp_path/task,s,task,tr,va,tb,vb,device='cuda',technical=True)
    assert report['deployable_hlt_only'] and report['transition']
    assert all(h['batch_sizes']==[128,2] for h in report['validation_history'])
    snap=torch.load(tmp_path/task/'transition.pt',weights_only=True)
    assert len(snap['cuda_rng'])==torch.cuda.device_count()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_installed_native_and_original_training_update_parity(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA required')
    if device=='cuda': monkeypatch.setattr(native,'model_config',lambda:deepcopy(PRODUCTION_CONFIG))
    tr,_,tb,_,spec=fixtures(tmp_path,130)
    result=resources.baseline(spec,tr,tb,device)
    assert result['one_update_model_equal'] and result['one_update_optimizer_equal'] and result['quality_gate'] is False


@pytest.fixture
def exact_scheduler(tmp_path,monkeypatch):
    *_,s=fixtures(tmp_path); commands=[]; reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p['stage']))
    monkeypatch.setattr(submission.time,'sleep',lambda *a:None)
    def sbatch(cmd,**kw):
        commands.append(cmd); return SimpleNamespace(stdout=str(8000+len(commands)),returncode=0,stderr='')
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    submission.launch(s)
    bootstrap=submission.launch(s,execute=True,plan_hash=submission.plan(s,'bootstrap')['content_hash'],authorization=c.AUTHORIZATION)
    acceptance=dict(content_hash='f'*64,minutes={task:120+i for i,task in enumerate(c.ORDER)})
    write_immutable_json(tmp_path/'setup/acceptance.json',acceptance)
    monkeypatch.setattr(worker,'completed_setup',lambda *a:acceptance)
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    return s,commands,reviews,bootstrap


def test_real_journals_exact_dependencies_and_idempotent_bounded_release(exact_scheduler):
    spec,commands,reviews,bootstrap=exact_scheduler
    assert bootstrap['jobs']=={'setup':'8001','release':'8002'} and '--dependency=afterok:8001' in commands[1]
    ledger=submission.finish(spec,'8002')
    assert len(commands)==21 and [cmd[-1] for cmd in commands[2:]]==list(c.ORDER)
    assert reviews==['bootstrap','science']
    assert all(not any(a.startswith('--dependency=') for a in cmd) for cmd in commands[2:])
    assert submission.finish(spec,'8002')==ledger and len(commands)==21
    assert submission.accepted_jobs(spec,'science')==ledger['jobs']
    assert submission.launch(spec,execute=True,plan_hash=submission.plan(spec,'bootstrap')['content_hash'],authorization=c.AUTHORIZATION)==bootstrap
    assert len(commands)==21


def test_lost_response_stops_duplicate_science_submission(exact_scheduler,monkeypatch):
    spec,commands,_,_=exact_scheduler
    def lost(cmd,**kw):
        commands.append(cmd); raise OSError('Lost accepted sbatch reply')
    monkeypatch.setattr(dag.subprocess,'run',lost)
    with pytest.raises(OSError): submission.finish(spec,'8002')
    with pytest.raises(RuntimeError,match='partial'): submission.finish(spec,'8002')
    assert len(commands)==3


def test_cpu_release_does_not_call_gpu_code(exact_scheduler,monkeypatch):
    spec,_,_,_=exact_scheduler; spec['reference']=dict(environment={})
    monkeypatch.setattr(site,'cpu_allocation',lambda s:'8002')
    monkeypatch.setattr(worker,'installed_environment',lambda:{})
    monkeypatch.setattr(worker,'require_backend',lambda:pytest.fail('CPU release entered GPU backend'))
    assert set(worker.run(spec,'release')['jobs'])==set(c.ORDER)


def test_source_inventory_and_additive_runtime_boundary(monkeypatch):
    project=Path(campaign.__file__).resolve().parents[3]; changes=['A\t'+p for p in campaign.RUNTIME]
    monkeypatch.setattr(campaign,'_source',lambda *a:{'content_hash':'a'*64})
    def git(cmd,**kwargs):
        output='\n'.join(campaign.REQUIRED) if cmd[3]=='ls-files' else '\n'.join(changes) if cmd[3]=='diff' else ''
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(campaign.subprocess,'run',git)
    assert campaign.current(project,'b'*40)['content_hash']=='a'*64
    changes.append('M\tsrc/hlt_classification/luka_embedding_aux/model.py')
    with pytest.raises(ValueError,match='pinned donor'): campaign.current(project,'b'*40)
