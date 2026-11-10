"""Scientific boundaries and production-path complementary teacher checks."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_embedding_aux import tiny
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.luka_complementary_teacher_hydra import contracts as c,model,explanations,folds,training
from hlt_classification.luka_complementary_teacher_hydra import inputs,resources,submission,campaign,worker,reporting,reuse
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from test_luka_attention import cache as large_cache


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    return dict(campaign.registration(),root=str(root),content_hash='c'*64,anchor=dict(evidence=dict(prepared='a'*64)),
        counts=dict(train=66,validation=22),nodes={t:dict(initialization_seed=23,sampler_seed=29) for t in c.FITS},
        recovery_anchors=dict(M0HLT=.95,OFFLINE=.98),reference=dict(environment={'fixture':True}),
        project_dir='/project',environment_dir='/env',bundle_root='/bundle',offline_root='/offline',offline_hash='d'*64)


def raw(train,device='cpu'):
    return {k:torch.as_tensor(v,device=device) for k,v in train.batch(np.arange(min(11,len(train)))).items() if k in ('features','vectors','mask','labels')}


def test_registry_fixed_budget_and_dependency_exclusions():
    assert len(c.FITS)==10 and len(c.TEACHERS)==3 and len(c.STUDENTS)==4
    for t in c.STUDENTS:
        assert sum(c.weights(t).values())==pytest.approx(5.)
        assert set(c.weights(t))==set(c.STUDENTS[t])
    assert c.DEPENDENCIES['TF']==c.FOLDS and c.DEPENDENCIES['TE']==('T0',)
    assert not c.recipe()['u100_objective']


def test_v2_execution_rejects_v1_receipts_and_keeps_science_budget():
    r=c.recipe()
    assert (r['schema_version'],r['explanation']['microbatch'],r['explanation']['diagnostic_rows'])==(2,64,8)
    assert (r['epochs'],r['batch_size'],r['explanation']['coefficient'])==(30,128,1.)
    assert campaign.registration()['fit_max_minutes']==2880
    old=c.artifact('ACCEPTANCE');old.update(contract='LUKA_COMPLEMENTARY_TEACHER_ACCEPTANCE/v1',schema_version=1)
    with pytest.raises(ValueError,match='contract mismatch'):c.validate(reseal(old),'ACCEPTANCE')


@pytest.mark.parametrize('task',c.FITS)
def test_native_initialization_and_head_seed_isolation(tmp_path,monkeypatch,task):
    tiny(monkeypatch); s=spec(tmp_path); m=model.new_model(s,task).eval()
    torch.manual_seed(23); native=DelphesParticleTransformer().eval()
    assert all(torch.equal(v,native.state_dict()[k]) for k,v in m.native_state().items())
    x=cache(coordinate='OFFLINE' if task in (*c.TEACHERS,*c.FOLDS) else 'D000').batch(np.arange(11))
    torch.testing.assert_close(_forward(m,x,'cpu',bf16=False),_forward(native,x,'cpu',bf16=False),rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_both_input_streams_second_order_and_eval_state(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); s=spec(tmp_path); a=model.new_model(s,'T0').to(device).eval(); b=model.new_model(s,'TE').to(device).train()
    for p in a.parameters(): p.requires_grad_(False)
    x=raw(cache(coordinate='OFFLINE'),device); state={k:v.clone() for k,v in b.state_dict().items()}
    penalty,diag=explanations.penalty(b,a,x)
    assert penalty.item()>0 and 0<diag['mask_fraction']<=1
    penalty.backward()
    assert b.training and all(torch.equal(v,b.state_dict()[k]) for k,v in state.items())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in b.mod.pair_embed.parameters())
    assert all(p.grad is None for p in a.parameters())
    assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
    assert explanations.finite_response(b,a,x)['step_standard_deviations']==.05


def test_folds_identity_stable_and_bounded_class_mass():
    ids=np.arange(66*32,dtype=np.uint16).astype(np.uint8).reshape(66,32)
    ids[:,0]=np.arange(66); labels=np.tile(np.arange(11,dtype=np.int64),6)
    f=folds.assignment(ids,labels); order=np.arange(65,-1,-1)
    np.testing.assert_array_equal(f[order],folds.assignment(ids[order],labels[order]))
    for k in range(11): assert np.bincount(f[labels==k],minlength=3).tolist()==[2,2,2]
    p=np.full((66,11),.1/10,np.float32); p[np.arange(66),labels]=.9
    p[:33]=1/11
    w=folds.difficulty(labels,p)
    assert w.min()>0 and w.max()/w.min()<=3
    for k in range(11): assert w[labels==k].mean()==pytest.approx(1.,abs=1e-6)


@pytest.mark.parametrize('task',['S0','SE','SF','SEF','TF','TE'])
def test_actual_step_losses_and_no_privileged_inference(tmp_path,monkeypatch,task):
    tiny(monkeypatch); s=spec(tmp_path); train=cache(coordinate='OFFLINE' if task in c.TEACHERS else 'D000')
    m=model.new_model(s,task); ix=np.arange(11)
    banks={t:SimpleNamespace(targets=np.ones((len(train),128),np.float32)) for t in c.STUDENTS.get(task,())}
    context={}
    if task=='TF': context['weights']=np.ones(len(train),np.float32)
    if task=='TE':
        context['reference']=model.new_model(s,'T0').eval()
        for p in context['reference'].parameters(): p.requires_grad_(False)
    values,_=training.step(m,task,optimizer(m),train.batch(ix),banks,ix,1.,'cpu',context=context,diagnose=True)
    assert set(values)==c.term_names(task) and np.isfinite(list(values.values())).all()
    native=DelphesParticleTransformer().eval(); native.load_state_dict(m.native_state(),strict=True)
    m.eval();m.capture=False
    torch.testing.assert_close(_forward(m,train.batch(ix),'cpu',bf16=False),_forward(native,train.batch(ix),'cpu',bf16=False))


def reseal(value):
    value=dict(value);value.pop('content_hash',None);return with_content_hash(value)


def test_penalty_padding_permutation_and_microbatch_equivalence(tmp_path,monkeypatch):
    tiny(monkeypatch);s=spec(tmp_path);a=model.new_model(s,'T0').eval();b=model.new_model(s,'TE').eval()
    for p in a.parameters(): p.requires_grad_(False)
    x=raw(cache(coordinate='OFFLINE'));loss,_=explanations.penalty(b,a,x)
    loss.backward();expected={n:p.grad.clone() for n,p in b.named_parameters() if p.grad is not None}
    b.zero_grad(set_to_none=True)
    for start in range(0,11,3):
        part={k:v[start:start+3] for k,v in x.items()};term,_=explanations.penalty(b,a,part);(term*len(part['labels'])/11).backward()
    for n,p in b.named_parameters():
        if n in expected: torch.testing.assert_close(p.grad,expected[n],rtol=2e-4,atol=2e-5)
    perm=torch.arange(x['features'].shape[-1]-1,-1,-1)
    changed={k:v[:,:,perm] if k!='labels' else v for k,v in x.items()}
    actual,_=explanations.penalty(b,a,changed);assert actual.item()==pytest.approx(loss.item(),rel=1e-4)
    padded={k:torch.nn.functional.pad(v,(0,3)) if k!='labels' else v for k,v in x.items()}
    actual,_=explanations.penalty(b,a,padded);assert actual.item()==pytest.approx(loss.item(),rel=1e-4)


@pytest.mark.parametrize('device,rows',[('cpu',75),('cuda',128),('cuda',75)])
def test_te_update_microbatch64_matches_8_full_partial_long_support(tmp_path,monkeypatch,device,rows):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if device=='cpu':tiny(monkeypatch)
    s=spec(tmp_path);length=128 if device=='cuda' else 5;rng=np.random.default_rng(90)
    x=dict(features=rng.normal(size=(rows,17,length)).astype(np.float32),
           vectors=rng.normal(size=(rows,4,length)).astype(np.float32),
           mask=np.ones((rows,1,length),bool),labels=np.arange(rows)%11)
    x['vectors'][:,3]=np.sqrt((x['vectors'][:,:3]**2).sum(1)+1)
    # Different lengths, full support, padding, and a partial 64-jet chunk.
    x['mask'][::3,:,length//2:]=False
    for k in ('features','vectors'):x[k]*=x['mask']
    reference=model.new_model(s,'T0').to(device).eval()
    for p in reference.parameters():p.requires_grad_(False)
    reference_state={k:v.clone() for k,v in reference.state_dict().items()}
    original_recipe=c.recipe
    outcomes=[]
    for micro in (8,64):
        def changed_recipe():
            r=original_recipe();r['explanation']['microbatch']=micro;return r
        monkeypatch.setattr(training,'recipe',changed_recipe)
        m=model.new_model(s,'TE').to(device);opt=optimizer(m)
        torch.manual_seed(781)
        values,diag=training.step(m,'TE',opt,x,{},np.arange(rows),3.,device,
            context=dict(reference=reference),diagnose=True)
        outcomes.append((values,diag,{n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None},
                         {n:b.detach().cpu().clone() for n,b in m.named_buffers()}))
        assert all(p.grad is None for p in reference.parameters())
        assert all(torch.equal(v,reference.state_dict()[k]) for k,v in reference_state.items())
        del m,opt
    a,b=outcomes
    assert a[0]['ce']==b[0]['ce']
    assert b[0]['explanation']==pytest.approx(a[0]['explanation'],rel=3e-4,abs=1e-6)
    assert b[1]['finite_response']==a[1]['finite_response']
    assert a[2].keys()==b[2].keys()
    for name,g in a[2].items():torch.testing.assert_close(b[2][name],g,rtol=5e-4,atol=3e-5)
    for name,v in a[3].items():torch.testing.assert_close(b[3][name],v,rtol=0,atol=0)


@pytest.mark.parametrize('task',['T0','TE','TF','S0','SEF'])
def test_production_depth_batch128_partial_cuda(tmp_path,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s=spec(tmp_path);tr=cache(coordinate='OFFLINE' if task in c.TEACHERS else 'D000');m=model.new_model(s,task).cuda()
    assert len(m.mod.blocks)==8 and len(m.mod.cls_blocks)==2
    banks={t:SimpleNamespace(targets=np.ones((len(tr),128),np.float32)) for t in c.STUDENTS.get(task,())};context={}
    if task=='TE':
        context['reference']=model.new_model(s,'T0').cuda().eval()
        for p in context['reference'].parameters(): p.requires_grad_(False)
    if task=='TF': context['weights']=np.ones(len(tr),np.float32)
    opt=optimizer(m)
    for size in (128,11):
        ix=np.arange(size)%len(tr);v,_=training.step(m,task,opt,tr.batch(ix),banks,ix,1.,'cuda',context=context)
        assert set(v)==c.term_names(task) and np.isfinite(list(v.values())).all()
    m.eval();m.capture=False;native=DelphesParticleTransformer().cuda().eval();native.load_state_dict(m.native_state(),strict=True)
    with torch.no_grad(): torch.testing.assert_close(_forward(native,tr.batch(np.arange(11)),'cuda',bf16=False),_forward(m,tr.batch(np.arange(11)),'cuda',bf16=False),rtol=0,atol=0)


def test_fold_manifest_role_identity_and_corruption(tmp_path):
    s=spec(tmp_path);(tmp_path/'setup').mkdir();tr=large_cache('train','OFFLINE',66);m=folds.build(s,tr)
    assert folds.load(s,tr)[0]==m
    for t in c.FOLDS:
        kept,held=folds.split(s,tr,t)
        assert len(kept)==44 and len(held)==22
        assert not set(map(bytes,kept.identities))&set(map(bytes,held.identities))
        with pytest.raises(PermissionError): training.check_inputs(s,t,kept,held,{},dict(parents=dict(folds=m['content_hash'])),False)
        with pytest.raises(PermissionError): training.check_inputs(s,t,tr,None,{},dict(parents=dict(folds=m['content_hash'])),False)
    tr.role='validation'
    with pytest.raises(PermissionError): folds.load(s,tr)
    p=tmp_path/'setup/folds.npz';data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
    with pytest.raises(ValueError): folds.load(s)


def diagnostic(s):
    profiles={}
    for t in c.FITS:
        profiles[t]=dict(initial_native_fp32_parity=True,miniature_complete=True,deployment_reload=True,
            largest_support_backward=True,partial_batch_backward=True,ordered_reference_joins=True,
            train_rows=44 if t in c.FOLDS else 66,validation_rows=0 if t in c.FOLDS else 22,
            export_rows=66 if t in c.TEACHERS else 22 if t in c.FOLDS else 0,
            step_samples=[.1]*12,step_seconds=float(np.mean([.1]*12)),validation_seconds=1.,probe_seconds=1.,load_seconds=1.,export_seconds=1.,
            second_order_both_inputs=t=='TE',targets_technical=t in c.STUDENTS or t=='TF',
            explanation_microbatch=c.recipe()['explanation']['microbatch'] if t=='TE' else None)
    return c.artifact('RESOURCES',parents=dict(spec=s['content_hash']),status='complete',error=None,profiles=profiles,
        environment=s['reference']['environment'],gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30),
        memory=dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))


@pytest.mark.parametrize('bad',['memory','time','missing','gpu','timing','fold_rows','second_order','microbatch'])
def test_admission_rejects_technical_failure_only(tmp_path,bad):
    s=spec(tmp_path);(tmp_path/'setup').mkdir();folds.build(s,large_cache('train','OFFLINE',66));d=diagnostic(s)
    assert set(resources.accept(s,d)['minutes'])==set(c.FITS)
    if bad=='memory': d['memory']['peak_rss_bytes']=32*2**30
    elif bad=='time': d['profiles']['TE'].update(step_samples=[1e6]*12,step_seconds=1e6)
    elif bad=='missing': del d['profiles']['TF']
    elif bad=='gpu': d['gpu']['name']='NVIDIA L40'
    elif bad=='timing': d['profiles']['TF']['validation_seconds']=-1
    elif bad=='fold_rows': d['profiles']['F0']['train_rows']=66
    elif bad=='microbatch': d['profiles']['TE']['explanation_microbatch']=8
    else: d['profiles']['TE']['second_order_both_inputs']=False
    with pytest.raises((ValueError,MemoryError,RuntimeError)):resources.accept(s,reseal(d))


def fake_scheduler(monkeypatch):
    calls=[];reviews=[];monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    monkeypatch.setattr(submission.time,'sleep',lambda _:None)
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(cmd,**kw): calls.append(cmd);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run);return calls,reviews


def test_durable_bootstrap_release_dag_and_idempotency(tmp_path,monkeypatch):
    s=spec(tmp_path);(tmp_path/'setup').mkdir();folds.build(s,large_cache('train','OFFLINE',66))
    a=resources.accept(s,diagnostic(s));write_immutable_json(tmp_path/'setup/acceptance.json',a)
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(s);p=submission.plan(s,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(s,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)
    boot=submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert boot['jobs']=={'setup':'9001','release':'9002'} and '--dependency=afterok:9001' in calls[1]
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0']);monkeypatch.setattr(worker,'completed_setup',lambda _:a)
    live=submission.finish(s,'9002')
    assert len(live['jobs'])==11 and len(calls)==13
    for t,deps in c.DEPENDENCIES.items():
        expected='--dependency=afterok:'+':'.join(live['jobs'][d] for d in deps)
        assert (expected in live['commands'][t])==bool(deps)
    assert any(x.startswith('--dependency=afterany:') for x in live['commands']['report'])
    assert submission.finish(s,'9002')==live and len(calls)==13
    assert [p['stage'] for p in reviews]==['bootstrap','envelope','science']
    with pytest.raises(PermissionError):submission.finish(s,'9999')


def test_failed_setup_and_ambiguous_claim_do_not_queue(tmp_path,monkeypatch):
    s=spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(s);p=submission.plan(s,'bootstrap')
    submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(s,'9002')
    assert len(calls)==2 and not (tmp_path/'submission_science').exists()
    other=tmp_path/'other';other.mkdir();s2=dict(s,root=str(other));submission.launch(s2);(other/'submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous'):submission.launch(s2,execute=True,plan_hash=submission.plan(s2,'bootstrap')['content_hash'],authorization=c.AUTHORIZATION)


@pytest.mark.parametrize('violation',['role','coordinate','identity','population','overlap','missing_target','u100'])
def test_hlt_training_target_firewall(tmp_path,violation):
    s=spec(tmp_path);s['counts']['train']=22;tr=cache();va=cache('validation')
    banks={'T0':SimpleNamespace(manifest=dict(role='train'),identities=tr.identities.copy(),labels=tr.labels.copy(),targets=np.zeros((22,128),np.float32))}
    if violation=='role':tr.role='validation'
    elif violation=='coordinate':tr.coordinate_name='OFFLINE'
    elif violation=='identity':banks['T0'].identities=tr.identities[::-1]
    elif violation=='population':s['counts']['train']=99
    elif violation=='overlap':va.identities=tr.identities.copy()
    elif violation=='missing_target':banks.clear()
    else:banks['u100']=banks['T0']
    with pytest.raises((ValueError,PermissionError)):training.check_inputs(s,'S0',tr,va,banks,{},False)


def fixture_bundle(tmp_path,monkeypatch,s,train,val):
    paths={}
    for name,strength in (('M0HLT',0.),('OFFLINE',.9),('matched',.2)):
        p=np.full((len(val),11),(1-strength)/11,np.float32);p[np.arange(len(val)),val.labels]+=strength
        dest=tmp_path/(name+'.npz');atomic_publish_bytes(dest,deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=p)))
        paths[f'reference/{name}/predictions.npz']=dest
    dest=tmp_path/'train_identity.npz';atomic_publish_bytes(dest,deterministic_npz_bytes(dict(identities=train.identities,labels=train.labels)))
    paths['targets/u100/targets.npz']=dest
    dest=tmp_path/'subset.npz';atomic_publish_bytes(dest,deterministic_npz_bytes(dict(indices=np.arange(len(val),dtype=np.int64))))
    paths['reference/historical/subset.npz']=dest;s['original50k_rows']=len(val)
    from hlt_classification.context_fusion.worker import save_state
    native=DelphesParticleTransformer().cuda().eval();cp=tmp_path/'native.pt';save_state(cp,native.state_dict())
    paths['reference/matched/checkpoint.pt']=cp
    # Real fixture replay uses the very same synthetic native checkpoint.
    from hlt_classification.jetclass2_delphes.runner import predict
    actual=predict(native,val,device='cuda',batch_size=128)
    paths['reference/matched/predictions.npz'].unlink()
    atomic_publish_bytes(paths['reference/matched/predictions.npz'],deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda n:paths[n]))
    return paths


def test_complete_cuda_setup_production_functions(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tiny(monkeypatch);s=spec(tmp_path);(tmp_path/'setup').mkdir();tr=large_cache('train','D000',66);va=cache('validation')
    fixture_bundle(tmp_path,monkeypatch,s,tr,va)
    monkeypatch.setattr(inputs,'views',lambda _,teacher:(large_cache('train','OFFLINE' if teacher else 'D000',66),cache('validation','OFFLINE' if teacher else 'D000')))
    monkeypatch.setattr(resources,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(resources,'gpu_identity',lambda:diagnostic(s)['gpu'])
    monkeypatch.setattr(resources,'memory_snapshot',lambda _:diagnostic(s)['memory'])
    a=resources.measure(s);assert a==resources.accepted(s) and set(a['minutes'])==set(c.FITS)
    for t in c.FITS:training.validate_fit(tmp_path/'setup'/('mini_'+t),s,t,technical=True)
    other=tmp_path/'failed';(other/'setup').mkdir(parents=True);s2=dict(s,root=str(other))
    def fail(*a,**k):raise RuntimeError('injected real-path failure')
    monkeypatch.setattr(training,'step',fail)
    with pytest.raises(RuntimeError,match='injected'):resources.measure(s2)
    d=load_json(other/'setup/resource_diagnostic.json');assert d['status']=='failed' and 'injected' in d['error']
    assert not (other/'setup/acceptance.json').exists()


def test_all_ten_scientific_workers_oof_banks_students_report(tmp_path,monkeypatch,capsys):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tiny(monkeypatch);s=spec(tmp_path);(tmp_path/'setup').mkdir();tr=large_cache('train','D000',66);va=cache('validation')
    fixture_bundle(tmp_path,monkeypatch,s,tr,va);folds.build(s,large_cache('train','OFFLINE',66))
    monkeypatch.setattr(inputs,'fit_views',lambda _,t:(large_cache('train','D000' if t in c.STUDENTS else 'OFFLINE',66),
        None if t in c.FOLDS else cache('validation','D000' if t in c.STUDENTS else 'OFFLINE')))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None);monkeypatch.setattr(worker,'require_backend',lambda:None)
    jobs={t:str(9100+i) for i,t in enumerate((*c.FITS,'report'))};job=['']
    monkeypatch.setattr(worker.site,'allocation',lambda _:job[0]);monkeypatch.setattr(worker.site,'cpu_allocation',lambda _:job[0])
    monkeypatch.setattr(worker,'gpu_identity',lambda:diagnostic(s)['gpu']);monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:diagnostic(s)['memory']);monkeypatch.setattr(worker,'completed_setup',lambda _:dict(gpu=diagnostic(s)['gpu']))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:jobs);monkeypatch.setattr(reporting,'states',lambda _:{t:['COMPLETED','0:0'] for t in jobs})
    for t in c.FITS:
        job[0]=jobs[t];worker.run(s,t)
        report,_=training.validate_fit(tmp_path/t/'fit',s,t)
        assert report['epochs']==30 and report['parents']==worker.fit_parents(s,t)
    w,m=folds.assemble(s);assert np.isfinite(w).all() and (w>0).all()
    assert m==load_json(tmp_path/'TF/difficulty.json')
    result=reporting.results(s);assert not result['pending'] and len(result['rows'])==18 and len(result['preparation'])==3
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    reporting.progress(s);reporting.results(s)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    job[0]=jobs['report'];worker.run(s,'report');assert load_json(tmp_path/'report/receipt.json')['gpu'] is None
    assert 'delta S0' in capsys.readouterr().out
    bad=load_json(tmp_path/'F0/oof.json');bad['trained_identity_hash']='0'*64;(tmp_path/'F0/oof.json').write_text(json.dumps(reseal(bad)))
    with pytest.raises(ValueError):folds.read_oof(s,'F0')


def test_runtime_scripts_and_documentation_links(tmp_path):
    import re
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in campaign.REQUIRED)
    for p in campaign.RUNTIME:
        text=(root/p).read_text()
        assert 'Fresh_check' not in text
        if p.endswith('.py'):compile(text,p,'exec')
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.exists():pytest.skip('Bash unavailable')
    for p in ('scripts/queue_luka_complementary_teacher_hydra.sh','sbatch/run_luka_complementary_teacher_hydra.sh'):
        subprocess.run([str(bash),'-n',p],cwd=root,check=True)
    for name in campaign.REQUIRED:
        if not name.endswith('.md'):continue
        path=root/name;text=path.read_text()
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' not in link:assert (path.parent/link.split('#')[0]).is_file(),link
        for i,body in enumerate(re.findall(r'```bash\n(.*?)\n```',text,re.S)):
            script=tmp_path/(path.stem+str(i)+'.sh');script.write_text(body,encoding='utf-8',newline='\n')
            subprocess.run([str(bash),'-n',str(script)],check=True)


def test_source_guard_allows_additive_sibling_but_rejects_donor_edits(tmp_path,monkeypatch):
    def git(*args):return subprocess.run(['git','-C',str(tmp_path),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init');git('config','user.email','test@example.invalid');git('config','user.name','Test')
    donor=tmp_path/'src/donor.py';donor.parent.mkdir();donor.write_text('x=1\n');git('add','.');git('commit','-m','base')
    base=git('rev-parse','HEAD');monkeypatch.setattr(campaign,'BASE',base)
    for name in campaign.REQUIRED:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture\n')
    sibling=tmp_path/'src/sibling.py';sibling.write_text('unrelated additive campaign\n')
    git('add','.');git('commit','-m','new campaign and sibling');commit=git('rev-parse','HEAD')
    monkeypatch.setattr(campaign,'_source',lambda *a:{'content_hash':'a'*64})
    monkeypatch.setattr(campaign,'__file__',str(tmp_path/'src/hlt_classification/luka_complementary_teacher_hydra/campaign.py'))
    assert campaign.current(tmp_path,commit)=={'content_hash':'a'*64}
    donor.write_text('x=2\n');git('add','.');git('commit','-m','modified donor')
    with pytest.raises(ValueError,match='preserve'):campaign.current(tmp_path,git('rev-parse','HEAD'))


def test_original_parent_runs_its_own_authenticator(tmp_path,monkeypatch):
    from hlt_classification.luka_teacher_sweep_hydra.contracts import artifact as old_artifact
    parent=old_artifact('SPEC',project_dir=str(tmp_path/'original'))
    path=tmp_path/'parent.json';write_immutable_json(path,parent);calls=[]
    def run(cmd,**kw):
        calls.append((cmd,kw));return SimpleNamespace(returncode=0,stdout='COMPLEMENTARY_PARENT='+parent['content_hash']+'\n',stderr='')
    monkeypatch.setattr(reuse.subprocess,'run',run)
    assert reuse.original(path)==parent
    cmd,kw=calls[0]
    assert cmd[-1]==parent['project_dir'] and kw['cwd']==parent['project_dir']
    assert kw['env']['PYTHONNOUSERSITE']=='1' and 'campaign.check' in cmd[4] and 'verify_all' in cmd[4]
    monkeypatch.setattr(reuse.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='bad source'))
    with pytest.raises(ValueError,match='authentication failed'):reuse.original(path)


def test_registration_round_trip_and_parent_substitution(tmp_path,monkeypatch):
    s=spec(tmp_path/'old');s['project_dir']=str(tmp_path/'old_code');s['environment_dir']=str(Path(__import__('sys').prefix).resolve())
    s['imported']=dict(node=dict(initialization_seed=23,sampler_seed=29));s['nodes']['T00']=dict(initialization_seed=24,sampler_seed=30)
    s['embedding_hash']='e'*64;s['qualification']={}
    path=tmp_path/'parent.json';write_immutable_json(path,s)
    monkeypatch.setattr(reuse,'original',lambda _:s);monkeypatch.setattr(campaign,'current',lambda *a:{'content_hash':'f'*64})
    monkeypatch.setattr(campaign.site,'environment',lambda _:None);monkeypatch.setattr(campaign,'storage',lambda _:None)
    root=tmp_path/'new';m=campaign.create(root,path,tmp_path/'new_code','a'*40,9)
    campaign.check(m)
    assert m['nodes']['TE']['initialization_seed']==24 and m['nodes']['SF']['initialization_seed']==23
    bad=copy.deepcopy(m);bad['offline_hash']='0'*64;bad=reseal(bad)
    (root/'campaign_spec.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='lineage'):campaign.check(bad)


def test_finite_bad_teacher_finishes_all_epochs(tmp_path,monkeypatch):
    tiny(monkeypatch);s=spec(tmp_path);s['counts']['train']=22;original=training.evaluate_probabilities
    def poor(y,p):
        m=original(y,p);m['macro_ovr_auc']=0.;return m
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    _,r=training.fit(tmp_path/'poor',s,'T0',cache(coordinate='OFFLINE'),cache('validation','OFFLINE'),{},device='cpu')
    assert r['epochs']==30 and r['validation']['macro_ovr_auc']==0.
