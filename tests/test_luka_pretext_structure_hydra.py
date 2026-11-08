"""Scientific definitions, label-free teachers, native students and durable DAG."""
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import hashlib
import copy
import numpy as np
import pytest
import torch
from test_luka_u100_pair_correction import dataset, prepared, all_prepared
from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.cms2jc2_response.bridge import p4_from_coordinates
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_pretext_structure_hydra import (contracts as c, structure as st, corruption as co,
    normalization as no, target_banks as tb, teacher_model as tm, teacher_training as tt,
    student_model as sm, student_training as tr, embedding_banks as eb, campaign, submission, resources)


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def particles(n=7,seed=0):
    rng=np.random.default_rng(seed); a=np.zeros((n,14),np.float64)
    a[:,:4]=p4_from_coordinates(rng.uniform(.2,10,n),rng.normal(0,.3,n),rng.normal(0,.3,n),rng.uniform(.01,.1,n))
    a[:,4]=rng.choice([-1,0,1],n); a[:,5]=np.where(a[:,4]!=0,0,2)
    a[:,6:8]=rng.normal(0,.2,(n,2)); a[:,8:10]=rng.uniform(.01,.1,(n,2)); a[:,10:]=a[:,4:5]!=0
    a[:,6:10]*=a[:,10:]
    return a


def native_cache(role,n=22):
    jets=[particles(5+i%4,i+(0 if role=='train' else 200)) for i in range(n)]
    native=[co.native(a) for a in jets]
    ids=np.array([np.frombuffer(hashlib.sha256(f'{role}/{i}'.encode()).digest(),np.uint8) for i in range(n)])
    labels=np.arange(n,dtype=np.int64)%11; offsets=np.r_[0,np.cumsum([len(a) for a in jets])].astype(np.int64)
    cache=RamCache([RamBlock(0,offsets,np.concatenate([a.features for a in native]),np.concatenate([a.vectors for a in native]),ids,labels)],
        role=role,coordinate_name='D000',foundation_sha256='a'*64)
    arrays=dict(identities=ids,labels=labels,offsets=offsets,values=np.concatenate(jets))
    return cache,arrays


def spec(root,n=22):
    return dict(campaign.registration(),root=str(root),project_dir='/project',content_hash='c'*64,
        anchor=dict(evidence=dict(prepared='a'*64)),reference=dict(environment={'fixture':True}),
        bundle_root='/bundle',environment_dir='/env',counts=dict(train=n,validation=n),
        recovery_anchors=dict(M0HLT=.95,OFFLINE=.98),original50k_rows=n,
        nodes={t:dict(node_id=t,initialization_seed=23+(200000 if t in c.TEACHERS else 0),sampler_seed=29) for t in c.FITS})


def small(monkeypatch):
    import hlt_classification.jetclass2_delphes.model as native
    import hlt_classification.luka_capacity_hydra.model as capacity
    original=native.model_config
    def config(): return dict(original(),num_layers=1,num_cls_layers=1,pair_embed_dims=[8,8,8])
    monkeypatch.setattr(native,'model_config',config)
    monkeypatch.setattr(capacity,'model_config',lambda width:config())


@pytest.fixture
def data(tmp_path):
    train,a=native_cache('train'); val,b=native_cache('validation')
    pt=tb.prepare_raw(tmp_path/'raw_train',lambda:iter([a]),train,'OFFLINE','1'*64)
    pv=tb.prepare_raw(tmp_path/'raw_val',lambda:iter([b]),val,'OFFLINE','2'*64)
    ts=tb.prepare_structure(tmp_path/'structure_train',pt)
    vs=tb.prepare_structure(tmp_path/'structure_val',pv,normalizers=ts.normalizers)
    stats=tb.prepare_pretext_statistics(tmp_path,pt,ts)
    return train,val,pt,pv,ts,vs,stats


def banks(train,task,ts):
    rng=np.random.default_rng(4)
    result={name:SimpleNamespace(manifest=c.artifact('FIXTURE_BANK',role='train',coordinate='U100' if name=='u100' else 'OFFLINE' if name=='ce_teacher' else 'PRETEXT_'+name),
        targets=rng.normal(size=(len(train),128)).astype(np.float32),identities=train.identities,labels=train.labels) for name in c.weights(task)}
    if task in c.STRUCTURES: result['structure']=ts
    return result


def test_registry_and_automatic_dag(tmp_path):
    s=spec(tmp_path); boot=submission.plan(s,'bootstrap'); env=submission.plan(s,'envelope')
    assert len(c.FITS)==len(set(c.FITS))==30 and len(c.STUDENTS)==22 and len(c.TEACHERS)==8
    assert [r['task_id'] for r in boot['commands']]==['setup','release']
    assert boot['commands'][1]['dependencies']==['setup']
    assert boot['upper_bound_commands']==env['commands']
    assert len(boot['commands'])+len(env['commands'])==33
    order=[]
    for row in env['commands'][:-1]:
        task=row['task_id']; assert row['dependencies']==list(c.dependencies(task))
        assert set(row['dependencies'])<=set(order); order.append(task)
        assert '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
    assert env['commands'][-1]['dependencies']==list(c.FITS)
    assert any(a.startswith('--dependency=afterany:') for a in env['commands'][-1]['command'])
    for task in c.STUDENTS:
        w=c.weights(task); assert w['u100']==5
        assert sum(v for n,v in w.items() if n!='u100')==pytest.approx(0 if task in c.STRUCTURES or task=='C_U' else 5)
    assert c.ramp(0)==0 and c.ramp(1.5)==.5 and c.ramp(4)==1


@pytest.mark.parametrize('n',[1,2,3,9])
def test_exact_correlations_independent_triples(n):
    a=particles(n); result=st.correlations(a); pt,_,phi,y=st.coordinates(a[:,:4]); z=pt/pt.sum()
    d=np.hypot(y[:,None]-y,st.wrap(phi[:,None]-phi))
    for j,beta in enumerate((.5,1,2)):
        e2=sum(z[i]*z[k]*d[i,k]**beta for i,k in combinations(range(n),2))
        e3=sum(z[i]*z[k]*z[l]*(d[i,k]*d[i,l]*d[k,l])**beta for i,k,l in combinations(range(n),3))
        assert np.expm1(result['C2'].values[j])*1e-4==pytest.approx(e2,abs=1e-12)
        assert np.expm1(result['C3'].values[j])*1e-4==pytest.approx(e3,abs=1e-12)
    for name in result:
        assert (np.diff(result[name].values[3:16])>=-1e-12).all()
        if n>=int(name[-1]): assert result[name].values[15]==pytest.approx(1)


def test_physics_permutation_phi_and_edge_cases():
    a=particles(11); reference=st.all_families(a); rotated=a.copy()
    angle=.77; rotated[:,0]=a[:,0]*np.cos(angle)-a[:,1]*np.sin(angle)
    rotated[:,1]=a[:,0]*np.sin(angle)+a[:,1]*np.cos(angle)
    for other in (a[::-1],rotated):
        for name,value in st.all_families(other).items():
            np.testing.assert_allclose(value.values,reference[name].values,atol=1e-10)
            np.testing.assert_array_equal(value.valid,reference[name].valid)
    single=st.branches(particles(1)); assert not single.values[0] and not single.values[5] and not single.values[12]
    zero=particles(2); zero[0,:4]=0; assert np.isfinite(st.branches(zero).values).all()
    zero[:,:4]=0
    with pytest.raises(ValueError): st.checked(zero)
    bad=particles(); bad[0,3]=0
    with pytest.raises(ValueError): st.checked(bad)


@pytest.mark.parametrize('task',c.TEACHERS)
def test_corruption_replay_and_no_hidden_frame(task):
    a=particles(14); identity=b'x'*32
    v,clean,removed=co.corrupt(a,identity,task,3)
    again=co.corrupt(a[::-1],identity,task,3)
    for x,y in zip((v,clean,removed),again): np.testing.assert_array_equal(x,y)
    assert 0<len(v)<len(a) and removed.any()
    assert co.native(v).features.shape==(len(v),17)
    target=co.set_target(clean[removed],v)
    assert target['kinematic'].shape[1]==6
    np.testing.assert_allclose(target['frame'][2],np.hypot(v[:,0],v[:,1]).sum())
    one=co.corrupt(particles(1),identity,task,1)
    assert len(one[0])==1 and not one[2].any()


def test_train_normalization_and_equal_blocks():
    m=no.Moments(3); m.add([[1,2,7],[3,4,7]],np.ones((2,3),bool))
    norm=m.finish(parents={},binary=[False]*3,blocks=[0,1,1])
    assert norm['mean']==[2,3,7] and norm['active']==[True,True,False]
    pred=torch.tensor([[2.,4.,100.]],requires_grad=True)
    loss=no.rows(pred,[[2.,3.,7.]],[[True]*3],norm).mean(); loss.backward()
    assert pred.grad[0,2]==0
    wrong=dict(norm,role='validation'); wrong=with_content_hash({k:v for k,v in wrong.items() if k!='content_hash'})
    with pytest.raises(PermissionError): no.rows(pred,[[2.,3.,7.]],[[True]*3],wrong)


@pytest.mark.parametrize('task',c.TEACHERS)
def test_teacher_real_gradient_and_label_blindness(tmp_path,monkeypatch,data,task):
    small(monkeypatch); train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path)
    model=tt.new_model(s,task,8,stats,ts.normalizers); opt=tt.optimizer(model)
    native_before={k:v.clone() for k,v in model.mod.fc.state_dict().items()}
    values=tt.step(model,opt,pt,ts,np.arange(11),1.,1,'cpu')
    assert np.isfinite(list(values.values())).all()
    assert all(p.grad is None for p in model.mod.fc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.mod.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.clean_heads.parameters())
    for k,v in native_before.items(): torch.testing.assert_close(v,model.mod.fc.state_dict()[k],rtol=0,atol=0)
    before=tt.evaluate(model,pv,vs,'cpu')
    shuffled=resources.PhysicalSubset(pv,np.arange(len(pv))); shuffled.labels=np.roll(shuffled.labels,1)
    after=tt.evaluate(model,shuffled,vs,'cpu'); assert before==after


@pytest.mark.parametrize('task',c.STUDENTS)
def test_student_heads_native_parity_and_objective(tmp_path,monkeypatch,data,task):
    small(monkeypatch); train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path)
    model=sm.new_model(s,task,ts.normalizers).eval(); after=torch.random.get_rng_state().clone()
    torch.manual_seed(s['nodes'][task]['initialization_seed']); native=sm.Native(128).eval()
    assert torch.equal(after,torch.random.get_rng_state())
    model.capture=True; raw=train.batch(np.arange(len(train))); logits=_forward(model,raw,'cpu',bf16=False)
    torch.testing.assert_close(logits,_forward(native,raw,'cpu',bf16=False),rtol=0,atol=0)
    targets={n:torch.randn(len(train),128,requires_grad=True) for n in c.weights(task)}
    loss,terms=sm.objective(model,task,logits,torch.from_numpy(raw['labels']),position=1.5,
        structure=ts if task in c.STRUCTURES else None,indices=np.arange(len(train)),**targets)
    assert set(terms)==c.term_names(task); loss.backward()
    assert all(v.grad is None for v in targets.values())
    for k,v in model.native_state().items(): torch.testing.assert_close(v,native.state_dict()[k],rtol=0,atol=0)


def test_selected_teacher_export_and_native_student_fit(tmp_path,monkeypatch,data):
    small(monkeypatch); train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path)
    model,r=tt.fit(tmp_path/'teacher',s,'T01',pt,pv,ts,vs,stats,8,device='cpu',technical=True)
    eb.export(tmp_path/'embeddings',model,pt,s,'T01',file_ref(tmp_path/'teacher/selected.pt'),device='cpu',technical=True)
    target=eb.Bank(tmp_path/'embeddings',train,s,'T01',file_ref(tmp_path/'teacher/selected.pt'),technical=True)
    b=banks(train,'E01',ts); b['T01']=target
    native,report=tr.fit(tmp_path/'student',s,'E01',train,val,b,device='cpu',technical=True)
    assert report['native_reload'] and report['epochs']==2
    assert all(k.startswith('mod.') for k in native.state_dict())
    with pytest.raises(ValueError): eb.Bank(tmp_path/'embeddings',train,s,'T01',file_ref(tmp_path/'teacher/selected.pt'),technical=False)


def test_set_loss_permutation_assignment_and_gradients(data):
    train,val,pt,pv,ts,vs,stats=data
    a=particles(6); v,clean,removed=co.corrupt(a,b'x'*32,'T01',1); target=co.set_target(clean[removed],v)
    d=tm.SetDecoder(8); h=torch.randn(2,128,requires_grad=True); scores=d(h)
    reverse={k:(x[::-1].copy() if isinstance(x,np.ndarray) and k!='frame' else x) for k,x in target.items()}
    x=tm.set_rows(scores,[target,target],stats['teachers']['T01']['set'])
    y=tm.set_rows(scores,[reverse,target],stats['teachers']['T01']['set'])
    torch.testing.assert_close(x,y,rtol=0,atol=1e-6); x.mean().backward(); assert h.grad.abs().sum()>0


def test_resource_caps_and_quality_not_gate(tmp_path):
    s=spec(tmp_path); profiles={t:dict(initial_factory_fp32_parity=True,clean_frontend_parity=True,miniature_complete=True,
        selected_reload=True,largest_support_backward=True,partial_batch_backward=True,train_rows=22,validation_rows=22,
        step_samples=[.1]*12,step_seconds=float(np.mean([.1]*12)),validation_seconds=1.,probe_seconds=1.,load_seconds=1.,export_seconds=1.,
        memory=dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30)) for t in c.FITS}
    for t,p in profiles.items():
        keys=('input_corruption','native_forward','auxiliary_forward_assignment','diagnostics','backward_optimizer') if t in c.TEACHERS else ('native_forward','targets_and_auxiliary','diagnostics','backward_optimizer')
        p['phase_samples']=[dict.fromkeys(keys,.01) for _ in range(3)]
    d=c.artifact('RESOURCES',parents=dict(spec=s['content_hash']),status='complete',error=None,profiles=profiles,
        environment=s['reference']['environment'],gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30))
    accepted=resources.accept(s,d); assert set(accepted['minutes'])==set(c.FITS)
    measured=submission.plan(s,'science',accepted); assert submission.within_envelope(submission.plan(s,'envelope'),measured)
    d['profiles']['T01']['memory']['peak_rss_bytes']=30*2**30
    d=with_content_hash({k:v for k,v in d.items() if k!='content_hash'})
    with pytest.raises(MemoryError): resources.accept(s,d)


@pytest.mark.parametrize('task',c.STUDENTS)
def test_all_student_production_fit_reload(tmp_path,monkeypatch,data,task):
    small(monkeypatch); train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path)
    model,report=tr.fit(tmp_path/task,s,task,train,val,banks(train,task,ts),device='cpu',technical=True)
    tr.validate_fit(tmp_path/task,s,task,technical=True)
    assert report['selected_epoch'] in (1,2)


@pytest.mark.parametrize('task',c.TEACHERS)
def test_all_teacher_production_fit_reload_and_export(tmp_path,monkeypatch,data,task):
    small(monkeypatch); train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path)
    model,report=tt.fit(tmp_path/task,s,task,pt,pv,ts,vs,stats,8,device='cpu',technical=True)
    manifest=eb.export(tmp_path/(task+'_bank'),model,pt,s,task,file_ref(tmp_path/task/'selected.pt'),device='cpu',technical=True)
    assert report['class_labels_used'] is False and manifest['native_reload']
    assert report['decoder_probe']['quality_gate'] is False
    eb.Bank(tmp_path/(task+'_bank'),train,s,task,file_ref(tmp_path/task/'selected.pt'),technical=True)


def test_u100_control_matches_original_update(tmp_path,monkeypatch,data):
    small(monkeypatch)
    from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
    from hlt_classification.luka_u100_contrastive.training import optimizer
    from torch.nn import functional as F
    train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path); ours=sm.new_model(s,'C_U')
    torch.manual_seed(s['nodes']['C_U']['initialization_seed']); old=AuxiliaryParT()
    raw=train.batch(np.arange(22)); bank=banks(train,'C_U',ts)
    torch.manual_seed(109); tr.step(ours,'C_U',optimizer(ours),raw,bank,np.arange(22),1.,'cpu')
    opt=optimizer(old)
    from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
    for group in opt.param_groups: group['lr']=learning_rate(1.)
    torch.manual_seed(109); old.train(); old.capture=True; logits=_forward(old,raw,'cpu',bf16=False)
    loss=F.cross_entropy(logits,torch.from_numpy(raw['labels']))+5*F.smooth_l1_loss(old.prediction,torch.from_numpy(bank['u100'].targets))
    loss.backward(); opt.step()
    for k,v in ours.native_state().items(): torch.testing.assert_close(v,old.native_state()[k],rtol=0,atol=0)
    for a,b in zip(ours.heads['u100'].parameters(),old.predictor.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires a real CUDA GPU')
@pytest.mark.parametrize('task',(*c.TEACHERS,'C_U','M04','S07','S08'))
def test_full_native_gpu_batch128_partial_and_support(tmp_path,data,task):
    train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path); ix=np.arange(128)%len(train)
    if task in c.TEACHERS:
        model=tt.new_model(s,task,298,stats,ts.normalizers).cuda(); opt=tt.optimizer(model)
        for indices in (ix,ix[:5]):
            values=tt.step(model,opt,pt,ts,indices,1.,1,'cuda'); assert np.isfinite(list(values.values())).all()
        if task in ('T01','T02','T03','T08'):
            # Fixed output support is independent of individual target count.
            assert model.decoder.maximum==298
    else:
        model=sm.new_model(s,task,ts.normalizers).cuda(); opt=tr.optimizer(model); b=banks(train,task,ts)
        for indices in (ix,ix[:5]):
            values,_=tr.step(model,task,opt,train.batch(indices),b,indices,1.,'cuda')
            assert np.isfinite(list(values.values())).all()
    del model,opt; resources.free()


def test_submission_partial_claim_blocks_blind_retry(tmp_path,monkeypatch):
    s=spec(tmp_path); plan=submission.plan(s,'bootstrap')
    monkeypatch.setattr(submission,'scheduler_review',lambda plan:None)
    from hlt_classification.data.cache_contracts import write_immutable_json
    folder=tmp_path/'submission_bootstrap'; folder.mkdir(); write_immutable_json(folder/'command_plan.json',plan)
    (folder/'live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous/partial'): submission.submit_stage(s,plan,execute=True)


def test_wrong_roles_and_corrupt_banks_fail(tmp_path,data):
    train,val,pt,pv,ts,vs,stats=data
    with pytest.raises(PermissionError): tb.prepare_pretext_statistics(tmp_path,pv,vs)
    with pytest.raises(PermissionError): tb.PhysicalBank(pt.folder,val,'OFFLINE')
    with pytest.raises(PermissionError): tb.prepare_structure(tmp_path/'bad_val',pv)
    target=ts.folder/'C2.npy'
    with target.open('r+b') as f: f.seek(-1,2); f.write(b'\xff')
    with pytest.raises(ValueError): tb.StructureBank(ts.folder,pt)


def test_no_split_ties_invalid_tracking_and_large_set():
    a=particles(298); a[:,6:]=0
    target=st.radial_identity_tracking(a)
    assert not target.valid[31:41].all()
    assert co.native(a).features.shape==(298,17)
    duplicate=np.repeat(particles(1),4,axis=0)
    np.testing.assert_array_equal(st.branches(duplicate).values,st.branches(duplicate[::-1]).values)
    vis,clean,remove=co.corrupt(a,b'a'*32,'T01',1)
    assert len(clean)==298 and len(co.set_target(clean[remove],vis)['pt'])==remove.sum()


def test_durable_release_exact_dependencies_and_idempotence(tmp_path,monkeypatch):
    from hlt_classification.luka_pretext_structure_hydra import worker
    from hlt_classification.data.cache_contracts import write_immutable_json
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    s=spec(tmp_path); calls=[]; reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    def run(command,**kw):
        calls.append(command); return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    submission.launch(s); p=submission.plan(s,'bootstrap')
    boot=submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert boot['jobs']==dict(setup='9001',release='9002')
    monkeypatch.setattr(submission,'accounting',lambda job:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'): submission.finish(s,'9002')
    assert len(calls)==2 and not (tmp_path/'submission_science').exists()
    acceptance=c.artifact('ACCEPTANCE',minutes={t:60 for t in c.FITS})
    (tmp_path/'setup').mkdir(); write_immutable_json(tmp_path/'setup/acceptance.json',acceptance)
    monkeypatch.setattr(submission,'accounting',lambda job:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:acceptance)
    science=submission.finish(s,'9002'); assert len(calls)==33 and len(science['jobs'])==31
    for task in c.FITS:
        deps=c.dependencies(task); command=science['commands'][task]
        if deps: assert '--dependency=afterok:'+':'.join(science['jobs'][t] for t in deps) in command
        else: assert not any(x.startswith('--dependency=') for x in command)
    assert submission.finish(s,'9002')==science and len(calls)==33
    assert [p['stage'] for p in reviews]==['bootstrap','envelope','science']


def test_runtime_scripts_and_source_guard(monkeypatch):
    import subprocess
    root=Path(__file__).resolve().parents[1]
    for p in campaign.RUNTIME:
        source=(root/p).read_text(); assert 'Fresh_check' not in source
        if p.endswith('.py'): compile(source,p,'exec')
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for p in ('scripts/queue_luka_pretext_structure_hydra.sh','sbatch/run_luka_pretext_structure_hydra.sh'):
            subprocess.run([str(bash),'-n',p],check=True)
    monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='a'*64))
    def git(command,**kwargs):
        if 'ls-files' in command: return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED))
        if 'diff' in command: return SimpleNamespace(stdout='A\tsrc/hlt_classification/luka_pretext_structure_hydra/contracts.py\n')
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(campaign.subprocess,'run',git)
    assert campaign.current(root,'b'*40)['content_hash']=='a'*64
    def bad(command,**kwargs):
        if 'diff' in command: return SimpleNamespace(stdout='M\tsrc/hlt_classification/luka_capacity_hydra/model.py\n')
        return git(command,**kwargs)
    monkeypatch.setattr(campaign.subprocess,'run',bad)
    with pytest.raises(ValueError): campaign.current(root,'b'*40)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_actual_setup_pipeline_with_simulated_site(tmp_path,monkeypatch):
    """Actual data preparation/miniatures/banks/profiles, simulated Slurm/site only."""
    from hlt_classification.luka_pretext_structure_hydra import inputs,worker
    from hlt_classification.luka_teacher_sweep_hydra.transfer import Delta
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.jetclass2_delphes.runner import predict
    small(monkeypatch); train,a=native_cache('train',1804); val,b=native_cache('validation')
    s=spec(tmp_path,1804); s['counts']['validation']=len(val)
    s.update(offline_root='/offline',offline_hash='d'*64)
    raw=SimpleNamespace(manifest=dict(content_hash='1'*64),shards=lambda source,cache:iter([a]))
    delta=SimpleNamespace(manifest=dict(content_hash='2'*64),shards=lambda source,cache:iter([b]))
    monkeypatch.setattr(inputs,'sources',lambda spec:(raw,delta))
    monkeypatch.setattr(inputs,'views',lambda *args:(train,val))
    monkeypatch.setattr(Delta,'__init__',lambda *a:None); monkeypatch.setattr(Delta,'load',lambda *a:train)
    fixture=tmp_path/'fixture'; fixture.mkdir(); native=sm.Native(128).cuda().eval()
    save_state(fixture/'checkpoint.pt',native.state_dict())
    atomic_publish_bytes(fixture/'predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,
        probabilities=predict(native,val,device='cuda',batch_size=128))))
    del native
    class Bundle:
        def __init__(self,*args): pass
        def verify_all(self): pass
        def path(self,key): return fixture/key.split('/')[-1]
    monkeypatch.setattr(inputs,'Bundle',Bundle)
    original_banks=inputs.banks
    monkeypatch.setattr(inputs,'embedding_bank',lambda spec,tr:banks(tr,'C_U',None)['u100'])
    def get_banks(spec,task,tr,**kwargs):
        if task=='C_CE': return banks(tr,task,None)
        return original_banks(spec,task,tr,**kwargs)
    monkeypatch.setattr(inputs,'banks',get_banks)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker.site,'allocation',lambda s:'902'); monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(setup='902'))
    gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)
    for module in (worker,resources):
        monkeypatch.setattr(module,'installed_environment',lambda:s['reference']['environment'])
        monkeypatch.setattr(module,'gpu_identity',lambda:gpu)
    # This Windows host lacks Linux peak RSS and is not Brown; allocations are
    # mocked, while CUDA work, timings, preparation, artifacts and replay are real.
    monkeypatch.setattr(resources,'memory_snapshot',lambda _:dict(peak_rss_bytes=4*2**30,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    worker.run(s,'setup'); acceptance=worker.completed_setup(s)
    assert set(acceptance['minutes'])==set(c.FITS)
    d=load_json(tmp_path/'setup/resource_diagnostic.json')
    assert all(len(p['step_samples'])==12 for p in d['profiles'].values())


@pytest.mark.parametrize('n',[1,2,3,12,64,298])
def test_exclusive_ca_against_pinned_fastjet(n):
    fj=pytest.importorskip('fastjet')
    assert fj.__version__=='3.5.1.5', 'Parity reference must be pinned'
    a=st.canonical(particles(n,103)); jets=[]
    for i,v in enumerate(a):
        p=fj.PseudoJet(*v[:4]); p.set_user_index(i); jets.append(p)
    # R=10 exceeds every separation in these fixtures, so beam removal never
    # changes the exclusive angular tree. Production uses no beam distance.
    sequence=fj.ClusterSequence(jets,fj.JetDefinition(fj.cambridge_algorithm,10.,fj.E_scheme))
    root=sequence.exclusive_jets(1)[0]; expected={}
    def visit(p):
        x,y=fj.PseudoJet(),fj.PseudoJet()
        key=frozenset([p.user_index()]) if not p.has_parents(x,y) else visit(x)|visit(y)
        expected[key]=np.array([p.px(),p.py(),p.pz(),p.E()]); return key
    visit(root)
    indices={tuple(v):i for i,v in enumerate(a)}; actual={}
    def ours(p):
        key=frozenset(indices[v] for v in p.key); actual[key]=p.p4
        if p.left is not None: ours(p.left); ours(p.right)
    ours(st.cluster(a)); assert actual.keys()==expected.keys()
    for key,value in actual.items(): np.testing.assert_allclose(value,expected[key],atol=1e-10,rtol=1e-12)


def test_two_prong_branching_hand_calculation():
    a=np.zeros((2,14)); a[:,:4]=p4_from_coordinates(np.array([3.,1.]),np.zeros(2),np.array([0.,.4]),np.zeros(2))
    target=st.branches(a)
    assert target.values[0]==1 and target.values[5]==0 and target.values[12]==1
    assert target.values[1]==pytest.approx(.25)
    assert target.values[2]==pytest.approx(np.log1p(.4/1e-3))
    assert target.values[3]==pytest.approx(np.log1p(.1/1e-4))
    assert target.values[10]==pytest.approx(np.log(2))
    assert target.values[13]==pytest.approx(.25)
    assert target.values[14]==pytest.approx(np.log1p(.4/1e-3))
    assert target.values[15]==pytest.approx(np.log1p(np.sqrt(6*(1-np.cos(.4)))/4/1e-4))


def test_full30_poor_teacher_still_completes(tmp_path,monkeypatch,data):
    small(monkeypatch); train,val,pt,pv,ts,vs,stats=data; s=spec(tmp_path)
    monkeypatch.setattr(tt,'evaluate',lambda *a:dict(primary=100.,clean_signature=50.))
    model,r=tt.fit(tmp_path/'poor',s,'T07',pt,pv,ts,vs,stats,8,device='cpu')
    assert r['epochs']==30 and r['selected_epoch']==1 and r['selection_loss']==150
    eb.export(tmp_path/'poor_bank',model,pt,s,'T07',file_ref(tmp_path/'poor/selected.pt'),device='cpu')
    assert eb.Bank(tmp_path/'poor_bank',train,s,'T07',file_ref(tmp_path/'poor/selected.pt')).manifest['technical'] is False


def test_array_budget_checked_before_allocation(tmp_path):
    train,a=native_cache('train'); requested=[]
    def reserve(size): requested.append(size); raise OSError('Insufficient reviewed quota')
    with pytest.raises(OSError,match='quota'):
        tb.prepare_raw(tmp_path/'budget',lambda:iter([a]),train,'OFFLINE','a'*64,reserve=reserve)
    assert requested[0]>a['values'].nbytes
    assert not list((tmp_path/'budget').glob('*.npy*'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_science_workers_and_read_only_partial_report(tmp_path,monkeypatch,capsys):
    from hlt_classification.luka_pretext_structure_hydra import inputs,worker,reporting
    from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes
    small(monkeypatch); train,a=native_cache('train'); val,b=native_cache('validation'); s=spec(tmp_path)
    raw=SimpleNamespace(manifest=dict(content_hash='1'*64),shards=lambda source,cache:iter([a]))
    delta=SimpleNamespace(manifest=dict(content_hash='2'*64),shards=lambda source,cache:iter([b]))
    monkeypatch.setattr(inputs,'sources',lambda spec:(raw,delta))
    (tmp_path/'setup').mkdir(); inputs.prepare(s,train,val)
    u=banks(train,'C_U',None)['u100']; monkeypatch.setattr(inputs,'embedding_bank',lambda *a:u)
    monkeypatch.setattr(inputs,'views',lambda *a:(train,val))
    fixtures=tmp_path/'reference'; fixtures.mkdir(); paths={}
    for name,strength in (('M0HLT',0.),('OFFLINE',.9),('matched',.2),('old',.3)):
        probabilities=np.full((22,11),(1-strength)/11,np.float32)
        probabilities[np.arange(22),val.labels]+=strength
        path=fixtures/(name+'.npz'); atomic_publish_bytes(path,deterministic_npz_bytes(dict(
            identities=val.identities,labels=val.labels,probabilities=probabilities)))
        paths['reference/'+name+'/predictions.npz']=path
    subset=fixtures/'subset.npz'; atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    paths['reference/historical/subset.npz']=subset
    s['reused']=dict(extra_control_predictions=dict(path=str(paths['reference/old/predictions.npz'])))
    bundle=SimpleNamespace(path=lambda key:paths[key],json=lambda key:u.manifest)
    monkeypatch.setattr(inputs,'Bundle',lambda *a:bundle)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    job=['901']; gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)
    monkeypatch.setattr(worker,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.site,'allocation',lambda _:job[0]); monkeypatch.setattr(worker.site,'cpu_allocation',lambda _:job[0])
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    monkeypatch.setattr(worker,'completed_setup',lambda _:dict(gpu=gpu))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(T07='901',E07='902',C_U='903',report='904'))
    monkeypatch.setattr(reporting,'status',lambda _:dict(T07=['COMPLETED','0:0'],E07=['COMPLETED','0:0']))
    worker.run(s,'T07'); job[0]='902'; worker.run(s,'E07'); job[0]='903'; worker.run(s,'C_U')
    assert load_json(tmp_path/'E07/startup_memory.json')['real_targets'] is True
    before={str(p):file_ref(p) for p in tmp_path.rglob('*') if p.is_file()}
    result=reporting.results(s); progress=reporting.progress(s)
    assert before=={str(p):file_ref(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert len(result['pending'])==27 and set(progress['best'])=={'T07','E07','C_U'}
    assert all(r['delta_C_CE'] is None for r in result['rows'])
    assert 'epoch 30' in capsys.readouterr().out
    job[0]='904'; worker.run(s,'report'); assert worker.receipt(s,'report')['gpu'] is None
    with pytest.raises(FileExistsError): worker.run(s,'report')


def test_validation_physical_export_real_root_roundtrip(all_prepared,tmp_path,monkeypatch):
    from hlt_classification.luka_pretext_structure_hydra import transfer
    from hlt_classification.luka_u100_pair_transfer import bundle as old_transfer
    from hlt_classification.luka_fullsim import regular_cache
    from hlt_classification.data.cache_contracts import write_immutable_json
    gate,m,p,inv,splits=all_prepared; prepared_root=Path(gate['root'])/'prepared'
    proof=dict(prepared=p['content_hash'],cache=m['content_hash'])
    origin=c.artifact('FIXTURE',root=str(tmp_path/'old'))
    path=tmp_path/'origin.json'; write_immutable_json(path,origin)
    monkeypatch.setattr(old_transfer,'authenticate',lambda *a:proof)
    monkeypatch.setattr(transfer,'COUNTS',p['counts'])
    out=tmp_path/'delta'; result=transfer.export(prepared_root,path,out,dict(content_hash='f'*64))
    raw=SimpleNamespace(manifest=dict(origin_proof=proof,conventions=p['conventions'],columns='p4_charge_category_tracking_mm_validity'))
    delta=transfer.ValidationDelta(out,result['manifest_hash'],raw)
    assert all(r['path'].startswith(('HLT/','OFFLINE/')) for r in delta.manifest['files'])
    for source,coordinate in (('HLT','D000'),('OFFLINE','OFFLINE')):
        cache=regular_cache.load(prepared_root,role='validation',coordinate=coordinate,max_ram_bytes=2**30,authenticated=(m,p,inv,splits))
        physical=tb.prepare_raw(tmp_path/source,lambda:list(delta.shards(source,cache)),cache,source,result['manifest_hash'])
        for i in range(len(cache)):
            actual=co.batch([physical.jet(i)]); expected=cache.batch(np.array([i]))
            for key in ('features','vectors','mask'): np.testing.assert_array_equal(actual[key],expected[key])
        with pytest.raises(PermissionError): list(delta.shards(source,SimpleNamespace(role='train')))
    member=out/delta.manifest['files'][0]['path']
    with member.open('r+b') as f: f.seek(-1,2); f.write(b'\xff')
    with pytest.raises(ValueError): list(delta.shards('HLT',regular_cache.load(prepared_root,role='validation',coordinate='D000',max_ram_bytes=2**30,authenticated=(m,p,inv,splits))))
