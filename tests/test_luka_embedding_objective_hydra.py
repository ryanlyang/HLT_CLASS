"""Scientific geometry, gradient routing, native parity and durable Hydra execution."""
from pathlib import Path
from types import SimpleNamespace
import copy
import subprocess
import numpy as np
import pytest
import torch
from torch import nn
from test_luka_u100_pretrain import cache
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, with_content_hash,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_embedding_objective_hydra import (
    contracts as c, model, geometry, losses, training, inputs, resources, campaign, submission, worker, reporting)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    node = dict(initialization_seed=23, sampler_seed=29)
    return dict(campaign.registration(), root=str(root), project_dir='/project', content_hash='c'*64,
        anchor=dict(evidence=dict(prepared='a'*64)), reference=dict(environment={'fixture':True}),
        bundle_root='/bundle', environment_dir='/env', counts=dict(train=22,validation=22),
        nodes={t:dict(node,node_id=t) for t in c.FITS}, imported=dict(node=node,teacher=dict(content_hash='t'*64)),
        teacher_context=dict(encoder_hash='e'*64,delta_hash='d'*64), embedding_hash='u'*64,
        recovery_anchors=dict(M0HLT=.95,OFFLINE=.98), original50k_rows=22,
        control=dict(report=dict(selected_pass=1,validation_history=[])))


def small(monkeypatch):
    from hlt_classification.jetclass2_delphes import model as native
    original = native.model_config
    cfg = lambda: dict(original(), num_layers=1, num_cls_layers=1, pair_embed_dims=[8,8,8])
    monkeypatch.setattr(native,'model_config',cfg); monkeypatch.setattr(resources,'model_config',cfg)


def banks(s, tr=None, va=None):
    tr, va = (cache() if tr is None else tr), (cache('validation') if va is None else va)
    rng = np.random.default_rng(45)
    def bank(view):
        manifest = (dict(content_hash=s['embedding_hash'],role='train',coordinate='U100') if view.role=='train'
                    else c.artifact('VALIDATION_BANK',parents=inputs.parents(s),role='validation',coordinate='U100'))
        return SimpleNamespace(targets=rng.normal(size=(len(view),128)).astype(np.float32),
            identities=view.identities.copy(),labels=view.labels.copy(),manifest=manifest)
    tb, vb = bank(tr), bank(va)
    arrays, ranks = geometry.statistics(tb.targets,tb.labels,rng.normal(size=(11,128)),rng.normal(size=11))
    arrays['scales'] = np.ones(5,np.float32)
    geom = geometry.Geometry(arrays,c.artifact('GEOMETRY',parents=geometry.parents(s),role='train',rows=len(tr),ranks=ranks))
    return {'u100':tb}, vb, geom


def diagnostic(s):
    return c.artifact('RESOURCES',parents=dict(spec=s['content_hash']),status='complete',error=None,
        environment=s['reference']['environment'],gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30),
        profiles={t:dict(initial_factory_fp32_parity=True,miniature_complete=True,deployment_reload=True,
            initial_factory_max_abs=0.,fp32_equivalence=dict(rtol=2e-5,atol=5e-6),native_architecture_verified=True,
            architecture=dict(parameters=2143615,additional_parameters=0),largest_support_backward=True,
            partial_batch_backward=True,real_targets=True,train_rows=s['counts']['train'],validation_rows=s['counts']['validation'],
            step_samples=[.3]*12,step_seconds=float(np.mean([.3]*12)),validation_seconds=1.,probe_seconds=1.,load_seconds=1.,
            memory=dict(peak_rss_bytes=4*2**30,gpu_peak_bytes=5*2**30)) for t in c.FITS})


def reseal(value):
    return with_content_hash({k:v for k,v in value.items() if k!='content_hash'})


def test_registry_independent_bounded_eight_fit_dag(tmp_path):
    s=spec(tmp_path); boot=submission.plan(s,'bootstrap'); env=submission.plan(s,'envelope')
    assert c.FITS==('R8','R2','R4','R1','R7','R3','R5','R6')
    assert len({str(c.cell(t)) for t in c.FITS})==8
    assert len(boot['commands'])==2 and len(env['commands'])==9
    assert boot['upper_bound_commands']==env['commands']
    assert '--dependency=afterok:${JOB_setup}' in boot['commands'][1]['command']
    for r in env['commands'][:-1]:
        assert not r['dependencies'] and not any(x.startswith('--dependency') for x in r['command'])
        assert '--gres=gpu:nvidia_rtx_a6000:1' in r['command']
    assert env['commands'][-1]['dependencies']==list(c.FITS)
    assert any(x.startswith('--dependency=afterany:') for x in env['commands'][-1]['command'])
    measured=submission.plan(s,'science',resources.accept(s,diagnostic(s)))
    assert submission.within_envelope(env,measured)
    measured['commands'][0]['command'].append('--gres=gpu:2')
    with pytest.raises(PermissionError): submission.within_envelope(env,measured)
    with pytest.raises(ValueError): c.cell('C0')


def test_statistics_respect_readout_common_shift_and_class_permutation(tmp_path):
    rng=np.random.default_rng(2); z=rng.normal(size=(220,128)); labels=np.arange(220)%11
    w=rng.normal(size=(11,128)); b=rng.normal(size=11)
    a,r=geometry.statistics(z,labels,w,b)
    d,q=geometry.statistics(z,labels,w+rng.normal(size=(1,128)),b+8)
    assert r==q and r['class_rank']==10 and r['discriminant_rank']==10
    for k in a: np.testing.assert_allclose(a[k],d[k],rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(a['class_projector']@a['class_projector'],a['class_projector'],atol=1e-6)
    np.testing.assert_allclose(a['discriminant_projector']@a['discriminant_projector'],a['discriminant_projector'],atol=1e-6)
    assert np.linalg.eigvalsh(a['whitener']).max()<=3.00001
    perm=rng.permutation(11); d,_=geometry.statistics(z,perm[labels],w[perm],b[perm])
    for k in ('coordinate','class_projector','discriminant_projector','whitener'):
        np.testing.assert_allclose(a[k],d[k],rtol=1e-5,atol=1e-6)
    with pytest.raises(ValueError): geometry.statistics(z,labels*0,w,b)


def test_degenerate_geometry_and_bounded_importance_are_results():
    a,r=geometry.statistics(np.zeros((22,128)),np.arange(22)%11,np.zeros((11,128)),np.zeros(11))
    assert r==dict(class_rank=0,discriminant_rank=0)
    assert np.all(a['coordinate']==1) and np.max(np.linalg.eigvalsh(a['whitener']))<=3.00001
    w=losses.importance(torch.tensor([[0.,0.,0.],[1e20,0.,1.]]))
    assert torch.allclose(w.mean(-1),torch.ones(2)) and w.min()>=1/16 and w.max()<=16


def test_subspace_loss_ignores_null_error_but_base_keeps_it(tmp_path):
    _,_,g=banks(spec(tmp_path)); g.class_projector.zero_(); g.class_projector[0,0]=1
    target=torch.zeros(2,128); p=torch.zeros_like(target); p[:,1]=2; p.requires_grad_()
    assert losses.component('class_span',p,target,g)==0
    assert losses.component('base',p,target,g)>0
    p2=p.detach().clone(); p2[:,0]=2
    assert losses.component('class_span',p2,target,g)>0


def test_fisher_matches_explicit_class_derivative(tmp_path):
    _,_,g=banks(spec(tmp_path)); p=torch.randn(5,128); t=torch.randn_like(p)
    prob=torch.softmax((t@g.readout.T+g.bias)/2,dim=1)
    gradient=(g.readout.unsqueeze(0)-(prob@g.readout).unsqueeze(1))/2
    expected=(prob.unsqueeze(-1)*gradient.square()).sum(1)
    weighted=(losses.importance(expected)*torch.nn.functional.smooth_l1_loss(p,t,reduction='none')).mean()
    torch.testing.assert_close(losses.component('fisher',p,t,g),weighted)


@pytest.mark.parametrize('task',c.FITS)
def test_objective_target_detach_and_exact_registered_mixture(tmp_path,task):
    _,_,g=banks(spec(tmp_path)); p=torch.randn(22,128,requires_grad=True); t=torch.randn_like(p,requires_grad=True)
    logits=torch.randn(22,11,requires_grad=True); y=torch.arange(22)%11
    g.scales.copy_(torch.tensor([.1,2,3,4,5]))
    total,terms=losses.objective(task,logits,y,p,t,g); cfg=c.cell(task)
    base=losses.component('base',p,t,g); f=cfg['fraction']
    extra=0 if cfg['term'] is None else g.scales[c.TERMS.index(cfg['term'])]*losses.component(cfg['term'],p,t,g)
    torch.testing.assert_close(terms['u100'],5*((1-f)*base+f*extra))
    total.backward(); assert t.grad is None and p.grad.norm()>0 and logits.grad.norm()>0
    assert all(not b.requires_grad for b in g.buffers())


def test_frozen_calibration_matches_initial_gradient_rms_and_does_not_train(tmp_path):
    _,_,g=banks(spec(tmp_path)); p=torch.randn(99,128); t=torch.randn_like(p)
    before=p.clone(); scales,diags=geometry.calibration(p,t,g)
    assert torch.equal(p,before) and not p.requires_grad and len(scales)==5
    for i,name in enumerate(c.TERMS):
        d=diags[name]; assert scales[i]==np.float32(d['scale']) and .1<=d['scale']<=10
        if not d['clipped']:
            assert d['scale']*d['alternative_gradient_rms']==pytest.approx(d['base_gradient_rms'])
    g.class_projector.zero_(); _,d=geometry.calibration(p,t,g)
    assert d['class_span']['degenerate'] and d['class_span']['scale']==1


def test_asymmetric_projection_keeps_ce_head_predictor_and_unused_parameters():
    m=nn.Module(); m.mod=nn.Module(); m.mod.encoder=nn.Parameter(torch.tensor([1.,2.])); m.mod.unused=nn.Parameter(torch.ones(1))
    m.mod.fc=nn.Linear(2,1); m.predictor=nn.Linear(2,1)
    ce=m.mod.encoder[0]+2*m.mod.encoder[1]+sum(p.sum() for p in m.mod.fc.parameters())
    aux=-3*m.mod.encoder[0]-m.mod.encoder[1]+sum(p.sum() for p in m.predictor.parameters())
    d=losses.backward(m,ce+aux,dict(ce=ce,u100=aux),protect=True)
    torch.testing.assert_close(m.mod.encoder.grad,torch.tensor([-1.,3.]))
    assert d['projected'] and d['dot_before']==-5 and abs(d['dot_after'])<1e-12
    assert m.mod.unused.grad is None
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in (*m.mod.fc.parameters(),*m.predictor.parameters()))


@pytest.mark.parametrize('cg,ag',([1.,2.],[0.,0.]))
def test_projection_nonconflict_or_zero_keeps_auxiliary(cg,ag):
    # Independent hand examples: same-direction and zero CE.
    ce=torch.tensor([cg,ag]); aux=2*ce+torch.tensor([0.,1.])
    adjusted,d=losses.project([ce],[aux]); assert not d['projected']; torch.testing.assert_close(adjusted[0],aux)


@pytest.mark.parametrize('task',c.FITS)
def test_full_thirty_epoch_fit_reload_and_role_firewall(tmp_path,monkeypatch,task):
    small(monkeypatch); s=spec(tmp_path); tr,va=cache(),cache('validation'); bb,vb,g=banks(s,tr,va)
    native,r=training.fit(tmp_path/task,s,task,tr,va,bb,g,vb,device='cpu')
    assert len(r['history'])==30 and r['native_reload'] and not r['oracle']
    assert all(k.startswith('mod.') for k in native.state_dict())
    assert any(k.startswith('predictor.') for k in torch.load(tmp_path/task/'selected_full.pt',weights_only=True))
    assert r['history'][0]['gradient_diagnostic']['ce_norm']>0
    assert r['history'][0]['projection']['updates']==(1 if c.cell(task)['protect_ce'] else 0)
    training.validate_fit(tmp_path/task,s,task)
    bad=copy.deepcopy(bb); bad['u100'].manifest['role']='validation'
    with pytest.raises(PermissionError): training.check_inputs(s,task,tr,va,bad,vb,g,False)
    with pytest.raises(PermissionError): training.check_inputs(s,task,tr,va,dict(bb,it=bb['u100']),vb,g,False)
    bad=copy.deepcopy(vb); bad.identities=bad.identities[::-1]
    with pytest.raises(PermissionError): training.evaluate(model.new_model(s,task),va,bad,g,'cpu')
    path=tmp_path/task/'selected.pt'; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): training.validate_fit(tmp_path/task,s,task)


@pytest.mark.parametrize('device',('cpu','cuda'))
def test_original_u100_complete_update_parity(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s=spec(tmp_path); bb,_,_=banks(s)
    r=resources.baseline(s,cache(),bb['u100'],device)
    assert r['model_update_equal'] and r['adamw_update_equal'] and r['fp32_forward_equal']


@pytest.mark.parametrize('task',c.FITS)
def test_full_depth_cuda_full_and_partial_batches(tmp_path,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s=spec(tmp_path); tr=cache(); bb,vb,g=banks(s); g=g.cuda()
    m=model.new_model(s,task).cuda(); opt=training.optimizer(m)
    assert len(m.mod.blocks)==8 and len(m.mod.cls_blocks)==2
    for size in (128,11):
        ix=np.arange(size)%len(tr)
        values,d=training.step(m,task,opt,tr.batch(ix),bb,ix,1.,'cuda',geometry=g,diagnose=True)
        assert np.isfinite(list(values.values())).all() and d['auxiliary_norm']>0
        if c.cell(task)['protect_ce']:
            assert d['dot_after']>=-1e-6*max(1,d['ce_norm']*d['auxiliary_norm'])
    m.capture=False; m.eval(); native=model.Native().cuda().eval(); native.load_state_dict(m.native_state(),strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(_forward(m,tr.batch(np.arange(11)),'cuda',bf16=True),
                                   _forward(native,tr.batch(np.arange(11)),'cuda',bf16=True),rtol=0,atol=0)


@pytest.mark.parametrize('bad',('memory','time','missing','gpu','negative','targets','samples'))
def test_resource_acceptance_never_gates_on_science_quality(tmp_path,bad):
    s=spec(tmp_path); d=diagnostic(s); t=c.FITS[0]
    assert set(resources.accept(s,d)['minutes'])==set(c.FITS)
    if bad=='memory': d['profiles'][t]['memory']['peak_rss_bytes']=32*2**30
    elif bad=='time': d['profiles'][t].update(step_samples=[1e6]*12,step_seconds=1e6)
    elif bad=='missing': del d['profiles'][t]
    elif bad=='gpu': d['gpu']['name']='NVIDIA L40'
    elif bad=='negative': d['profiles'][t]['load_seconds']=-1
    elif bad=='targets': d['profiles'][t]['real_targets']=False
    else: d['profiles'][t]['step_samples']=[.3]
    with pytest.raises((ValueError,MemoryError,RuntimeError)): resources.accept(s,reseal(d))


def fake_scheduler(monkeypatch):
    calls=[]; reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command,**kwargs):
        calls.append(command); return SimpleNamespace(stdout=str(8000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_auto_release_exact_ids_and_no_duplicate_submission(tmp_path,monkeypatch):
    s=spec(tmp_path); (tmp_path/'setup').mkdir(); a=resources.accept(s,diagnostic(s))
    write_immutable_json(tmp_path/'setup/acceptance.json',a)
    calls,reviews=fake_scheduler(monkeypatch); submission.launch(s); p=submission.plan(s,'bootstrap')
    with pytest.raises(PermissionError): submission.launch(s,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    result=submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert result['jobs']==dict(setup='8001',release='8002')
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    result=submission.finish(s,'8002'); assert len(result['jobs'])==9 and len(calls)==11
    assert submission.finish(s,'8002')==result and len(calls)==11
    assert [p['stage'] for p in reviews]==['bootstrap','envelope','science']


def test_failed_setup_and_ambiguous_submission_preserved(tmp_path,monkeypatch):
    s=spec(tmp_path); calls,_=fake_scheduler(monkeypatch); submission.launch(s)
    p=submission.plan(s,'bootstrap'); submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'): submission.finish(s,'8002')
    assert len(calls)==2 and not (tmp_path/'submission_science').exists()
    other=tmp_path/'other'; other.mkdir(); s=spec(other); submission.launch(s)
    (other/'submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous'):
        submission.launch(s,execute=True,plan_hash=submission.plan(s,'bootstrap')['content_hash'],authorization=c.AUTHORIZATION)


def test_source_guard_preserves_existing_runtime(monkeypatch):
    root=Path(__file__).resolve().parents[1]; monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='a'*64))
    def git(cmd,**k):
        if 'ls-files' in cmd:return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED))
        if 'diff' in cmd:return SimpleNamespace(stdout='A\tsrc/hlt_classification/luka_embedding_objective_hydra/model.py\n')
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(campaign.subprocess,'run',git)
    assert campaign.current(root,'b'*40)['content_hash']=='a'*64
    def bad(cmd,**k):
        return SimpleNamespace(stdout='M\tsrc/hlt_classification/luka_embedding_aux/model.py\n') if 'diff' in cmd else git(cmd,**k)
    monkeypatch.setattr(campaign.subprocess,'run',bad)
    with pytest.raises(ValueError): campaign.current(root,'b'*40)


def test_runtime_shell_and_python_parse():
    root=Path(__file__).resolve().parents[1]
    for path in campaign.RUNTIME:
        source=(root/path).read_text(); assert 'Fresh_check' not in source
        if path.endswith('.py'): compile(source,path,'exec')
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for p in ('scripts/queue_luka_embedding_objective_hydra.sh','sbatch/run_luka_embedding_objective_hydra.sh'):
            subprocess.run([str(bash),'-n',p],check=True)


def prepared_setup(tmp_path, monkeypatch, device='cpu'):
    small(monkeypatch); s=spec(tmp_path); tr,va=cache(),cache('validation'); bb,_,_=banks(s,tr,va)
    (tmp_path/'setup').mkdir(exist_ok=True)
    teacher=model.Native().eval().requires_grad_(False)
    monkeypatch.setattr(inputs.donor,'encoder',lambda tc,d:teacher.to(d))
    monkeypatch.setattr(inputs.donor,'oracle_validation',lambda tc:cache('validation','U100'))
    monkeypatch.setattr(inputs,'moments',lambda s:(np.zeros(128,np.float64),np.ones(128,np.float64)))
    vb=inputs.build(s,va,device); g=geometry.build(s,tr,bb['u100'],device)
    return s,tr,va,bb,vb,g


def test_build_geometry_validation_roles_hashes_and_calibration(tmp_path,monkeypatch):
    s,tr,va,bb,vb,g=prepared_setup(tmp_path,monkeypatch)
    assert g.manifest['rows']==22 and g.manifest['role']=='train'
    assert vb.manifest['usage']=='evaluation_only'
    assert inputs.fit_parents(s,'R2')['geometry']==g.manifest['content_hash']
    before={k:v.clone() for k,v in g.named_buffers()}
    m=model.new_model(s,'R2'); opt=training.optimizer(m); ix=np.arange(22)
    training.step(m,'R2',opt,tr.batch(ix),bb,ix,1.,'cpu',geometry=g)
    assert all(torch.equal(before[k],v) for k,v in g.named_buffers())
    native,r=training.fit(tmp_path/'R4',s,'R4',tr,va,bb,g,vb,device='cpu',technical=True)
    assert r['parents']==inputs.fit_parents(s,'R4')
    with pytest.raises(PermissionError): geometry.build(s,va,vb,'cpu')
    with pytest.raises(PermissionError): inputs.validation_bank(s,tr)
    m=load_json(tmp_path/'setup/geometry.json'); m['role']='validation'
    (tmp_path/'setup/geometry.json').write_text(__import__('json').dumps(reseal(m)))
    with pytest.raises(ValueError,match='role'): geometry.load(s)


def test_geometry_payload_corruption_rejected(tmp_path,monkeypatch):
    s,*_=prepared_setup(tmp_path,monkeypatch)
    p=tmp_path/'setup/geometry.npz'; p.write_bytes(p.read_bytes()+b'corruption')
    with pytest.raises(ValueError): geometry.load(s)


def test_diagnostics_do_not_change_native_eval_probabilities(tmp_path,monkeypatch):
    s,tr,va,bb,vb,g=prepared_setup(tmp_path,monkeypatch)
    m=model.new_model(s,'R2').eval(); original={k:v.clone() for k,v in m.state_dict().items()}
    rng=torch.get_rng_state(); a,d=training.evaluate(m,va,vb,g,'cpu')
    b=training.predict(m,va,device='cpu',batch_size=128)
    assert np.array_equal(a,b) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items())
    assert d['predicted_teacher_ce']>0


def test_complete_real_functions_setup_on_synthetic_cuda(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,tr,va,bb,vb,g=prepared_setup(tmp_path,monkeypatch,'cuda')
    # Input authentication/Slurm are external fixture boundaries. The production
    # geometry, validation targets, updates, setup measurements and fits run here.
    monkeypatch.setattr(inputs,'views',lambda *a:(tr,va))
    monkeypatch.setattr(inputs,'banks',lambda *a:bb)
    monkeypatch.setattr(inputs,'build',lambda *a:vb)
    monkeypatch.setattr(geometry,'build',lambda *a:g)
    monkeypatch.setattr(resources,'installed_environment',lambda:s['reference']['environment'])
    gpu=diagnostic(s)['gpu']; monkeypatch.setattr(resources,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(resources,'memory_snapshot',lambda torch:dict(peak_rss_bytes=4*2**30,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(resources,'replay',lambda *a:None)
    # Fixture contains22 rows; repeat only benchmark dispatch rows, not training population.
    base_batch=tr.batch; tr.batch=lambda ix:base_batch(np.asarray(ix)%len(tr))
    original_step=training.step
    def step(m,t,opt,raw,banks,ix,pos,device,**kw):
        return original_step(m,t,opt,raw,banks,np.asarray(ix)%len(tr),pos,device,**kw)
    monkeypatch.setattr(training,'step',step)
    # Timing sampler requires at least14 full batches: use a deterministic tiled
    # permutation here; each miniature still authenticates the22 unique rows.
    class Rng:
        def permutation(self,n): return np.tile(np.arange(n),100)
    original_rng=np.random.default_rng
    monkeypatch.setattr(resources.np.random,'default_rng',lambda seed=None: Rng() if seed==20261108 else original_rng(seed))
    a=resources.measure(s)
    assert set(a['minutes'])==set(c.FITS)
    resources.check_baseline(s)
    for task in c.FITS:
        r,_=training.validate_fit(tmp_path/'setup'/('mini_'+task),s,task,technical=True)
        assert r['parents']==inputs.fit_parents(s,task) and r['native_reload']


def test_reporting_reads_completed_and_partial_without_writes(tmp_path,monkeypatch):
    s,tr,va,bb,vb,g=prepared_setup(tmp_path,monkeypatch)
    task='R8'; folder=tmp_path/task; folder.mkdir()
    _,r=training.fit(folder/'fit',s,task,tr,va,bb,g,vb,device='cpu')
    write_immutable_json(folder/'receipt.json',{'fixture':'Slurm boundary'})
    arrays=dict(identities=va.identities,labels=va.labels,probabilities=np.full((22,11),1/11,np.float32))
    for name in ('M0HLT','OFFLINE','matched','control'):
        a=copy.deepcopy(arrays)
        if name=='OFFLINE': a['probabilities']=.1*a['probabilities']+.9*np.eye(11,dtype=np.float32)[va.labels]
        atomic_publish_bytes(tmp_path/(name+'.npz'),deterministic_npz_bytes(a))
    atomic_publish_bytes(tmp_path/'subset.npz',deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    class Bundle:
        def __init__(self,*a): pass
        def path(self,k): return tmp_path/('subset.npz' if 'subset' in k else k.split('/')[1]+'.npz')
    monkeypatch.setattr(inputs,'Bundle',Bundle); monkeypatch.setattr(reporting,'reuse_check',lambda *a,**k:tmp_path/'control.npz')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{})
    monkeypatch.setattr(worker,'receipt',lambda *a:None)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=reporting.results(s); progress=reporting.progress(s)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert result['pending']==[t for t in c.FITS if t!=task]
    assert progress['best'][task]['epoch']==r['selected_epoch']
    rows=[row for row in result['rows'] if row['model']==task]
    assert len(rows)==2 and all(row['delta_C0']==pytest.approx(row['auc']-.5) for row in rows)
    p=folder/'fit/epoch-01.json'; p.write_text('{}')
    with pytest.raises(ValueError): reporting.progress(s)


def test_registration_reuses_matched_control_and_rejects_drift(tmp_path,monkeypatch):
    import sys
    parent=tmp_path/'parent'; parent.mkdir(); project=tmp_path/'code'; project.mkdir()
    old=spec(parent); old.update(environment_dir=str(Path(sys.prefix).resolve()),project_dir=str(parent),
        bundle_root=str(tmp_path/'bundle'),reused=dict(C0={'fixture':'control'}),qualification={'fixture':True})
    path=parent/'campaign_spec.json'; write_immutable_json(path,old)
    tc=dict(delta_root=str(tmp_path/'delta'),encoder_root=str(tmp_path/'encoder'))
    monkeypatch.setattr(campaign,'original',lambda p:load_json(p)); monkeypatch.setattr(campaign,'context',lambda p:tc)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='f'*64))
    monkeypatch.setattr(campaign,'load_anchors',lambda s:dict(M0HLT=.95,OFFLINE=.98))
    monkeypatch.setattr(inputs,'validate_sources',lambda s:None); monkeypatch.setattr(campaign.site,'environment',lambda *a:None)
    monkeypatch.setattr(campaign,'storage',lambda s:None)
    s=campaign.create(tmp_path/'new',path,project,'b'*40,5); campaign.check(s)
    assert s['control']==old['reused']['C0'] and len(s['nodes'])==8
    assert {n['initialization_seed'] for n in s['nodes'].values()}=={23}
    with pytest.raises(ValueError,match='fresh separate'): campaign.create(tmp_path/'new',path,project,'b'*40,5)
    bad=copy.deepcopy(s); bad['nodes']['R8']['sampler_seed']+=1
    with pytest.raises(ValueError): campaign.check(reseal(bad))


@pytest.mark.parametrize('task',('R4','R8'))
def test_science_worker_startup_discard_full_fit_and_exact_receipt(tmp_path,monkeypatch,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,tr,va,bb,vb,g=prepared_setup(tmp_path,monkeypatch,'cuda')
    a=resources.accept(s,diagnostic(s)); calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.site,'allocation',lambda s:'901')
    monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(worker,'gpu_identity',lambda:a['gpu'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    monkeypatch.setattr(worker,'memory_snapshot',lambda torch:dict(peak_rss_bytes=4*2**30,gpu_peak_bytes=5*2**30))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{task:'901'})
    monkeypatch.setattr(inputs,'views',lambda *a:(tr,va)); monkeypatch.setattr(inputs,'banks',lambda *a:bb)
    original=training.fit
    def fit(*args,**kwargs):
        calls.append((Path(args[0]).name,kwargs.get('technical',False)))
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'fit',fit)
    r=worker.run(s,task)
    assert r['job_id']=='901' and calls==[('startup',True),('fit',False)]
    assert worker.receipt(s,task)==r
    parent=load_json(tmp_path/task/'fit/training_report.json')['parents']
    assert parent==inputs.fit_parents(s,task)
    before=(tmp_path/task/'receipt.json').read_bytes()
    with pytest.raises(FileExistsError): worker.run(s,task)
    assert (tmp_path/task/'receipt.json').read_bytes()==before
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{task:'902'})
    with pytest.raises(ValueError,match='ownership'): worker.receipt(s,task)


def test_selected_diagnostics_are_not_checkpoint_quality_gates(tmp_path,monkeypatch):
    s,tr,va,bb,vb,g=prepared_setup(tmp_path,monkeypatch)
    original=training.evaluate
    def evaluate(*args,**kwargs):
        probs,diag=original(*args,**kwargs)
        # Very poor but finite auxiliary diagnostics are not a technical failure.
        diag.update(embedding=1e10,class_span=1e10,complement=1e10,predicted_teacher_ce=1e10,
                    predicted_teacher_accuracy=0.)
        return probs,diag
    monkeypatch.setattr(training,'evaluate',evaluate)
    _,r=training.fit(tmp_path/'bad_quality',s,'R1',tr,va,bb,g,vb,device='cpu')
    assert r['epochs']==30 and all(row['auxiliary_validation']['embedding']==1e10 for row in r['history'])


@pytest.mark.parametrize('resource',('peak_rss_bytes','gpu_peak_bytes'))
def test_worker_preserves_training_peak_across_deployment_counter_reset(tmp_path,monkeypatch,resource):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,tr,va,bb,vb,g=prepared_setup(tmp_path,monkeypatch,'cuda')
    a=resources.accept(s,diagnostic(s)); task='R8'
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.site,'allocation',lambda s:'901')
    monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(worker,'gpu_identity',lambda:a['gpu'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{task:'901'})
    monkeypatch.setattr(inputs,'views',lambda *a:(tr,va))
    monkeypatch.setattr(inputs,'banks',lambda *a:bb)
    # The post-deployment process counter is small, but training exceeded its
    # envelope. The worker must preserve the earlier peak and reject the fit.
    memory=dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30)
    peak=dict(memory); peak[resource]=64*2**30
    monkeypatch.setattr(worker,'memory_snapshot',lambda torch:memory.copy())
    monkeypatch.setattr(resources,'miniature',lambda *a,**k:dict(fit_memory=peak))
    with pytest.raises(MemoryError,match='memory allowance'): worker.run(s,task)
    saved=load_json(tmp_path/task/'startup_memory.json')
    assert saved['memory'][resource]==64*2**30
    assert not (tmp_path/task/'fit').exists()
