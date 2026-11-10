"""Conditional subclass targets, unchanged native inference and durable Hydra execution."""
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
from hlt_classification.luka_subclass_hydra import (
    contracts as c, model, targets, losses, training, inputs, resources, campaign, submission, worker, reporting, reuse)


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


def banks(s,tr=None,va=None):
    tr = cache() if tr is None else tr
    rng=np.random.default_rng(45)
    source=Path(s['root'])/'teacher_bank.json'
    if not source.exists(): write_immutable_json(source,dict(content_hash='o'*64))
    s['teacher_refs']=dict(bank=file_ref(source),checkpoint=dict(sha256='k'*64))
    u=SimpleNamespace(targets=rng.normal(size=(len(tr),128)).astype(np.float32),
        identities=tr.identities.copy(),labels=tr.labels.copy(),
        manifest=dict(content_hash=s['embedding_hash'],role='train',coordinate='U100'))
    q=np.zeros((len(tr),10),np.float32);q[tr.labels==0,:]=.1;q[tr.labels!=0,:5]=.2
    q[:,0]+=.1;q/=q.sum(1,keepdims=True)
    b=SimpleNamespace(targets=q,identities=tr.identities.copy(),labels=tr.labels.copy(),
        manifest=c.artifact('SUBCLASS_BANK',parents=targets.parents(s),role='train',coordinate='OFFLINE'))
    return dict(u100=u,subclass=b)


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


def test_registry_independent_bounded_ten_fit_dag(tmp_path):
    s=spec(tmp_path); boot=submission.plan(s,'bootstrap'); env=submission.plan(s,'envelope')
    assert set(c.WEIGHTS.values())=={.03,.1,.3,.5,1,2,5,10,40,100}
    assert len({str(c.cell(t)) for t in c.FITS})==10
    assert len(boot['commands'])==2 and len(env['commands'])==11
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


@pytest.mark.parametrize('task',c.FITS)
def test_full_thirty_epoch_fit_reload_and_role_firewall(tmp_path,monkeypatch,task):
    small(monkeypatch); s=spec(tmp_path); tr,va=cache(),cache('validation'); bb=banks(s,tr,va)
    native,r=training.fit(tmp_path/task,s,task,tr,va,bb,device='cpu')
    assert len(r['history'])==30 and r['native_reload'] and not r['oracle']
    assert all(k.startswith('mod.') for k in native.state_dict())
    assert any(k.startswith('predictor.') for k in torch.load(tmp_path/task/'selected_full.pt',weights_only=True))
    assert r['history'][0]['gradient_diagnostic']['ce_norm']>0
    assert r['history'][0]['subclass_coefficient']==pytest.approx(c.WEIGHTS[task]/3)
    assert r['history'][2]['subclass_coefficient']==c.WEIGHTS[task]
    training.validate_fit(tmp_path/task,s,task)
    bad=copy.deepcopy(bb); bad['u100'].manifest['role']='validation'
    with pytest.raises(PermissionError): training.check_inputs(s,task,tr,va,bad,False)
    with pytest.raises(PermissionError): training.check_inputs(s,task,tr,va,dict(bb,it=bb['u100']),False)
    path=tmp_path/task/'selected.pt'; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): training.validate_fit(tmp_path/task,s,task)


@pytest.mark.parametrize('device',('cpu','cuda'))
def test_original_u100_complete_update_parity(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s=spec(tmp_path); bb=banks(s)
    r=resources.baseline(s,cache(),bb['u100'],device)
    assert r['model_update_equal'] and r['adamw_update_equal'] and r['fp32_forward_equal']


@pytest.mark.parametrize('task',c.FITS)
def test_full_depth_cuda_full_and_partial_batches(tmp_path,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s=spec(tmp_path); tr=cache(); bb=banks(s)
    m=model.new_model(s,task).cuda(); opt=training.optimizer(m)
    assert len(m.mod.blocks)==8 and len(m.mod.cls_blocks)==2
    for size in (128,11):
        ix=np.arange(size)%len(tr)
        values,d,stats=training.step(m,task,opt,tr.batch(ix),bb,ix,1.,'cuda',diagnose=True)
        assert np.isfinite(list(values.values())).all() and d['weighted_subclass_norm']>0
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
    result=submission.finish(s,'8002'); assert len(result['jobs'])==11 and len(calls)==13
    assert submission.finish(s,'8002')==result and len(calls)==13
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
        if 'diff' in cmd:return SimpleNamespace(stdout='A\tsrc/hlt_classification/luka_subclass_hydra/model.py\n')
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
        for p in ('scripts/queue_luka_subclass_hydra.sh','sbatch/run_luka_subclass_hydra.sh'):
            subprocess.run([str(bash),'-n',p],check=True)


def prepared_setup(tmp_path,monkeypatch,device='cpu'):
    from test_luka_attention import cache as data_cache
    from hlt_classification.luka_embedding_aux import targets as off
    small(monkeypatch);s=spec(tmp_path);tr=data_cache(coordinate='D000',rows=110);va=cache('validation')
    s['counts']['train']=len(tr);bb=banks(s,tr,va)
    root=tmp_path/'T00';root.mkdir();bankdir=root/'bank';bankdir.mkdir();(tmp_path/'setup').mkdir(exist_ok=True)
    teacher=model.Native();torch.save(teacher.state_dict(),root/'selected.pt')
    refs=dict(checkpoint=file_ref(root/'selected.pt'))
    rng=np.random.default_rng(51)
    atomic_publish_bytes(bankdir/'targets.npz',deterministic_npz_bytes(dict(identities=tr.identities,
        labels=tr.labels,targets=rng.normal(size=(len(tr),128)).astype(np.float32),mean=np.zeros(128),scale=np.ones(128))))
    m=off.artifact('BANK',parents=dict(recipe=off.recipe()['content_hash'],prepared=tr.foundation_sha256,
        teacher=refs['checkpoint']['sha256']),teacher=refs['checkpoint'],role='train',coordinate='OFFLINE',width=128,
        rows=len(tr),payload=file_ref(bankdir/'targets.npz',root=bankdir))
    write_immutable_json(bankdir/'manifest.json',m)
    for name in ('report','receipt'):write_immutable_json(root/(name+'.json'),dict(fixture=True))
    refs.update(bank=file_ref(bankdir/'manifest.json'),targets=file_ref(bankdir/'targets.npz'),
        report=file_ref(root/'report.json'),receipt=file_ref(root/'receipt.json'))
    s['teacher_refs']=refs
    class Bundle:
        def __init__(self,*a):pass
        def json(self,key):return dict(content_hash=s['embedding_hash'],role='train',coordinate='U100',
            parents=dict(prepared=tr.foundation_sha256,teacher=s['imported']['teacher']['content_hash']))
    monkeypatch.setattr(inputs,'Bundle',Bundle)
    offline=inputs.offline_bank(s,tr);bb['subclass']=targets.build(s,tr,offline)
    return s,tr,va,bb


def test_complete_real_functions_setup_on_synthetic_cuda(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,tr,va,bb=prepared_setup(tmp_path,monkeypatch,'cuda')
    # Input authentication/Slurm are external fixture boundaries. The production
    # frozen TRAIN targets, updates, setup measurements and fits run here.
    monkeypatch.setattr(inputs,'views',lambda *a:(tr,va))
    monkeypatch.setattr(inputs,'banks',lambda *a:bb)
    monkeypatch.setattr(inputs,'offline_bank',lambda *a:None)
    monkeypatch.setattr(targets,'build',lambda *a:bb['subclass'])
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
    s,tr,va,bb=prepared_setup(tmp_path,monkeypatch)
    task='S100'; folder=tmp_path/task; folder.mkdir()
    _,r=training.fit(folder/'fit',s,task,tr,va,bb,device='cpu')
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
    old['offline_root']=str(tmp_path/'offline')
    path=parent/'campaign_spec.json'; write_immutable_json(path,old)
    refs={'fixture':'teacher'}
    monkeypatch.setattr(campaign,'original',lambda p:load_json(p)); monkeypatch.setattr(campaign,'references',lambda p:refs)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='f'*64))
    monkeypatch.setattr(campaign,'load_anchors',lambda s:dict(M0HLT=.95,OFFLINE=.98))
    monkeypatch.setattr(inputs,'validate_sources',lambda s:None); monkeypatch.setattr(campaign.site,'environment',lambda *a:None)
    monkeypatch.setattr(campaign,'storage',lambda s:None)
    s=campaign.create(tmp_path/'new',path,project,'b'*40,5); campaign.check(s)
    assert s['control']==old['control'] and len(s['nodes'])==10
    assert {n['initialization_seed'] for n in s['nodes'].values()}=={23}
    with pytest.raises(ValueError,match='fresh separate'): campaign.create(tmp_path/'new',path,project,'b'*40,5)
    bad=copy.deepcopy(s); bad['nodes']['S100']['sampler_seed']+=1
    with pytest.raises(ValueError): campaign.check(reseal(bad))


@pytest.mark.parametrize('task',('S0p5','S100'))
def test_science_worker_startup_discard_full_fit_and_exact_receipt(tmp_path,monkeypatch,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,tr,va,bb=prepared_setup(tmp_path,monkeypatch,'cuda')
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


@pytest.mark.parametrize('resource',('peak_rss_bytes','gpu_peak_bytes'))
def test_worker_preserves_training_peak_across_deployment_counter_reset(tmp_path,monkeypatch,resource):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,tr,va,bb=prepared_setup(tmp_path,monkeypatch,'cuda')
    a=resources.accept(s,diagnostic(s)); task='S100'
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


@pytest.mark.parametrize('task',c.FITS)
def test_conditional_objective_matches_hand_formula_and_target_detach(task):
    labels=torch.tensor([0,1,10]);logits=torch.randn(3,11,requires_grad=True)
    p=torch.randn(3,128,requires_grad=True);t=torch.randn_like(p,requires_grad=True)
    z=torch.randn(3,60,requires_grad=True);q=torch.zeros(3,10)
    q[0,:]=.1;q[1:,:5]=.2;q.requires_grad_()
    loss,parts=losses.objective(task,logits,labels,p,t,z,q,1.5)
    kl=[]
    for i,(offset,k) in enumerate(((0,10),(10,5),(55,5))):
        kl.append((q[i,:k].detach()*(q[i,:k].detach().log()-z[i,offset:offset+k].log_softmax(0))).sum())
    expected=torch.nn.functional.cross_entropy(logits,labels)+5*torch.nn.functional.smooth_l1_loss(p,t.detach())
    expected+=c.WEIGHTS[task]*.5*torch.stack(kl).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward();assert q.grad is None and t.grad is None and p.grad.norm()>0
    for i,(offset,k) in enumerate(((0,10),(10,5),(55,5))):
        assert z.grad[i,offset:offset+k].norm()>0
        z.grad[i,offset:offset+k]=0
    assert z.grad.count_nonzero()==0
    assert c.coefficient(task,3)==c.coefficient(task,30)==c.WEIGHTS[task]
    for value in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):c.coefficient(task,value)


def test_gradient_probes_are_observation_only(tmp_path,monkeypatch):
    small(monkeypatch);s=spec(tmp_path);bb=banks(s);tr=cache();task='S100'
    results=[]
    for diagnose in (False,True):
        m=model.new_model(s,task);opt=training.optimizer(m)
        rng=torch.get_rng_state().clone()
        parts,diag,stats=training.step(m,task,opt,tr.batch(np.arange(22)),bb,np.arange(22),3.,'cpu',diagnose=diagnose)
        results.append((copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone(),parts,stats))
        assert (diag is not None)==diagnose
    from hlt_classification.luka_embedding_curriculum_hydra.resources import equal_state
    assert equal_state(results[0],results[1])
    m=model.new_model(s,task);m.capture=True
    _forward(m,tr.batch(np.arange(22)),'cpu',bf16=False)
    q=torch.as_tensor(bb['subclass'].targets)
    losses.conditional_kl(m.subclass_logits,torch.as_tensor(tr.labels),q).backward()
    assert all(p.grad is None for p in m.mod.fc.parameters())
    assert all(p.grad is None for p in m.predictor.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.shared_parameters(m))


@pytest.mark.parametrize('bad',('role','coordinate','identity','target','mean','checkpoint','prototype','recipe'))
def test_real_target_bank_corruption_and_role_rejection(tmp_path,monkeypatch,bad):
    s,tr,va,bb=prepared_setup(tmp_path,monkeypatch)
    q=bb['subclass'];assert not q.targets.flags.writeable
    assert all(inputs.fit_parents(s,t)['subclass']==q.manifest['content_hash'] for t in c.FITS)
    with pytest.raises(PermissionError):targets.Bank(s,va)
    with pytest.raises(FileExistsError):targets.build(s,tr,inputs.offline_bank(s,tr))
    if bad in ('role','coordinate','identity'):
        wrong=copy.deepcopy(tr)
        if bad=='role':wrong.role='validation'
        elif bad=='coordinate':wrong.coordinate_name='U100'
        else:wrong.identities=wrong.identities[::-1]
        with pytest.raises((ValueError,PermissionError)):targets.Bank(s,wrong)
    elif bad=='recipe':
        p=tmp_path/'setup/subclasses/manifest.json';v=load_json(p);v['recipe']['population']+=1
        p.write_text(__import__('json').dumps(reseal(v)))
        with pytest.raises(ValueError):targets.check_manifest(s)
    else:
        key={'target':'targets','mean':'targets','checkpoint':'checkpoint'}.get(bad)
        path=(Path(s['teacher_refs'][key]['path']) if key else tmp_path/'setup/subclasses/prototypes.npz')
        path.write_bytes(path.read_bytes()+b'changed')
        with pytest.raises(ValueError):
            targets.Bank(s,tr) if key is None else inputs.offline_bank(s,tr)


def test_no_quality_gate_for_subclass_statistics(tmp_path,monkeypatch):
    small(monkeypatch);s=spec(tmp_path);tr,va=cache(),cache('validation');bb=banks(s)
    monkeypatch.setattr(losses,'statistics',lambda *a:dict(agreement=0.,entropy=0.))
    _,r=training.fit(tmp_path/'poor',s,'S100',tr,va,bb,device='cpu')
    assert len(r['history'])==30 and all(e['train_subclass']['agreement']==0 for e in r['history'])


def test_pinned_teacher_authentication_only_t00(tmp_path,monkeypatch):
    old=dict(contract='LUKA_TEACHER_SWEEP_SPEC/v1',content_hash='a'*64,project_dir=str(tmp_path))
    path=tmp_path/'campaign_spec.json';write_immutable_json(path,old);calls=[]
    def run(cmd,**kwargs):
        calls.append((cmd,kwargs));return SimpleNamespace(returncode=0,stdout='SUBCLASS_PARENT='+old['content_hash']+'\n',stderr='')
    monkeypatch.setattr(reuse.subprocess,'run',run)
    assert reuse.original(path)==old
    code=calls[0][0][4]
    assert "validate_teacher(s,'T00')" in code and "'S00'" not in code and 'check_storage=False' in code
    assert calls[0][1]['env']['PYTHONNOUSERSITE']=='1'
    monkeypatch.setattr(reuse.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='stale source'))
    with pytest.raises(ValueError,match='stale source'):reuse.original(path)
