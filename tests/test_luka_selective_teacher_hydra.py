"""Physical field isolation, native training, target lineage and durable release."""
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
from test_luka_u100_pair_correction import dataset, prepared, all_prepared
from test_luka_frozen_adapter_hydra import export_origin
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, with_content_hash,
    deterministic_npz_bytes, atomic_publish_bytes)
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_selective_teacher_hydra import (
    contracts as c, model, targets, training, resources, campaign, submission, worker, transfer,
    inputs, reporting, views, validation_cache, reuse)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    return dict(campaign.registration(), root=str(root), project_dir='/project', content_hash='c'*64,
        anchor=dict(evidence=dict(prepared='a'*64)), counts=dict(train=22, validation=22),
        nodes={t: dict(initialization_seed=23, sampler_seed=29) for t in c.FITS},
        recovery_anchors=dict(M0HLT=.95, OFFLINE=.98), reference=dict(environment={'fixture': True}),
        validation_hash='d'*64, bundle_root='/bundle', validation_root='/validation', environment_dir='/env')


def banks(train, task):
    rng = np.random.default_rng(8)
    return {k: SimpleNamespace(targets=rng.normal(size=(len(train),128)).astype(np.float32),
        identities=train.identities, labels=train.labels, manifest=c.artifact('TEST_BANK', role='train', name=k))
        for k in c.weights(task)} if task in c.STUDENTS else {}


def diagnostic(s):
    return c.artifact('RESOURCES', parents=dict(spec=s['content_hash']), status='complete', error=None,
        environment=s['reference']['environment'], gpu=dict(name='NVIDIA RTX A6000', compute_capability=[8,6], total_memory_bytes=48*2**30),
        memory=dict(peak_rss_bytes=4*2**30, gpu_peak_bytes=5*2**30), profiles={t: dict(
            initial_native_fp32_parity=True, miniature_complete=True, deployment_reload=True,
            largest_support_backward=True, partial_batch_backward=True, train_rows=s['counts']['train'],
            validation_rows=s['counts']['validation'], step_samples=[.3]*12, step_seconds=float(np.mean([.3]*12)),
            validation_seconds=1., probe_seconds=1., load_seconds=1., export_seconds=1.) for t in resources.PROFILE_TASKS})


def reseal(d):
    return with_content_hash({k:v for k,v in d.items() if k != 'content_hash'})


@pytest.mark.parametrize('task', (*c.TEACHERS, '', 'KIT'))
def test_feature_splice_exactly_equals_physical_hybrid_frontend(task):
    from hlt_classification.cms2jc2_response.bridge import Particles
    from hlt_classification.cms_proxy_ladder.views import build_view
    from hlt_classification.literature_context.transform import build_inputs
    from hlt_classification.jetclass2_delphes.cache import RamBlock
    rng = np.random.default_rng(133)
    def particles(n):
        xyz = rng.normal(size=(n,3))*7
        p4 = np.column_stack((xyz,np.sqrt((xyz*xyz).sum(1)+.3)))
        cat = rng.integers(0,6,n)
        charge = np.where(np.isin(cat,(0,3,4)),1,0)
        valid = (rng.random((n,4))>.3) & (charge[:,None]!=0)
        valid[:,2:] &= valid[:,:2]
        tr = rng.normal(size=(n,4))*.1; tr[:,2:]=abs(tr[:,2:]); tr[~valid]=0
        return Particles(p4,charge,cat,tr,valid,tuple(map(str,range(n))))
    h, off = particles(7), particles(5)
    mapping = np.array([3,-1,0,4,2,-1,1],np.int32)
    u = build_view(identity='a'*64,proxy=h,offline=off,coordinate='U100',mapping=mapping)
    # Deliberately mixed identity/tracking can violate the *physical* Particles
    # applicability contract. Evaluate the unchanged frontend on raw fields;
    # this is a declared feature ablation, not a new detector particle endpoint.
    class Fields(SimpleNamespace):
        def __len__(self): return len(self.p4)
        @property
        def pt(self): return np.hypot(self.p4[:,0],self.p4[:,1])
    physical = Fields(p4=u.p4 if 'K' in task else h.p4,
        charge=u.charge if 'I' in task else h.charge, category=u.category if 'I' in task else h.category,
        tracking=u.tracking if 'T' in task else h.tracking, valid=u.valid if 'T' in task else h.valid)
    def block(p):
        x = build_inputs(p)
        return RamBlock(0,np.array([0,len(p)],np.int64),x.features,x.vectors,np.zeros((1,32),np.uint8),np.zeros(1,np.int64))
    hb,ub=block(h),block(u); expected=block(physical); actual=views.splice(hb,ub,task)
    np.testing.assert_array_equal(actual.features,expected.features)
    np.testing.assert_array_equal(actual.vectors,expected.vectors)
    assert np.any(hb.features[:,15:17]!=ub.features[:,15:17])
    np.testing.assert_array_equal(hb.features,block(h).features)  # sources remain immutable
    for field in 'IT':
        if field not in task: np.testing.assert_array_equal(actual.features[:,c.GROUPS[field]],hb.features[:,c.GROUPS[field]])
    # Unmatched local physical fields remain HLT. K relative features may change
    # on an unmatched particle because the jet's visible axis changed.
    np.testing.assert_array_equal(actual.features[mapping<0,5:15],hb.features[mapping<0,5:15])
    bad=copy.deepcopy(ub); bad.identities[0,0]=1
    with pytest.raises(ValueError): views.splice(hb,bad,task)


def test_hybrid_fingerprint_and_role_firewall():
    h,u=cache(),cache(coordinate='U100')
    x=views.hybrid(h,u,'IT'); a=views.fingerprint(x)
    x.blocks[0].features[0,11]+=.1
    assert a['sha256']!=views.fingerprint(x)['sha256']
    with pytest.raises(PermissionError): views.hybrid(h,cache('validation','U100'),'IT')
    with pytest.raises(ValueError): views.splice(h.blocks[0],u.blocks[0],'UNKNOWN')


def test_twelve_fits_decreasing_target_order_fixed_weights_and_dag(tmp_path):
    s=spec(tmp_path); boot=submission.plan(s,'bootstrap'); env=submission.plan(s,'envelope')
    assert len(c.TEACHERS)==4 and len(c.STUDENTS)==8
    assert [len(c.STUDENTS[t])+1 for t in c.STUDENTS]==[6,5,4,3,2,2,2,2]
    assert [r['task_id'] for r in env['commands']]==[*c.FITS,'report']
    assert boot['upper_bound_commands']==env['commands']
    assert '--dependency=afterok:${JOB_setup}' in boot['commands'][1]['command']
    for row in env['commands'][:-1]:
        t=row['task_id']; deps=[k for k in c.STUDENTS[t] if k in c.TEACHERS] if t in c.STUDENTS else []
        assert row['dependencies']==deps
        assert '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
        if t in c.STUDENTS:
            w=c.weights(t); assert w['u100']==5 and sum(w.values())==pytest.approx(10)
    assert env['commands'][-1]['dependencies']==list(c.FITS)
    accepted=resources.accept(s,diagnostic(s)); measured=submission.plan(s,'science',accepted)
    assert submission.within_envelope(env,measured)
    changed=copy.deepcopy(measured); changed['commands'][4]['command'].append('--gres=gpu:2')
    with pytest.raises(PermissionError): submission.within_envelope(env,changed)
    accepted['minutes']['S6']=2881
    with pytest.raises(ValueError): submission.plan(s,'science',accepted)


@pytest.mark.parametrize('task',c.FITS)
def test_native_parity_gradient_routes_and_inference_without_targets(tmp_path,monkeypatch,task):
    tiny(monkeypatch); s=spec(tmp_path)
    m=model.new_model(s,task).eval(); rng=torch.random.get_rng_state().clone()
    torch.manual_seed(23); native=DelphesParticleTransformer().eval()
    assert torch.equal(rng,torch.random.get_rng_state())
    for k,v in native.state_dict().items(): torch.testing.assert_close(m.native_state()[k],v,rtol=0,atol=0)
    raw=cache().batch(np.arange(22)); y=torch.from_numpy(raw['labels']); m.capture=True
    logits=_forward(m,raw,'cpu',bf16=False)
    torch.testing.assert_close(logits,_forward(native,raw,'cpu',bf16=False),rtol=0,atol=0)
    tt={k:torch.randn(22,128,requires_grad=True) for k in c.term_names(task)-{'ce'}}
    loss,terms=model.objective(m,task,logits,y,**tt)
    assert set(terms)==c.term_names(task)
    diag=training.gradients(terms,m)
    assert diag['weighted_encoder_norms']['ce']>0
    for k,v in terms.items():
        if k=='ce': continue
        assert diag['weighted_encoder_norms'][k]>0
        gs=torch.autograd.grad(v,list(m.mod.fc.parameters()),retain_graph=True,allow_unused=True)
        assert all(g is None for g in gs)
        expected=c.weights(task)[k]*torch.nn.functional.smooth_l1_loss(m.predictions[k],tt[k].detach())
        torch.testing.assert_close(v,expected,rtol=0,atol=0)
    loss.backward(); assert all(v.grad is None for v in tt.values())
    m.capture=False
    with torch.no_grad(): torch.testing.assert_close(_forward(m,raw,'cpu',bf16=False),_forward(native,raw,'cpu',bf16=False),rtol=0,atol=0)
    with pytest.raises(PermissionError): model.objective(m,task,logits,y,forbidden=torch.zeros(22,128))


@pytest.mark.parametrize('task',c.TEACHERS)
def test_two_epoch_fit_export_and_selective_bank_lineage(tmp_path,monkeypatch,task):
    tiny(monkeypatch); s=spec(tmp_path); tr,va=cache(coordinate=task),cache('validation',task)
    native,r=training.fit(tmp_path/task,s,task,tr,va,{},device='cpu',technical=True)
    ref=file_ref(tmp_path/task/'selected.pt')
    targets.export(tmp_path/'bank',native,tr,teacher_ref=ref,device='cpu',spec_hash=s['content_hash'])
    b=targets.Bank(tmp_path/'bank',cache(),teacher_ref=ref,coordinate=task,spec_hash=s['content_hash'])
    assert b.targets.shape==(22,128) and r['native_reload'] and r['oracle']
    training.validate_fit(tmp_path/task,s,task,technical=True)
    for bad in ('validation','wrong_view','wrong_spec'):
        with pytest.raises((ValueError,PermissionError)):
            targets.Bank(tmp_path/'bank',cache('validation') if bad=='validation' else cache(),teacher_ref=ref,
                coordinate='IT' if bad=='wrong_view' and task!='IT' else 'K' if bad=='wrong_view' else task,
                spec_hash='b'*64 if bad=='wrong_spec' else s['content_hash'])
    with pytest.raises(PermissionError): targets.export(tmp_path/'forbidden',native,va,teacher_ref=ref,device='cpu',spec_hash=s['content_hash'])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_thirty_teacher_student_pipeline(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); s=spec(tmp_path); tr,va=cache(coordinate='K'),cache('validation','K')
    native,r=training.fit(tmp_path/'teacher',s,'K',tr,va,{},device=device)
    ref=file_ref(tmp_path/'teacher/selected.pt')
    targets.export(tmp_path/'bank',native,tr,teacher_ref=ref,device=device,spec_hash=s['content_hash'])
    h,hv=cache(),cache('validation'); bb=banks(h,'S2K')
    bb['k']=targets.Bank(tmp_path/'bank',h,teacher_ref=ref,coordinate='K',spec_hash=s['content_hash'])
    _,r=training.fit(tmp_path/'student',s,'S2K',h,hv,bb,device=device)
    assert len(r['history'])==30 and not r['oracle']
    training.validate_fit(tmp_path/'student',s,'S2K')
    p=tmp_path/'bank/targets.npz'; x=bytearray(p.read_bytes()); x[-1]^=1; p.write_bytes(x)
    with pytest.raises(ValueError): targets.Bank(tmp_path/'bank',h,teacher_ref=ref,coordinate='K',spec_hash=s['content_hash'])


@pytest.mark.parametrize('task',c.FITS)
def test_full_native_batch128_and_partial11_cuda(tmp_path,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s=spec(tmp_path); tr=cache(coordinate=task if task in c.TEACHERS else 'D000'); bb=banks(tr,task)
    m=model.new_model(s,task).cuda(); opt=training.optimizer(m)
    assert len(m.mod.blocks)==8 and len(m.mod.cls_blocks)==2
    for size in (128,11):
        ix=np.arange(size)%len(tr)
        vals,_=training.step(m,task,opt,tr.batch(ix),bb,ix,1.,'cuda')
        assert set(vals)==c.term_names(task) and np.isfinite(list(vals.values())).all()
    m.eval(); m.capture=False; native=DelphesParticleTransformer().cuda().eval()
    native.load_state_dict(m.native_state(),strict=True)
    with torch.no_grad(): torch.testing.assert_close(_forward(native,tr.batch(np.arange(11)),'cuda',bf16=False),
        _forward(m,tr.batch(np.arange(11)),'cuda',bf16=False),rtol=0,atol=0)


@pytest.mark.parametrize('bad',['memory','time','missing','gpu','nonfinite'])
def test_resource_failures_not_quality(tmp_path,bad):
    s=spec(tmp_path); d=diagnostic(s)
    assert resources.accept(s,d)['minutes'].keys()==set(c.FITS)
    if bad=='memory': d['memory']['peak_rss_bytes']=32*2**30
    elif bad=='time': d['profiles']['S6'].update(step_samples=[1e6]*12,step_seconds=1e6)
    elif bad=='missing': del d['profiles']['S2I']
    elif bad=='gpu': d['gpu']['name']='NVIDIA L40'
    else: d['profiles']['K']['step_samples'][0]=-1
    with pytest.raises((ValueError,MemoryError,RuntimeError)): resources.accept(s,reseal(d))


def fake_scheduler(monkeypatch):
    calls=[]; reviewed=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviewed.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command,**kwargs):
        calls.append(command); return SimpleNamespace(stdout=str(8000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviewed


def test_durable_release_exact_multi_teacher_ids_and_idempotent_completed_submission(tmp_path,monkeypatch):
    s=spec(tmp_path); (tmp_path/'setup').mkdir(); a=resources.accept(s,diagnostic(s))
    write_immutable_json(tmp_path/'setup/acceptance.json',a)
    calls,reviewed=fake_scheduler(monkeypatch)
    submission.launch(s); p=submission.plan(s,'bootstrap')
    with pytest.raises(PermissionError): submission.launch(s,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)
    ledger=submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert ledger['jobs']==dict(setup='8001',release='8002')
    monkeypatch.setattr(submission,'accounting',lambda job:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    result=submission.finish(s,'8002')
    assert len(result['jobs'])==13 and len(calls)==15
    for student,teachers in c.STUDENTS.items():
        ids=[result['jobs'][t] for t in teachers if t in c.TEACHERS]
        assert '--dependency=afterok:'+':'.join(ids) in result['commands'][student]
    assert submission.finish(s,'8002')==result and len(calls)==15
    assert [x['stage'] for x in reviewed]==['bootstrap','envelope','science']


def test_failed_setup_and_ambiguous_submission_do_not_duplicate(tmp_path,monkeypatch):
    s=spec(tmp_path); calls,_=fake_scheduler(monkeypatch); submission.launch(s)
    p=submission.plan(s,'bootstrap')
    submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda job:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'): submission.finish(s,'8002')
    assert len(calls)==2 and not (tmp_path/'submission_science').exists()
    (tmp_path/'other').mkdir()
    other=spec(tmp_path/'other'); submission.launch(other)
    (Path(other['root'])/'submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous'):
        submission.launch(other,execute=True,plan_hash=submission.plan(other,'bootstrap')['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==2


def test_runtime_and_scripts_parse():
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in campaign.RUNTIME)
    for p in campaign.RUNTIME:
        text=(root/p).read_text(); assert 'Fresh_check' not in text
        if p.endswith('.py'): compile(text,p,'exec')
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for p in ('scripts/queue_luka_selective_teacher_hydra.sh','sbatch/run_luka_selective_teacher_hydra.sh'):
            subprocess.run([str(bash),'-n',p],check=True)


def test_validation_delta_rebuild_exact_bytes_relocation_and_corruption(all_prepared,tmp_path,monkeypatch):
    import shutil
    gate,m,p,inv,splits=all_prepared
    source=Path(gate['root'])/'prepared'; copied=tmp_path/'prepared'
    shutil.copytree(source,copied)
    meta=(m,p,inv,splits)
    anchor=dict(prepared_root=str(copied),raw_container=gate['input_container'],prepared=p['content_hash'],cache=m['content_hash'])
    # Only this isolated synthetic copy is removed; original fixture remains.
    for ref in m['blocks']['validation']['U100']: (copied/ref['path']).unlink()
    rebuilt=validation_cache.load(anchor,meta,16*2**30)
    assert len(rebuilt)==p['counts']['validation']
    assert not any((copied/ref['path']).exists() for ref in m['blocks']['validation']['U100'])
    proof=dict(counts=p['counts'],prepared=p['content_hash'],cache=m['content_hash'],campaign='b'*64)
    old=c.artifact('TEST_ORIGIN',root=str(tmp_path/'old'),anchor=dict(evidence=anchor))
    path=tmp_path/'origin.json'; write_immutable_json(path,old)
    monkeypatch.setattr(transfer.old_transfer,'authenticate',lambda *a:proof)
    monkeypatch.setattr(transfer,'COUNTS',p['counts'])
    monkeypatch.setattr(transfer.regular_cache,'metadata',lambda _:meta)
    monkeypatch.setattr(transfer.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    out=tmp_path/'delta'; result=transfer.export(path,out,dict(content_hash='e'*64))
    assert result['payload_bytes']==sum(r['bytes'] for r in m['blocks']['validation']['U100'])
    assert not (out/'inputs/train').exists()
    monkeypatch.setattr(transfer,'Bundle',lambda _:SimpleNamespace(manifest=dict(origin=proof),json=lambda _:m))
    monkeypatch.setattr(transfer,'metadata',lambda _:meta)
    d=transfer.Delta(out,result['manifest_hash'],'/bundle'); d.verify_all()
    moved=d.load(dict(counts=p['counts'],memory_mb=32000),'validation')
    for a,b in zip(moved.blocks,rebuilt.blocks):
        np.testing.assert_array_equal(a.features,b.features); np.testing.assert_array_equal(a.vectors,b.vectors)
    for role in ('train','test'):
        with pytest.raises(PermissionError): d.load({},role)
    with pytest.raises((ValueError,PermissionError)): d.path('../outside')
    with pytest.raises(ValueError): transfer.Delta(out,'0'*64,'/bundle')
    ref=m['blocks']['validation']['U100'][0]; dest=copied/ref['path']; dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(b'corrupt')
    with pytest.raises(ValueError): validation_cache.load(anchor,meta,16*2**30)
    dest=out/next(iter(d.rows)); x=bytearray(dest.read_bytes()); x[-1]^=1; dest.write_bytes(x)
    with pytest.raises(ValueError): d.verify_all()


def test_current_source_rejects_modified_donors(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(campaign,'__file__',str(tmp_path/'src/hlt_classification/luka_selective_teacher_hydra/campaign.py'))
    change=['A\tsrc/hlt_classification/luka_selective_teacher_hydra/model.py']
    def git(args,**kwargs): return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED) if 'ls-files' in args else '\n'.join(change))
    monkeypatch.setattr(campaign.subprocess,'run',git)
    campaign.current(tmp_path,'a'*40)
    change[0]='M\tsrc/hlt_classification/luka_teacher_sweep_hydra/model.py'
    with pytest.raises(ValueError,match='donor runtime'): campaign.current(tmp_path,'a'*40)


def test_old_teacher_sweep_authentication_uses_original_process(tmp_path,monkeypatch):
    project=tmp_path/'old_code'; project.mkdir(); path=tmp_path/'old.json'
    old=dict(contract='LUKA_TEACHER_SWEEP_SPEC/v1',project_dir=str(project),content_hash='a'*64)
    path.write_text(json.dumps(old)); calls=[]
    def run(args,**kwargs):
        calls.append((args,kwargs)); return SimpleNamespace(returncode=0,stdout='SELECTIVE_PARENT='+old['content_hash']+'\n',stderr='')
    monkeypatch.setattr(reuse.subprocess,'run',run)
    assert reuse.original(path)==old
    cmd,kw=calls[0]; assert kw['cwd']==str(project) and kw['env']['PYTHONPATH']==str(project/'src')
    assert 'worker.validate_teacher' in cmd[4] and "('T00','S00')" in cmd[4]
    monkeypatch.setattr(reuse.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='not complete'))
    with pytest.raises(ValueError,match='not complete'): reuse.original(path)


def validation_chain(tmp_path,delta,digest,**bindings):
    """Group A owns contrastive keys; its cross-message parent owns U100 validation."""
    from hlt_classification.luka_cross_message_hydra.contracts import artifact as cross_artifact
    from hlt_classification.luka_next_supervision_hydra.contracts import artifact as group_artifact
    shared=dict(bundle_root=str(tmp_path/'bundle'),reference={},imported={},anchor={})
    shared.update(bindings)
    cross=cross_artifact('SPEC',version=3,delta_root=str(delta),delta_hash=digest,counts=c.COUNTS,**shared)
    cross_path=tmp_path/'cross.json'; write_immutable_json(cross_path,cross)
    contrast=tmp_path/'contrast'; contrast.mkdir()
    write_immutable_json(contrast/'manifest.json',with_content_hash(dict(
        contract='LUKA_U100_CONTRASTIVE_HYDRA_TRANSFER/v1',schema_version=1)))
    group=group_artifact('SPEC',parents=dict(parent=cross['content_hash']),parent_spec=file_ref(cross_path),
        scope='group_A_only',counts=c.COUNTS,delta_root=str(contrast),delta_hash='c'*64,**shared)
    group_path=tmp_path/'group.json'; write_immutable_json(group_path,group)
    return dict(group_a_spec=file_ref(group_path),parents=dict(group_a=group['content_hash']),**shared)


def test_existing_validation_is_default_and_new_export_is_explicit(tmp_path,monkeypatch):
    old=validation_chain(tmp_path,tmp_path,'d'*64)
    assert campaign.validation_source(old)==('existing_frozen_adapter',str(tmp_path),'d'*64)
    assert campaign.validation_source(old,tmp_path,'e'*64)==('selective_export',str(tmp_path.resolve()),'e'*64)
    with pytest.raises(ValueError): campaign.validation_source(old,tmp_path,None)
    with pytest.raises(ValueError): transfer.selected('unknown',tmp_path,'d'*64,'/bundle')
    import hlt_classification.luka_frozen_adapter_hydra.inputs as legacy
    seen=[]
    monkeypatch.setattr(transfer,'Bundle',lambda root:root)
    monkeypatch.setattr(legacy,'Delta',lambda r,h,b:SimpleNamespace(root=Path(r),manifest={'legacy':True},verify_all=lambda:seen.append('verified')))
    monkeypatch.setattr(legacy,'oracle_validation',lambda s:seen.append((s['delta_root'],s['delta_hash'])))
    d=transfer.selected('existing_frozen_adapter',tmp_path,'d'*64,'/bundle')
    assert d.verify_all()=={'legacy':True}
    d.load({},'validation'); assert seen==['verified',(str(tmp_path),'d'*64)]
    with pytest.raises(PermissionError): d.load({},'train')


def test_registration_roundtrip_binds_reused_controls_and_default_delta(tmp_path,monkeypatch):
    import sys
    delta=tmp_path/'delta'; delta.mkdir(); project=tmp_path/'code'; project.mkdir()
    parent=tmp_path/'parent.json'; parent.write_text('{}')
    old={k:{'fixture':k} for k in campaign.COPIED}
    old.update(validation_chain(tmp_path,delta,'d'*64,reference=dict(environment={})))
    old.update(root=str(tmp_path/'old'),project_dir=str(tmp_path/'old_code'),environment_dir=sys.prefix,
        content_hash='a'*64,control={'fixed':'C0'})
    refs={'fixed':'T00/S00'}
    monkeypatch.setattr(campaign,'original',lambda _:old)
    monkeypatch.setattr(campaign,'references',lambda _:refs)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='b'*64))
    monkeypatch.setattr(campaign,'nodes',lambda _:dict(fixture=True))
    monkeypatch.setattr(campaign,'anchors',lambda _:dict(M0HLT=.95,OFFLINE=.98))
    monkeypatch.setattr(campaign.site,'environment',lambda _:None)
    monkeypatch.setattr(campaign.reuse,'check',lambda *a,**kw:None)
    monkeypatch.setattr(campaign,'storage',lambda _:None)
    seen=[]
    def choose(kind,root,digest,bundle):
        seen.append((kind,root,digest)); return SimpleNamespace(root=Path(root),verify_all=lambda:None)
    monkeypatch.setattr(campaign,'selected',choose)
    root=tmp_path/'new'
    s=campaign.create(root,parent,None,None,project,'e'*40,9)
    campaign.check(s)
    assert s['validation_kind']=='existing_frozen_adapter' and s['reused']==refs
    assert seen==[('existing_frozen_adapter',str(delta),'d'*64)]*2
    assert s['parents']['teacher_sweep']==old['content_hash']
    changed=copy.deepcopy(s); changed['validation_root']=str(tmp_path/'contrast'); changed=reseal(changed)
    (root/'campaign_spec.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Existing validation source differs'): campaign.check(changed)
    bad=copy.deepcopy(s); bad['reused']={'substituted':'wrong'}; bad=reseal(bad)
    (root/'campaign_spec.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='registration differs'): campaign.check(bad)


def test_validation_source_uses_real_frozen_delta_not_contrastive_keys(export_origin,tmp_path,monkeypatch):
    from hlt_classification.luka_frozen_adapter_hydra import inputs as legacy
    f=export_origin; dest=tmp_path/'validation'
    exported=legacy.export(f.spec_path,dest)
    old=validation_chain(tmp_path,dest,exported['manifest_hash'])
    monkeypatch.setattr(transfer,'Bundle',lambda _:f.bundle)
    kind,root,digest=campaign.validation_source(old)
    d=transfer.selected(kind,root,digest,old['bundle_root'])
    assert d.verify_all()['contract']=='LUKA_FROZEN_ADAPTER_HYDRA_TRANSFER/v1'
    assert d.root==dest.resolve()
    assert d.original.path('prepared/inputs/validation/U100/0000.npz').read_bytes()==f.payload.read_bytes()
    # Never accept the key-only manifest, even if someone binds its correct hash.
    contrast=tmp_path/'contrast'; contrast_hash=load_json(contrast/'manifest.json')['content_hash']
    with pytest.raises(ValueError,match='contract mismatch'):
        transfer.selected(kind,contrast,contrast_hash,old['bundle_root'])
    shard=d.original.path('prepared/inputs/validation/U100/0000.npz'); shard.write_bytes(b'corrupted')
    with pytest.raises(ValueError): d.verify_all()


@pytest.mark.parametrize('problem',[
    'group_file','cross_file','group_parent_hash','cross_parent_hash','cross_contract',
    'group_scope','cross_counts','group_bundle_root','cross_bundle_root','cross_anchor','cross_imported'])
def test_validation_source_rejects_changed_parent_chain(tmp_path,problem):
    old=validation_chain(tmp_path,tmp_path,'d'*64)
    group_path=Path(old['group_a_spec']['path']); group=load_json(group_path)
    cross_path=Path(group['parent_spec']['path']); cross=load_json(cross_path)
    if problem=='group_file': group_path.write_text('{}')
    elif problem=='cross_file': cross_path.write_text('{}')
    elif problem=='group_parent_hash': old['parents']['group_a']='0'*64
    else:
        if problem=='cross_parent_hash': group['parents']['parent']='0'*64
        elif problem=='group_scope': group['scope']='group_B_only'
        elif problem=='group_bundle_root': group['bundle_root']='/other'
        else:
            field=problem.removeprefix('cross_')
            cross[field]='wrong' if field=='contract' else dict(train=1,validation=1) if field=='counts' else '/other'
            cross=reseal(cross); cross_path.write_text(json.dumps(cross))
            group['parent_spec']=file_ref(cross_path); group['parents']['parent']=cross['content_hash']
        group=reseal(group); group_path.write_text(json.dumps(group))
        old['group_a_spec']=file_ref(group_path); old['parents']['group_a']=group['content_hash']
    with pytest.raises(ValueError): campaign.validation_source(old)


@pytest.mark.parametrize('violation',['role','coordinate','identity','population','overlap','missing_target'])
def test_student_input_and_target_firewall(tmp_path,violation):
    s=spec(tmp_path); tr,va=cache(),cache('validation'); bb=banks(tr,'S6')
    if violation=='role': tr.role='validation'
    elif violation=='coordinate': tr.coordinate_name='K'
    elif violation=='identity': bb['k'].identities=tr.identities[::-1]
    elif violation=='population': s['counts']['train']=100
    elif violation=='missing_target': del bb['it']
    else: va.identities=tr.identities.copy()
    with pytest.raises((ValueError,PermissionError)): training.check_inputs(s,'S6',tr,va,bb,False)


def test_six_target_join_reuses_original_offline_contract(tmp_path,monkeypatch):
    from hlt_classification.luka_embedding_aux import targets as offline
    from hlt_classification.context_fusion.worker import save_state
    tiny(monkeypatch); s=spec(tmp_path); h=cache(); native=DelphesParticleTransformer()
    checkpoint=tmp_path/'offline.pt'; save_state(checkpoint,native.state_dict()); offref=file_ref(checkpoint)
    offline.export(tmp_path/'old_bank',native,cache(coordinate='OFFLINE'),teacher_ref=offref,device='cpu')
    s['reused']=dict(offline_checkpoint=offref,offline_bank=file_ref(tmp_path/'old_bank/manifest.json'))
    for task in c.TEACHERS:
        folder=tmp_path/task; (folder/'fit').mkdir(parents=True)
        save_state(folder/'fit/selected.pt',native.state_dict())
        targets.export(folder/'bank',native,cache(coordinate=task),teacher_ref=file_ref(folder/'fit/selected.pt'),device='cpu',spec_hash=s['content_hash'])
    calls=[]; monkeypatch.setattr(worker,'validate_teacher',lambda _,t:calls.append(t))
    u=banks(h,'S2K')['u100']; monkeypatch.setattr(inputs,'embedding_bank',lambda *a:u)
    actual=inputs.banks(s,'S6',h)
    assert set(actual)=={'u100','k','i','t','it','offline'} and calls==list(c.TEACHERS)
    assert actual['offline'].manifest['coordinate']=='OFFLINE'
    assert actual['offline'].manifest['contract']!=actual['k'].manifest['contract']
    for t in c.TEACHERS: assert actual[t.lower()].manifest['coordinate']==t
    technical=inputs.banks(s,'S6',h,technical=True)
    assert not technical['k'].targets.any() and technical['offline'].manifest==actual['offline'].manifest


def test_poor_finite_teacher_still_completes_thirty(tmp_path,monkeypatch):
    tiny(monkeypatch); s=spec(tmp_path); evaluate=training.evaluate_probabilities
    def poor(y,p):
        value=evaluate(y,p); value['macro_ovr_auc']=0.; return value
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    _,r=training.fit(tmp_path/'poor',s,'T',cache(coordinate='T'),cache('validation','T'),{},device='cpu')
    assert r['epochs']==30 and r['validation']['macro_ovr_auc']==0.


def test_setup_all_twelve_measurement_paths_and_persistent_failure(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from test_luka_attention import cache as large
    tiny(monkeypatch); s=spec(tmp_path); s['counts']['train']=1793; (tmp_path/'setup').mkdir()
    h,hv=large('train','D000',1793),cache('validation'); paths={}
    for name,data in (('targets/u100/targets.npz',dict(identities=h.identities,labels=h.labels)),
        ('reference/matched/predictions.npz',dict(identities=hv.identities,labels=hv.labels))):
        path=tmp_path/(str(len(paths))+'.npz'); atomic_publish_bytes(path,deterministic_npz_bytes(data)); paths[name]=path
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda name:paths[name]))
    monkeypatch.setattr(inputs,'views',lambda _,task,**kw:(large('train',task if task in c.TEACHERS else 'D000',1793),
        cache('validation',task if task in c.TEACHERS else 'D000')))
    monkeypatch.setattr(inputs,'banks',lambda _,task,tr,**kw:banks(tr,task))
    monkeypatch.setattr(resources,'installed_environment',lambda:s['reference']['environment'])
    gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)
    monkeypatch.setattr(resources,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(resources,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    a=resources.measure(s)
    assert a==resources.accepted(s) and set(a['minutes'])==set(c.FITS)
    for task in c.FITS: training.validate_fit(tmp_path/'setup'/('mini_'+task),s,task,technical=True)
    other=tmp_path/'failed'; (other/'setup').mkdir(parents=True); s2=dict(s,root=str(other))
    def fail(*a,**kw): raise RuntimeError('injected measurement error')
    monkeypatch.setattr(training,'step',fail)
    with pytest.raises(RuntimeError,match='injected'): resources.measure(s2)
    d=load_json(other/'setup/resource_diagnostic.json')
    assert d['status']=='failed' and 'injected' in d['error'] and not (other/'setup/acceptance.json').exists()


def test_scientific_workers_actual_multi_targets_and_report(tmp_path,monkeypatch,capsys):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); s=spec(tmp_path); h,hv=cache(),cache('validation'); u=banks(h,'S2K')['u100']
    s.update(imported={},control={},original50k_rows=22,reused={})
    paths={}
    for name,strength in (('M0HLT',0.),('OFFLINE',.9),('matched',.2),('C0',.4),('S00',.5),('T00',.8)):
        p=np.full((22,11),(1-strength)/11,np.float32); p[np.arange(22),hv.labels]+=strength
        path=tmp_path/(name+'.npz'); atomic_publish_bytes(path,deterministic_npz_bytes(dict(identities=hv.identities,labels=hv.labels,probabilities=p)))
        paths[f'reference/{name}/predictions.npz']=path
    s['reused']=dict(offline_predictions=file_ref(paths['reference/T00/predictions.npz']),extra_control_predictions=file_ref(paths['reference/S00/predictions.npz']))
    subset=tmp_path/'subset.npz'; atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    paths['reference/historical/subset.npz']=subset
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda n:paths[n],json=lambda _:u.manifest))
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:u)
    monkeypatch.setattr(inputs,'views',lambda _,task:(cache(coordinate=task),cache('validation',task)) if task in c.TEACHERS else (h,hv))
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    job=['901']; gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.site,'allocation',lambda _:job[0]); monkeypatch.setattr(worker.site,'cpu_allocation',lambda _:job[0])
    monkeypatch.setattr(worker,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    monkeypatch.setattr(worker,'completed_setup',lambda _:dict(gpu=gpu)); monkeypatch.setattr(resources,'accepted',lambda _:dict(gpu=gpu))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(I='901',T='902',S3='903',report='904'))
    monkeypatch.setattr(reporting,'states',lambda _:dict(I=['COMPLETED','0:0'],T=['COMPLETED','0:0'],S3=['COMPLETED','0:0']))
    monkeypatch.setattr(reporting,'reuse_check',lambda *a,**kw:paths['reference/C0/predictions.npz'])
    worker.run(s,'I'); job[0]='902'; worker.run(s,'T'); job[0]='903'; worker.run(s,'S3')
    reporting.progress(s); job[0]='904'; worker.run(s,'report')
    result=load_json(tmp_path/'report/results.json')
    assert len(result['pending'])==9 and len(result['rows'])==18
    assert {r['model'] for r in result['rows'] if r['oracle']}=={'OFFLINE','T00_OFFLINE','I','T'}
    assert 'delta_S00=' in capsys.readouterr().out
    assert load_json(tmp_path/'report/receipt.json')['gpu'] is None
    assert load_json(tmp_path/'S3/startup_memory.json')['real_selected_teacher_targets']
    with pytest.raises(FileExistsError): worker.run(s,'report')
    receipt=tmp_path/'I/receipt.json'; bad=load_json(receipt); bad['job_id']='999'; receipt.write_text(json.dumps(reseal(bad)))
    with pytest.raises(ValueError,match='ownership'): worker.validate_teacher(s,'I')
