"""Real pinned Weaver gradients, role isolation, deployment and campaign DAG."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.luka_frozen_adapter_hydra import model as m, training as t, contracts as c, inputs


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    old=torch.get_num_threads(); torch.set_num_threads(1)
    config=native.model_config(); config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=8,num_layers=8,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    yield
    torch.set_num_threads(old)


def raw(): return cache(coordinate='D000').batch(np.arange(3))


def oracle_raw(value): return dict(value,u_features=value['features']+.1,u_vectors=value['vectors'])


@pytest.mark.parametrize('site',['bias','edge'])
def test_frozen_teacher_zero_and_gradients(site):
    torch.manual_seed(7); teacher=m.AdapterParT(site,'teacher'); value=raw()
    state={k:v.clone() for k,v in teacher.mod.state_dict().items()}
    base=native.DelphesParticleTransformer(); base.mod.load_state_dict(state); base.eval()
    teacher.eval()
    with torch.inference_mode():
        torch.testing.assert_close(t.forward(teacher,oracle_raw(value),'cpu'),t.forward(base,value,'cpu'),rtol=1e-5,atol=2e-6)
        assert torch.count_nonzero(teacher.correction_value)==0
    opt=t.optimizer(teacher)
    for _ in range(2): t.step(teacher,opt,oracle_raw(value),'cpu',lr=.0003)
    assert not teacher.mod.training
    assert all(torch.equal(v,state[k]) for k,v in teacher.mod.state_dict().items())
    assert teacher.correction.left.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.mod.parameters())


@pytest.mark.parametrize('site',['bias','edge'])
@pytest.mark.parametrize('mode',['aux','ce','both'])
def test_student_gradients_and_deployment(site,mode):
    value=raw(); teacher=m.AdapterParT(site,'teacher').eval()
    with torch.no_grad(): teacher.correction.left.bias.fill_(.2)
    teacher.requires_grad_(False); student=m.AdapterParT(site,mode,5)
    state={k:v.clone() for k,v in student.mod.state_dict().items()}
    args={} if mode=='ce' else dict(teacher=teacher,oracle_raw=oracle_raw(value),scale=torch.full((8 if site=='bias' else 4,),.05),coefficient=1.)
    t.step(student,t.optimizer(student),value,'cpu',lr=.001,embedding=torch.zeros(3,128),**args)
    assert student.correction.left.weight.grad is not None
    assert student.correction.left.weight.grad.abs().sum()>0
    assert any(not torch.equal(v,state[k]) for k,v in student.mod.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    deployed=t.deployment(dict(site=site,mode=mode,embedding=5),student.deployment_state(),'cpu')
    assert not hasattr(deployed,'predictor')
    student.eval(); student.capture=False; student.supervision=False
    torch.testing.assert_close(t.forward(deployed,value,'cpu'),t.forward(student,value,'cpu'),rtol=0,atol=0)
    assert not any(k.startswith('predictor.') for k in student.deployment_state())
    if mode=='aux': assert all(k.startswith('mod.') for k in student.deployment_state())
    with pytest.raises(PermissionError): t.forward(student,oracle_raw(value),'cpu')


def test_row_center_and_per_jet_loss():
    mask=torch.tensor([[True,True,False],[True,False,False]])
    values=torch.randn(2,3,3,8,requires_grad=True); centered=m.center(values,mask)
    torch.testing.assert_close(centered.sum(2),torch.zeros(2,3,8),atol=1e-6,rtol=0)
    assert not centered[~mask].any()
    target=torch.ones(2,3,3,8,requires_grad=True)
    result=m.loss(torch.zeros_like(target,requires_grad=True),target,mask,torch.ones(8))
    assert result==.5
    result.backward(); assert target.grad is None


def test_pair_context_and_permutation():
    value=raw(); features=torch.tensor(value['features']).transpose(1,2)
    vectors=torch.tensor(value['vectors']); mask=torch.tensor(value['mask'][:,0])
    ctx=torch.randn(features.shape[0],features.shape[1],128)
    head=m.Correction(128,8,False)
    with torch.no_grad(): head.left.weight.normal_(std=.01); head.geometry[-1].weight.normal_(std=.01)
    p=head(ctx,vectors,mask); permutation=torch.arange(ctx.shape[1]-1,-1,-1)
    q=head(ctx[:,permutation],vectors[:,:,permutation],mask[:,permutation])
    torch.testing.assert_close(q,p[:,permutation][:,:,permutation],atol=1e-6,rtol=1e-5)
    changed=vectors.clone(); changed[:,0]*=1.01; changed[:,3]*=1.01
    assert not torch.equal(p,head(ctx,changed,mask))


def test_edge_contraction_is_pair_message():
    a=torch.rand(2,8,5,5).softmax(-1); coeff=torch.randn(2,5,5,4); basis=torch.randn(8,4,16)
    compact=torch.einsum('bhij,bijr->bhir',a,coeff)
    efficient=torch.einsum('bhir,hrd->bihd',compact,basis)
    dense=torch.einsum('bhij,bijhd->bihd',a,torch.einsum('bijr,hrd->bijhd',coeff,basis))
    torch.testing.assert_close(efficient,dense)


def test_registered_comparisons():
    assert len(c.TASKS)==14 and len(c.ARMS)==12
    for task in c.TASKS:
        cfg=c.settings(task); expected=[] if cfg['mode'] in ('ce','teacher') else ['teacher_'+cfg['site']]
        assert c.dependencies(task)==expected
    assert c.dependencies('report')==list(c.TASKS)


@pytest.mark.parametrize('task',['teacher_bias','teacher_edge','bias_both_e5','edge_both_e5','bias_aux_e0','edge_ce_e0'])
def test_fit_publication(tmp_path,task):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    u=cache(coordinate='U100'); vu=cache(role='validation',coordinate='U100')
    ot,ov=inputs.OracleCache(tr,u),inputs.OracleCache(va,vu)
    setting=c.settings(task); model=m.AdapterParT(**setting)
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)))
    teacher=None; scale=None; parent=None
    if setting['mode'] in ('aux','both'):
        teacher=m.AdapterParT(setting['site'],'teacher').eval().requires_grad_(False)
        scale=torch.full((8 if setting['site']=='bias' else 4,),.05); parent='b'*64
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64)) if setting['embedding'] else None
    report=t.fit(tmp_path/task,spec,task,model,ot if task in c.TEACHERS else tr,ov if task in c.TEACHERS else va,
        device='cpu',bank=bank,teacher=teacher,oracle_train=ot if teacher else None,scale=scale,teacher_parent=parent,technical=True)
    assert report['epochs']==1 and report['technical']
    t.validate_fit(tmp_path/task,spec,task,technical=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('site',['bias','edge'])
def test_cuda_bf16_frozen_backward(site):
    teacher=m.AdapterParT(site,'teacher').cuda()
    t.step(teacher,t.optimizer(teacher),oracle_raw(raw()),'cuda',lr=.0003)
    assert teacher.correction.left.weight.grad.abs().sum()>0


def test_render_independent_fits(tmp_path):
    from hlt_classification.luka_frozen_adapter_hydra import campaign,submission
    spec=dict(campaign.registration(),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    envelope=submission.render(spec,'envelope')
    plan=submission.render(spec,'science',acceptance=dict(content_hash='b'*64,minutes={t:90 for t in c.TASKS}))
    assert submission.within_envelope(envelope,plan)
    assert len(plan['commands'])==15
    for row in plan['commands']:
        task=row['task_id']; command=row['command']
        assert row['dependencies']==c.dependencies(task)
        dependencies=[v for v in command if v.startswith('--dependency=')]
        if task=='report': assert dependencies[0].startswith('--dependency=afterany:')
        elif c.dependencies(task):
            assert dependencies==['--dependency=afterok:${JOB_'+c.dependencies(task)[0]+'}']
            assert '--kill-on-invalid-dep=yes' in command
        else: assert not dependencies
        assert '--partition=gpus' in command and '--gres=gpu:nvidia_l40:1' in command


def test_delta_exact_members_and_tampering(tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import with_content_hash,write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    root=tmp_path/'delta'; root.mkdir(); (root/'M0HLT.pt').write_bytes(b'checkpoint')
    path=root/'prepared/inputs/validation/U100/0000.npz'; path.parent.mkdir(parents=True); path.write_bytes(b'array')
    shard=dict(file_ref(path,root=root/'prepared'),file_index=0)
    cache_manifest=with_content_hash(dict(blocks=dict(validation=dict(U100=[shard]))))
    anchor=dict(cache=cache_manifest['content_hash'],prepared='p'*64,models=dict(M0HLT=dict(checkpoint=file_ref(root/'M0HLT.pt'))))
    old=dict(content_hash='o'*64,anchor=dict(evidence=anchor),reference=dict(content_hash='r'*64,environment={}),source_commit='s'*40)
    class Bundle:
        def json(self,name): return old if name=='origin/campaign_spec.json' else cache_manifest
    proof=dict(campaign=old['content_hash'],reference='r'*64,prepared='p'*64,cache=cache_manifest['content_hash'],
        source_commit='s'*40,control_job='21993056',counts=c.COUNTS,environment={},original_source_replay=True)
    rows=inputs.expected_files(old,cache_manifest)
    manifest=c.artifact('TRANSFER',parents=dict(origin=old['content_hash'],transfer=c.MANIFEST_HASH),
        original_replay=proof,files=rows,payload_bytes=sum(r['bytes'] for r in rows),roles=['oracle_initialization','oracle_validation_U100'])
    write_immutable_json(root/'manifest.json',manifest)
    d=inputs.Delta(root,manifest['content_hash'],Bundle()); d.verify_all()
    with pytest.raises(PermissionError): d.path('prepared/inputs/train/OFFLINE/0000.npz')
    with pytest.raises(ValueError): inputs.Delta(root,'0'*64,Bundle())
    (root/'M0HLT.pt').write_bytes(b'corrupted!')
    with pytest.raises(ValueError): d.path('M0HLT.pt')


@pytest.fixture
def export_origin(tmp_path,monkeypatch):
    """Real export/copy/validation path; only remote original-source replay is mocked."""
    from hlt_classification.data.cache_contracts import write_immutable_json,with_content_hash
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.luka_fullsim.alljets_data import artifact as original_artifact
    root=tmp_path/'source'; prepared=root/'prepared'
    payload=prepared/'inputs/validation/U100/0000.npz'; payload.parent.mkdir(parents=True)
    payload.write_bytes(b'opaque authenticated U100 payload')
    checkpoint=root/'M0HLT.pt'; checkpoint.write_bytes(b'opaque authenticated selected checkpoint')
    sentinel=root/'do-not-transfer.root'; sentinel.write_bytes(b'unrelated raw data')
    shard=dict(file_ref(payload,root=prepared),file_index=0)
    cache_manifest=original_artifact('CACHE',blocks=dict(validation=dict(U100=[shard])))
    cache_path=prepared/'cache_manifest.json'; write_immutable_json(cache_path,cache_manifest)
    anchor=dict(prepared_root=str(prepared),cache=cache_manifest['content_hash'],prepared='a'*64,
        models=dict(M0HLT=dict(checkpoint=file_ref(checkpoint))))
    old=with_content_hash(dict(contract='LUKA_U100_PAIR_CORRECTION_SPEC/v1',schema_version=1,
        counts=c.COUNTS,anchor=dict(evidence=anchor),source_commit='c'*40,
        reference=dict(content_hash='b'*64,environment=dict(python='3.10'))))
    spec_path=root/'campaign_spec.json'; write_immutable_json(spec_path,old)
    def authenticate(path,spec):
        assert path==spec_path and spec==old
        return dict(campaign=old['content_hash'],reference=old['reference']['content_hash'],
            prepared=anchor['prepared'],cache=anchor['cache'],source_commit=old['source_commit'],
            control_job='21993056',counts=c.COUNTS,environment=old['reference']['environment'],original_source_replay=True)
    monkeypatch.setattr(inputs.transfer,'authenticate',authenticate)
    bundle=SimpleNamespace(json=lambda name:old if name=='origin/campaign_spec.json' else cache_manifest)
    return SimpleNamespace(root=root,prepared=prepared,payload=payload,checkpoint=checkpoint,
        cache=cache_manifest,cache_path=cache_path,spec=old,spec_path=spec_path,bundle=bundle)


def test_export_roundtrip_preserves_sources_and_minimal_scope(export_origin,tmp_path):
    f=export_origin; before={p:p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
    out=tmp_path/'export'; result=inputs.export(f.spec_path,out)
    d=inputs.Delta(out,result['manifest_hash'],f.bundle); d.verify_all()
    assert set(d.rows)=={'M0HLT.pt','prepared/inputs/validation/U100/0000.npz'}
    assert result['files']==2 and result['payload_bytes']==sum(len(before[p]) for p in (f.checkpoint,f.payload))
    assert d.path('M0HLT.pt').read_bytes()==before[f.checkpoint]
    assert d.path('prepared/inputs/validation/U100/0000.npz').read_bytes()==before[f.payload]
    assert before=={p:p.read_bytes() for p in before}
    with pytest.raises(ValueError,match='fresh separate'): inputs.export(f.spec_path,out)


@pytest.mark.parametrize('problem',['contract','schema','checksum','checkpoint','shard','missing_shard'])
def test_export_rejects_invalid_evidence_before_output(export_origin,tmp_path,problem):
    from hlt_classification.data.cache_contracts import with_content_hash
    import json
    f=export_origin; out=tmp_path/'never'
    if problem in ('contract','schema','checksum'):
        value=dict(f.cache)
        if problem=='contract': value['contract']='LUKA_FULLSIM_REGULAR_CACHE/v1'
        elif problem=='schema': value['schema_version']=2
        else: value['serial_reference_passed']=False
        if problem!='checksum':
            value.pop('content_hash'); value=with_content_hash(value)
        f.cache_path.write_text(json.dumps(value))
    elif problem=='checkpoint': f.checkpoint.write_bytes(b'corrupt checkpoint')
    elif problem=='shard': f.payload.write_bytes(b'corrupt U100 payload')
    else: f.payload.unlink()
    with pytest.raises(ValueError): inputs.export(f.spec_path,out)
    assert not out.exists()


def test_oracle_input_alignment_and_student_guard():
    h=cache(coordinate='D000'); u=cache(coordinate='U100'); inputs.aligned(h,u)
    u.labels=u.labels[::-1].copy()
    with pytest.raises(ValueError): inputs.OracleCache(h,u)
    model=m.AdapterParT('bias','ce')
    with pytest.raises(PermissionError): t.step(model,t.optimizer(model),raw(),'cpu',lr=.001,
        oracle_raw=oracle_raw(raw()))


def test_train_statistics_dont_use_validation():
    teacher=m.AdapterParT('bias','teacher')
    va=cache(role='validation',coordinate='D000'); u=cache(role='validation',coordinate='U100')
    with pytest.raises(PermissionError): t.rms(teacher,inputs.OracleCache(va,u),'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('task',['teacher_bias','teacher_edge','bias_both_e5','edge_both_e5','bias_aux_e5','edge_aux_e5'])
def test_cuda_bf16_fit_reload(tmp_path,task):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    ot=inputs.OracleCache(tr,cache(coordinate='U100')); ov=inputs.OracleCache(va,cache(role='validation',coordinate='U100'))
    setting=c.settings(task); model=m.AdapterParT(**setting)
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)))
    teacher=None; scale=None; parent=None
    if setting['mode'] in ('aux','both'):
        teacher=m.AdapterParT(setting['site'],'teacher').cuda().eval().requires_grad_(False)
        scale=torch.full((8 if setting['site']=='bias' else 4,),.05,device='cuda'); parent='b'*64
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64)) if setting['embedding'] else None
    t.fit(tmp_path/task,spec,task,model,ot if task in c.TEACHERS else tr,ov if task in c.TEACHERS else va,
        device='cuda',bank=bank,teacher=teacher,oracle_train=ot if teacher else None,scale=scale,teacher_parent=parent,technical=True)


def test_full_thirty_epoch_fit(tmp_path):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    ot=inputs.OracleCache(tr,cache(coordinate='U100'))
    task='edge_both_e5'; model=m.AdapterParT(**c.settings(task))
    teacher=m.AdapterParT('edge','teacher').eval().requires_grad_(False)
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)))
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64))
    report=t.fit(tmp_path/task,spec,task,model,tr,va,device='cpu',bank=bank,teacher=teacher,oracle_train=ot,
        scale=torch.full((4,),.05),teacher_parent='b'*64)
    assert len(report['validation_history'])==30
    t.validate_fit(tmp_path/task,spec,task)


def test_exact_submission_dry_run(tmp_path):
    from hlt_classification.luka_frozen_adapter_hydra import campaign,submission
    spec=dict(campaign.registration(),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    value=submission._submit(spec,submission.render(spec,'envelope'),execute=False)
    assert value['dry_run'] and len(value['jobs'])==15
    assert not submission.accepted_jobs(spec,'setup')


def test_partial_output_is_not_reusable(tmp_path,monkeypatch):
    from hlt_classification.luka_frozen_adapter_hydra import worker
    path=tmp_path/'teacher_bias'; path.mkdir(); (path/'selected.pt').write_bytes(b'partial')
    with pytest.raises(FileNotFoundError): worker.receipt(dict(root=str(tmp_path),content_hash='a'*64),'teacher_bias')


def test_source_scope_and_missing_files_fail(tmp_path,monkeypatch):
    from hlt_classification.luka_frozen_adapter_hydra import campaign
    commands=[]
    monkeypatch.setattr(campaign,'_source',lambda *args:dict(content_hash='a'*64))
    def fake(args,**kwargs):
        commands.append(args)
        return SimpleNamespace(stdout='M\tsrc/hlt_classification/luka_u100_pair_hydra/worker.py\n' if 'diff' in args else '')
    monkeypatch.setattr(campaign.subprocess,'run',fake)
    with pytest.raises(ValueError): campaign.current(tmp_path,'f'*40)
    assert any('merge-base' in command for command in commands)


def test_multichunk_checkpoint_backward():
    torch.manual_seed(11)
    head=m.Correction(128,8,False)
    with torch.no_grad(): head.left.weight.normal_(std=.01); head.geometry[-1].weight.normal_(std=.01)
    h=torch.randn(2,71,128,requires_grad=True)
    p=torch.randn(2,3,71); energy=(p.square().sum(1)+1).sqrt(); p=torch.cat((p,energy[:,None]),1)
    mask=torch.ones(2,71,dtype=torch.bool); mask[1,51:]=False
    result=head(h,p,mask)
    with torch.no_grad(): expected=head(h,p,mask)
    torch.testing.assert_close(result,expected)
    result.square().sum().backward()
    assert torch.isfinite(h.grad).all() and h.grad[:,:51].abs().sum()>0
    assert not h.grad[1,51:].any()


def test_exact_dispatch_and_failed_setup(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    from hlt_classification.luka_frozen_adapter_hydra import campaign,submission,worker
    calls=[]
    def run(args,**kwargs):
        calls.append(args); return SimpleNamespace(returncode=0,stdout=str(5000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(dag.subprocess,'run',run)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'check_site',lambda *a:None)
    monkeypatch.setattr(submission.resources,'accepted',lambda s:c.artifact('ACCEPTANCE',minutes={t:120 for t in c.TASKS}))
    monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    spec=dict(campaign.registration(),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    review=submission.review(spec); submission.dispatch(spec,review['content_hash'],c.AUTHORIZATION)
    assert len(calls)==16
    jobs=submission.accepted_jobs(spec,'science')
    for cmd in calls[1:-1]:
        task=cmd[-1]; deps=c.dependencies(task)
        matches=[s for s in cmd if s.startswith('--dependency=')]
        assert matches==(['--dependency=afterok:'+jobs[deps[0]]] if deps else [])
    with pytest.raises(FileExistsError): submission.dispatch(spec,review['content_hash'],c.AUTHORIZATION)
    failed=tmp_path/'failed'; failed.mkdir(); spec2=dict(spec,root=str(failed))
    rev=submission.review(spec2); monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='no scientific fits'): submission.dispatch(spec2,rev['content_hash'],c.AUTHORIZATION)
    assert len(calls)==17


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_shared_setup_and_worker(tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.luka_frozen_adapter_hydra import campaign,worker,submission,resources
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    u=cache(coordinate='U100'); vu=cache(role='validation',coordinate='U100')
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64))
    model=native.DelphesParticleTransformer().cuda().eval(); checkpoint=tmp_path/'M0HLT.pt'
    save_state(checkpoint,{k:v.cpu() for k,v in model.state_dict().items()})
    p=t.predict(model,va,'cuda'); ref=tmp_path/'reference.npz'
    atomic_publish_bytes(ref,deterministic_npz_bytes(dict(identities=va.identities,labels=va.labels,probabilities=p)))
    del model
    folder=tmp_path/'run'; folder.mkdir()
    spec=dict(campaign.registration(),root=str(folder),project_dir='/project',content_hash='a'*64,
        imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)),bundle_root='/bundle',
        anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda _:ref,verify_all=lambda:None))
    monkeypatch.setattr(inputs,'delta',lambda _:SimpleNamespace(path=lambda _:checkpoint,verify_all=lambda:None))
    monkeypatch.setattr(inputs,'load_view',lambda s,role,coordinate:tr if role=='train' and coordinate=='D000' else u if role=='train' else va)
    monkeypatch.setattr(inputs,'oracle_validation',lambda s:vu)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    monkeypatch.setattr(worker.portability,'setup_record',lambda *a:c.artifact('TEST_PORTABILITY'))
    monkeypatch.setattr(worker.portability,'validate_setup',lambda *a:None)
    jobs={'setup':'7000',**{task:str(7001+i) for i,task in enumerate(c.TASKS)}}
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:jobs)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    current=['setup']; monkeypatch.setattr(worker.site,'allocation',lambda s:jobs[current[0]])
    worker.run(spec,'setup')
    accepted=worker.completed_setup(spec)
    assert len(accepted['measurements'])==14
    for task in ('teacher_bias','bias_both_e5'):
        current[0]=task; worker.run(spec,task); worker.receipt(spec,task)
    report,_=t.validate_fit(folder/'bias_both_e5/fit',spec,'bias_both_e5')
    parent,_=t.validate_fit(folder/'teacher_bias/fit',spec,'teacher_bias')
    assert report['parents']['teacher']==parent['content_hash']
