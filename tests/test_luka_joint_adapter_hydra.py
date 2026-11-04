"""Real pinned Weaver gradients, role isolation, deployment and campaign DAG."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.luka_joint_adapter_hydra import model as m, training as t, contracts as c, inputs


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
def test_joint_teacher_zero_and_gradients(site):
    torch.manual_seed(7); teacher=m.AdapterParT(site,'teacher'); value=raw()
    state={k:v.clone() for k,v in teacher.mod.state_dict().items()}
    base=native.DelphesParticleTransformer(); base.mod.load_state_dict(state); base.eval()
    teacher.eval()
    with torch.inference_mode():
        torch.testing.assert_close(t.forward(teacher,oracle_raw(value),'cpu'),t.forward(base,value,'cpu'),rtol=1e-5,atol=2e-6)
        assert torch.count_nonzero(teacher.correction_value)==0
    opt=t.optimizer(teacher)
    for _ in range(2): t.step(teacher,opt,oracle_raw(value),'cpu',lr=.0003)
    assert teacher.mod.training
    assert any(not torch.equal(v,state[k]) for k,v in teacher.mod.state_dict().items())
    assert teacher.correction.left.weight.grad.abs().sum()>0
    assert teacher.mod.fc[0].weight.grad.abs().sum()>0


@pytest.mark.parametrize('site',['bias','edge'])
@pytest.mark.parametrize('mode',['aux','final_aux','both'])
def test_student_gradients_and_deployment(site,mode):
    value=raw(); teacher=m.AdapterParT(site,'teacher').eval()
    with torch.no_grad(): teacher.correction.left.bias.fill_(.2)
    teacher.requires_grad_(False); student=m.AdapterParT(site,mode,5,1.)
    state={k:v.clone() for k,v in student.mod.state_dict().items()}
    args=dict(teacher=teacher,oracle_raw=oracle_raw(value),scale=torch.full((8 if site=='bias' else 4,),.05),coefficient=1.)
    t.step(student,t.optimizer(student),value,'cpu',lr=.001,embedding=torch.zeros(3,128),**args)
    assert student.correction.left.weight.grad is not None
    assert student.correction.left.weight.grad.abs().sum()>0
    assert any(not torch.equal(v,state[k]) for k,v in student.mod.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    deployed=t.deployment(dict(site=site,mode=mode,embedding=5,correction_weight=1.),student.deployment_state(),'cpu')
    assert not hasattr(deployed,'predictor')
    student.eval(); student.capture=False; student.supervision=False
    torch.testing.assert_close(t.forward(deployed,value,'cpu'),t.forward(student,value,'cpu'),rtol=0,atol=0)
    assert not any(k.startswith('predictor.') for k in student.deployment_state())
    if mode in ('aux','final_aux'): assert all(k.startswith('mod.') for k in student.deployment_state())
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


@pytest.mark.parametrize('task',['teacher_bias','teacher_edge','bias_both_l1','edge_both_l1','bias_aux_l1','edge_final_aux_l1'])
def test_fit_publication(tmp_path,task):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    u=cache(coordinate='U100'); vu=cache(role='validation',coordinate='U100')
    ot,ov=inputs.OracleCache(tr,u),inputs.OracleCache(va,vu)
    setting=c.settings(task); model=m.AdapterParT(**setting)
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)))
    teacher=None; scale=None; parent=None
    if setting['mode'] in ('aux','both','final_aux'):
        teacher=m.AdapterParT(setting['site'],'teacher').eval().requires_grad_(False)
        scale=torch.full((8 if setting['site']=='bias' else 4,),.05); parent='b'*64
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64)) if setting['embedding'] else None
    report=t.fit(tmp_path/task,spec,task,model,ot if task in c.TEACHERS else tr,ov if task in c.TEACHERS else va,
        device='cpu',bank=bank,teacher=teacher,oracle_train=ot if teacher else None,scale=scale,teacher_parent=parent,technical=True)
    assert report['epochs']==1 and report['technical']
    t.validate_fit(tmp_path/task,spec,task,technical=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('site',['bias','edge'])
def test_cuda_bf16_joint_backward(site):
    teacher=m.AdapterParT(site,'teacher').cuda()
    t.step(teacher,t.optimizer(teacher),oracle_raw(raw()),'cuda',lr=.0003)
    assert teacher.correction.left.weight.grad.abs().sum()>0


def test_render_independent_fits(tmp_path):
    from hlt_classification.luka_joint_adapter_hydra import campaign,submission
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












def test_train_statistics_dont_use_validation():
    teacher=m.AdapterParT('bias','teacher')
    va=cache(role='validation',coordinate='D000'); u=cache(role='validation',coordinate='U100')
    with pytest.raises(PermissionError): t.rms(teacher,inputs.OracleCache(va,u),'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('task',['teacher_bias','teacher_edge','bias_both_l1','edge_both_l1','bias_aux_l1','edge_aux_l1'])
def test_cuda_bf16_fit_reload(tmp_path,task):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    ot=inputs.OracleCache(tr,cache(coordinate='U100')); ov=inputs.OracleCache(va,cache(role='validation',coordinate='U100'))
    setting=c.settings(task); model=m.AdapterParT(**setting)
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)))
    teacher=None; scale=None; parent=None
    if setting['mode'] in ('aux','both','final_aux'):
        teacher=m.AdapterParT(setting['site'],'teacher').cuda().eval().requires_grad_(False)
        scale=torch.full((8 if setting['site']=='bias' else 4,),.05,device='cuda'); parent='b'*64
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64)) if setting['embedding'] else None
    t.fit(tmp_path/task,spec,task,model,ot if task in c.TEACHERS else tr,ov if task in c.TEACHERS else va,
        device='cuda',bank=bank,teacher=teacher,oracle_train=ot if teacher else None,scale=scale,teacher_parent=parent,technical=True)


def test_full_thirty_epoch_fit(tmp_path):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    ot=inputs.OracleCache(tr,cache(coordinate='U100'))
    task='edge_both_l1'; model=m.AdapterParT(**c.settings(task))
    teacher=m.AdapterParT('edge','teacher').eval().requires_grad_(False)
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)))
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64))
    report=t.fit(tmp_path/task,spec,task,model,tr,va,device='cpu',bank=bank,teacher=teacher,oracle_train=ot,
        scale=torch.full((4,),.05),teacher_parent='b'*64)
    assert len(report['validation_history'])==30
    t.validate_fit(tmp_path/task,spec,task)


@pytest.mark.parametrize('site',['bias','edge'])
def test_final_aux_loss_reaches_last_block_but_intermediate_aux_does_not(site):
    value=raw()
    for mode,expected in (('aux',False),('final_aux',True)):
        model=m.AdapterParT(site,mode,5,.3); model.supervision=True
        # The zero-initialized output intentionally blocks context gradients on
        # the very first step. Use a learned nonzero predictor for this test.
        with torch.no_grad(): model.correction.left.weight.normal_(std=.02)
        t.forward(model,value,'cpu')
        loss=m.loss(model.correction_value,torch.ones_like(model.correction_value),
            torch.from_numpy(value['mask'][:,0]),torch.ones(8 if site=='bias' else 4))
        loss.backward()
        grads=[p.grad for p in model.mod.blocks[7].parameters() if p.grad is not None]
        assert any(g.abs().sum()>0 for g in grads)==expected
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.mod.blocks[0].parameters())


def test_scratch_teacher_and_student_native_initialization_match():
    imported=dict(node=parent_nodes()[0]); states=[]
    for task in ('teacher_bias','teacher_edge','bias_both_l0p3','edge_final_aux_l1'):
        model=m.new_model(imported,c.settings(task)); states.append(model.mod.state_dict())
        if task in c.TEACHERS: assert all(p.requires_grad for p in model.mod.parameters())
    assert all(all(torch.equal(v,state[k]) for k,v in states[0].items()) for state in states[1:])
    with pytest.raises(ValueError,match='scratch'):
        m.new_model(imported,c.settings('teacher_bias'),states[0])


@pytest.mark.parametrize('task',['bias_final_aux_l0p3','edge_final_aux_l0p3','bias_both_l0p3','edge_aux_l1'])
def test_registered_weight_and_ramp(tmp_path,task,monkeypatch):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    ot=inputs.OracleCache(tr,cache(coordinate='U100'))
    setting=c.settings(task); teacher=m.AdapterParT(setting['site'],'teacher').eval().requires_grad_(False)
    with torch.no_grad(): teacher.correction.left.bias.fill_(.2)
    model=m.AdapterParT(**setting)
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='c'*64))
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)))
    coefficients=[]; original=t.step
    def step(*args,**kwargs):
        coefficients.append(kwargs['coefficient']); return original(*args,**kwargs)
    monkeypatch.setattr(t,'step',step)
    result=t.fit(tmp_path/task,spec,task,model,tr,va,device='cpu',bank=bank,teacher=teacher,oracle_train=ot,
        scale=torch.full((8 if setting['site']=='bias' else 4,),.05),teacher_parent='b'*64,technical=True)
    assert coefficients==[pytest.approx(setting['correction_weight']/3)]
    row=result['validation_history'][0]
    assert row['weighted_correction']==pytest.approx(coefficients[0]*row['correction_loss'],rel=1e-5)
    assert result['settings']['embedding']==5


def test_exact_submission_dry_run(tmp_path):
    from hlt_classification.luka_joint_adapter_hydra import campaign,submission
    spec=dict(campaign.registration(),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    value=submission._submit(spec,submission.render(spec,'envelope'),execute=False)
    assert value['dry_run'] and len(value['jobs'])==15
    assert not submission.accepted_jobs(spec,'setup')


def test_partial_output_is_not_reusable(tmp_path,monkeypatch):
    from hlt_classification.luka_joint_adapter_hydra import worker
    path=tmp_path/'teacher_bias'; path.mkdir(); (path/'selected.pt').write_bytes(b'partial')
    with pytest.raises(FileNotFoundError): worker.receipt(dict(root=str(tmp_path),content_hash='a'*64),'teacher_bias')


def test_source_scope_and_missing_files_fail(tmp_path,monkeypatch):
    from hlt_classification.luka_joint_adapter_hydra import campaign
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
    from hlt_classification.luka_joint_adapter_hydra import campaign,submission,worker
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


def test_finish_reuses_setup_and_is_idempotent(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.luka_joint_adapter_hydra import campaign,submission,worker
    calls=[]
    def run(args,**kwargs):
        calls.append(args); return SimpleNamespace(returncode=0,stdout=str(6000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(dag.subprocess,'run',run)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'check_site',lambda *a:None)
    monkeypatch.setattr(submission.resources,'accepted',lambda s:c.artifact('ACCEPTANCE',minutes={t:120 for t in c.TASKS}))
    monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    spec=dict(campaign.registration(),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    review=submission.review(spec)
    claim=tmp_path/'submission/dispatcher_claim'; claim.mkdir()
    write_immutable_json(claim/'intent.json',c.artifact('DISPATCH_CLAIM',parents=dict(review=review['content_hash'])))
    submission._submit(spec,review['setup'],execute=True)
    with pytest.raises(PermissionError): submission.finish(spec,'0'*64,c.AUTHORIZATION)
    assert len(calls)==1
    result=submission.finish(spec,review['content_hash'],c.AUTHORIZATION)
    assert len(calls)==16
    assert submission.finish(spec,review['content_hash'],c.AUTHORIZATION)==result
    assert len(calls)==16


@pytest.mark.parametrize('issue',['partial','failed','not_done'])
def test_finish_does_not_retry_partial_or_unsuccessful_setup(tmp_path,monkeypatch,issue):
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.luka_joint_adapter_hydra import campaign,submission,worker
    spec=dict(campaign.registration(),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'check_site',lambda *a:None)
    monkeypatch.setattr(submission.resources,'accepted',lambda s:c.artifact('ACCEPTANCE',minutes={t:120 for t in c.TASKS}))
    monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(setup='7000'))
    status={'partial':['COMPLETED','0:0'],'failed':['FAILED','1:0'],'not_done':['RUNNING','0:0']}[issue]
    monkeypatch.setattr(submission,'accounting',lambda j:status)
    review=submission.review(spec)
    claim=tmp_path/'submission/dispatcher_claim'; claim.mkdir()
    write_immutable_json(claim/'intent.json',c.artifact('DISPATCH_CLAIM',parents=dict(review=review['content_hash'])))
    if issue=='partial': (tmp_path/'submission_science/live_claim').mkdir(parents=True)
    monkeypatch.setattr(submission,'_submit',lambda *a,**k:pytest.fail('Must not submit anything'))
    with pytest.raises(RuntimeError): submission.finish(spec,review['content_hash'],c.AUTHORIZATION)


def test_control_reference_checks_and_original_source_replay(tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.luka_frozen_adapter_hydra.contracts import artifact as old_artifact
    from hlt_classification.luka_joint_adapter_hydra import references as r
    old=old_artifact('SPEC',source_commit=r.CONTROL_COMMIT,counts=c.COUNTS,root=str(tmp_path),
        project_dir='/pinned/original',bundle_root=str(tmp_path/'bundle'),delta_hash='d'*64,
        reference={'r':1},imported={'node':parent_nodes()[0]},anchor={'a':1})
    path=tmp_path/'campaign_spec.json'; write_immutable_json(path,old)
    calls=[]
    def replay(path,project,program,marker):
        calls.append((project,program)); return file_ref(path),dict(spec=old['content_hash'],rows={},pending=list(r.CONTROL_TASKS))
    monkeypatch.setattr(r,'replay',replay)
    control=r.register(path,old,tmp_path/'bundle','d'*64)
    spec=dict(old,control_campaign=control)
    assert r.check(spec)==old
    assert r.completed(spec)['pending']==list(r.CONTROL_TASKS)
    assert all(project=='/pinned/original' for project,_ in calls)
    assert 'campaign.check(s,check_storage=False)' in calls[0][1]
    assert 'worker.receipt(s,task)' in calls[0][1]
    with pytest.raises(ValueError): r.register(path,dict(old,anchor={'bad':1}),tmp_path/'bundle','d'*64)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError): r.check(spec)


def test_campaign_create_and_check_bind_controls_without_retraining(tmp_path,monkeypatch):
    from pathlib import Path
    from hlt_classification.luka_joint_adapter_hydra import campaign,references
    from hlt_classification.data.cache_contracts import with_content_hash
    bundle=tmp_path/'bundle'; bundle.mkdir(); delta=tmp_path/'delta'; delta.mkdir()
    project=tmp_path/'project'; project.mkdir(); precision=tmp_path/'precision/spec.json'
    control=tmp_path/'old/spec.json'
    origin=dict(reference=dict(environment={}),imported=dict(node=parent_nodes()[0]),anchor=dict(evidence={}))
    b=SimpleNamespace(root=bundle,json=lambda _:origin)
    d=SimpleNamespace(root=delta,manifest=dict(content_hash='d'*64),verify_all=lambda:None)
    monkeypatch.setattr(campaign.inputs,'Bundle',lambda _:b)
    monkeypatch.setattr(campaign.inputs,'Delta',lambda *a:d)
    monkeypatch.setattr(campaign.inputs,'delta',lambda *a:d)
    monkeypatch.setattr(campaign.site,'environment',lambda *a:None)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='s'*64))
    monkeypatch.setattr(campaign.portability,'authenticate',lambda *a:dict(content_hash='p'*64))
    monkeypatch.setattr(campaign.portability,'validate_evidence',lambda *a:None)
    reused=c.artifact('CONTROL_REFERENCE',parents=dict(spec='c'*64),reference_retrained=False)
    monkeypatch.setattr(references,'register',lambda *a:reused)
    monkeypatch.setattr(references,'check',lambda s:None)
    monkeypatch.setattr(campaign,'storage',lambda s:None)
    spec=campaign.create(tmp_path/'run',bundle,delta,'d'*64,precision,project,'f'*40,28,control)
    assert spec['storage_budget_bytes']==8*2**30 and spec['available_quota_gib']==28
    assert spec['parents']['controls']==reused['content_hash']
    assert not spec['reference_retrained'] and spec['max_concurrent_fits'] is None
    assert campaign.check(spec)==spec
    with pytest.raises(ValueError,match='fresh'):
        campaign.create(tmp_path/'run',bundle,delta,'d'*64,precision,project,'f'*40,28,control)
    changed=dict(spec); changed['parents']=dict(spec['parents'],controls='0'*64)
    changed.pop('content_hash'); changed=with_content_hash(changed)
    with pytest.raises(ValueError): campaign.check(changed)


def test_production_configuration_native_identity_and_final_hook(monkeypatch):
    from hlt_classification.models.particle_transformer import canonical_particle_transformer_config
    config=canonical_particle_transformer_config(); config.update(input_dim=17,num_classes=11,trim=False)
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    value=raw()
    for task in ('teacher_bias','teacher_edge','bias_final_aux_l0p3','edge_final_aux_l1'):
        model=m.new_model(dict(node=parent_nodes()[0]),c.settings(task)).eval()
        base=native.DelphesParticleTransformer().eval(); base.mod.load_state_dict(model.mod.state_dict())
        with torch.no_grad():
            model.supervision=True
            output=t.forward(model,oracle_raw(value) if task in c.TEACHERS else value,'cpu')
            torch.testing.assert_close(output,t.forward(base,value,'cpu'),atol=2e-6,rtol=1e-5)
            assert model.correction_value.shape[:3]==(3,value['features'].shape[-1],value['features'].shape[-1])


def test_progress_reads_published_epochs_without_inference(tmp_path,monkeypatch,capsys):
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.luka_joint_adapter_hydra import progress,campaign
    spec=dict(root=str(tmp_path),bundle_root='/bundle',content_hash='a'*64,counts=dict(validation=11))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    bundle=SimpleNamespace(json=lambda n:dict(recovery_anchors=dict(M0HLT=.9,OFFLINE=1.)) if n.startswith('origin')
        else dict(validation_history=[dict(pass_number=i,validation=dict(macro_ovr_auc=.94)) for i in range(1,31)]))
    monkeypatch.setattr(progress.inputs,'Bundle',lambda _:bundle)
    folder=tmp_path/'bias_final_aux_l0p3/fit'; folder.mkdir(parents=True)
    write_immutable_json(folder/'epoch_01.json',c.artifact('EPOCH',parents=dict(spec='a'*64),
        task='bias_final_aux_l0p3',pass_number=1,validation=dict(macro_ovr_auc=.95,rows=11)))
    (tmp_path/'bias_final_aux_l0p3-123.out').write_text('JOINT_ADAPTER bias_final_aux_l0p3 epoch=2/30 batch=2/4\n')
    progress.show(spec); text=capsys.readouterr().out
    assert '0.95000000' in text and '50.0%' in text and 'batch 2/4' in text


def test_shell_launcher_returns_before_detached_submission_finishes(tmp_path):
    import os,subprocess,sys,time
    from pathlib import Path
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def sh(p):
        p=Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    repo=tmp_path/'project'; (repo/'scripts').mkdir(parents=True)
    name='queue_luka_joint_adapter_hydra.sh'
    script=repo/'scripts'/name
    script.write_bytes((Path(__file__).resolve().parents[1]/'scripts'/name).read_bytes())
    base=tmp_path/'inputs'; envdir=base/'environment_pair_hydra_v1/venv'; (envdir/'bin').mkdir(parents=True)
    (envdir.parent/'environment-check.json').write_text('{}')
    precision=base/'runs/pair_precision_ad55d3cd_r1/campaign_spec.json'; precision.parent.mkdir(parents=True); precision.write_text('{}')
    control=tmp_path/'control.json'; control.write_text('{}')
    delta=tmp_path/'delta'; delta.mkdir(); done=tmp_path/'done'
    binpath=tmp_path/'bin'; binpath.mkdir()
    bodies={
        'git':'case "$*" in *"rev-parse HEAD"*) printf "%040d\\n" 1;; esac\n',
        'python3':'case "$*" in *delta_root*) printf "%s\\n" "$FIXTURE_DELTA";; *delta_hash*) printf "%064d\\n" 1;; esac\n',
        'setsid':'exec "$@"\n',
    }
    for name,body in bodies.items():
        p=binpath/name; p.write_bytes(('#!/usr/bin/env bash\n'+body).encode()); p.chmod(0o755)
    runner=envdir/'bin/python'
    runner.write_text('''#!/usr/bin/env bash
case "$*" in
*" -c "*) printf '%064d\\n' 2;;
*" create "*) while [[ "$1" != --root ]]; do shift; done; shift; mkdir -p "$1"; printf '{}' > "$1/campaign_spec.json";;
*" review "*) while [[ "$1" != --spec ]]; do shift; done; shift; mkdir -p "${1%/*}/submission"; printf '{}' > "${1%/*}/submission/review.json";;
*" dispatch "*) sleep 1; printf 'DONE\\n'; printf done > "$FIXTURE_DONE";;
*) exit 3;;
esac
''',newline='\n'); runner.chmod(0o755)
    startup=tmp_path/'env.sh'; startup.write_text('''export PATH="${FIXTURE_BIN}:${PATH}"
# MSYS mkdir -p probes restricted Windows ancestors. Fixture parents already
# exist, so use a direct mkdir for each new leaf; production launcher is intact.
mkdir() {
    if [[ "$1" == -p ]]; then
        shift
        for directory in "$@"; do
            [[ "$directory" == -- ]] && continue
            [[ -d "$directory" ]] || command mkdir -- "$directory" || return
        done
    else
        command mkdir "$@"
    fi
}
''',newline='\n')
    environ=dict(os.environ,BASH_ENV=sh(startup),FIXTURE_BIN=sh(binpath),FIXTURE_DELTA=sh(delta),FIXTURE_DONE=sh(done))
    args=[bash,str(script),sh(base),sh(control),'8']
    result=subprocess.run(args,env=environ,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert 'Dispatcher PID:' in result.stdout
    deadline=time.monotonic()+10
    while not done.exists() and time.monotonic()<deadline: time.sleep(.05)
    assert done.read_text()=='done'
    logs=list((base/'runs').glob('joint_adapter_*/submit.*.log'))
    assert len(logs)==1 and 'DONE' in logs[0].read_text()
    assert subprocess.run(args,env=environ,capture_output=True,text=True,timeout=10).returncode!=0
    # setsid is stubbed here: this tests process handoff, not survival of a
    # real Brown SSH disconnect or institutional login-node cleanup policy.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_shared_setup_and_worker(tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.luka_joint_adapter_hydra import campaign,worker,submission,resources,references
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
    for task in ('teacher_bias','bias_both_l1'):
        current[0]=task; worker.run(spec,task); worker.receipt(spec,task)
    report,_=t.validate_fit(folder/'bias_both_l1/fit',spec,'bias_both_l1')
    parent,_=t.validate_fit(folder/'teacher_bias/fit',spec,'teacher_bias')
    assert report['parents']['teacher']==parent['content_hash']
    # Completed results combine new selected fits with replayed controls, without
    # another forward or writes to the scientific run.
    subset=tmp_path/'subset.npz'
    atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(len(va),dtype=np.int64))))
    spec['original50k_rows']=len(va)
    spec['portability_evidence']=dict(refs=dict(predictions=dict(path=str(ref))))
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(
        path=lambda n:subset if n.endswith('subset.npz') else ref,
        json=lambda n:bank.manifest))
    monkeypatch.setattr(references,'completed',lambda s:dict(spec='z'*64,pending=['edge_ce_e5'],
        rows=dict(bias_ce_e5=dict(predictions=dict(path=str(ref)),selected_epoch=1,report='c'*64,privileged_oracle=False))))
    monkeypatch.setattr(t,'predict',lambda *a:pytest.fail('Results must not infer'))
    result=worker.results(spec)
    row=next(r for r in result['rows'] if r['model']=='bias_both_l1')
    assert row['architecture_control']=='OLD_bias_ce_e5'
    assert row['validation_delta_architecture_control'] is not None
    assert not row['privileged_oracle']
    assert 'teacher_edge' in result['pending'] and result['pending_controls']==['edge_ce_e5']
    worker.print_results(result)
