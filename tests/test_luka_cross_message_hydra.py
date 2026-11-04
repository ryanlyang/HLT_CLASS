"""Installed Weaver message training, transfer, deployment and exact Hydra DAG."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_luka_attention import cache,parent_nodes
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.luka_cross_message_hydra import contracts as c,model as m,inputs,training as t


@pytest.fixture(autouse=True)
def compact_native(monkeypatch):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    config=native.model_config(); config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=8,num_layers=8,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    yield
    torch.set_num_threads(previous)


def caches(device='cpu'):
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    source=native.DelphesParticleTransformer().to(device).eval().requires_grad_(False)
    ot=inputs.build_cache(tr,cache(coordinate='U100'),source,device,budget_bytes=2**30)
    ov=inputs.build_cache(va,cache(role='validation',coordinate='U100'),source,device,budget_bytes=2**30)
    return tr,va,ot,ov,source


def spec_for(tr,va,root=None):
    from hlt_classification.luka_cross_message_hydra.campaign import registration
    value=registration()
    value.update(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),encoder_hash='e'*64,
        counts=dict(train=len(tr),validation=len(va)),root=str(root))
    return value


def bank_for(tr):
    return SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='b'*64))


@pytest.mark.parametrize('width',[16,32])
def test_native_identity_and_joint_teacher_gradients(width):
    tr,va,ot,ov,source=caches(); cfg=c.settings(f'teacher_m{width}')
    value=m.new_model(dict(node=parent_nodes()[0]),cfg); baseline=native.DelphesParticleTransformer()
    baseline.mod.load_state_dict(value.mod.state_dict()); baseline.eval(); value.eval()
    ix=np.arange(3)
    torch.testing.assert_close(t.forward(value,ot.batch(ix),'cpu'),t.forward(baseline,tr.batch(ix),'cpu'),rtol=0,atol=0)
    opt=t.optimizer(value)
    for _ in range(3): t.step(value,opt,ot.batch(ix),'cpu',lr=.001)
    assert value.message.attention.in_proj_weight.grad.abs().sum()>0
    assert value.mod.blocks[0].attn.in_proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in source.parameters())
    assert not source.training


@pytest.mark.parametrize('mode',['both','aux','ce'])
def test_student_is_hlt_only_and_deployable(mode):
    tr,va,ot,ov,_=caches(); task='m32_ce_s0' if mode=='ce' else f'm32_{mode}_l0p3_s0'
    cfg=c.settings(task); student=m.new_model(dict(node=parent_nodes()[0]),cfg)
    teacher=m.MessageParT(**c.settings('teacher_m32')).eval().requires_grad_(False)
    ix=np.arange(3); extra={} if mode=='ce' else dict(teacher=teacher,oracle_raw=ot.batch(ix),scale=torch.ones(32),coefficient=.3)
    for _ in range(2): t.step(student,t.optimizer(student),tr.batch(ix),'cpu',lr=.001,embedding=torch.zeros(3,128),**extra)
    assert student.message.net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())
    if mode!='aux': assert student.expansion.weight.grad.abs().sum()>0
    student.eval(); student.capture=False; student.supervision=False
    deployed=t.deployment(cfg,student.deployment_state(),'cpu')
    torch.testing.assert_close(t.forward(student,tr.batch(ix),'cpu'),t.forward(deployed,tr.batch(ix),'cpu'),rtol=0,atol=0)
    assert not hasattr(deployed,'predictor')
    if mode=='aux': assert not hasattr(deployed,'message')
    with pytest.raises(PermissionError): t.forward(student,ot.batch(ix),'cpu')
    if mode=='ce':
        with pytest.raises(PermissionError):
            t.step(student,t.optimizer(student),tr.batch(ix),'cpu',lr=.001,embedding=torch.zeros(3,128),teacher=teacher)


def test_matched_initialization_and_no_privileged_control_weights():
    imp=dict(node=parent_nodes()[0])
    for seed in (0,1):
        a=m.new_model(imp,c.settings(f'm32_ce_s{seed}'))
        b=m.new_model(imp,c.settings(f'm32_both_l0p3_s{seed}'))
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert not any('attention' in k for k in a.message.state_dict())
    a=m.new_model(imp,c.settings('m32_ce_s0')); b=m.new_model(imp,c.settings('m32_ce_s1'))
    assert any(not torch.equal(v,b.mod.state_dict()[k]) for k,v in a.mod.state_dict().items())
    with pytest.raises(ValueError): m.new_model(imp,c.settings('m32_ce_s0'),a.state_dict())


@pytest.mark.parametrize('task',['teacher_m32','m32_both_l0p3_s0'])
def test_message_head_initialization_preserves_native_cpu_and_cuda_rng(task):
    from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
    torch.manual_seed(9341); AuxiliaryParT()
    cpu=torch.get_rng_state().clone()
    cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    torch.manual_seed(9341); m.MessageParT(**c.settings(task))
    assert torch.equal(torch.get_rng_state(),cpu)
    if cuda:
        assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))


def test_loss_padding_equal_jet_weight_and_detached_target():
    mask=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
    p=torch.zeros(2,3,16,requires_grad=True); target=torch.ones(2,3,16,requires_grad=True)
    target.data[~mask]=float('nan')
    value=m.loss(p,target,mask,torch.ones(16)); assert value==.5
    value.backward(); assert target.grad is None and p.grad[~mask].count_nonzero()==0


def test_cross_read_permutation_and_padded_memory():
    head=m.CrossRead(32).eval(); mask=torch.tensor([[1,1,0]],dtype=torch.bool)
    ctx=torch.randn(1,3,128); mem=torch.randn_like(ctx); features=torch.randn(1,17,3)
    before=head(ctx,features,mask,mem)
    mem[:,2]=float('nan'); after=head(ctx,features,mask,mem)
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    order=torch.tensor([1,0,2]); perm=head(ctx[:,order],features[:,:,order],mask[:,order],mem[:,order])
    torch.testing.assert_close(perm,before[:,order],atol=1e-6,rtol=1e-5)


def test_token_cache_alignment_budget_and_frozen_encoder():
    tr,_,ot,_,source=caches()
    assert ot.tokens.dtype==np.float32 and not source.training
    ix=np.array([2,0]); raw=ot.batch(ix)
    assert np.array_equal(raw['features'],tr.batch(ix)['features'])
    assert np.array_equal(raw['u_logits'],ot.logits[ix])
    with pytest.raises(MemoryError): inputs.build_cache(tr,cache(coordinate='U100'),source,'cpu',budget_bytes=1)
    with pytest.raises(ValueError): inputs.build_cache(tr,cache(role='validation',coordinate='U100'),source,'cpu',budget_bytes=2**30)


@pytest.mark.parametrize('task',c.TASKS)
def test_production_miniature_fit_and_readback(tmp_path,task):
    tr,va,ot,ov,_=caches(); spec=spec_for(tr,va,tmp_path); cfg=c.settings(task); oracle=task in c.TEACHERS
    model=m.new_model(spec['imported'],cfg); teacher=None; scale=None; parent=None
    if cfg['mode'] in ('both','aux'):
        teacher=m.MessageParT(**c.settings(c.teacher_task(task))).eval().requires_grad_(False)
        scale=torch.ones(cfg['width']); parent='c'*64
    report=t.fit(tmp_path/task,spec,task,model,ot if oracle else tr,ov if oracle else va,device='cpu',
        bank=None if oracle else bank_for(tr),teacher=teacher,oracle_train=ot if teacher else None,
        scale=scale,teacher_parent=parent,technical=True)
    checked,_=t.validate_fit(tmp_path/task,spec,task,technical=True)
    assert checked==report and report['epochs']==1
    if oracle: assert report['message_disabled']['rows']==len(va)
    else: assert report['message_disabled'] is None and report['validation_history'][0]['kd_loss']==0


@pytest.mark.parametrize('task',['teacher_m32','m32_both_l0p3_s0'])
def test_full30_production_fit(tmp_path,task):
    tr,va,ot,ov,_=caches(); spec=spec_for(tr,va,tmp_path); cfg=c.settings(task); oracle=task in c.TEACHERS
    teacher=None if oracle else m.MessageParT(**c.settings('teacher_m32')).eval().requires_grad_(False)
    report=t.fit(tmp_path/task,spec,task,m.new_model(spec['imported'],cfg),ot if oracle else tr,ov if oracle else va,
        device='cpu',bank=None if oracle else bank_for(tr),teacher=teacher,oracle_train=ot if teacher else None,
        scale=None if oracle else torch.ones(32),teacher_parent=None if oracle else 'c'*64)
    assert report['epochs']==30 and len(report['validation_history'])==30


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_bf16_message_training_and_reload():
    tr,va,ot,ov,source=caches('cuda'); ix=np.arange(3)
    teacher=m.MessageParT(**c.settings('teacher_m32')).cuda(); opt=t.optimizer(teacher)
    for _ in range(2): t.step(teacher,opt,ot.batch(ix),'cuda',lr=.001)
    teacher.eval().requires_grad_(False)
    cfg=c.settings('m32_both_l0p3_s0'); student=m.MessageParT(**cfg).cuda()
    values=t.step(student,t.optimizer(student),tr.batch(ix),'cuda',lr=.001,embedding=torch.zeros(3,128,device='cuda'),
        teacher=teacher,oracle_raw=ot.batch(ix),scale=torch.ones(32,device='cuda'),coefficient=.3)
    assert np.isfinite(values).all()
    student.eval(); student.capture=False; student.supervision=False
    deployed=t.deployment(cfg,student.deployment_state(),'cuda')
    torch.testing.assert_close(t.forward(student,tr.batch(ix),'cuda'),t.forward(deployed,tr.batch(ix),'cuda'),rtol=0,atol=0)


def test_registration_and_controls():
    assert len(c.ARMS)==12 and len(c.TEACHERS)==2
    for task in c.TASKS:
        cfg=c.settings(task)
        assert c.dependencies(task)==([] if cfg['mode'] in ('teacher','ce') else [c.teacher_task(task)])
        if task in c.ARMS:
            assert cfg['embedding']==5
            expected=f'm{cfg["width"]}_ce_s{cfg["seed_index"]}' if cfg['mode']!='aux' else 'U100_W5_HYDRA_REPLAY'
            assert c.architecture_control(task)==expected


def fake_origin(tmp_path):
    from hlt_classification.data.cache_contracts import with_content_hash,write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    sources=tmp_path/'original'; sources.mkdir()
    (sources/'selected.pt').write_bytes(b'checkpoint'); (sources/'predictions.npz').write_bytes(b'predictions')
    teacher=with_content_hash(dict(coordinate='U100',node=dict(node_id='COARSE_U100',context_coordinate=None),
        prepared='p'*64,files=dict(checkpoint=file_ref(sources/'selected.pt'),predictions=file_ref(sources/'predictions.npz'))))
    old=with_content_hash(dict(contract='LUKA_U100_PAIR_CORRECTION_SPEC/v1',counts=c.COUNTS,source_commit='f'*40,
        anchor=dict(evidence=dict(prepared='p'*64,cache='c'*64)),imported=dict(teacher=teacher),
        reference=dict(content_hash='r'*64,environment={})))
    path=sources/'campaign_spec.json'; write_immutable_json(path,old)
    return old,path


def test_encoder_export_and_relocated_verification(tmp_path,monkeypatch):
    old,path=fake_origin(tmp_path)
    monkeypatch.setattr(inputs.transfer,'authenticate',lambda p,s:inputs.origin_proof(s))
    output=tmp_path/'delta'; result=inputs.export(path,output)
    b=SimpleNamespace(json=lambda _:old)
    value=inputs.EncoderDelta(output,result['manifest_hash'],b); value.verify_all()
    assert set(value.rows)=={'U100.pt','validation_predictions.npz'}
    assert result['files']==2 and result['payload_bytes']<128*2**20
    with pytest.raises(PermissionError): value.path('other.pt')
    with pytest.raises(ValueError): inputs.export(path,output)
    (output/'U100.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): value.verify_all()


def test_encoder_export_stops_before_writing_if_origin_or_payload_invalid(tmp_path,monkeypatch):
    old,path=fake_origin(tmp_path)
    monkeypatch.setattr(inputs.transfer,'authenticate',lambda p,s:dict(inputs.origin_proof(s),source_commit='0'*40))
    output=tmp_path/'bad'
    with pytest.raises(ValueError): inputs.export(path,output)
    assert not output.exists()
    monkeypatch.setattr(inputs.transfer,'authenticate',lambda p,s:inputs.origin_proof(s))
    Path(old['imported']['teacher']['files']['checkpoint']['path']).write_bytes(b'bad')
    with pytest.raises(ValueError): inputs.export(path,output)
    assert not output.exists()


@pytest.mark.parametrize('gpu_profile',['l40','a6000'])
def test_campaign_binds_both_deltas_and_fixed_budget(tmp_path,monkeypatch,gpu_profile):
    from hlt_classification.luka_cross_message_hydra import campaign
    roots={key:tmp_path/key for key in ('bundle','delta','encoder','project')}
    for path in roots.values(): path.mkdir()
    origin=dict(reference=dict(environment={}),imported=dict(node=parent_nodes()[0]),anchor=dict(evidence={}))
    b=SimpleNamespace(root=roots['bundle'],json=lambda _:origin)
    d=SimpleNamespace(root=roots['delta'],manifest=dict(content_hash='d'*64),verify_all=lambda:None)
    e=SimpleNamespace(root=roots['encoder'],manifest=dict(content_hash='e'*64),verify_all=lambda:None)
    monkeypatch.setattr(inputs,'Bundle',lambda _:b); monkeypatch.setattr(inputs,'Delta',lambda *a:d)
    monkeypatch.setattr(inputs,'EncoderDelta',lambda *a:e); monkeypatch.setattr(inputs,'delta',lambda *a:d)
    monkeypatch.setattr(inputs,'encoder_delta',lambda *a:e)
    monkeypatch.setattr(campaign.site,'environment',lambda *a:None)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='s'*64))
    monkeypatch.setattr(campaign.portability,'authenticate',lambda *a:dict(content_hash='p'*64))
    monkeypatch.setattr(campaign.portability,'validate_evidence',lambda *a:None)
    monkeypatch.setattr(campaign,'storage',lambda *a:None)
    spec=campaign.create(tmp_path/'run',roots['bundle'],roots['delta'],'d'*64,tmp_path/'precision/spec.json',
        roots['project'],'f'*40,28,roots['encoder'],'e'*64,gpu_profile=gpu_profile)
    assert spec['storage_budget_bytes']==8*2**30 and spec['parents']['encoder']=='e'*64
    assert spec['schema_version']==(3 if gpu_profile=='a6000' else 1)
    campaign.check(spec)
    with pytest.raises(ValueError): campaign.check(dict(spec,encoder_hash='0'*64))


def test_shared_setup_runs_all_fourteen_real_miniatures(tmp_path,monkeypatch):
    from hlt_classification.luka_cross_message_hydra import resources
    tr,va,ot,ov,_=caches(); spec=spec_for(tr,va,tmp_path); (tmp_path/'setup').mkdir()
    monkeypatch.setattr(resources,'miniature_indices',lambda data:np.arange(len(data)))
    monkeypatch.setattr(resources,'gpu_identity',lambda:dict(name='test',total_memory_bytes=2**40))
    value=resources.measure(spec,tr,va,ot,ov,bank_for(tr),1.,device='cpu')
    assert set(value['minutes'])==set(c.TASKS) and value['quality_gate'] is False
    assert resources.accepted(spec)==value


def test_auxiliary_message_loss_reaches_only_upstream_native_blocks():
    tr,va,ot,ov,_=caches(); ix=np.arange(3); student=m.MessageParT(**c.settings('m32_aux_l0p3_s0'))
    student.eval(); student.supervision=True
    t.forward(student,tr.batch(ix),'cpu')
    mask=torch.from_numpy(tr.batch(ix)['mask'][:,0])
    m.loss(student.correction_value,torch.ones_like(student.correction_value),mask,torch.ones(32)).backward()
    assert student.mod.blocks[0].attn.in_proj.weight.grad.abs().sum()>0
    assert student.mod.blocks[3].attn.in_proj.weight.grad.abs().sum()>0
    assert student.mod.blocks[4].attn.in_proj.weight.grad is None


def test_statistics_are_train_only():
    tr,va,ot,ov,_=caches(); teacher=m.MessageParT(**c.settings('teacher_m32'))
    with pytest.raises(PermissionError): t.rms(teacher,ov,'cpu')


def test_full_production_architecture_identity(monkeypatch):
    from hlt_classification.models.particle_transformer import canonical_particle_transformer_config
    config=canonical_particle_transformer_config(); config.update(input_dim=17,num_classes=11,trim=False)
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    tr,va,ot,ov,_=caches(); ix=np.arange(3)
    for task in ('teacher_m32','m32_both_l0p3_s0','m16_ce_s0','m32_aux_l1_s0'):
        value=m.new_model(dict(node=parent_nodes()[0]),c.settings(task)).eval()
        baseline=native.DelphesParticleTransformer().eval(); baseline.mod.load_state_dict(value.mod.state_dict())
        value.supervision=True
        with torch.no_grad():
            actual=t.forward(value,(ot if task in c.TEACHERS else tr).batch(ix),'cpu')
            torch.testing.assert_close(actual,t.forward(baseline,tr.batch(ix),'cpu'),rtol=0,atol=0)
        assert value.correction_value.shape==(3,tr.batch(ix)['features'].shape[-1],c.settings(task)['width'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_setup_workers_and_authenticated_results(tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.luka_cross_message_hydra import worker,campaign,submission
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    u=cache(coordinate='U100'); vu=cache(role='validation',coordinate='U100'); bank=bank_for(tr)
    source=native.DelphesParticleTransformer().cuda().eval(); checkpoint=tmp_path/'U100.pt'
    save_state(checkpoint,{k:v.cpu() for k,v in source.state_dict().items()})
    ref=tmp_path/'oracle.npz'
    atomic_publish_bytes(ref,deterministic_npz_bytes(dict(identities=va.identities,labels=va.labels,
        probabilities=t.predict(source,vu,'cuda'))))
    del source
    folder=tmp_path/'run'; folder.mkdir(); spec=spec_for(tr,va,folder)
    spec.update(project_dir='/project',bundle_root='/bundle',anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(verify_all=lambda:None))
    monkeypatch.setattr(inputs,'delta',lambda _:SimpleNamespace(verify_all=lambda:None))
    monkeypatch.setattr(inputs,'encoder_delta',lambda _:SimpleNamespace(verify_all=lambda:None,
        path=lambda n:checkpoint if n=='U100.pt' else ref))
    monkeypatch.setattr(inputs,'load_view',lambda s,role,coordinate:tr if role=='train' and coordinate=='D000' else u if role=='train' else va)
    monkeypatch.setattr(inputs,'oracle_validation',lambda s:vu)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    monkeypatch.setattr(worker.portability,'setup_record',lambda *a:c.artifact('TEST_PORTABILITY'))
    monkeypatch.setattr(worker.portability,'validate_setup',lambda *a:None)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    jobs={'setup':'7000',**{task:str(7001+i) for i,task in enumerate(c.TASKS)}}
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:jobs)
    current=['setup']; monkeypatch.setattr(worker.site,'allocation',lambda s:jobs[current[0]])
    worker.run(spec,'setup'); assert len(worker.completed_setup(spec)['minutes'])==14
    for task in ('teacher_m32','m32_both_l0p3_s0'):
        current[0]=task; worker.run(spec,task); worker.receipt(spec,task)
    # The architecture control must never load frozen features or a selected teacher.
    monkeypatch.setattr(inputs,'oracle_cache',lambda *a:pytest.fail('Control opened oracle tokens'))
    monkeypatch.setattr(worker,'load_teacher',lambda *a:pytest.fail('Control opened teacher weights'))
    current[0]='m32_ce_s0'; worker.run(spec,current[0]); worker.receipt(spec,current[0])
    report,_=t.validate_fit(folder/'m32_both_l0p3_s0/fit',spec,'m32_both_l0p3_s0')
    parent,_=t.validate_fit(folder/'teacher_m32/fit',spec,'teacher_m32')
    assert report['parents']['teacher']==parent['content_hash']
    subset=tmp_path/'subset.npz'
    atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(len(va),dtype=np.int64))))
    spec.update(original50k_rows=len(va),portability_evidence=dict(refs=dict(predictions=dict(path=str(ref)))))
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda n:subset if n.endswith('subset.npz') else ref,
        json=lambda n:bank.manifest))
    monkeypatch.setattr(t,'predict',lambda *a:pytest.fail('Results must not infer'))
    result=worker.results(spec); rows={r['model']:r for r in result['rows']}
    row=rows['m32_both_l0p3_s0']; control=rows['m32_ce_s0']
    assert row['architecture_control']=='m32_ce_s0' and not row['privileged_oracle']
    assert row['validation_delta_architecture_control']==row['validation']['macro_ovr_auc']-control['validation']['macro_ovr_auc']
    assert rows['teacher_m32_message_off']['selected_epoch']==rows['teacher_m32']['selected_epoch']
    assert rows['teacher_m32_message_off']['diagnostic'] and 'teacher_m16' in result['pending']
    worker.print_results(result)


def test_progress_uses_published_epoch_without_inference(tmp_path,monkeypatch,capsys):
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.luka_cross_message_hydra import progress,campaign
    spec=dict(root=str(tmp_path),bundle_root='/bundle',content_hash='a'*64,counts=dict(validation=11))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    bundle=SimpleNamespace(json=lambda n:dict(recovery_anchors=dict(M0HLT=.9,OFFLINE=1.)) if n.startswith('origin')
        else dict(validation_history=[dict(pass_number=i,validation=dict(macro_ovr_auc=.94)) for i in range(1,31)]))
    monkeypatch.setattr(progress.inputs,'Bundle',lambda _:bundle)
    task='m32_both_l0p3_s0'; folder=tmp_path/task/'fit'; folder.mkdir(parents=True)
    write_immutable_json(folder/'epoch_01.json',c.artifact('EPOCH',parents=dict(spec='a'*64),
        task=task,pass_number=1,validation=dict(macro_ovr_auc=.95,rows=11)))
    (tmp_path/(task+'-123.out')).write_text('CROSS_MESSAGE '+task+' epoch=2/30 batch=2/4\n')
    progress.show(spec); output=capsys.readouterr().out
    assert '0.95000000' in output and '50.0%' in output and 'batch 2/4' in output


def test_exact_submission_dry_run(tmp_path):
    from hlt_classification.luka_cross_message_hydra import campaign,submission
    spec=dict(campaign.registration(),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    value=submission._submit(spec,submission.render(spec,'envelope'),execute=False)
    assert value['dry_run'] and len(value['jobs'])==15
    assert not submission.accepted_jobs(spec,'setup')


def test_finish_reuses_setup_and_is_idempotent(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.luka_cross_message_hydra import campaign,submission,worker
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
    from hlt_classification.luka_cross_message_hydra import campaign,submission,worker
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


def test_shell_launcher_returns_before_detached_submission_finishes(tmp_path):
    import os,subprocess,sys,time
    from pathlib import Path
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def sh(p):
        p=Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    repo=tmp_path/'project'; (repo/'scripts').mkdir(parents=True)
    name='queue_luka_cross_message_hydra.sh'
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
        'python3':'case "$*" in *content_hash*) printf "%064d\\n" 3;; *delta_root*) printf "%s\\n" "$FIXTURE_DELTA";; *delta_hash*) printf "%064d\\n" 1;; esac\n',
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
*" dispatch "*) [[ "$*" == *"--authorization AUTHORIZE_CROSS_MESSAGE_HYDRA_V1"* ]] || exit 9; sleep 1; printf 'DONE\\n'; printf done > "$FIXTURE_DONE";;
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
    args=[bash,str(script),sh(base),sh(control),sh(delta),'8']
    result=subprocess.run(args,env=environ,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert 'Dispatcher PID:' in result.stdout
    deadline=time.monotonic()+10
    while not done.exists() and time.monotonic()<deadline: time.sleep(.05)
    assert done.read_text()=='done'
    logs=list((base/'runs').glob('cross_message_*/submit.*.log'))
    assert len(logs)==1 and 'DONE' in logs[0].read_text()
    assert subprocess.run(args,env=environ,capture_output=True,text=True,timeout=10).returncode!=0
    # setsid is stubbed here: this tests process handoff, not survival of a
    # real Brown SSH disconnect or institutional login-node cleanup policy.
