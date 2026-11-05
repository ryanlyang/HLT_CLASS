"""Shared-route science, installed-Weaver kernels and exact submission contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import json
import subprocess
import sys
import pytest
import torch
from test_luka_attention import cache,parent_nodes
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.luka_cross_message_hydra import inputs as donor_inputs,model as donor_model
from hlt_classification.luka_cross_message_hydra.training import forward,predict,optimizer
from hlt_classification.luka_shared_message_hydra import contracts as c,model as m,training as t
from hlt_classification.luka_shared_message_hydra import campaign,submission,resources,worker,inputs,reporting
from hlt_classification.data.cache_contracts import load_json,with_content_hash,write_immutable_json


@pytest.fixture(autouse=True)
def compact_native(monkeypatch):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    config=native.model_config(); config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=8,num_layers=8,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    yield
    torch.set_num_threads(previous)


def fixtures():
    tr=cache(coordinate='D000'); va=cache(role='validation',coordinate='D000')
    source=native.DelphesParticleTransformer().eval().requires_grad_(False)
    oracle=donor_inputs.build_cache(tr,cache(coordinate='U100'),source,'cpu',budget_bytes=2**30)
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='b'*64))
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=len(tr),validation=len(va)),
        encoder_hash='e'*64,teacher_proof=dict(report_hash='f'*64),embedding_hash='b'*64,
        anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    return tr,va,oracle,bank,spec


def test_registration_and_schedule():
    assert len(c.TASKS)==12
    assert sum(a['mode']=='joint' for a in c.ARMS.values())==9
    assert c.recipe()['embedding_weight']==5
    a=c.ARMS['joint_b0p1_s0']
    assert c.coefficients(a,3)==(.25,0)
    assert c.coefficients(a,4.5)==(.25,.05)
    assert c.coefficients(a,6)==(.25,.1)
    assert c.coefficients(c.ARMS['joint_fade_s0'],15)==(.125,.05)
    assert c.coefficients(c.ARMS['joint_fade_s0'],20)==(0,0)
    with pytest.raises(ValueError): c.coefficients(a,float('nan'))


def test_native_parity_shared_prefix_and_stripped_deployment():
    tr,va,oracle,bank,spec=fixtures(); task=c.TASKS[0]
    model=m.new_model(spec['imported'],task).eval(); base=native.DelphesParticleTransformer().eval()
    base.mod.load_state_dict(model.mod.state_dict())
    raw=tr.batch(np.arange(3))
    torch.testing.assert_close(forward(model,raw,'cpu'),forward(base,raw,'cpu'),rtol=0,atol=0)
    model.train(); model.capture=True
    calls=[0,0]
    def pre(*args): calls[0]+=1
    def post(*args): calls[1]+=1
    h1=model.mod.blocks[0].register_forward_hook(pre); h2=model.mod.blocks[7].register_forward_hook(post)
    data={k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')}
    out=model.training_forward(**data,u_tokens=torch.from_numpy(oracle.batch(np.arange(3))['u_tokens']))
    assert calls==[1,2]; h1.remove(); h2.remove()
    assert out[1].shape==out[0].shape==(3,11)
    model.eval(); model.capture=False
    state=model.deployment_state(); restored=m.deployment(state,'cpu')
    assert not any(k.startswith(('predictor.','privileged.')) for k in state)
    assert not hasattr(restored,'privileged') and not hasattr(restored,'predictor')
    torch.testing.assert_close(forward(restored,raw,'cpu'),forward(model,raw,'cpu'),rtol=0,atol=0)
    with pytest.raises(PermissionError): forward(restored,oracle.batch(np.arange(3)),'cpu')


def test_matched_weights_rng_and_learned_geometry():
    imp=dict(node=parent_nodes()[0])
    for s in (0,1):
        a=m.new_model(imp,f'control_s{s}'); cpu=torch.get_rng_state().clone()
        b=m.new_model(imp,f'joint_b0p1_s{s}')
        assert torch.equal(cpu,torch.get_rng_state())
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    raw=cache(coordinate='D000').batch(np.arange(3)); model=m.new_model(imp,'joint_b0p1_s0')
    model.capture=True
    with torch.no_grad(): model.expansion.weight.normal_(std=.01)
    forward(model,raw,'cpu').square().mean().backward()
    assert model.message.pair_map.weight.grad.abs().sum()>0
    assert model.mod.blocks[0].attn.in_proj.weight.grad.abs().sum()>0


@pytest.mark.parametrize('task',c.TASKS)
def test_objective_gradients_and_real_miniature(tmp_path,task):
    tr,va,oracle,bank,spec=fixtures(); cfg=c.ARMS[task]
    teacher=None
    if cfg['mode']=='frozen':
        teacher=donor_model.MessageParT(width=16,mode='teacher',embedding=0).eval().requires_grad_(False)
    report=t.fit(tmp_path/task,spec,task,tr,va,bank,device='cpu',technical=True,
        oracle_train=None if cfg['mode']=='control' else oracle,teacher=teacher)
    assert report['technical'] and report['selected_pass']==1
    row=report['validation_history'][0]
    assert row['embedding_loss']>0
    assert row['weighted_agreement']>0 if cfg['beta'] else row['weighted_agreement']==0
    assert row['ce_privileged']>0 if cfg['mode']=='joint' else row['ce_privileged']==0
    assert all(p.grad is None for p in teacher.parameters()) if teacher else True


def test_agreement_gradient_direction_and_mask():
    mask=torch.tensor([[True,True,False]])
    a=torch.randn(1,3,16,requires_grad=True); b=torch.randn(1,3,16,requires_grad=True)
    value=t.agreement(m.normalize(a,mask),m.normalize(b,mask),mask)
    value.backward(); assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert a.grad[:,2].abs().sum()==0
    a.grad=None; b.grad=None
    t.agreement(m.normalize(a,mask),m.normalize(b,mask).detach(),mask).backward()
    assert a.grad.abs().sum()>0 and b.grad is None


def queue_spec(root):
    return dict(campaign.registration(),content_hash='a'*64,root=str(root),project_dir='/code',
        environment_dir='/env',reference=dict(environment=native.installed_environment()),
        qualification=dict(acceptance=dict(gpu=dict(name='NVIDIA RTX A6000',total_memory_bytes=48*2**30))))


def test_durable_bootstrap_independent_fits_and_cpu_report(tmp_path):
    spec=queue_spec(tmp_path); bootstrap=submission.plan(spec,'bootstrap')
    a=dict(content_hash='b'*64,minutes={task:600 for task in c.TASKS})
    science=submission.plan(spec,'science',a)
    assert [r['task_id'] for r in bootstrap['commands']]==['setup','release']
    release=bootstrap['commands'][1]
    assert release['dependencies']==['setup'] and '--dependency=afterok:${JOB_setup}' in release['command']
    assert '--partition=compute' in release['command'] and not any('--gres=' in x for x in release['command'])
    assert len(science['commands'])==13
    for row in science['commands'][:-1]:
        assert row['dependencies']==[] and '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
        assert '--mem=32000M' in row['command'] and '--cpus-per-task=6' in row['command']
        assert not any(x.startswith(('--dependency=','--nodelist=')) for x in row['command'])
    assert science['commands'][-1]['dependencies']==list(c.TASKS)
    assert any(x.startswith('--dependency=afterany:') for x in science['commands'][-1]['command'])


def test_exact_launch_release_idempotence_and_ambiguous_partial(tmp_path,monkeypatch):
    spec=queue_spec(tmp_path); monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    reviews=[]; monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p['stage']))
    calls=[]
    def run(cmd,**kw):
        calls.append(cmd); return SimpleNamespace(stdout=str(9300+len(calls)),returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    dry=submission.launch(spec); assert dry['dry_run'] and not calls
    p=submission.plan(spec,'bootstrap')
    with pytest.raises(PermissionError): submission.launch(spec,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    live=submission.launch(spec,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert live['jobs']==dict(setup='9301',release='9302')
    assert '--dependency=afterok:9301' in calls[1]
    assert submission.launch(spec,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)==live
    assert len(calls)==2
    a=dict(content_hash='b'*64,minutes={task:500 for task in c.TASKS})
    write_immutable_json(tmp_path/'setup/acceptance.json',a)
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    science=submission.finish(spec)
    assert len(calls)==15 and set(science['jobs'])==set(c.TASKS)|{'report'}
    assert reviews==['bootstrap','science']
    assert submission.finish(spec)==science and len(calls)==15
    # A lost sbatch response must never cause automatic duplicate submission.
    other=tmp_path/'partial'; other.mkdir(); partial=queue_spec(other)
    submission.launch(partial); (other/'submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous/partial'):
        submission.launch(partial,execute=True,plan_hash=submission.plan(partial,'bootstrap')['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==15


def test_failed_setup_cannot_release_science(tmp_path,monkeypatch):
    spec=queue_spec(tmp_path); monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    bootstrap=submission.plan(spec,'bootstrap')
    write_immutable_json(tmp_path/'submission_bootstrap/live_claim/intent.json',
        c.artifact('SUBMIT_CLAIM',parents=dict(plan=bootstrap['content_hash']),authorization=c.AUTHORIZATION))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(setup='123'))
    def failed(s): raise FileNotFoundError('No authenticated setup receipt')
    monkeypatch.setattr(worker,'completed_setup',failed)
    monkeypatch.setattr(submission,'submit_stage',lambda *a,**k:pytest.fail('Science must not be submitted'))
    with pytest.raises(FileNotFoundError): submission.finish(spec)


def timing_report(spec,task,seconds=.2):
    return dict(task=task,technical=True,epochs=1,content_hash='r'*64,parents=dict(spec=spec['content_hash']),
        counts=dict(train=258,validation=258),validation_history=[dict(batch_seconds=[seconds,seconds,.05],
        batch_sizes=[128,128,2],validation_seconds=.2,validation=dict(macro_ovr_auc=.05))])


def test_warmed_resource_projection_memory_and_no_quality_gate(tmp_path):
    spec=queue_spec(tmp_path); task=c.TASKS[0]; report=timing_report(spec,task)
    warm=timing_report(spec,task,500); gpu=spec['qualification']['acceptance']['gpu']
    memory=dict(gpu_peak_bytes=8*2**30,peak_rss_bytes=14*2**30)
    a=resources.resource_record(spec,task,report,warm,memory,gpu,60)
    assert a['passed'] and a['quality_gate'] is False and a['minutes']<600
    slow=resources.resource_record(spec,task,warm,report,memory,gpu,60)
    assert not slow['passed'] and slow['violations'][0]['resource']=='minutes'
    memory=dict(gpu_peak_bytes=48*2**30,peak_rss_bytes=32*2**30)
    bad=resources.resource_record(spec,task,report,warm,memory,gpu,60)
    assert {v['resource'] for v in bad['violations']}=={'gpu_peak_bytes','peak_rss_bytes'}
    report['validation_history'][0]['batch_seconds'][0]=float('nan')
    with pytest.raises(ValueError): resources.resource_record(spec,task,report,warm,memory,gpu,60)


def test_teacher_proof_runs_original_source(tmp_path,monkeypatch):
    parent=dict(project_dir='/original',root=str(tmp_path),content_hash='p'*64)
    root=tmp_path/'teacher_m16'; (root/'fit').mkdir(parents=True)
    for path in ('receipt.json','fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz'):
        (root/path).write_bytes(b'original')
    def run(cmd,**kw):
        assert cmd[:4]==[sys.executable,'-B','-s','-c']
        assert 'worker.receipt' in cmd[4] and 'training.validate_fit' in cmd[4] and 'sys.path.insert' in cmd[4]
        assert kw['cwd']=='/original' and kw['env']['PYTHONPATH']==str(Path('/original/src'))
        return SimpleNamespace(returncode=0,stderr='',stdout='SHARED_TEACHER='+json.dumps(dict(receipt='r'*64,job_id='145410',report='t'*64)))
    monkeypatch.setattr(subprocess,'run',run)
    value=campaign.teacher_proof('/parent.json',parent)
    assert value['report_hash']=='t'*64 and value['job_id']=='145410' and len(value['files'])==5


def test_input_roles_and_fade_skip_privileged_route(monkeypatch):
    with pytest.raises(PermissionError): inputs.tokens({},cache(role='validation'))
    with pytest.raises(PermissionError): inputs.load_hlt({},'test')
    tr,va,oracle,bank,spec=fixtures(); task='joint_fade_s0'; model=m.new_model(spec['imported'],task)
    def forbidden(*a,**k): pytest.fail('Faded privileged producer ran')
    monkeypatch.setattr(model.privileged,'forward',forbidden)
    ix=np.arange(3)
    values=t.step(model,optimizer(model),tr.batch(ix),'cpu',setting=c.ARMS[task],position=25,
        embedding=bank.targets[ix],oracle_raw=oracle.batch(ix))
    row=dict(zip(t.LOSS_KEYS,values))
    assert row['ce_privileged']==row['agreement_loss']==row['weighted_agreement']==0
    assert row['embedding_loss']>0 and row['weighted_ce']==row['ce_hlt']


@pytest.mark.parametrize('task',['joint_b0p1_s0','control_s0','joint_student_only_s0'])
def test_production_width_cuda_full_and_partial_publication(tmp_path,monkeypatch,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    # Restore the authoritative production widths, not the compact test fixture.
    config=native.canonical_particle_transformer_config()
    config.update(input_dim=17,num_classes=11,trim=False)
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    tr=cache(coordinate='D000',rows=130); va=cache(role='validation',coordinate='D000',rows=130)
    source=native.DelphesParticleTransformer().cuda().eval().requires_grad_(False)
    oracle=donor_inputs.build_cache(tr,cache(coordinate='U100',rows=130),source,'cuda',budget_bytes=2**30)
    del source
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((130,128),np.float32),manifest=dict(content_hash='b'*64))
    spec=dict(content_hash='a'*64,imported=dict(node=parent_nodes()[0]),counts=dict(train=130,validation=130),
        encoder_hash='e'*64,embedding_hash='b'*64,anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    report=t.fit(tmp_path/task,spec,task,tr,va,bank,device='cuda',technical=True,
                 oracle_train=None if task=='control_s0' else oracle)
    assert report['validation_history'][0]['batch_sizes']==[128,2]
    state=torch.load(tmp_path/task/'selected.pt',weights_only=True)
    assert not any(k.startswith(('predictor.','privileged.')) for k in state)


def test_new_spec_reuses_parent_and_rejects_mutation(tmp_path,monkeypatch):
    locations={k:tmp_path/k for k in ('parent','project','bundle','delta','encoder','source')}
    for path in locations.values(): path.mkdir()
    parent=dict(content_hash='p'*64,root=str(locations['parent']),project_dir=str(locations['source']),
        environment_dir=sys.prefix,bundle_root=str(locations['bundle']),delta_root=str(locations['delta']),
        encoder_root=str(locations['encoder']),delta_hash='d'*64,encoder_hash='e'*64,
        reference=dict(environment=native.installed_environment()),imported=dict(node=parent_nodes()[0]),
        anchor=dict(evidence=dict(prepared='a'*64)))
    path=locations['parent']/'campaign_spec.json'; path.write_text(json.dumps(parent))
    qualification=dict(content_hash='q'*64); teacher=dict(content_hash='t'*64,report_hash='r'*64)
    monkeypatch.setattr(campaign,'authenticate_parent',lambda p:(parent,qualification))
    monkeypatch.setattr(campaign,'teacher_proof',lambda *a:teacher)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='s'*64))
    monkeypatch.setattr(campaign,'Bundle',lambda p:SimpleNamespace(json=lambda name:dict(content_hash='b'*64)))
    spec=campaign.create(tmp_path/'new',path,locations['project'],'f'*40,4)
    assert campaign.check(spec)==parent and spec['embedding_hash']=='b'*64
    assert spec['recipe']['embedding_weight']==5 and spec['teacher_proof']==teacher
    with pytest.raises(ValueError): campaign.create(tmp_path/'new',path,locations['project'],'f'*40,4)
    with pytest.raises(ValueError): campaign.create(locations['parent']/'unsafe',path,locations['project'],'f'*40,4)
    bad=deepcopy(spec); bad['recipe']['embedding_weight']=0
    with pytest.raises(ValueError): campaign.check(with_content_hash(bad))


def test_source_rejects_donor_runtime_changes(monkeypatch):
    project=Path(campaign.__file__).resolve().parents[3]
    monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='s'*64))
    changes=['A\t'+p for p in sorted(campaign.RUNTIME)]
    def run(cmd,**kw):
        return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED if cmd[3]=='ls-files' else changes if cmd[3]=='diff' else []))
    monkeypatch.setattr(subprocess,'run',run)
    assert campaign.current(project,'f'*40)['content_hash']=='s'*64
    changes.append('M\tsrc/hlt_classification/luka_cross_message_hydra/model.py')
    with pytest.raises(ValueError,match='additive'): campaign.current(project,'f'*40)


def test_joint_agreement_can_train_both_producers_without_ce_target_leak():
    tr,va,oracle,bank,spec=fixtures(); ix=np.arange(3)
    for detach in (False,True):
        model=m.new_model(spec['imported'],'joint_b0p1_s0')
        cfg=dict(c.ARMS['joint_b0p1_s0'],alpha=0.,detach_target=detach)
        t.step(model,optimizer(model),tr.batch(ix),'cpu',setting=cfg,position=7,
            embedding=bank.targets[ix],oracle_raw=oracle.batch(ix))
        assert model.message.compress[-1].weight.grad.abs().sum()>0
        gradient=model.privileged.compress[-1].weight.grad
        if detach: assert gradient is None or gradient.abs().sum()==0
        else: assert gradient.abs().sum()>0


def test_fresh_full_fit_and_tamper_rejection(tmp_path):
    tr,va,oracle,bank,spec=fixtures(); task='joint_fade_s0'
    report=t.fit(tmp_path/'fit',spec,task,tr,va,bank,device='cpu',oracle_train=oracle)
    assert report['epochs']==30 and len(report['validation_history'])==30
    assert report['validation_history'][0]['weighted_agreement']==0
    assert report['validation_history'][6]['weighted_agreement']>0
    assert report['validation_history'][-1]['ce_privileged']==0
    assert all(r['embedding_loss']>0 for r in report['validation_history'])
    (tmp_path/'fit/selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): t.validate_fit(tmp_path/'fit',spec,task)


def test_setup_measurement_warms_fresh_models_and_authenticates_all_fits(tmp_path,monkeypatch):
    # Run one real production pair here; all twelve objectives have individual
    # executable miniature tests above. Limit only measurement orchestration.
    task='joint_b0p1_s0'; tr=cache(coordinate='D000',rows=257); va=cache(role='validation',coordinate='D000',rows=257)
    source=native.DelphesParticleTransformer().eval().requires_grad_(False)
    oracle=donor_inputs.build_cache(tr,cache(coordinate='U100',rows=257),source,'cpu',budget_bytes=2**30)
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((257,128),np.float32),manifest=dict(content_hash='b'*64))
    spec=queue_spec(tmp_path); spec.update(imported=dict(node=parent_nodes()[0]),encoder_hash='e'*64,embedding_hash='b'*64,
        anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    (tmp_path/'setup').mkdir()
    monkeypatch.setattr(resources,'TASKS',(task,))
    monkeypatch.setattr(resources,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(resources,'peaks',lambda device:dict(gpu_peak_bytes=0,peak_rss_bytes=2**30))
    initial=[]; original=t.new_model
    def observed(*args):
        model=original(*args); initial.append({k:v.clone() for k,v in model.state_dict().items()}); return model
    monkeypatch.setattr(t,'new_model',observed)
    a=resources.measure(spec,tr,va,oracle,bank,1.,device='cpu')
    assert resources.accepted(spec)==a
    assert len(initial)==2 and all(torch.equal(v,initial[1][k]) for k,v in initial[0].items())
    (tmp_path/'setup'/('warm_'+task)/'selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): resources.accepted(spec)


def test_cpu_worker_profile_and_wrong_job_rejected(monkeypatch,tmp_path):
    spec=queue_spec(tmp_path); monkeypatch.setenv('SLURM_JOB_ID','123')
    record='JobId=123 JobState=RUNNING Partition=compute Account=default QOS=normal NumCPUs=1 NumNodes=1 AllocTRES=cpu=1,mem=4000M,node=1'
    def run(cmd,**kwargs):
        return SimpleNamespace(stdout='ClusterName = hydra\n' if cmd[-1]=='config' else record)
    monkeypatch.setattr(subprocess,'run',run)
    assert worker.cpu_allocation(spec)=='123'
    monkeypatch.setenv('SLURM_JOB_ID','456')
    with pytest.raises(PermissionError): worker.cpu_allocation(spec)


def test_hlt_message_is_equivariant_and_ignores_padded_keys():
    predictor=m.HLTRelationalPredictor().eval()
    context=torch.randn(2,5,128); features=torch.randn(2,17,5); pair=torch.randn(2,8,5,5)
    mask=torch.tensor([[True,True,True,False,False],[True]*5])
    original=predictor(context,features,mask,pair)
    permutation=torch.tensor([2,0,4,1,3])
    moved=predictor(context[:,permutation],features[:,:,permutation],mask[:,permutation],pair[:,:,permutation][:,:,:,permutation])
    torch.testing.assert_close(moved,original[:,permutation],atol=2e-6,rtol=2e-5)
    changed=features.clone(); changed[0,:,3:]=10000
    changed_pair=pair.clone(); changed_pair[0,:,:,3:]=-10000
    torch.testing.assert_close(predictor(context,changed,mask,changed_pair)[0,:3],original[0,:3],atol=1e-6,rtol=1e-5)
    assert original[0,3:].abs().sum()==0


@pytest.mark.parametrize('rows',[256,257,258,500])
def test_miniature_preserves_worst_full_batch_and_safe_partial(rows):
    tr=cache(coordinate='D000',rows=rows)
    ix=resources.miniature_rows(tr)
    lengths=np.concatenate([np.diff(b.offsets) for b in tr.blocks])
    largest=np.argsort(-lengths,kind='stable')[:128]
    np.testing.assert_array_equal(ix[:128],largest)
    assert len(ix)>128 and 2<=len(ix)%128<=127 and len(np.unique(ix))==len(ix)
    assert set(tr.labels[ix])==set(range(11))


@pytest.mark.parametrize('task',['control_s0','joint_b0p1_s0'])
def test_exact_worker_publishes_fresh_science_and_only_train_tokens(tmp_path,monkeypatch,task):
    tr,va,oracle,bank,fit_spec=fixtures(); spec=queue_spec(tmp_path); spec.update(fit_spec,root=str(tmp_path))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:spec)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.execution,'allocation',lambda s:'1234')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{task:'1234'})
    monkeypatch.setattr(worker,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:spec['qualification']['acceptance'])
    calls=[]
    def load(s,role): calls.append(('hlt',role)); return tr if role=='train' else va
    def tokens(s,train):
        assert train is tr; calls.append(('tokens',train.role)); return oracle
    monkeypatch.setattr(inputs,'load_hlt',load); monkeypatch.setattr(inputs,'tokens',tokens)
    monkeypatch.setattr(inputs,'bank',lambda s,train:bank)
    original=t.fit
    def cpu_fit(*args,**kwargs): return original(*args,**dict(kwargs,device='cpu'))
    monkeypatch.setattr(t,'fit',cpu_fit)
    value=worker.run(spec,task)
    assert worker.receipt(spec,task)==value and value['job_id']=='1234'
    expected=[('hlt','train'),('hlt','validation')]+([('tokens','train')] if task.startswith('joint') else [])
    assert calls==expected
    report,_=t.validate_fit(tmp_path/task/'fit',spec,task)
    assert report['epochs']==30 and report['technical'] is False
    with pytest.raises(FileExistsError): worker.run(spec,task)
