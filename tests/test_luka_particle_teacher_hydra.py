"""Aligned full oracle tokens, native students and authenticated Hydra execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_attention import cache,parent_nodes
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.luka_cross_message_hydra import model as teacher_model
from hlt_classification.luka_cross_message_hydra.training import forward,predict,optimizer
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.luka_particle_teacher_hydra import contracts as c,model as m,training as t
from hlt_classification.luka_particle_teacher_hydra import campaign,submission,resources,worker,inputs,reporting
from hlt_classification.data.cache_contracts import load_json,with_content_hash,write_immutable_json


@pytest.fixture(autouse=True)
def compact_native(monkeypatch):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    config=native.model_config(); config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=8,num_layers=8,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    yield
    torch.set_num_threads(previous)


def fixtures(root,rows=22,device='cpu'):
    tr=cache(coordinate='D000',rows=rows); va=cache(role='validation',coordinate='D000',rows=rows)
    rich=cache(coordinate='U100',rows=rows)
    encoder=native.DelphesParticleTransformer().to(device).eval().requires_grad_(False)
    teacher=teacher_model.MessageParT(width=16,mode='teacher',embedding=0).to(device).eval().requires_grad_(False)
    with torch.no_grad(): teacher.expansion.weight.normal_(std=.1)
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((len(tr),128),np.float32),manifest=dict(content_hash='b'*64))
    spec=dict(campaign.registration(),root=str(root),content_hash='a'*64,imported=dict(node=parent_nodes()[0]),
        counts=dict(train=len(tr),validation=len(va)),encoder_hash='e'*64,teacher_proof=dict(report_hash='f'*64),
        embedding_hash='b'*64,anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    targets=inputs.build_targets(spec,tr,rich,encoder,teacher,device,budget_bytes=2**30)
    write_immutable_json(root/'setup/targets.json',targets.metadata)
    return tr,va,targets,bank,spec


def test_registration_and_schedule():
    assert len(c.TASKS)==12 and sum(not a['depths'] for a in c.ARMS.values())==2
    assert c.recipe()['embedding_weight']==5 and c.recipe()['student_depth']==8
    a=c.ARMS['final_l1_s0']
    assert c.coefficient(a,0)==0 and c.coefficient(a,1.5)==.5 and c.coefficient(a,3)==1
    with pytest.raises(ValueError): c.coefficient(a,float('nan'))


def test_native_parity_and_matched_rng_and_export():
    imp=dict(node=parent_nodes()[0]); raw=cache(coordinate='D000').batch(np.arange(4))
    for seed in (0,1):
        control=m.new_model(imp,f'control_s{seed}').eval(); rng=torch.get_rng_state().clone()
        student=m.new_model(imp,f'final_l1_s{seed}').eval()
        assert torch.equal(rng,torch.get_rng_state())
        for key,value in control.state_dict().items(): assert torch.equal(value,student.state_dict()[key])
        torch.manual_seed(c.seed(imp,c.ARMS[f'control_s{seed}'],'initialization_seed'))
        baseline=AuxiliaryParT().eval()
        for key,value in baseline.state_dict().items(): assert torch.equal(value,control.state_dict()[key])
        expected=forward(baseline,raw,'cpu')
        student.capture=True
        torch.testing.assert_close(expected,forward(student,raw,'cpu'),atol=0,rtol=0)
        restored=m.deployment(student.deployment_state(),'cpu')
        torch.testing.assert_close(expected,forward(restored,raw,'cpu'),atol=0,rtol=0)
        assert not hasattr(restored,'token_heads') and not hasattr(restored,'predictor')
        with pytest.raises(PermissionError): forward(restored,dict(raw,u_tokens=np.zeros((4,4,128),np.float32)),'cpu')


def test_teacher_targets_are_full_active_oracle_outputs_and_normalized(tmp_path,monkeypatch):
    tr=cache(coordinate='D000'); rich=cache(coordinate='U100')
    spec=dict(content_hash='a'*64,counts=dict(train=len(tr)),encoder_hash='e'*64,
        teacher_proof=dict(report_hash='f'*64),anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    encoder=native.DelphesParticleTransformer().eval().requires_grad_(False)
    teacher=teacher_model.MessageParT(width=16,mode='teacher',embedding=0).eval().requires_grad_(False)
    with torch.no_grad(): teacher.expansion.weight.normal_(std=.2)
    original=teacher.oracle; calls=[]; capture={}
    handles=[teacher.mod.blocks[d-1].register_forward_hook(lambda _m,_a,out,d=d:capture.update({d:out.detach().clone()})) for d in (6,8)]
    def spy(*a,**kw): calls.append(kw['u_tokens'].shape); return original(*a,**kw)
    monkeypatch.setattr(teacher,'oracle',spy)
    targets=inputs.build_targets(spec,tr,rich,encoder,teacher,'cpu',budget_bytes=2**30)
    for handle in handles: handle.remove()
    assert len(calls)==1 and all(p.grad is None for p in teacher.parameters())
    raw=tr.batch(np.arange(len(tr))); valid=raw['mask'][:,0]
    for d in (6,8):
        np.testing.assert_array_equal(targets.arrays[d],capture[d].numpy()[valid])
        stats=targets.metadata['statistics'][str(d)]; values=targets.arrays[d].astype(np.float64)
        np.testing.assert_allclose(stats['mean'],values.mean(0),atol=1e-12)
        np.testing.assert_allclose(stats['scale'],np.maximum(values.std(0),.001),atol=1e-12)
        norm=targets.batch(np.arange(len(tr)),valid,[d])[d][valid]
        np.testing.assert_allclose(norm.mean(0),0.,atol=1e-6)
    repeated=inputs.build_targets(spec,tr,rich,encoder,teacher,'cpu',budget_bytes=2**30)
    assert repeated.metadata==targets.metadata
    # Removing privileged contribution changes the target definition.
    with torch.no_grad(): teacher.expansion.weight.zero_()
    other=inputs.build_targets(spec,tr,rich,encoder,teacher,'cpu',budget_bytes=2**30)
    assert other.metadata['array_hashes']!=targets.metadata['array_hashes']
    with pytest.raises(MemoryError): inputs.build_targets(spec,tr,rich,encoder,teacher,'cpu',budget_bytes=1)
    teacher.message_enabled=False
    with pytest.raises(ValueError): inputs.build_targets(spec,tr,rich,encoder,teacher,'cpu',budget_bytes=2**30)
    with pytest.raises(PermissionError): inputs.targets(spec,cache(role='validation'))
    with pytest.raises(PermissionError): inputs.load_hlt(spec,'test')


def test_loss_jet_weighting_centering_padding_permutation_and_stop_gradient():
    mask=torch.tensor([[True,False,False],[True,True,True]])
    p=torch.zeros(2,3,128,requires_grad=True); target=torch.zeros_like(p,requires_grad=True)
    with torch.no_grad(): target[0,0]=2.; target[1]=1.; target[0,1:]=float('nan')
    loss,detail=m.particle_loss({8:p},{8:target},mask)
    assert loss.item()==pytest.approx(1.)  # mean of 1.5 and .5, each jet equal
    loss.backward(); assert target.grad is None and p.grad[0,1:].abs().sum()==0
    mixed,_=m.particle_loss({8:p},{8:target},mask,centered=True)
    assert mixed.item()==pytest.approx(.5)  # each target is constant within its jet
    perm=torch.tensor([2,0,1])
    changed,_=m.particle_loss({8:p[:,perm]},{8:target[:,perm]},mask[:,perm])
    torch.testing.assert_close(changed,loss)
    multi,_=m.particle_loss({6:p,8:p},{6:target,8:target},mask)
    torch.testing.assert_close(multi,loss)


def test_particle_loss_reaches_encoder_through_head(tmp_path):
    tr,va,targets,bank,spec=fixtures(tmp_path); student=m.new_model(spec['imported'],'multi_l1_s0')
    student.capture=True; raw=tr.batch(np.arange(4)); forward(student,raw,'cpu')
    target={d:torch.from_numpy(a) for d,a in targets.batch(np.arange(4),raw['mask'][:,0],[6,8]).items()}
    loss,_=m.particle_loss(student.token_predictions,target,torch.from_numpy(raw['mask'][:,0]))
    loss.backward()
    assert student.mod.blocks[0].attn.in_proj.weight.grad.abs().sum()>0
    assert all(head[-1].weight.grad.abs().sum()>0 for head in student.token_heads.values())
    assert student.mod.fc[0].weight.grad is None  # loss did not pass through classifier


@pytest.mark.parametrize('task',c.TASKS)
def test_all_production_objectives_and_replay(tmp_path,task):
    tr,va,targets,bank,spec=fixtures(tmp_path)
    report=t.fit(tmp_path/task,spec,task,tr,va,bank,device='cpu',technical=True,
        targets=targets if c.ARMS[task]['depths'] else None)
    row=report['validation_history'][0]
    assert row['embedding_loss']>0 and report['selected_pass']==1
    assert (row['particle_loss']>0)==bool(c.ARMS[task]['depths'])
    assert row['weighted_particle']==pytest.approx(row['particle_loss']*c.ARMS[task]['coefficient'],abs=1e-5)
    for d in (6,8): assert (row[f'd{d}_loss']>0)==(d in c.ARMS[task]['depths'])
    state=torch.load(tmp_path/task/'selected.pt',weights_only=True)
    assert all(k.startswith('mod.') for k in state)


@pytest.mark.parametrize('task',['control_s0','multi_l1_s0','centered_l1_s0'])
def test_production_width_cuda_full_partial(tmp_path,monkeypatch,task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    config=native.canonical_particle_transformer_config(); config.update(input_dim=17,num_classes=11,trim=False)
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))
    tr,va,targets,bank,spec=fixtures(tmp_path,130,'cuda')
    report=t.fit(tmp_path/task,spec,task,tr,va,bank,device='cuda',technical=True,
        targets=targets if c.ARMS[task]['depths'] else None)
    assert report['validation_history'][0]['batch_sizes']==[128,2]


def test_full_fit_and_target_metadata_tamper(tmp_path):
    tr,va,targets,bank,spec=fixtures(tmp_path); task='final_l1_s0'
    report=t.fit(tmp_path/'fit',spec,task,tr,va,bank,device='cpu',targets=targets)
    assert report['epochs']==30 and len(report['validation_history'])==30
    first=report['validation_history'][0]
    assert first['weighted_particle']==pytest.approx(first['particle_loss']/3,abs=1e-6)
    path=tmp_path/'setup/targets.json'; metadata=load_json(path)
    metadata['statistics']['8']['mean'][0]+=1; path.write_text(json.dumps(with_content_hash(metadata)))
    with pytest.raises(ValueError): t.validate_fit(tmp_path/'fit',spec,task)


def test_setup_warmed_measurement_and_authenticated_acceptance(tmp_path,monkeypatch):
    tr,va,targets,bank,spec=fixtures(tmp_path,258)
    spec.update(reference=dict(environment=native.installed_environment()),
        qualification=dict(acceptance=dict(gpu=dict(name='NVIDIA RTX A6000',total_memory_bytes=48*2**30))))
    monkeypatch.setattr(resources,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(resources,'peaks',lambda device:dict(gpu_peak_bytes=0,peak_rss_bytes=2**30))
    # Each objective is tested above; orchestration here exercises both roles.
    monkeypatch.setattr(resources,'TASKS',('multi_l1_s0','control_s0'))
    initial=[]; original=t.new_model
    def observe(*args):
        model=original(*args); initial.append({k:v.clone() for k,v in model.state_dict().items()}); return model
    monkeypatch.setattr(t,'new_model',observe)
    value=resources.measure(spec,tr,va,targets,bank,1.,device='cpu')
    assert resources.accepted(spec)==value
    assert len(initial)==4
    for a,b in ((0,1),(2,3)): assert all(torch.equal(v,initial[b][k]) for k,v in initial[a].items())
    (tmp_path/'setup/warm_multi_l1_s0/selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): resources.accepted(spec)


@pytest.mark.parametrize('task',['control_s0','final_l1_s0'])
def test_exact_worker_fit_with_native_deployment(tmp_path,monkeypatch,task):
    tr,va,targets,bank,spec=fixtures(tmp_path)
    spec.update(reference=dict(environment=native.installed_environment()),
        qualification=dict(acceptance=dict(gpu=dict(name='NVIDIA RTX A6000',total_memory_bytes=48*2**30))))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:spec)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.execution,'allocation',lambda s:'1234')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{task:'1234'})
    monkeypatch.setattr(worker,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:spec['qualification']['acceptance'])
    calls=[]
    def load(s,role): calls.append(role); return tr if role=='train' else va
    def get_targets(s,train):
        assert c.ARMS[task]['depths'] and train is tr; calls.append('targets'); return targets
    monkeypatch.setattr(inputs,'load_hlt',load); monkeypatch.setattr(inputs,'targets',get_targets)
    monkeypatch.setattr(inputs,'bank',lambda s,train:bank)
    original=t.fit
    monkeypatch.setattr(t,'fit',lambda *a,**kw:original(*a,**dict(kw,device='cpu')))
    value=worker.run(spec,task)
    assert worker.receipt(spec,task)==value
    assert calls==['train','validation']+(['targets'] if c.ARMS[task]['depths'] else [])
    assert t.validate_fit(tmp_path/task/'fit',spec,task)[0]['epochs']==30
    with pytest.raises(FileExistsError): worker.run(spec,task)


def test_target_identity_rebuild_and_role_failures(tmp_path):
    tr,va,targets,bank,spec=fixtures(tmp_path)
    raw=tr.batch(np.arange(4))
    with pytest.raises(ValueError): targets.batch(np.arange(4),~raw['mask'][:,0],[8])
    targets.metadata=deepcopy(targets.metadata); targets.metadata['array_hashes']['8']='0'*64
    targets.metadata=with_content_hash(targets.metadata)
    with pytest.raises(ValueError,match='alignment or metadata'):
        t.fit(tmp_path/'bad',spec,'final_l1_s0',tr,va,bank,device='cpu',targets=targets,technical=True)
    with pytest.raises(ValueError,match='role'):
        t.fit(tmp_path/'bad2',spec,'control_s0',tr,va,bank,device='cpu',targets=targets,technical=True)



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


def test_resource_admission_uses_warmed_cost_and_never_quality(tmp_path):
    spec=queue_spec(tmp_path); task='final_l1_s0'; r=timing_report(spec,task)
    warm=timing_report(spec,task,seconds=100.)
    memory=dict(gpu_peak_bytes=6*2**30,peak_rss_bytes=14*2**30)
    gpu=spec['qualification']['acceptance']['gpu']
    value=resources.resource_record(spec,task,r,warm,memory,gpu,200.)
    assert value['passed'] and value['quality_gate'] is False
    assert value['train_step_seconds']==.2  # exclude cold-start warmup
    slow=resources.resource_record(spec,task,timing_report(spec,task,seconds=10.),warm,memory,gpu,200.)
    assert not slow['passed'] and slow['violations'][0]['resource']=='minutes'
    large=resources.resource_record(spec,task,r,warm,dict(memory,peak_rss_bytes=32*2**30),gpu,200.)
    assert not large['passed'] and large['violations'][0]['resource']=='peak_rss_bytes'
    with pytest.raises(ValueError,match='finite'):
        resources.resource_record(spec,task,timing_report(spec,task,seconds=float('nan')),warm,memory,gpu,0.)


def test_exact_setup_worker_publishes_reusable_targets_and_miniatures(tmp_path,monkeypatch):
    tr,va,targets,bank,spec=fixtures(tmp_path/'fixture',258)
    spec['root']=str(tmp_path/'run'); Path(spec['root']).mkdir()
    spec.update(reference=dict(environment=native.installed_environment()),
        qualification=dict(acceptance=dict(gpu=dict(name='NVIDIA RTX A6000',total_memory_bytes=48*2**30))))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:spec)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.execution,'allocation',lambda s:'1234')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'setup':'1234'})
    monkeypatch.setattr(worker,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(resources,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(resources,'peaks',lambda device:dict(gpu_peak_bytes=0,peak_rss_bytes=2**30))
    verified=[]
    def source(name):
        return SimpleNamespace(verify_all=lambda:verified.append(name))
    spec['bundle_root']='bundle'
    monkeypatch.setattr(inputs.donor,'Bundle',lambda _:source('bundle'))
    monkeypatch.setattr(inputs.donor,'delta',lambda _:source('delta'))
    monkeypatch.setattr(inputs.donor,'encoder_delta',lambda _:source('encoder'))
    monkeypatch.setattr(inputs,'load_hlt',lambda s,role:tr if role=='train' else va)
    monkeypatch.setattr(inputs,'targets',lambda *a:targets)
    monkeypatch.setattr(inputs,'bank',lambda *a:bank)
    monkeypatch.setattr(resources,'TASKS',('final_l1_s0','control_s0'))
    monkeypatch.setattr(worker,'TASKS',('final_l1_s0','control_s0'))
    original=resources.measure
    monkeypatch.setattr(resources,'measure',lambda *a:original(*a,device='cpu'))
    result=worker.run(spec,'setup')
    assert verified==['bundle','delta','encoder']
    assert worker.receipt(spec,'setup')==result
    assert worker.completed_setup(spec)['parents']['targets']==targets.metadata['content_hash']
    outputs={r['path'] for r in result['outputs']}
    assert {'targets.json','acceptance.json','warm_final_l1_s0/receipt.json','mini_control_s0/receipt.json'}<=outputs
    with pytest.raises(FileExistsError): worker.run(spec,'setup')


def test_epoch_progress_is_read_only_and_uses_matched_seed(tmp_path,monkeypatch,capsys):
    spec=queue_spec(tmp_path); spec['bundle_root']='bundle'
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    class Bundle:
        def __init__(self,path): pass
        def json(self,path):
            if path.startswith('origin/'):
                return dict(recovery_anchors=dict(M0HLT=.90,OFFLINE=.95))
            return dict(validation_history=[dict(pass_number=1,validation=dict(macro_ovr_auc=.91))])
    monkeypatch.setattr(reporting,'Bundle',Bundle)
    for task,auc in (('control_s0',.92),('control_s1',.925),('final_l1_s1',.93)):
        write_immutable_json(tmp_path/task/'fit/epoch_01.json',c.artifact('EPOCH',parents=dict(spec=spec['content_hash']),
            task=task,pass_number=1,validation=dict(rows=spec['counts']['validation'],macro_ovr_auc=auc)))
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    reporting.progress(spec)
    output=capsys.readouterr().out
    part=output.split('\nfinal_l1_s1\n')[1].split('\ncontrol_s1')[0]
    assert 'control_s1' in part and '0.92500000' in part and '60.0%' in part
    assert before=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    corrupt=tmp_path/'final_l1_s1/fit/epoch_01.json'
    row=load_json(corrupt); row['task']='control_s0'; corrupt.write_text(json.dumps(with_content_hash(row)))
    with pytest.raises(ValueError,match='published epoch'): reporting.progress(spec)


def test_teacher_proof_runs_original_source(tmp_path,monkeypatch):
    parent=dict(project_dir='/original',root=str(tmp_path),content_hash='p'*64)
    root=tmp_path/'teacher_m16'; (root/'fit').mkdir(parents=True)
    for path in ('receipt.json','fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz'):
        (root/path).write_bytes(b'original')
    def run(cmd,**kw):
        assert cmd[:4]==[sys.executable,'-B','-s','-c']
        assert 'worker.receipt' in cmd[4] and 'training.validate_fit' in cmd[4] and 'sys.path.insert' in cmd[4]
        assert kw['cwd']=='/original' and kw['env']['PYTHONPATH']==str(Path('/original/src'))
        return SimpleNamespace(returncode=0,stderr='',stdout='PARTICLE_TEACHER_PROOF='+json.dumps(dict(receipt='r'*64,job_id='145410',report='t'*64)))
    monkeypatch.setattr(subprocess,'run',run)
    value=campaign.teacher_proof('/parent.json',parent)
    assert value['report_hash']=='t'*64 and value['job_id']=='145410' and len(value['files'])==5


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


@pytest.mark.parametrize('allocated_cpus',[1,2])
def test_cpu_worker_profile_and_wrong_job_rejected(monkeypatch,tmp_path,allocated_cpus):
    spec=queue_spec(tmp_path); monkeypatch.setenv('SLURM_JOB_ID','123')
    for name in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'): monkeypatch.delenv(name,raising=False)
    record=('JobId=123 JobState=RUNNING Partition=compute Account=default QOS=normal '
        f'NumCPUs={allocated_cpus} NumNodes=1 NumTasks=1 CPUs/Task=1 '
        f'ReqTRES=cpu=1,mem=4000M,node=1 AllocTRES=cpu={allocated_cpus},mem=4000M,node=1')
    def run(cmd,**kwargs):
        return SimpleNamespace(stdout='ClusterName = hydra\n' if cmd[-1]=='config' else record)
    monkeypatch.setattr(subprocess,'run',run)
    assert worker.cpu_allocation(spec)=='123'
    monkeypatch.setenv('SLURM_JOB_ID','456')
    with pytest.raises(PermissionError): worker.cpu_allocation(spec)


@pytest.mark.parametrize('field,value',[
    ('JobState','PENDING'),('Partition','gpus'),('Account','other'),('QOS','other'),
    ('NumNodes','2'),('NumTasks','2'),('CPUs/Task','2'),('NumCPUs','1'),
    ('ReqTRES','cpu=2,mem=4000M,node=1'),('ReqTRES','cpu=1,mem=4000M,node=1,gres/gpu=1'),
    ('ReqTRES','cpu=1,mem=8000M,node=1'),('ReqTRES','mem=4000M,node=1'),
    ('AllocTRES','cpu=3,mem=4000M,node=1'),('AllocTRES','cpu=2,mem=8000M,node=1'),
    ('AllocTRES','cpu=2,mem=4000M,node=1,gres/gpu=1'),('AllocTRES','cpu=2,mem=4000M,node=2'),
])
def test_cpu_rounding_does_not_relax_other_resources(monkeypatch,tmp_path,capsys,field,value):
    spec=queue_spec(tmp_path); monkeypatch.setenv('SLURM_JOB_ID','123')
    for name in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'): monkeypatch.delenv(name,raising=False)
    fields=dict(JobId='123',JobState='RUNNING',Partition='compute',Account='default',QOS='normal',
        NumCPUs='2',NumNodes='1',NumTasks='1',**{'CPUs/Task':'1'},
        ReqTRES='cpu=1,mem=4000M,node=1',AllocTRES='cpu=2,mem=4000M,node=1')
    fields[field]=value
    def run(cmd,**kwargs):
        return SimpleNamespace(stdout='ClusterName = hydra\n' if cmd[-1]=='config'
            else ' '.join(k+'='+v for k,v in fields.items()))
    monkeypatch.setattr(subprocess,'run',run)
    with pytest.raises(PermissionError,match='observed|rounding|GPU resources'): worker.cpu_allocation(spec)
    output=capsys.readouterr().out
    assert 'CPU allocation evidence' in output and 'ReqTRES' in output and value in output


def test_cpu_tasks_environment_fallback_and_conflicts(monkeypatch,tmp_path):
    spec=queue_spec(tmp_path); monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setenv('SLURM_NTASKS','1'); monkeypatch.setenv('SLURM_CPUS_PER_TASK','1')
    monkeypatch.setenv('SLURM_MEM_PER_NODE','4000')
    record=('JobId=123 JobState=RUNNING Partition=compute Account=default QOS=normal '
        'NumCPUs=2 NumNodes=1 NumTasks=(null) CPUs/Task=1 '
        'ReqTRES=cpu=1,mem=4000M,node=1 AllocTRES=cpu=2,mem=4000M,node=1')
    cluster='hydra'
    def run(cmd,**kwargs):
        return SimpleNamespace(stdout=f'ClusterName = {cluster}\n' if cmd[-1]=='config' else record)
    monkeypatch.setattr(subprocess,'run',run)
    assert worker.cpu_allocation(spec)=='123'
    for name,wrong,expected in (('SLURM_NTASKS','2','1'),('SLURM_CPUS_PER_TASK','2','1'),('SLURM_MEM_PER_NODE','8000','4000')):
        monkeypatch.setenv(name,wrong)
        with pytest.raises(PermissionError,match=name): worker.cpu_allocation(spec)
        monkeypatch.setenv(name,expected)
    monkeypatch.delenv('SLURM_NTASKS')
    with pytest.raises(PermissionError,match='NumTasks'): worker.cpu_allocation(spec)
    monkeypatch.setenv('SLURM_NTASKS','1'); cluster='sporc'
    with pytest.raises(PermissionError,match='ClusterName'): worker.cpu_allocation(spec)


@pytest.mark.parametrize('rows',[256,257,258,500])
def test_miniature_preserves_worst_full_batch_and_safe_partial(rows):
    tr=cache(coordinate='D000',rows=rows)
    ix=resources.miniature_rows(tr)
    lengths=np.concatenate([np.diff(b.offsets) for b in tr.blocks])
    largest=np.argsort(-lengths,kind='stable')[:128]
    np.testing.assert_array_equal(ix[:128],largest)
    assert len(ix)>128 and 2<=len(ix)%128<=127 and len(np.unique(ix))==len(ix)
    assert set(tr.labels[ix])==set(range(11))
