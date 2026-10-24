"""Combined supervision correctness, original-target parity and direct worker/DAG checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_attention import parent_nodes
from test_luka_particle_prediction import tiny
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,with_content_hash,
    write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward,predict
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_u100_embedding.contracts import artifact as teacher_artifact
from hlt_classification.luka_u100_embedding import targets as jet_targets
from hlt_classification.luka_particle_prediction import banks as token_banks
from hlt_classification.luka_embedding_aux.model import head_input, AuxiliaryParT
from hlt_classification.luka_particle_prediction.model import token_forward, ParticlePredictor
from hlt_classification.luka_joint_prediction import contracts as c,model,banks,training,worker,campaign,submission,source


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def spec_fixture(root):
    accepted=dict(content_hash='c'*64,resources=dict(execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000),
        preflight=dict(projections=dict(aligned=120,set=150),environment={},gpu={}))
    return dict(root=str(root),project_dir='/project',source_commit='e'*40,source_snapshot=dict(content_hash='d'*64),
        anchor=dict(evidence=dict(prepared='a'*64,prepared_root='/prepared',raw_container='/raw',environment={})),
        acceptance=accepted,content_hash='b'*64,**{**campaign.policy(dict(nodes=parent_nodes()),accepted),
        'counts':dict(train=22,validation=22),'original50k_rows':22})


def bank_fixture(root,device='cpu'):
    teacher=DelphesParticleTransformer().to(device).eval(); save_state(root/'teacher.pt',teacher.state_dict())
    record=teacher_artifact('TEACHER',coordinate='U100',target_role='train',prepared='a'*64,
        files=dict(checkpoint=file_ref(root/'teacher.pt')))
    tr=cache(); rich=cache(coordinate='U100')
    banks.export(root/'targets',teacher,rich,tr,record,device=device)
    return tr,banks.Bank(root/'targets',tr,record),record,teacher


def test_registered_objective_scope_and_two_direct_debug_jobs():
    spec=spec_fixture(Path('/study')); r=c.recipe(); plan=submission.render(spec)
    assert (r['ce_coefficient'],r['jet_coefficient'],r['token_coefficient'])==(1.,4.,3.)
    assert r['logit_kd'] is False and r['warm_start'] is False and r['epochs']==30 and r['batch_size']==128
    assert spec['preflight_jobs']==0 and spec['technical_training_epochs']==0
    assert plan['jobs']==2 and plan['preflight_jobs']==0
    assert [row['task_id'] for row in plan['commands']]==['aligned','set']
    for row in plan['commands']:
        assert row['dependencies']==[] and '--partition=debug' in row['command']
        assert '--mem=90000M' in row['command'] and '--export=NONE' in row['command']
        assert not any(a.startswith('--dependency') for a in row['command'])
        assert not any('preflight' in a for a in row['command'])
    for arm in c.ARMS:
        node=spec['nodes'][arm]
        assert (node['jet_weight'],node['token_weight'])==(4.,3.) and 'auxiliary_weight' not in node
    wrong=deepcopy(spec['acceptance']); wrong['preflight']['projections']['set']=1440
    with pytest.raises(ValueError,match='exceeds'): campaign.policy(dict(nodes=parent_nodes()),wrong)
    wrong=deepcopy(spec['acceptance']); wrong['resources']['execution_site']['partition']='tier3'
    with pytest.raises(ValueError,match='debug'): campaign.policy(dict(nodes=parent_nodes()),wrong)


@pytest.mark.parametrize('arm',c.ARMS)
def test_objective_exact_weights_and_detached_targets(arm):
    torch.manual_seed(25)
    logits=torch.randn(2,11,requires_grad=True); p=torch.randn(2,3,128,requires_grad=True)
    t=torch.randn_like(p,requires_grad=True); j=torch.randn(2,128,requires_grad=True)
    jt=torch.randn_like(j,requires_grad=True); mask=torch.tensor([[True,False,False],[True,True,True]])
    t.data[0,1:]=float('nan'); p.data[0,1:]=float('nan')
    total,ce,jet,token=model.objective(logits,torch.tensor([0,1]),p,t,j,jt,mask,arm)
    torch.testing.assert_close(total,ce+4*jet+3*token)
    torch.testing.assert_close(jet,torch.nn.functional.smooth_l1_loss(j,jt))
    total.backward()
    assert jt.grad is None and t.grad is None
    assert p.grad[0,1:].abs().sum()==0 and j.grad.abs().sum()>0
    with pytest.raises(ValueError): model.objective(logits,torch.tensor([0,1]),p,t,j,jt*float('nan'),mask,arm)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bf16',[False,True])
def test_installed_full_weaver_dual_tap_single_forward_and_native_parity(device,bf16):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    node=c.node(dict(nodes=parent_nodes()),'aligned'); torch.manual_seed(node['initialization_seed'])
    native=DelphesParticleTransformer().to(device).eval()
    joint=model.new_model(node).to(device).eval(); joint.capture=True
    for k,v in joint.native_state().items(): torch.testing.assert_close(v,native.state_dict()[k].cpu(),rtol=0,atol=0)
    raw=cache().batch(np.arange(4)); called=[]
    handle=joint.mod.blocks[0].register_forward_hook(lambda *_:called.append(1))
    y=_forward(native,raw,device,bf16=bf16); z=_forward(joint,raw,device,bf16=bf16); handle.remove()
    assert called==[1] and joint.jet_prediction.shape==(4,128) and joint.prediction.shape==(4,16,128)
    assert joint.jet_prediction.dtype==joint.prediction.dtype==torch.float32
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    y.square().mean().backward(); z.square().mean().backward()
    for k,p in native.named_parameters():
        other=dict(joint.named_parameters())[k].grad
        if p.grad is None: assert other is None
        else: torch.testing.assert_close(p.grad,other,rtol=0,atol=0)
    x,v,mask=[torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')]
    with torch.inference_mode():
        l,t,j=model.joint_forward(native,x,v,mask)
        lj,jj=head_input(native,x,v,mask); lt,tt=token_forward(native,x,v,mask)
    torch.testing.assert_close(t,tt,rtol=0,atol=0); torch.testing.assert_close(j,jj,rtol=0,atol=0)
    torch.testing.assert_close(l,lj,rtol=0,atol=0); torch.testing.assert_close(l,lt,rtol=0,atol=0)
    joint.zero_grad(set_to_none=True); _forward(joint,raw,device,bf16=bf16)
    _,_,jl,tl=model.objective(z,torch.zeros(4,dtype=torch.long,device=device),joint.prediction,
        torch.zeros_like(joint.prediction),joint.jet_prediction,torch.zeros_like(joint.jet_prediction),mask,'set')
    (4*jl+3*tl).backward()
    assert joint.predictor[0].weight.grad.abs().sum()>0 and joint.jet_predictor[0].weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in joint.mod.blocks[7].parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in joint.mod.cls_blocks.parameters())
    assert all(p.grad is None for p in joint.mod.fc.parameters())
    assert all(not k.startswith(('predictor.','jet_predictor.')) for k in joint.native_state())


def test_both_heads_match_donor_initializations_and_preserve_rng():
    torch.manual_seed(123); joint=model.JointPredictor(); after=torch.get_rng_state()
    torch.manual_seed(123); native=DelphesParticleTransformer(); assert torch.equal(after,torch.get_rng_state())
    torch.manual_seed(123); jet=AuxiliaryParT()
    torch.manual_seed(123); token=ParticlePredictor()
    for k,v in jet.predictor.state_dict().items(): torch.testing.assert_close(v,joint.jet_predictor.state_dict()[k],rtol=0,atol=0)
    for k,v in token.predictor.state_dict().items(): torch.testing.assert_close(v,joint.predictor.state_dict()[k],rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_joint_bank_parity_normalization_scope_and_corruption(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); tr,bank,record,teacher=bank_fixture(tmp_path,device)
    rich=cache(coordinate='U100')
    old=token_banks.export(tmp_path/'old_tokens',teacher,rich,tr,record,device=device)
    jet_targets.export(tmp_path/'old_jets',teacher,rich,teacher_record=record,device=device)
    assert bank.manifest['outputs'][0]==old['outputs'][0]
    a=load_npz_arrays(tmp_path/'old_jets/targets.npz')
    np.testing.assert_allclose(bank.jet_batch(np.arange(len(tr))),a['targets'],rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(bank.jet_mean,a['mean'],rtol=1e-6,atol=1e-7)
    mask=tr.batch(np.arange(len(tr)))['mask'][:,0]
    np.testing.assert_allclose(bank.batch(np.arange(len(tr)),16)[mask].mean(0),0.,atol=3e-6)
    np.testing.assert_allclose(bank.jet_batch(np.arange(len(tr))).mean(0),0.,atol=3e-6)
    assert not np.array_equal(bank.jet_mean,bank.mean)
    bank.close()
    for role,co in (('validation','U100'),('train','OFFLINE')):
        with pytest.raises(PermissionError): banks.export(tmp_path/'wrong',teacher,cache(role,co),tr,record,device='cpu')
    changed=deepcopy(tr); changed.identities=changed.identities[::-1].copy()
    with pytest.raises(ValueError): banks.Bank(tmp_path/'targets',changed,record)
    with (tmp_path/'targets/jets.npy').open('r+b') as handle:
        handle.seek(-1,2); handle.write(b'x')
    with pytest.raises(ValueError): banks.Bank(tmp_path/'targets',tr,record)


@pytest.mark.parametrize('arm',c.ARMS)
def test_complete30_and_both_loss_histories_native_readback(tmp_path,monkeypatch,arm):
    tiny(monkeypatch); tr,bank,_,_=bank_fixture(tmp_path); spec=spec_fixture(tmp_path)
    report,_=worker.fit(tmp_path/arm,spec,arm,tr,cache('validation'),bank,device='cpu')
    actual,pred=worker.validate_fit(tmp_path/arm,spec,arm,bank.manifest['content_hash'])
    assert report==actual and len(report['validation_history'])==30 and pred['probabilities'].shape==(22,11)
    for r in report['validation_history']:
        assert r['total_loss']==r['ce_loss']+4*r['jet_loss']+3*r['token_loss']
    bad=deepcopy(report); bad['validation_history'][0]['weighted_token_loss']*=2
    with pytest.raises(ValueError): training.validate_report(with_content_hash(bad),spec,arm,bank.manifest['content_hash'],technical=False)
    assert set(torch.load(tmp_path/arm/'selected.pt',weights_only=True))==set(DelphesParticleTransformer().state_dict())
    bank.close()


def test_weak_auc_still_completes30(tmp_path,monkeypatch):
    tiny(monkeypatch); tr,bank,_,_=bank_fixture(tmp_path); spec=spec_fixture(tmp_path)
    original=training.evaluate_probabilities; seen=[]
    def poor(labels,probabilities):
        seen.append(1); return dict(original(labels,probabilities),macro_ovr_auc=.4 if len(seen)==1 else .1)
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    node=spec['nodes']['aligned']; report,_=training.train(model.new_model(node),tr,cache('validation'),bank,node,device='cpu')
    assert report['selected_pass']==1 and report['passes']==30
    training.validate_report(report,spec,'aligned',bank.manifest['content_hash'],technical=False); bank.close()


def test_exact_two_job_submission_and_duplicate_guards(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda *_:None)
    monkeypatch.setattr(submission,'check_site',lambda *_:None)
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='x',authorization=c.AUTHORIZATION)
    plan=submission.submit(spec); calls=[]
    def fake(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(spec)==dict(aligned='101',set='102') and len(calls)==2
    assert not any('--dependency' in arg for cmd in calls for arg in cmd)
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==2


def test_wrong_task_and_exact_job_fail_before_fit(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path)
    with pytest.raises(ValueError,match='no preflight'): worker.run(spec,'preflight')
    monkeypatch.setattr(campaign,'check',lambda *_:None)
    monkeypatch.setattr(worker,'execution',lambda *_:'999')
    monkeypatch.setattr(worker,'installed_environment',lambda:{})
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(aligned='101'))
    with pytest.raises(ValueError,match='exact job'): worker.run(spec,'aligned')
    assert not (tmp_path/'aligned').exists()


def test_source_frozen_donors_and_complete_registration(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else ''))
    assert source.current(project,'b'*40)==dict(content_hash='a'*64)
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else 'M\tsrc/hlt_classification/luka_particle_prediction/model.py'))
    with pytest.raises(ValueError,match='runtime changed'): source.current(project,'b'*40)
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout=''))
    with pytest.raises(ValueError,match='complete'): source.current(project,'b'*40)
    compile(source.REPLAY,'source_replay','exec')
    assert 'worker.validate_preflight(s)' in source.REPLAY and 'worker.results(s)' in source.REPLAY


def acceptance_fixture(tmp_path,monkeypatch):
    from hlt_classification.luka_particle_prediction.contracts import artifact as old_artifact
    root=tmp_path/'old_token'
    parent=dict(nodes=parent_nodes(),root=str(tmp_path/'parent/science'),project_dir=str(tmp_path/'old_project'))
    write_immutable_json(tmp_path/'parent/science/campaign_spec.json',parent)
    atomic_publish_bytes(tmp_path/'teacher.pt',b'fixture')
    teacher=dict(coordinate='U100',files=dict(checkpoint=file_ref(tmp_path/'teacher.pt')))
    anchor=dict(spec=file_ref(tmp_path/'parent/science/campaign_spec.json'),evidence=dict(environment={},
        prepared_root=str(tmp_path/'prepared'),raw_container=str(tmp_path/'raw')))
    reference=dict(spec=dict(path=str(tmp_path/'reference/campaign_spec.json')))
    resources=dict(execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000)
    old=old_artifact('SPEC',root=str(root),source_commit=source.BASE,counts=c.COUNTS,
        source_snapshot={},project_dir=str(tmp_path/'old_project'),reference=reference,anchor=anchor,teacher=teacher,**resources)
    write_immutable_json(root/'campaign_spec.json',old)
    pre=old_artifact('PREFLIGHT',parents=dict(spec=old['content_hash']),technical_checks_passed=True,
        environment={},projections=dict(aligned=120,set=150),gpu={})
    bank=old_artifact('BANK',checkpoint=teacher['files']['checkpoint'],rows=c.COUNTS['train'],coordinate='U100',role='train')
    write_immutable_json(root/'preflight/receipt.json',pre)
    write_immutable_json(root/'preflight/bank_manifest.json',bank)
    for arm in c.ARMS:
        write_immutable_json(root/arm/'receipt.json',{})
        atomic_publish_bytes(root/arm/'fit/validation_predictions.npz',b'fixture')
    value=c.artifact('ACCEPTANCE',spec=file_ref(root/'campaign_spec.json'),reference=reference,anchor=anchor,teacher=teacher,
        resources=resources,counts=c.COUNTS,preflight=pre,bank=bank,
        files=dict(preflight=file_ref(root/'preflight/receipt.json'),bank=file_ref(root/'preflight/bank_manifest.json'),
            **{a:file_ref(root/a/'receipt.json') for a in c.ARMS}),
        predictions={'TOKEN_'+a.upper():file_ref(root/a/'fit/validation_predictions.npz') for a in c.ARMS})
    monkeypatch.setattr(source.donor,'validate_reference',lambda _:None)
    return value,parent


def test_acceptance_binding_and_original_source_replay(tmp_path,monkeypatch):
    value,parent=acceptance_fixture(tmp_path,monkeypatch); source.validate_acceptance(value)
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(source.donor,'matched_core',lambda *a:None)
    fields={k:v for k,v in value.items() if k in ('reference','anchor','teacher','resources','counts','preflight','bank','files','predictions')}
    monkeypatch.setattr(source,'replay',lambda *a:(value['spec'],fields))
    assert source.import_acceptance(value['spec']['path'])==(value,parent)
    for change in ('teacher','preflight','predictions','counts'):
        bad=deepcopy(value)
        if change=='teacher': bad['teacher']['coordinate']='OFFLINE'
        elif change=='preflight': bad['preflight']['technical_checks_passed']=False
        elif change=='predictions': bad['predictions']['TOKEN_SET']=bad['predictions']['TOKEN_ALIGNED']
        else: bad['counts']['train']-=1
        with pytest.raises(ValueError): source.validate_acceptance(with_content_hash(bad))


def test_fresh_root_quota_recipe_and_source_validation(tmp_path,monkeypatch):
    accepted,parent=acceptance_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='d'*64))
    monkeypatch.setattr(source,'import_acceptance',lambda *a:(accepted,parent))
    args=(accepted['spec']['path'],tmp_path/'project','e'*40)
    for quota in (1,0,True):
        with pytest.raises(ValueError): campaign.create(tmp_path/'bad',*args,available_quota_gib=quota)
    with pytest.raises(PermissionError): campaign.create(tmp_path/'old_token/subrun',*args,available_quota_gib=2)
    assert not (tmp_path/'bad').exists()
    spec=campaign.create(tmp_path/'new',*args,available_quota_gib=2); campaign.check(spec)
    for key,wrong in (('preflight_jobs',1),('train_minutes',dict(aligned=1,set=1)),('recipe',{})):
        with pytest.raises(ValueError): campaign.validate_spec(with_content_hash(dict(spec,**{key:wrong})))
    with pytest.raises(FileExistsError): campaign.create(tmp_path/'new',*args,available_quota_gib=2)
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='f'*64))
    with pytest.raises(ValueError,match='Source changed'): campaign.check(spec)


def test_scratch_locality_no_home_fallback(tmp_path,monkeypatch):
    root=tmp_path/'checkpoints/study'; root.mkdir(parents=True)
    spec=spec_fixture(root); local=tmp_path/'scratch'; local.mkdir(); spec['scratch_base']=str(local)
    for kind in ('ceph','nfs','tmpfs','xfs'):
        monkeypatch.setattr(worker,'mountinfo',lambda:f'1 0 8:1 / {local.as_posix()} rw - {kind} /dev/test rw\n')
        if kind=='xfs': assert worker.scratch_directory(spec)==local
        else:
            with pytest.raises(PermissionError,match='node-local disk'): worker.scratch_directory(spec)
    spec['scratch_base']=str(root)
    with pytest.raises(PermissionError): worker.scratch_directory(spec)


def test_ambiguous_submission_cannot_retry_duplicate(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda *_:None)
    monkeypatch.setattr(submission,'check_site',lambda *_:None)
    plan=submission.submit(spec); calls=[]
    def ambiguous(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout='not-a-job-id\n',stderr='')
    monkeypatch.setattr(subprocess,'run',ambiguous)
    with pytest.raises((ValueError,RuntimeError)):
        submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError):
        submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


def test_production_two_workers_cuda_and_retirement(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    # Full installed architecture (not a replacement model); synthetic22-row views.
    root=tmp_path/'checkpoints/study'; root.mkdir(parents=True); spec=spec_fixture(root)
    native=DelphesParticleTransformer().cuda().eval(); save_state(tmp_path/'teacher.pt',native.state_dict())
    richval=cache('validation','U100'); val=cache('validation')
    probabilities=predict(native,richval,device='cuda',batch_size=128)
    pred=dict(identities=val.identities,labels=val.labels,probabilities=probabilities)
    atomic_publish_bytes(tmp_path/'reference.npz',deterministic_npz_bytes(pred)); ref=file_ref(tmp_path/'reference.npz')
    teacher=teacher_artifact('TEACHER',coordinate='U100',target_role='train',prepared='a'*64,
        files=dict(checkpoint=file_ref(tmp_path/'teacher.pt'),predictions=ref))
    spec['teacher']=teacher
    spec['anchor']['evidence']['models']={n:dict(predictions=ref) for n in ('M0HLT','OFFLINE')}
    spec['reference']=dict(files=dict(predictions=ref),reference=dict(files=dict(predictions=ref)))
    spec['anchor']['evidence']['environment']=dict(fixture=True)
    spec['acceptance']['predictions']={n:ref for n in ('TOKEN_ALIGNED','TOKEN_SET')}
    spec['acceptance']['bank']=token_banks.export(tmp_path/'original_bank',native,cache(coordinate='U100'),cache(),teacher,device='cuda')
    gpu=dict(total_memory_bytes=40*2**30,name='fixture A100')
    spec['acceptance']['preflight'].update(environment=dict(fixture=True),gpu=gpu)
    scratch=tmp_path/'scratch'; scratch.mkdir()
    monkeypatch.setattr(worker,'scratch_directory',lambda _:scratch)
    monkeypatch.setattr(campaign,'check',lambda *_:None); monkeypatch.setattr(campaign,'validate_spec',lambda *_:None)
    monkeypatch.setattr(source,'current',lambda *_:spec['source_snapshot'])
    monkeypatch.setattr(worker,'installed_environment',lambda:dict(fixture=True)); monkeypatch.setattr(worker,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _t:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=max(1,torch.cuda.max_memory_allocated())))
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda *_:(None,None,None,None))
    monkeypatch.setattr(worker,'data',lambda _s,r,co,_m:cache(r,co))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *_:np.arange(22,dtype=np.int64))
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(aligned='101',set='102'))
    assert worker.results(spec)['pending']==['aligned','set']
    for task,job in (('aligned','101'),('set','102')):
        monkeypatch.setattr(worker,'execution',lambda *_:job)
        result=worker.run(spec,task)
        assert not result['new_preflight_performed'] and result['scratch_payloads_retired'] and not list(scratch.iterdir())
        assert len(load_json(root/task/'fit/training_report.json')['validation_history'])==30
        if task=='aligned': assert worker.results(spec)['pending']==['set']
    result=worker.results(spec); assert not result['pending'] and len(result['rows'])==9
    assert load_json(root/'aligned/bank_manifest.json')==load_json(root/'set/bank_manifest.json')
    # Re-signing publication cannot disguise cross-arm target drift.
    bpath=root/'set/bank_manifest.json'; original=load_json(bpath)
    changed=deepcopy(original); changed['outputs'][2]['sha256']='f'*64
    bpath.write_text(json.dumps(with_content_hash(changed)))
    top=root/'set/receipt.json'; old=load_json(top); altered=deepcopy(old)
    altered['outputs']=[file_ref(bpath,root=bpath.parent) if r['path']=='bank_manifest.json' else r for r in altered['outputs']]
    top.write_text(json.dumps(with_content_hash(altered)))
    with pytest.raises(ValueError,match='identical jet AND token'): worker.results(spec)
    bpath.write_text(json.dumps(original)); top.write_text(json.dumps(old))
    failed=dict(spec,root=str(root.parent/'failed')); Path(failed['root']).mkdir()
    monkeypatch.setattr(worker,'execution',lambda *_:'101')
    def fail(*a,**kw): raise ValueError('injected failure')
    monkeypatch.setattr(worker,'fit',fail)
    with pytest.raises(ValueError,match='injected'): worker.run(failed,'aligned')
    assert not list(scratch.iterdir()) and not (Path(failed['root'])/'aligned/receipt.json').exists()


def test_cli_shell_and_plan_are_direct_and_weighted():
    root=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,'-B',str(root/'scripts/luka_joint_prediction.py'),'run','--help'],capture_output=True,text=True)
    assert proc.returncode==0 and '{aligned,set}' in proc.stdout and 'preflight' not in proc.stdout
    wrapper=(root/'sbatch/run_luka_joint_prediction.sh').read_text()
    assert '"${PROJECT_DIR}/sbatch/common.sh"' in wrapper and 'PYTHONNOUSERSITE=1' in wrapper
    assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in wrapper and 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in wrapper
    helper=(root/'scripts/queue_luka_joint_prediction.sh').read_text()
    assert '</dev/tty' in helper and c.AUTHORIZATION in helper
    assert 'scancel' not in helper and 'scontrol' not in helper
    assert all((root/p).is_file() for p in source.REQUIRED)
