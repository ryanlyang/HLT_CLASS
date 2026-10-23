"""Real Weaver token taps, exact set loss, scratch banks, fixed30 and fail-closed DAG."""
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_attention import parent_nodes
from hlt_classification.data.cache_contracts import (load_json,with_content_hash,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward,predict
from hlt_classification.luka_u100_embedding.contracts import artifact as teacher_artifact
from hlt_classification.luka_particle_prediction import contracts as c,model,banks,training,worker,campaign,submission,source


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny(monkeypatch):
    from hlt_classification.jetclass2_delphes import model as native
    config=native.model_config()
    config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=2,num_layers=8,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(config))


def spec_fixture(root):
    return dict(root=str(root),project_dir='/project',source_commit='e'*40,source_snapshot=dict(content_hash='d'*64),
        anchor=dict(evidence=dict(prepared='a'*64,prepared_root='/prepared',raw_container='/raw',environment={})),
        content_hash='b'*64,**{**campaign.policy(dict(nodes=parent_nodes())),
        'counts':dict(train=22,validation=22),'original50k_rows':22})


def bank_fixture(root,device='cpu'):
    teacher=DelphesParticleTransformer().to(device).eval(); save_state(root/'teacher.pt',teacher.state_dict())
    record=teacher_artifact('TEACHER',coordinate='U100',target_role='train',prepared='a'*64,
        files=dict(checkpoint=file_ref(root/'teacher.pt')))
    tr=cache(); rich=cache(coordinate='U100')
    banks.export(root/'targets',teacher,rich,tr,record,device=device)
    return tr,banks.Bank(root/'targets',tr,record),record


def test_exact_set_loss_against_exhaustive_matching():
    torch.manual_seed(20)
    p=torch.randn(2,3,128,requires_grad=True); t=torch.randn_like(p,requires_grad=True)
    mask=torch.ones(2,3,dtype=torch.bool)
    expected=sum(min(float(torch.nn.functional.smooth_l1_loss(p[j],t[j,list(order)]).detach())
                     for order in permutations(range(3))) for j in range(2))/2
    got=model.feature_loss(p,t,mask,'set')
    assert float(got.detach())==pytest.approx(expected,abs=1e-7)
    got.backward(); assert p.grad is not None and t.grad is None


def test_unordered_reordering_and_no_many_to_one_cheat():
    t=torch.arange(3.).reshape(1,3,1).expand(1,3,128)
    p=t[:,[2,0,1]].clone().requires_grad_(); mask=torch.ones(1,3,dtype=torch.bool)
    assert model.feature_loss(p,t,mask,'set')==0
    assert model.feature_loss(p,t,mask,'aligned')>0
    duplicate=t[:,[0,0,0]]
    assert model.feature_loss(duplicate,t,mask,'set')>0
    assert model.feature_loss(p,t[:,[1,2,0]],mask,'set')==0


@pytest.mark.parametrize('arm',c.ARMS)
def test_padding_equal_jet_reduction_and_gradient(arm):
    p=torch.zeros(2,4,128,requires_grad=True); t=torch.ones_like(p,requires_grad=True)
    mask=torch.tensor([[True,False,False,False],[True,True,True,True]])
    t.data[0,1:]=float('nan'); p.data[0,1:]=float('nan')
    aux=model.feature_loss(p,t,mask,arm); assert aux==.5
    aux.backward(); assert t.grad is None
    assert p.grad[0,1:].abs().sum()==0
    assert p.grad[0,0].abs().sum()==pytest.approx(float(p.grad[1].abs().sum()))
    logits=torch.randn(2,11,requires_grad=True)
    loss,ce,feature=model.objective(logits,torch.tensor([0,1]),p,t,mask,arm)
    torch.testing.assert_close(loss,ce+5*feature)


@pytest.mark.parametrize('arm',c.ARMS)
def test_invalid_support_finite_and_shape(arm):
    p=torch.zeros(1,2,128); mask=torch.ones(1,2,dtype=torch.bool)
    for bad in (torch.full_like(p,float('nan')),torch.full_like(p,float('inf'))):
        with pytest.raises(ValueError): model.feature_loss(p,bad,mask,arm)
    with pytest.raises(ValueError): model.feature_loss(p,p,torch.zeros_like(mask),arm)
    with pytest.raises(ValueError): model.feature_loss(p,p,mask.float(),arm)
    with pytest.raises(ValueError): model.feature_loss(p,p,mask[:,None].expand(-1,2,-1),arm)
    with pytest.raises(ValueError): model.feature_loss(p[:,:,:127],p,mask,arm)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bf16',[False,True])
def test_installed_full_model_parity_tap_and_aux_gradients(device,bf16):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    node=c.node(dict(nodes=parent_nodes()),'aligned'); torch.manual_seed(node['initialization_seed'])
    native=DelphesParticleTransformer().to(device).eval()
    student=model.new_model(node).to(device).eval(); student.capture=True
    for k,v in student.native_state().items(): torch.testing.assert_close(v,native.state_dict()[k].cpu(),rtol=0,atol=0)
    raw=cache().batch(np.arange(4)); expected=[]
    hook=native.mod.blocks[7].register_forward_hook(lambda _m,_a,out:expected.append(out))
    y=_forward(native,raw,device,bf16=bf16); actual=_forward(student,raw,device,bf16=bf16); hook.remove()
    torch.testing.assert_close(y,actual,rtol=0,atol=0)
    assert student.prediction.shape==(4,16,128)
    x,v,mask=[torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')]
    _,tokens=model.token_forward(native,x,v,mask)
    assert tokens.shape==(4,16,128)
    y.square().mean().backward(); actual.square().mean().backward()
    for k,p in native.named_parameters():
        q=dict(student.named_parameters())[k]
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    student.zero_grad(set_to_none=True)
    _forward(student,raw,device,bf16=bf16)
    target=torch.randn_like(student.prediction)
    loss=model.feature_loss(student.prediction,target,mask,'set'); loss.backward()
    assert student.predictor[0].weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.mod.blocks[7].parameters())
    assert all(p.grad is None for p in student.mod.fc.parameters())
    assert not any(k.startswith('predictor.') for k in student.native_state())


def test_bank_normalization_identity_corruption_and_scope(tmp_path,monkeypatch):
    tiny(monkeypatch); tr,bank,record=bank_fixture(tmp_path)
    rows=bank.batch(np.arange(len(tr)),16); mask=tr.batch(np.arange(len(tr)))['mask'][:,0]
    np.testing.assert_allclose(rows[mask].mean(0),0.,atol=3e-6)
    np.testing.assert_allclose(rows[mask].std(0),1.,atol=3e-6)
    bank.close()
    for wrong in (cache('validation','U100'),cache(coordinate='OFFLINE')):
        with pytest.raises(PermissionError): banks.export(tmp_path/'wrong',DelphesParticleTransformer(),wrong,tr,record,device='cpu')
    changed=deepcopy(tr); changed.identities=changed.identities[::-1].copy()
    with pytest.raises(ValueError): banks.Bank(tmp_path/'targets',changed,record)
    with pytest.raises(ValueError): banks.Bank(tmp_path/'targets',tr,with_content_hash(dict(record,prepared='z'*64)))
    with (tmp_path/'targets/tokens.npy').open('r+b') as handle:
        handle.seek(-1,2); handle.write(b'x')
    with pytest.raises(ValueError): banks.Bank(tmp_path/'targets',tr,record)


@pytest.mark.parametrize('arm',c.ARMS)
def test_full30_scientific_fit_and_native_readback(tmp_path,monkeypatch,arm):
    tiny(monkeypatch); tr,bank,_=bank_fixture(tmp_path); spec=spec_fixture(tmp_path)
    report,receipt=worker.fit(tmp_path/arm,spec,arm,tr,cache('validation'),bank,device='cpu')
    report2,arrays=worker.validate_fit(tmp_path/arm,spec,arm,bank.manifest['content_hash'])
    assert len(report['validation_history'])==30 and report==report2
    assert arrays['probabilities'].shape==(22,11)
    assert not any(k.startswith('predictor.') for k in torch.load(tmp_path/arm/'selected.pt',weights_only=True))
    bad=with_content_hash(dict(report,passes=29))
    with pytest.raises(ValueError): training.validate_report(bad,spec,arm,bank.manifest['content_hash'],technical=False)
    bank.close()


def test_weak_auc_never_stops_training(tmp_path,monkeypatch):
    tiny(monkeypatch); tr,bank,_=bank_fixture(tmp_path); spec=spec_fixture(tmp_path)
    original=training.evaluate_probabilities; calls=[]
    def poor(labels,probs):
        calls.append(1); return dict(original(labels,probs),macro_ovr_auc=.4 if len(calls)==1 else .1)
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    node=spec['nodes']['set']
    report,_=training.train(model.new_model(node),tr,cache('validation'),bank,node,device='cpu')
    assert report['passes']==30 and report['selected_pass']==1
    training.validate_report(report,spec,'set',bank.manifest['content_hash'],technical=False); bank.close()


def test_debug_dag_exact_dependency_and_authorization(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path); plan=submission.render(spec)
    assert plan['jobs']==3 and plan['scientific_fits']==2
    assert [r['task_id'] for r in plan['commands']]==['preflight','aligned','set']
    for row in plan['commands']:
        assert '--partition=debug' in row['command'] and '--no-requeue' in row['command']
        assert '--export=NONE' in row['command'] and '--mem=90000M' in row['command']
        if row['task_id']!='preflight': assert '--dependency=afterok:${JOB_preflight}' in row['command']
    monkeypatch.setattr(campaign,'check',lambda *_:None)
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)


def test_exact_dry_review_live_journal_and_duplicate_block(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda *_:None)
    monkeypatch.setattr(submission,'check_site',lambda *_:None)
    plan=submission.submit(spec)
    with pytest.raises(PermissionError):
        submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization='wrong')
    calls=[]
    def fake(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(spec)==dict(preflight='101',aligned='102',set='103')
    assert all('--dependency=afterok:101' in row for row in calls[1:])
    assert all('--kill-on-invalid-dep=yes' in row for row in calls)
    with pytest.raises(FileExistsError):
        submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==3


def test_no_science_without_exact_successful_preflight(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(preflight='101'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'ryreu')
    valid='101|lukapp_preflight|ryreu|reu-aisocial|COMPLETED|0:0\n'
    for output in ('',valid.replace('COMPLETED','FAILED'),valid.replace('0:0','1:0'),
                   valid.replace('ryreu','other'),valid.replace('lukapp_preflight','other'),
                   valid.replace('reu-aisocial','other'),valid+valid):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=output))
        with pytest.raises(ValueError): submission.successful_preflight(spec,'101')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=valid))
    submission.successful_preflight(spec,'101')
    with pytest.raises(ValueError): submission.successful_preflight(spec,'999')
    monkeypatch.setattr(campaign,'check',lambda *_:None)
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('Science must not execute without evidence'))
    with pytest.raises(FileNotFoundError): worker.run(spec,'aligned')
    assert not (tmp_path/'aligned').exists()


def test_clean_source_registration_and_no_existing_runtime_edits(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else ''))
    assert source.current(project,'b'*40)==dict(content_hash='a'*64)
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else 'M\tsrc/hlt_classification/jetclass2_delphes/model.py'))
    with pytest.raises(ValueError,match='runtime changed'): source.current(project,'b'*40)
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout=''))
    with pytest.raises(ValueError,match='complete'): source.current(project,'b'*40)
    compile(source.REPLAY,'reference_replay','exec')
    assert 'direct.results(s)' in source.REPLAY


def test_fresh_root_and_immutable_scientific_registration(tmp_path,monkeypatch):
    parent=dict(nodes=parent_nodes(),root=str(tmp_path/'donor/science'),project_dir=str(tmp_path/'old_project'))
    write_immutable_json(tmp_path/'donor/science/campaign_spec.json',parent)
    anchor=dict(spec=file_ref(tmp_path/'donor/science/campaign_spec.json'),
        evidence=dict(prepared='a'*64,prepared_root=str(tmp_path/'prepared'),raw_container=str(tmp_path/'raw')))
    reference=c.artifact('REFERENCE',anchor=anchor,teacher=dict(coordinate='U100'),
        spec=dict(path=str(tmp_path/'reference/campaign_spec.json')))
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='c'*64))
    monkeypatch.setattr(source,'import_reference',lambda *a:(reference,parent))
    monkeypatch.setattr(source,'validate_reference',lambda *_:None)
    args=(tmp_path/'reference/campaign_spec.json',tmp_path/'project','b'*40)
    for quota in (0,1,True):
        with pytest.raises(ValueError): campaign.create(tmp_path/'wrong',*args,available_quota_gib=quota)
    assert not (tmp_path/'wrong').exists()
    with pytest.raises(PermissionError): campaign.create(tmp_path/'donor/subrun',*args,available_quota_gib=2)
    spec=campaign.create(tmp_path/'study',*args,available_quota_gib=2)
    campaign.check(spec)
    assert spec['execution_site']['partition']=='debug' and len(spec['tasks'])==3
    with pytest.raises(FileExistsError): campaign.create(tmp_path/'study',*args,available_quota_gib=2)
    bad=with_content_hash(dict(spec,train_minutes=60))
    with pytest.raises(ValueError): campaign.validate_spec(bad)


def test_scratch_requires_canonical_local_disk_not_home_or_tmpfs(tmp_path,monkeypatch):
    root=tmp_path/'checkpoints/study'; root.mkdir(parents=True)
    spec=spec_fixture(root); local=tmp_path/'scratch'; local.mkdir()
    spec['scratch_base']=str(local)
    def mounts(kind):
        return f'1 0 8:1 / {local.as_posix()} rw - {kind} /dev/fixture rw\n'
    for kind in ('ceph','nfs','tmpfs'):
        monkeypatch.setattr(worker,'mountinfo',lambda:mounts(kind))
        with pytest.raises(PermissionError,match='node-local disk'): worker.scratch_directory(spec)
    monkeypatch.setattr(worker,'mountinfo',lambda:mounts('xfs'))
    assert worker.scratch_directory(spec)==local
    spec['scratch_base']=str(root)
    with pytest.raises(PermissionError): worker.scratch_directory(spec)


def test_projection_and_single_seed_registration():
    spec=spec_fixture(Path('/x'))
    assert spec['counts']==dict(train=22,validation=22)
    for a in c.ARMS:
        n=dict(spec['nodes'][a]); n.pop('arm'); n.pop('node_id'); n.pop('branch')
        reference=dict(c.old_node(dict(nodes=parent_nodes()))); reference.pop('node_id'); reference.pop('branch')
        assert n==reference
    assert worker.projected_minutes(600,[dict(train_seconds=3600,validation_seconds=60)])>1440
    assert c.recipe()['logit_kd'] is False and c.recipe()['auxiliary_coefficient']==5


def test_production_workers_cuda_miniature(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); root=tmp_path/'checkpoints'/'study'; root.mkdir(parents=True); spec=spec_fixture(root)
    native=DelphesParticleTransformer().cuda().eval(); save_state(tmp_path/'teacher.pt',native.state_dict())
    richval=cache('validation','U100'); val=cache('validation')
    probabilities=predict(native,richval,device='cuda',batch_size=128)
    pred=dict(identities=val.identities,labels=val.labels,probabilities=probabilities)
    atomic_publish_bytes(tmp_path/'reference.npz',deterministic_npz_bytes(pred))
    ref=file_ref(tmp_path/'reference.npz')
    teacher=teacher_artifact('TEACHER',coordinate='U100',target_role='train',prepared='a'*64,
        files=dict(checkpoint=file_ref(tmp_path/'teacher.pt'),predictions=ref))
    spec['teacher']=teacher
    spec['anchor']['evidence']['models']={n:dict(predictions=ref) for n in ('M0HLT','OFFLINE')}
    spec['reference']=dict(files=dict(predictions=ref),reference=dict(files=dict(predictions=ref)))
    spec['anchor']['evidence']['environment']=dict(fixture=True)
    scratch=tmp_path/'scratch'; scratch.mkdir()
    monkeypatch.setattr(worker,'scratch_directory',lambda _:scratch)
    monkeypatch.setattr(campaign,'check',lambda *_:None); monkeypatch.setattr(campaign,'validate_spec',lambda *_:None)
    monkeypatch.setattr(source,'current',lambda *_:spec['source_snapshot'])
    monkeypatch.setattr(worker,'installed_environment',lambda:dict(fixture=True))
    monkeypatch.setattr(worker,'gpu_identity',lambda:dict(total_memory_bytes=40*2**30,name='fixture A100'))
    monkeypatch.setattr(worker,'memory_snapshot',lambda _t:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=max(1,torch.cuda.max_memory_allocated())))
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda *_:(None,None,None,None))
    monkeypatch.setattr(worker,'data',lambda _s,r,co,_m:cache(r,co))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *_:np.arange(22,dtype=np.int64))
    monkeypatch.setattr(submission,'accepted_jobs',lambda _s:dict(preflight='101',aligned='102',set='103'))
    monkeypatch.setattr(submission,'successful_preflight',lambda *_:None)
    for task,job in (('preflight','101'),('aligned','102'),('set','103')):
        monkeypatch.setattr(worker,'execution',lambda *_:job)
        result=worker.run(spec,task)
        assert result['scratch_payloads_retired'] and not list(scratch.iterdir())
        if task=='preflight': assert worker.results(spec)['pending']==['aligned','set']
        if task=='aligned': assert worker.results(spec)['pending']==['set']
    result=worker.results(spec); assert not result['pending'] and len(result['rows'])==7
    for arm in c.ARMS:
        history=load_json(root/arm/'fit/training_report.json')['validation_history']
        assert len(history)==30
        assert load_json(root/arm/'bank_manifest.json')==load_json(root/'preflight/bank_manifest.json')
    tokenfile=root/'aligned/bank_manifest.json'
    saved_bank=load_json(tokenfile); changed=deepcopy(saved_bank); changed['outputs'][0]['sha256']='f'*64
    tokenfile.write_text(json.dumps(with_content_hash(changed)))
    # Even re-signing a new top-level receipt cannot authorize different targets.
    top=root/'aligned/receipt.json'; saved_top=load_json(top); changed_top=deepcopy(saved_top)
    changed_top['outputs']=[file_ref(tokenfile,root=tokenfile.parent) if r['path']=='bank_manifest.json'
                            else r for r in changed_top['outputs']]
    top.write_text(json.dumps(with_content_hash(changed_top)))
    with pytest.raises(ValueError,match='exact preflight targets'): worker.results(spec)
    tokenfile.write_text(json.dumps(saved_bank))
    top.write_text(json.dumps(saved_top))
    # Technical failure retires only our temporary bank, never accepts a gate.
    failed=dict(spec,root=str(root.parent/'failed')); Path(failed['root']).mkdir()
    monkeypatch.setattr(worker,'execution',lambda *_:'101')
    def fail(*_): raise ValueError('injected technical failure')
    monkeypatch.setattr(worker,'parity_and_stress',fail)
    with pytest.raises(ValueError,match='injected'): worker.run(failed,'preflight')
    assert not list(scratch.iterdir()) and not (Path(failed['root'])/'preflight/receipt.json').exists()
    assert (Path(failed['root'])/'preflight/claim.json').exists()
    evidence=root/'preflight/receipt.json'; saved=load_json(evidence)
    saved['projections']['set']=1441; evidence.write_text(json.dumps(with_content_hash(saved)))
    with pytest.raises(ValueError): worker.validate_preflight(spec,accounting=False)


def test_cli_and_shell_contracts():
    root=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,'-B',str(root/'scripts/luka_particle_prediction.py'),'--help'],capture_output=True,text=True)
    assert proc.returncode==0 and 'results' in proc.stdout
    wrapper=(root/'sbatch/run_luka_particle_prediction.sh').read_text()
    assert '"${PROJECT_DIR}/sbatch/common.sh"' in wrapper
    assert 'PYTHONNOUSERSITE=1' in wrapper and 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in wrapper
    assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in wrapper
    helper=(root/'scripts/queue_luka_particle_prediction.sh').read_text()
    assert '</dev/tty' in helper and c.AUTHORIZATION in helper
    assert 'scancel' not in helper and 'scontrol' not in helper
