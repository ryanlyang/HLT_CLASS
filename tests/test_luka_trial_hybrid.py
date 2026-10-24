"""Four full branches, exact losses, native targets, direct DAG and owned retirement."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_embedding_aux import tiny
from test_luka_attention import parent_nodes
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,with_content_hash,
    write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment,distillation_loss
from hlt_classification.jetclass2_delphes.runner import _forward,predict
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_hybrid_embedding.training import objective as old_objective
from hlt_classification.luka_trial_hybrid import contracts as c,model,training,banks,campaign,source,worker,submission


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def bare_spec(path):
    acceptance=dict(content_hash='b'*64,resources=dict(execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000),
        preflight=dict(projections=dict(aligned=120,set=150),environment={},gpu={}))
    return dict(root=str(path),project_dir='/project',source_commit='c'*40,source_snapshot=dict(content_hash='d'*64),
        anchor=dict(evidence=dict(prepared='a'*64,cache='e'*64,prepared_root='/prepared',raw_container='/raw',models={},environment={})),
        acceptance=acceptance,content_hash='f'*64,available_quota_gib=8,**campaign.policy(dict(nodes=parent_nodes()),acceptance))


def science_fixture(tmp_path,monkeypatch):
    root=tmp_path/'study'; root.mkdir(); spec=bare_spec(root)
    env=installed_environment(); spec['anchor']['evidence']['environment']=env; spec['acceptance']['preflight']['environment']=env
    spec['counts']=dict(train=22,validation=22); spec['original50k_rows']=22
    tr={co:cache('train',co) for co in ('OFFLINE',*c.STAGES.values())}
    val={co:cache('validation',co) for co in ('OFFLINE',*c.STAGES.values())}
    for name,co in (('OFFLINE','OFFLINE'),('M0HLT','D000')):
        folder=tmp_path/name; folder.mkdir(); torch.manual_seed(42 if name=='OFFLINE' else 43)
        native=DelphesParticleTransformer(); save_state(folder/'selected.pt',native.state_dict())
        write_immutable_json(folder/'report.json',dict(old=True))
        spec['anchor']['evidence']['models'][name]=dict(checkpoint=file_ref(folder/'selected.pt'),report=file_ref(folder/'report.json'))
    spec['acceptance']['anchor']=spec['anchor']; spec['acceptance']['teacher']={}; spec['acceptance']['reference']={}
    spec=with_content_hash(spec); write_immutable_json(root/'campaign_spec.json',spec)
    jobs={row['task_id']:str(101+i) for i,row in enumerate(c.tasks())}
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(campaign,'validate_spec',lambda *a,**kw:None)
    monkeypatch.setattr(worker,'execution',lambda _,t:jobs[t])
    monkeypatch.setattr(worker,'data',lambda _,r,co,meta:(tr if r=='train' else val)[co])
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda _:None)
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot'])
    monkeypatch.setattr(worker,'storage',lambda _:0)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:jobs)
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    def refs(folder,s,v,meta):
        folder.mkdir(); p=np.full((len(v),11),1/11,np.float32)
        for name in worker.REFERENCES:
            atomic_publish_bytes(folder/(name+'.npz'),deterministic_npz_bytes(dict(identities=v.identities,labels=v.labels,probabilities=p)))
        atomic_publish_bytes(folder/'original50k.npz',deterministic_npz_bytes(dict(indices=np.arange(len(v),dtype=np.int64))))
    monkeypatch.setattr(worker,'snapshot_references',refs)
    return spec,tr,val


def test_exact_four_arm_twenty_fit_graph_and_schedules():
    spec=bare_spec(Path('/study')); plan=submission.render(spec)
    assert len(plan['commands'])==22 and plan['scientific_fits']==20 and plan['preflight_jobs']==0
    assert spec['preflight_jobs']==spec['technical_training_epochs']==spec['startup_training_steps']==0
    assert spec['counts']==dict(train=299234,validation=74895)
    assert c.coefficients('c25',1)==(.25,.75)
    assert c.coefficients('c50',30)==(.5,.5)
    assert c.coefficients('c75',12)==(.75,.25)
    assert c.coefficients('anneal',1)==(.25,.75) and c.coefficients('anneal',13)==(.625,.375)
    assert all(c.coefficients('anneal',i)==(1.,0.) for i in range(25,31))
    assert c.teachers('UE100')==['OFFLINE','M0HLT'] and 'M0HLT' not in c.teachers('DE000')
    for arm in c.ARMS:
        for stage in c.STAGES:
            node=spec['nodes'][arm][stage]; r=c.recipe(arm,stage)
            assert sum(r['weights'].values())==pytest.approx(5.) and r['epochs']==30 and r['batch_size']==128
            assert node['deployable'] is (stage=='DE000')
            assert node['initialization_seed']==spec['nodes']['c25'][stage]['initialization_seed']
            row=next(r for r in plan['commands'] if r['task_id']==c.task_id(arm,stage))
            assert row['dependencies']==['targets']+[c.task_id(arm,s) for s in c.ancestors(stage)]
    for row in plan['commands']:
        assert '--partition=debug' in row['command'] and '--export=NONE' in row['command']
        assert '--kill-on-invalid-dep=yes' in row['command']
    assert not any('--gres=' in arg for arg in plan['commands'][-1]['command'])
    assert len(worker.target_payloads())==18 and len(set(worker.target_payloads()))==18
    with pytest.raises(ValueError): c.coefficients('c25',31)
    with pytest.raises(ValueError): c.coefficients('bad',1)


@pytest.mark.parametrize('arm',c.ARMS)
@pytest.mark.parametrize('stage',list(c.STAGES))
def test_exact_loss_gradients_temperature_scaling_and_detached_targets(arm,stage):
    x=torch.randn(4,11,requires_grad=True); y=torch.arange(4)
    q=torch.softmax(torch.randn(4,11)/2,-1).requires_grad_(True)
    p={t:torch.randn(4,128,requires_grad=True) for t in c.teachers(stage)}
    z={t:torch.randn(4,128,requires_grad=True) for t in c.teachers(stage)}
    loss,ce,kd,aux=training.objective(x,y,p,z,q,arm=arm,stage=stage,epoch=13)
    cw,kw=c.coefficients(arm,13)
    expected=cw*torch.nn.functional.cross_entropy(x,y)+kw*4*torch.nn.functional.kl_div(torch.log_softmax(x/2,-1),q.detach(),reduction='batchmean')
    expected+=sum(5/len(z)*torch.nn.functional.smooth_l1_loss(p[t],z[t].detach()) for t in c.teachers(stage))
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss,x,retain_graph=True)[0],torch.autograd.grad(expected,x,retain_graph=True)[0])
    if arm=='c25': torch.testing.assert_close(loss,old_objective(x,y,p,z,stage=stage,probabilities=q)[0],rtol=0,atol=0)
    loss.backward(); assert q.grad is None and all(v.grad is None for v in z.values())
    assert all(v.grad is not None and v.grad.abs().sum()>0 for v in p.values())
    zero=training.objective(x,y,p,z,q,arm='anneal',stage=stage,epoch=30)[0]
    torch.testing.assert_close(torch.autograd.grad(zero,x)[0],torch.autograd.grad(torch.nn.functional.cross_entropy(x,y),x)[0])


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bf16',[False,True])
def test_native_full_architecture_parity_and_shared_encoder_aux_gradients(device,bf16):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA not available')
    node=c.nodes(dict(nodes=parent_nodes()))['c25']['DE000']
    torch.manual_seed(node['initialization_seed']); native=DelphesParticleTransformer().to(device).eval()
    net=model.new_model(node).to(device).eval(); net.capture=True; raw=cache().batch(np.arange(4))
    assert net.registered==c.recipe('c25','DE000') and net.registered['epochs']==30
    assert all(torch.equal(v,native.state_dict()[k].cpu()) for k,v in net.native_state().items())
    with parity_backend(device):
        x=_forward(native,raw,device,bf16=bf16); y=_forward(net,raw,device,bf16=bf16)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert set(net.predictions)==set(c.teachers('DE000')) and all(v.shape==(4,128) for v in net.predictions.values())
        loss=sum(v.square().mean() for v in net.predictions.values()); loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in net.named_parameters() if n.startswith('mod.'))
        assert net.mod.fc[0].weight.grad is None
        assert all(p.grad is not None for p in net.predictors.parameters())
    assert not any(k.startswith('predictor') for k in net.native_state())


def test_frozen_native_export_and_train_only_normalization_and_joins(tmp_path,monkeypatch):
    tiny(monkeypatch); spec,tr,_=science_fixture(tmp_path,monkeypatch)
    worker.run(spec,'targets',device='cpu')
    folder=banks.directory(spec,'c25','OFFLINE'); rec=banks.teacher_record(spec,'c25','OFFLINE')
    m,a=banks.load(folder,tr['U000'],rec)
    np.testing.assert_allclose(a['targets'].mean(0),0,atol=1e-6)
    assert a['mean'].dtype==a['scale'].dtype==np.float64 and np.min(a['scale'])>=1e-3
    n=worker.native(rec['files']['checkpoint'],'cpu')
    np.testing.assert_allclose(a['probabilities'],predict(n,tr['OFFLINE'],device='cpu',temperature=2.,batch_size=128),rtol=1e-5,atol=1e-6)
    b=banks.Bundle(spec,'c75','UE000',tr['U000'])
    assert set(b.targets)=={'OFFLINE','M0HLT'} and b.manifest['arm']=='c75'
    with pytest.raises(PermissionError): banks.load(folder,cache('validation','D000'),rec)
    bad=deepcopy(tr['U000']); bad.identities=bad.identities[::-1].copy()
    with pytest.raises(ValueError): banks.load(folder,bad,rec)
    m=load_json(folder/'manifest.json'); m['temperature']=1.
    (folder/'manifest.json').write_text(json.dumps(with_content_hash(m)))
    with pytest.raises(ValueError): banks.load(folder,tr['U000'],rec)


def test_bad_support_and_resource_allowance_rejected():
    s=bare_spec(Path('/study')); accepted=deepcopy(s['acceptance'])
    accepted['preflight']['projections']['aligned']=1440
    with pytest.raises(ValueError,match='24h'): campaign.policy(dict(nodes=parent_nodes()),accepted)
    accepted=deepcopy(s['acceptance']); accepted['resources']['execution_site']['partition']='tier3'
    with pytest.raises(ValueError): campaign.policy(dict(nodes=parent_nodes()),accepted)
    summaries={r:{co:dict(maximum=600,resident_bytes=100) for co in ('OFFLINE',*c.STAGES.values())} for r in ('train','validation')}
    with pytest.raises(ValueError): campaign.check_support(summaries)


def test_campaign_registration_and_immutable_recipe_checks(tmp_path,monkeypatch):
    project=tmp_path/'project'; project.mkdir()
    root=tmp_path/'new'; template=bare_spec(root); accepted=deepcopy(template['acceptance'])
    accepted['anchor']=template['anchor']; accepted['spec']=dict(path=str(tmp_path/'token/campaign_spec.json'))
    accepted=with_content_hash(accepted)
    parent=dict(nodes=parent_nodes(),root=str(tmp_path/'old/science'),project_dir=str(tmp_path/'oldproject'))
    token=tmp_path/'token/campaign_spec.json'
    summaries={r:{co:dict(maximum=256,resident_bytes=1024**3) for co in ('OFFLINE',*c.STAGES.values())}
               for r in ('train','validation')}
    monkeypatch.setattr(source,'current',lambda *a:template['source_snapshot'])
    monkeypatch.setattr(source,'import_acceptance',lambda _: (accepted,parent))
    monkeypatch.setattr(source,'validate_acceptance',lambda _:None)
    monkeypatch.setattr(campaign,'support',lambda _:summaries)
    monkeypatch.setattr(campaign,'storage',lambda _:0)
    with pytest.raises(ValueError,match='Reserve'): campaign.create(root,token,project,'c'*40,available_quota_gib=7)
    assert not root.exists()
    with pytest.raises(PermissionError): campaign.create(project/'overlap',token,project,'c'*40,available_quota_gib=8)
    spec=campaign.create(root,token,project,'c'*40,available_quota_gib=8)
    campaign.validate_spec(spec); campaign.check(spec)
    with pytest.raises(FileExistsError): campaign.create(root,token,project,'c'*40,available_quota_gib=8)
    bad=deepcopy(spec); bad['recipes']['c75']['DE000']['auxiliary_total']=6.
    bad=with_content_hash(bad); (root/'campaign_spec.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Registered'): campaign.validate_spec(bad)


def test_submission_full_dry_live_exact_dependencies_and_duplicate_refusal(tmp_path,monkeypatch):
    spec=bare_spec(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    calls=[]
    def sbatch(cmd,**kw):
        calls.append(cmd); return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',sbatch)
    plan=submission.submit(spec); assert not calls
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==22 and len(submission.accepted_jobs(spec))==22
    assert not any('${JOB_' in arg for cmd in calls for arg in cmd)
    assert '--dependency=afterok:101:102' in calls[2]
    assert '--dependency=afterok:101' in calls[6]
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)


def test_ambiguous_submission_is_not_blindly_retried(tmp_path,monkeypatch):
    spec=bare_spec(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(submission,'check_site',lambda _:None)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='not-a-job\n',stderr=''))
    plan=submission.submit(spec)
    with pytest.raises(ValueError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)


def test_exact_failed_ancestor_accounting_blocks_descendants(tmp_path,monkeypatch):
    s=bare_spec(tmp_path); monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(targets='123'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'ryreu')
    for text in ('','123|lukahy_targets|ryreu|reu-aisocial|FAILED|1:0\n','123|other|ryreu|reu-aisocial|COMPLETED|0:0\n'):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
        with pytest.raises(ValueError): submission.successful(s,'targets','123')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='123|lukahy_targets|ryreu|reu-aisocial|COMPLETED|0:0\n'))
    submission.successful(s,'targets','123')


def test_source_guard_no_legacy_runtime_edits_and_required_files(monkeypatch):
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in args else 'M\tsrc/old.py\n'))
    with pytest.raises(ValueError,match='runtime'): source.current(Path(__file__).resolve().parents[1],'b'*40)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=''))
    with pytest.raises(ValueError,match='Commit'): source.current(Path(__file__).resolve().parents[1],'b'*40)


def test_poor_science_finishes_and_starts_from_scratch(tmp_path,monkeypatch):
    tiny(monkeypatch); spec,tr,val=science_fixture(tmp_path,monkeypatch); worker.run(spec,'targets',device='cpu')
    b=banks.Bundle(spec,'anneal','UE000',tr['U000'])
    monkeypatch.setattr(training,'predict',lambda *a,**kw:np.full((22,11),1/11,np.float32))
    student,report,state=training.train(tr['U000'],val['U000'],bank=b,node=spec['nodes']['anneal']['UE000'],device='cpu')
    training.validate_report(report,spec,'anneal','UE000',b.manifest['content_hash'])
    assert report['passes']==30 and report['selected_pass']==1
    torch.manual_seed(999)
    _,again,other=training.train(tr['U000'],val['U000'],bank=b,node=spec['nodes']['anneal']['UE000'],device='cpu')
    assert all(torch.equal(v,other[k]) for k,v in state.items())
    assert all(r['validation']['macro_ovr_auc']==.5 for r in report['validation_history'])
    bad=deepcopy(report); bad['validation_history'][20]['weighted_kd_loss']+=1.
    with pytest.raises(ValueError): training.validate_report(with_content_hash(bad),spec,'anneal','UE000',b.manifest['content_hash'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual installed-Weaver CUDA required')
def test_full_architecture_all_twenty_workers_600_epochs_and_retirement(tmp_path,monkeypatch):
    spec,tr,val=science_fixture(tmp_path,monkeypatch)
    worker.run(spec,'targets')
    assert len(worker.results(spec)['pending'])==20
    for arm in c.ARMS:
        for stage in c.STAGES:
            task=c.task_id(arm,stage); row=worker.run(spec,task)
            report=load_json(Path(spec['root'])/task/'fit/training_report.json')
            assert report['passes']==30 and len(report['validation_history'])==30
            assert row['task']==task
            state=torch.load(Path(spec['root'])/task/'fit/selected.pt',weights_only=True)
            assert not any(k.startswith('predictor') for k in state)
            if stage=='UE000':
                b=banks.Bundle(spec,arm,'UE100',tr['U100'])
                assert set(b.targets)=={'OFFLINE','M0HLT'}
                wrong=deepcopy(b.manifest); wrong['arm']=next(a for a in c.ARMS if a!=arm)
                with pytest.raises(ValueError): banks.validate_bundle(with_content_hash(wrong),spec,arm,'UE100',payloads=True)
    before=worker.results(spec); assert before['pending']==[] and len(before['rows'])==27
    checkpoints={r['task_id']:file_ref(Path(spec['root'])/r['task_id']/'fit/selected.pt')
                 for r in spec['tasks'] if r['task_id'] not in ('targets','report')}
    done=worker.run(spec,'report'); assert done['contract']=='LUKA_TRIAL_HYBRID_COMPLETE/v1'
    assert worker.results(spec)==before
    assert all(not (Path(spec['root'])/p).exists() for p in worker.target_payloads())
    for t,ref in checkpoints.items(): assert file_ref(ref['path'])==ref
    for role in ('validation','original50k'):
        assert all(r[role]['rows']==22 for r in before['rows'])
    # Native retained results remain authenticated even though disposable TRAIN targets are gone.
    last=Path(spec['root'])/'anneal_DE000/fit/validation_predictions.npz'
    last.write_bytes(last.read_bytes()+b'damage')
    with pytest.raises(ValueError): worker.results(spec)


def test_retirement_refuses_unfinished_and_partial_retirement(tmp_path,monkeypatch):
    s=bare_spec(tmp_path)
    with pytest.raises(ValueError): worker.retire(s,dict(pending=['x']),'123')
    report=tmp_path/'report'; report.mkdir()
    write_immutable_json(report/'retirement.json',{})
    with pytest.raises(ValueError,match='Partial'): worker.retired_evidence(s)


@pytest.mark.parametrize('damage',['nonfinite','dtype','labels','probabilities','moments'])
def test_bank_semantics_reject_even_rehashed_invalid_payload(tmp_path,monkeypatch,damage):
    tiny(monkeypatch); spec,tr,_=science_fixture(tmp_path,monkeypatch)
    worker.run(spec,'targets',device='cpu')
    folder=banks.directory(spec,'c25','OFFLINE'); rec=banks.teacher_record(spec,'c25','OFFLINE')
    a=load_npz_arrays(folder/'targets.npz'); m=load_json(folder/'manifest.json')
    if damage=='nonfinite': a['targets'][0,0]=np.nan
    elif damage=='dtype': a['targets']=a['targets'].astype(np.float64)
    elif damage=='labels': a['labels'][0]=(a['labels'][0]+1)%11
    elif damage=='probabilities': a['probabilities'][0]*=2.
    elif damage=='moments': a['mean'][0]+=1.
    (folder/'targets.npz').write_bytes(deterministic_npz_bytes(a))
    m['payload']=file_ref(folder/'targets.npz',root=folder)
    (folder/'manifest.json').write_text(json.dumps(with_content_hash(m)))
    with pytest.raises(ValueError): banks.load(folder,tr['U000'],rec)


def test_native_teacher_branch_transplant_and_checkpoint_identity_rejected(tmp_path,monkeypatch):
    tiny(monkeypatch); spec,tr,_=science_fixture(tmp_path,monkeypatch)
    worker.run(spec,'targets',device='cpu'); worker.run(spec,'c25_UE000',device='cpu')
    old=Path(spec['root'])/'c25_UE000'; new=Path(spec['root'])/'c50_UE000/fit'; new.mkdir(parents=True)
    for name in ('selected.pt','training_report.json'):
        (new/name).write_bytes((old/'fit'/name).read_bytes())
    with pytest.raises(ValueError,match='branch'): banks.teacher_record(spec,'c50','UE000')
    record=banks.teacher_record(spec,'c25','OFFLINE'); model=worker.native(record['files']['checkpoint'],'cpu')
    with torch.no_grad(): next(model.parameters()).add_(.01)
    with pytest.raises(ValueError,match='exact selected'): banks.check_selected(model,record)


def test_retirement_rejects_noncanonical_owned_leaf_before_any_deletion(tmp_path,monkeypatch):
    root=tmp_path/'study'; root.mkdir(); (root/'report').mkdir(); (root/'banks').mkdir()
    leaf=root/'banks/targets.npz'; leaf.write_bytes(b'keep me')
    s=bare_spec(root); original=Path.resolve
    monkeypatch.setattr(worker,'target_payloads',lambda:['banks/targets.npz'])
    # Simulate a symlinked parent resolving outside the owned campaign, including on Windows.
    monkeypatch.setattr(Path,'resolve',lambda p,*a,**kw:tmp_path/'external.npz' if p==leaf else original(p,*a,**kw))
    with pytest.raises(PermissionError): worker.retire(s,dict(pending=[]),'123')
    assert leaf.read_bytes()==b'keep me' and not (root/'report/retirement.json').exists()


def test_missing_ancestor_or_wrong_environment_stops_before_worker_output(tmp_path,monkeypatch):
    tiny(monkeypatch); spec,_,_=science_fixture(tmp_path,monkeypatch)
    with pytest.raises(FileNotFoundError): worker.run(spec,'c25_UE100',device='cpu')
    assert not (Path(spec['root'])/'c25_UE100').exists()
    monkeypatch.setattr(worker,'installed_environment',lambda:dict(wrong=True))
    with pytest.raises(ValueError,match='environment'): worker.run(spec,'targets',device='cpu')
    assert not (Path(spec['root'])/'targets').exists()


def test_cli_shell_and_no_hidden_preflight():
    project=Path(__file__).resolve().parents[1]
    script=(project/'scripts/queue_luka_trial_hybrid.sh').read_text()
    assert 'AUTHORIZE_LUKA_TRIAL_HYBRID_20_FITS_DEBUG_NO_PREFLIGHT_V1' in script
    assert '</dev/tty' in script and 'nohup' in script
    assert 'scancel' not in script and 'scontrol update' not in script
    text=(project/'sbatch/run_luka_trial_hybrid.sh').read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text and 'PYTHONNOUSERSITE=1' in text
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in text
    assert len(source.REQUIRED)==16
