"""Cumulative science, exact ancestry, no-preflight DAG and real four-stage workers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat_multiteacher import fixture as multi_fixture
from test_luka_fullstat import fullstat, durable
from hlt_classification.data.cache_contracts import (
    load_json, load_npz_arrays, write_immutable_json, with_content_hash)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.regular_campaign import nodes as original_nodes
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_multiteacher import model as multi_model, training as multi_training
from hlt_classification.luka_fullstat_cumulative_embedding import (
    contracts as c, campaign, source, banks, model, training, worker, submission)


def fixture(path, *, export=True):
    old, tr, val, views = multi_fixture(path/'previous', export=export)
    accepted = dict(technical_checks_passed=True, gpu=old['reuse']['gpu'],
                    environment=old['reuse']['environment'])
    preflight = dict(accepted, train_minutes=700, recipe=dict(batch_size=128),
                     measurements=[dict(coordinate=co) for co in c.STAGES.values()])
    write_immutable_json(path/'preflight.json', preflight)
    write_immutable_json(path/'setup.json', accepted)
    refs = old['reference']
    reuse = c.artifact('REUSE', counts=c.COUNTS, anchor=old['anchor'],
        references={n: refs for n in ('M0HLT','OFFLINE','U100_EMBED_W5','MULTI4_EMBED_W5')},
        base_node=old['reuse']['base_node'], execution_site=old['execution_site'],
        offline_refs=old['teachers']['OFFLINE']['refs'], offline_bank=old['reuse_banks']['OFFLINE'],
        gpu=old['reuse']['gpu'], environment=old['reuse']['environment'],
        summaries={r:{co:dict(maximum=4,resident_bytes=10000) for co in c.STAGES.values()}
                   for r in ('train','validation')},
        original_preflight=file_ref(path/'preflight.json'), multi_setup=file_ref(path/'setup.json'),
        multi_train_minutes=1089, original_train_minutes=700, protected_roots=[old['root']])
    root = path/'cumulative'; root.mkdir()
    s = c.artifact('SPEC', parents=dict(source='c'*64,reuse=reuse['content_hash']),
        root=str(root), project_dir='/project', source_commit='b'*40, source_snapshot=dict(content_hash='c'*64),
        reuse=reuse, available_result_bytes=12*2**30, **campaign.policy(reuse))
    s = with_content_hash(dict(s,counts=dict(train=len(tr),validation=len(val))))
    write_immutable_json(root/'campaign_spec.json',s)
    caches={}
    for role, original in [('train',tr),('validation',val)]:
        caches[role]={}
        for co in c.STAGES.values():
            value=deepcopy(original);value.coordinate_name=co
            caches[role][co]=value
    return s,caches


def mock_execution(s, caches, monkeypatch):
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:os.environ['SLURM_JOB_ID'])
    monkeypatch.setattr(worker,'metadata',lambda _: (dict(content_hash=s['anchor']['evidence']['prepared'],
        summaries=s['reuse']['summaries']),None,None))
    monkeypatch.setattr(worker,'data',lambda s,r,co,**kw:caches[r][co])
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:{'train_'+stage:str(101+i) for i,stage in enumerate(c.STAGES)})
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def test_exact_graph_weights_fresh_coordinate_seeds_and_sealed_roles(tmp_path):
    s,_=fixture(tmp_path,export=False)
    originals={n['coordinate']:n for n in original_nodes()}
    assert list(c.STAGES)==['UE100','DE066','DE033','DE000']
    for i,(stage,co) in enumerate(c.STAGES.items()):
        r,n=s['recipes'][stage],s['nodes'][stage]
        assert r['teacher_order']==['OFFLINE']+list(c.STAGES)[:i]
        assert r['weights']=={t:5/(i+1) for t in r['teacher_order']}
        assert r['epochs']==20 and r['batch_size']==128 and r['ce_coefficient']==1 and r['kd_coefficient']==0
        assert not any(r[k] for k in ('attention','relational','warm_start','teacher_embedding_average'))
        assert n['coordinate']==co and n['teacher'] is None and n['context_coordinate'] is None
        assert n['deployable'] is (stage=='DE000')
        for key in ('initialization_seed','sampler_seed','u','f'):assert n[key]==originals[co][key]
        assert c.tasks()[i]['dependencies']==['train_'+t for t in list(c.STAGES)[:i]]
    assert s['train_minutes']==1209 and s['new_scientific_fits']==4
    assert s['preflight_jobs']==s['setup_jobs']==s['startup']['technical_epochs']==0
    assert s['startup']['benchmark_steps']==0 and not s['startup']['full_validation_pass']
    assert s['final_test_sealed'] and not s['final_test_accessed']


@pytest.mark.parametrize('key,value',[('counts',{}),('gpu',dict(name='A100')),('references',{}),
    ('original_train_minutes',True),('multi_train_minutes',2800),('original_train_minutes',699)])
def test_inapplicable_reuse_or_walltime_refused(tmp_path,key,value):
    s,_=fixture(tmp_path,export=False)
    with pytest.raises(ValueError):campaign.policy(with_content_hash(dict(s['reuse'],**{key:value})))


def test_source_evidence_payload_and_resident_memory_cannot_be_bypassed(tmp_path):
    s,_=fixture(tmp_path,export=False)
    bad=deepcopy(s['reuse']);bad['summaries']['train']['U100']['resident_bytes']=200*2**30
    with pytest.raises(MemoryError):campaign.policy(with_content_hash(bad))
    p=Path(s['reuse']['original_preflight']['path']);p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError):campaign.policy(s['reuse'])


@pytest.mark.parametrize('stage',c.STAGES)
def test_loss_exact_mean_weight_and_detached_targets(stage):
    x=torch.randn(5,11,requires_grad=True);y=torch.arange(5)
    ps={t:torch.randn(5,128,requires_grad=True) for t in c.teachers(stage)}
    ts={t:torch.randn(5,128,requires_grad=True) for t in c.teachers(stage)}
    loss,ce,aux=training.objective(x,y,ps,ts,stage=stage)
    weight=5/len(ts)
    torch.testing.assert_close(loss,ce+sum(weight*aux[t] for t in ts),rtol=0,atol=0)
    loss.backward()
    for t in ts:
        assert ts[t].grad is None
        torch.testing.assert_close(ps[t].grad,weight*(ps[t].detach()-ts[t].detach()).clamp(-1,1)/ps[t].numel())
    with pytest.raises(ValueError):training.objective(x,y,ps,{},stage=stage)
    ts[next(iter(ts))]=torch.full((5,128),float('nan'))
    with pytest.raises(ValueError):training.objective(x,y,ps,ts,stage=stage)


@pytest.mark.parametrize('bf16',[False,True])
def test_final_matches_accepted_four_head_kernel_and_optimizer_step(tmp_path,monkeypatch,bf16):
    tiny(monkeypatch)
    s,cs=fixture(tmp_path,export=False)
    if bf16 and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    device='cuda' if bf16 else 'cpu'
    old=multi_model.new_model(s['reuse']['base_node']);rng=torch.get_rng_state().clone()
    new=model.new_model(s['nodes']['DE000'])
    assert torch.equal(rng,torch.get_rng_state())
    mapped=dict(OFFLINE='OFFLINE',UE100='COARSE_U100',DE066='COARSE_D066',DE033='COARSE_D033')
    assert all(torch.equal(v,old.native_state()[k]) for k,v in new.native_state().items())
    for t,old_t in mapped.items():
        assert all(torch.equal(v,old.predictors[old_t].state_dict()[k]) for k,v in new.predictors[t].state_dict().items())
    old.to(device).train();new.to(device).train()
    old.capture=new.capture=True
    raw=cs['train']['D000'].batch(np.arange(4));y=torch.from_numpy(raw['labels']).to(device)
    targets={t:torch.randn(4,128,device=device) for t in mapped}
    with parity_backend(device):
        seed=torch.get_rng_state();cuda_seed=torch.cuda.get_rng_state() if device=='cuda' else None
        logits=_forward(old,raw,device,bf16=bf16)
        a,*_=multi_training.objective(logits,y,old.predictions,{mapped[t]:v for t,v in targets.items()})
        torch.set_rng_state(seed)
        if cuda_seed is not None:torch.cuda.set_rng_state(cuda_seed)
        next_logits=_forward(new,raw,device,bf16=bf16)
        b,*_=training.objective(next_logits,y,new.predictions,targets,stage='DE000')
        torch.testing.assert_close(logits,next_logits,rtol=0,atol=0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.backward();b.backward()
        for k,p in new.named_parameters():
            old_k=k
            for t,old_t in mapped.items():old_k=old_k.replace('predictors.'+t+'.','predictors.'+old_t+'.')
            torch.testing.assert_close(p.grad,dict(old.named_parameters())[old_k].grad,rtol=0,atol=0)
        for m in (old,new):torch.optim.AdamW(m.parameters(),lr=1e-3).step()
        assert all(torch.equal(v,old.native_state()[k]) for k,v in new.native_state().items())
    new.eval();new.capture=False
    for head in new.predictors.values():
        for p in head.parameters():p.data.fill_(float('nan'))
    with torch.no_grad():assert torch.isfinite(_forward(new,raw,device,bf16=bf16)).all()
    DelphesParticleTransformer().load_state_dict(new.native_state(),strict=True)


@pytest.mark.parametrize('role,co',[('final_test','D000'),('unused','U100'),('train','OFFLINE'),('train','U050')])
def test_only_registered_roles_and_views(role,co):
    with pytest.raises(PermissionError):worker.data({},role,co)


def test_real_prepared_cache_all_rungs(durable,monkeypatch):
    output,meta=durable
    s=dict(memory_mb=180000,counts=meta[1]['counts'],anchor=dict(evidence=dict(
        prepared=meta[0]['content_hash'],cache_root=str(output))))
    monkeypatch.setattr(worker,'metadata',lambda _:meta)
    for role in ('train','validation'):
        original=worker.data(s,role,'D000')
        for co in c.STAGES.values():
            value=worker.data(s,role,co)
            np.testing.assert_array_equal(value.identities,original.identities)
            np.testing.assert_array_equal(value.labels,original.labels)


def test_offline_bank_import_keeps_bytes_and_cross_view_join(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,cs=fixture(tmp_path)
    before=deepcopy(s['offline_bank'])
    for co in c.STAGES.values():
        m,a=banks.arrays(before,cs['train'][co],refs=s['offline_refs'],teacher_id='OFFLINE')
        assert m['contract'].startswith('LUKA_FULLSTAT_EMBEDDING_SWEEP_')
        assert a['targets'].shape==(len(cs['train'][co]),128) and not a['targets'].flags.writeable
        np.testing.assert_allclose(a['targets'].mean(0),0,atol=1e-6)
    assert file_ref(before['path'])==before
    for changed in (deepcopy(cs['train']['U100']),deepcopy(cs['validation']['U100'])):
        changed.identities=changed.identities[::-1]
        with pytest.raises(ValueError):banks.arrays(before,changed,refs=s['offline_refs'],teacher_id='OFFLINE')
    with pytest.raises(PermissionError):banks.Bundle(s,'UE100',cs['validation']['U100'])


def test_new_selected_native_bank_export_semantics_and_failed_joins(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,cs=fixture(tmp_path)
    tr,val=cs['train']['U100'],cs['validation']['U100']
    bundle=banks.publish(s,'UE100',tr)
    root=Path(s['root']);directory=root/'train_UE100'
    with parity_backend('cpu'):worker.fit(directory,s,'UE100',tr,val,bundle,device='cpu')
    report=worker.validate_fit(directory,s,'UE100',bundle.manifest['content_hash'])
    refs=banks.refs_for(s,'UE100')
    teacher=DelphesParticleTransformer()
    teacher.load_state_dict(torch.load(refs['checkpoint']['path'],weights_only=True))
    output=root/'targets/UE100'
    banks.export(output,teacher,tr,refs=refs,device='cpu',teacher_id='UE100')
    manifest=output/'manifest.json';original=manifest.read_bytes()
    for key,value in [('teacher','COARSE_U100'),('coordinate','D033'),('role','validation'),
                      ('rows',1),('width',64),('final_test_accessed',True)]:
        manifest.write_text(json.dumps(with_content_hash(dict(json.loads(original),**{key:value}))))
        with pytest.raises((ValueError,PermissionError)):
            banks.arrays(file_ref(manifest),cs['train']['D066'],refs=refs,teacher_id='UE100')
    manifest.write_bytes(original)
    _,a=banks.arrays(file_ref(manifest),cs['train']['D066'],refs=refs,teacher_id='UE100')
    from hlt_classification.luka_embedding_aux.model import head_input
    from hlt_classification.luka_embedding_aux.targets import standardize
    raw=tr.batch(np.arange(len(tr)))
    with torch.inference_mode():
        _,native=head_input(teacher,**{k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')})
    expected,mean,scale=standardize(native.numpy())
    np.testing.assert_allclose(a['targets'],expected,atol=1e-5)
    np.testing.assert_allclose(a['mean'],mean,atol=1e-6)
    np.testing.assert_allclose(a['scale'],scale,atol=1e-6)
    with pytest.raises(ValueError,match='checkpoint'):
        banks.export(tmp_path/'wrong',DelphesParticleTransformer(),tr,refs=refs,device='cpu',teacher_id='UE100')
    for forbidden in (cs['validation']['U100'],cs['train']['D000']):
        with pytest.raises(PermissionError):banks.export(tmp_path/'bad',teacher,forbidden,refs=refs,device='cpu',teacher_id='UE100')
    for key in ('identities','labels'):
        wrong=deepcopy(cs['train']['D066']);setattr(wrong,key,getattr(wrong,key)[::-1])
        with pytest.raises(ValueError):banks.arrays(file_ref(manifest),wrong,refs=refs,teacher_id='UE100')
    bad=deepcopy(report);bad['validation_history'][0]['weighted_auxiliary_losses']['OFFLINE']=0
    with pytest.raises(ValueError):worker.report_valid(with_content_hash(bad),s,'UE100',bundle.manifest['content_hash'])
    (output/'targets.npz').write_bytes((output/'targets.npz').read_bytes()+b'damage')
    with pytest.raises(ValueError):banks.arrays(file_ref(manifest),cs['train']['D066'],refs=refs,teacher_id='UE100')


@pytest.mark.parametrize('measured',[dict(peak_rss_bytes=0,gpu_peak_bytes=1),
    dict(peak_rss_bytes=180000*1024**2,gpu_peak_bytes=1),dict(peak_rss_bytes=1,gpu_peak_bytes=48*2**30)])
def test_memory_failure_no_automatic_fallback(tmp_path,measured):
    s,_=fixture(tmp_path,export=False)
    with pytest.raises(MemoryError):worker.limits(s,measured)


def test_dry_review_and_exact_four_job_submission(tmp_path,monkeypatch):
    s,_=fixture(tmp_path,export=False)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s)
    assert plan['jobs']==4 and plan['setup_jobs']==plan['preflight_jobs']==0
    assert [r['task_id'] for r in plan['commands']]==['train_'+stage for stage in c.STAGES]
    for i,r in enumerate(plan['commands']):
        assert '--kill-on-invalid-dep=yes' in r['command'] and '--no-requeue' in r['command']
        if i:assert '--dependency=afterok:'+':'.join('${JOB_train_'+t+'}' for t in list(c.STAGES)[:i]) in r['command']
    for auth,h in [(None,plan['content_hash']),(c.AUTHORIZATION,'bad')]:
        with pytest.raises(PermissionError):submission.submit(s,execute=True,authorization=auth,plan_hash=h)
    calls=[]
    def fake(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,authorization=c.AUTHORIZATION,plan_hash=plan['content_hash'])
    assert submission.accepted_jobs(s)=={'train_'+stage:str(101+i) for i,stage in enumerate(c.STAGES)}
    assert '--dependency=afterok:101:102:103' in calls[3]
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,authorization=c.AUTHORIZATION,plan_hash=plan['content_hash'])
    assert len(calls)==4


def test_ambiguous_submission_stays_claimed_without_retry(tmp_path,monkeypatch):
    s,_=fixture(tmp_path,export=False)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s);calls=[]
    def ambiguous(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='not-an-exact-job-id\n',stderr='')
    monkeypatch.setattr(subprocess,'run',ambiguous)
    with pytest.raises(Exception):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


def test_failed_accounting_and_missing_parent_prevent_training(tmp_path,monkeypatch):
    s,cs=fixture(tmp_path,export=False)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(train_UE100='101',train_DE066='102'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'rlyang')
    valid='101|lukafsc_train_UE100|rlyang|default|COMPLETED|0:0\n'
    for text in ('',valid.replace('COMPLETED','FAILED'),valid.replace('0:0','1:0'),valid.replace('rlyang','other')):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
        with pytest.raises(ValueError):submission.successful(s,'train_UE100','101')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=valid))
    submission.successful(s,'train_UE100','101')
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'102')
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('No fit without completed ancestor'))
    with pytest.raises(ValueError,match='ancestor incomplete'):worker.run(s,'train_DE066')
    assert not (Path(s['root'])/'train_DE066_attempt').exists()


def test_actual_full_architecture_cuda_four_stage_production_miniature(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('No CUDA; not Oscar resource certification')
    s,cs=fixture(tmp_path)
    mock_execution(s,cs,monkeypatch)
    source_bank=deepcopy(s['offline_bank'])
    for i,stage in enumerate(c.STAGES):
        monkeypatch.setenv('SLURM_JOB_ID',str(101+i))
        receipt=worker.run(s,'train_'+stage)
        assert receipt['stage']==stage
        report=load_json(Path(s['root'])/('train_'+stage)/'training_report.json')
        assert report['passes']==20 and len(report['validation_history'])==20
        assert all(set(r['auxiliary_losses'])==set(c.teachers(stage)) for r in report['validation_history'])
        startup=load_json(Path(s['root'])/('train_'+stage+'_attempt')/'startup.json')
        assert startup['stress_steps']==3 and startup['technical_epochs']==startup['benchmark_steps']==0
        assert not startup['full_validation_pass']
        state=torch.load(Path(s['root'])/('train_'+stage)/'selected.pt',weights_only=True)
        assert not any(k.startswith('predictors.') for k in state)
        if stage!='DE000':
            a=load_npz_arrays(Path(s['root'])/'targets'/stage/'targets.npz')
            np.testing.assert_allclose(a['targets'].mean(0),0,atol=1e-6)
        if i>0:
            assert receipt['parents'][list(c.STAGES)[i-1]]==previous_receipt['content_hash']
        previous_receipt=receipt
        summary=worker.results(s)
        assert summary['pending']==list(c.STAGES)[i+1:]
    assert file_ref(source_bank['path'])==source_bank
    assert not summary['final_test_accessed'] and not summary['fresh_matched_control']
    assert [r['node_id'] for r in summary['rows']]==['M0HLT','OFFLINE','U100_EMBED_W5','MULTI4_EMBED_W5',*c.STAGES]
    assert summary['rows'][-1]['input']=='native_HLT'
    assert all(r['input']=='richer_input_oracle' for r in summary['rows'][4:7])
    # Distinct selected ancestors keep independent latent axes and moments.
    first=load_npz_arrays(Path(s['root'])/'targets/UE100/targets.npz')
    second=load_npz_arrays(Path(s['root'])/'targets/DE066/targets.npz')
    assert not np.array_equal(first['mean'],second['mean'])
    assert not np.array_equal(first['targets'],second['targets'])
    # Rehashed semantic damage in a distant ancestor is rejected before final results.
    task=Path(s['root'])/'train_UE100/task_receipt.json';before=task.read_bytes()
    task.write_text(json.dumps(with_content_hash(dict(load_json(task),job_id='999'))))
    with pytest.raises(ValueError):worker.results(s)
    task.write_bytes(before)
    target=Path(s['root'])/'targets/DE066/targets.npz';target.write_bytes(target.read_bytes()+b'damage')
    with pytest.raises(ValueError):worker.completed(s,'DE000')


def test_low_auc_does_not_stop_twenty_epoch_fit(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,cs=fixture(tmp_path)
    tr,val=cs['train']['U100'],cs['validation']['U100']
    bank=banks.publish(s,'UE100',tr)
    monkeypatch.setattr(training,'predict',lambda *a,**kw:np.full((len(val),11),1/11,np.float32))
    report,state=training.train(tr,val,bank=bank,node=s['nodes']['UE100'],device='cpu')
    worker.report_valid(report,s,'UE100',bank.manifest['content_hash'])
    assert report['passes']==20 and report['selected_pass']==1
    assert all(r['validation']['macro_ovr_auc']==.5 for r in report['validation_history'])
    assert state


def test_reporting_ranks_ties_and_censoring():
    labels=np.array([0,0,0,1,1,1,1,1,1,1,1,1,1])
    p=np.zeros((13,11));p[:,0]=1
    scores=np.array([.85,.65,.2,1.,.9,.8,.7,.6,.5,.4,.3,.2,.1])
    p[:,0]=1-scores;p[:,1]=scores
    assert worker.rejection(labels,p,.3)['X_bb']==3
    assert worker.rejection(labels,p,.5)['X_bb']==1.5
    p[0,1]=.8;p[0,0]=.2
    assert worker.rejection(labels,p,.3)['X_bb']==3 # include exact-threshold tie
    p[:3,1]=0;p[:3,0]=1
    assert worker.rejection(labels,p,.5)['X_bb'] is None
    with pytest.raises(ValueError):worker.rejection(labels,p,.99)


def test_source_replay_and_changed_donor_runtime_fail_closed(tmp_path,monkeypatch):
    original=tmp_path/'old';project=tmp_path/'new';root=tmp_path/'parent';root.mkdir()
    for directory in (original,project):
        (directory/'src').mkdir(parents=True);(directory/'src/kernel.py').write_text('# accepted\n')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),
        source_commit=source.DONOR,counts=c.COUNTS))
    evidence=dict(jobs=dict(train_multi4='123'),execution_site=dict(account='default'))
    def replay(args,**kw):
        if args[0]=='git':return SimpleNamespace(stdout='src/kernel.py\n')
        if args[0]=='sacct':return SimpleNamespace(stdout='123|lukafsm_train_multi4|rlyang|default|COMPLETED|0:0\n')
        assert kw['cwd']==original and kw['env']['PYTHONPATH']==str(original/'src')
        return SimpleNamespace(returncode=0,stdout='CUMULATIVE_REUSE='+json.dumps(evidence)+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',replay);monkeypatch.setattr(source.getpass,'getuser',lambda:'rlyang')
    c.validate(source.authenticate(root/'campaign_spec.json',project),'REUSE')
    compile(source.REPLAY,'replay','exec')
    assert 'campaign.check(s); result=worker.results(s)' in source.REPLAY
    (project/'src/kernel.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='Changed donor'):source.authenticate(root/'campaign_spec.json',project)


@pytest.mark.parametrize('output,returncode',[
    ('',1),('',0),('CUMULATIVE_REUSE={}\nCUMULATIVE_REUSE={}\n',0)])
def test_missing_failed_or_ambiguous_original_replay_blocks_reuse(tmp_path,monkeypatch,output,returncode):
    original=tmp_path/'old';project=tmp_path/'new';root=tmp_path/'parent';root.mkdir()
    for directory in (original,project):
        (directory/'src').mkdir(parents=True);(directory/'src/kernel.py').write_text('')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),
        source_commit=source.DONOR,counts=c.COUNTS))
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='src/kernel.py\n')
        if args[0]=='git' else SimpleNamespace(returncode=returncode,stdout=output,stderr='evidence rejected'))
    with pytest.raises(ValueError):source.authenticate(root/'campaign_spec.json',project)


def test_additive_runtime_guard_and_actual_donor_compatibility(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    changed=subprocess.run(['git','diff','--name-only','--diff-filter=MDR',source.DONOR,source.BASELINE,
        '--','src','scripts','sbatch'],cwd=project,check=True,capture_output=True,text=True).stdout.splitlines()
    assert not changed
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else '\n'.join(source.ADDITIONS|source.ENTRYPOINTS) if 'diff' in cmd else ''))
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    for p in source.REQUIRED:assert (project/p).is_file(),p
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else 'src/old.py' if 'diff' in cmd else ''))
    with pytest.raises(ValueError):source.current(project,'b'*40)


def test_shell_cli_docs_syntax():
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():bash=shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    for p in ('scripts/queue_luka_fullstat_cumulative_embedding.sh','sbatch/run_luka_fullstat_cumulative_embedding.sh'):
        subprocess.run([str(bash),'-n',p],cwd=project,check=True,capture_output=True,timeout=30)
    book=(project/'docs/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_RUNBOOK.md').read_text()
    for block in re.findall(r'^```bash\n(.*?)^```',book,re.M|re.S):
        subprocess.run([str(bash),'-n'],input=block,check=True,text=True,capture_output=True,timeout=30)
    subprocess.run([sys.executable,'-B','scripts/luka_fullstat_cumulative_embedding.py','--help'],
        cwd=project,check=True,capture_output=True,timeout=60)
    for p in source.REQUIRED:
        text=(project/p).read_text();assert all(line==line.rstrip() for line in text.splitlines()),p
        if p.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/p).parent.joinpath(link).is_file(),(p,link)
