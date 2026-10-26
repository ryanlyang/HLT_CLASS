"""FullStat four-rung relational objective and immutable production DAG."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat_splithead import fixture as old_fixture
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_splithead import worker as old_worker
from hlt_classification.luka_fullstat_relational_ladder import contracts as c, source, campaign, banks, kd_banks, model, training, worker, submission


def fixture(path, export=False):
    old, caches = old_fixture(path, export=export)
    imported = {t: deepcopy(old['imported_banks'][t]) for t in ('OFFLINE','M0HLT')}
    if export:
        old_worker.export_target(old,'M0HLT','kd',caches['train']['D000'],device='cpu')
        imported['M0HLT']['kd'] = file_ref(Path(old['root'])/'logit_targets/M0HLT/manifest.json')
    else:
        imported['M0HLT']['kd'] = {}
    receipt = path/'synthetic_completed_parent.json'
    write_immutable_json(receipt, {'synthetic': True})
    r = old['reuse']
    reuse = c.artifact('REUSE', parent_spec=file_ref(Path(old['root'])/'campaign_spec.json'),
        source_commit=source.DONOR, counts=dict(c.COUNTS), accepted=r,
        anchor=r['anchor'], references=r['references'], external_refs={t:r['external_refs'][t] for t in imported},
        imported_banks=imported, base_node=r['base_node'], execution_site=r['execution_site'],
        environment=r['environment'], gpu=r['gpu'], summaries=r['summaries'],
        completed_parent_receipt=file_ref(receipt),
        jobs=dict(finalize_DE033='102',train_DE000='103'), protected_roots=[old['root']],
        runtime_prefix=source.RUNTIME, parent_contract=source.PARENT_CONTRACTS[1])
    root=path/'relational'; root.mkdir()
    spec=c.artifact('SPEC',parents=dict(source='c'*64,reuse=reuse['content_hash']),
        root=str(root),project_dir='/project',source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),
        reuse=reuse,available_result_bytes=16*2**30,**campaign.policy(reuse))
    spec=with_content_hash(dict(spec,counts=old['counts']))
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec,caches


class SmallSampler:
    """Only the 22-row worker miniature; real2048 tests below are separate."""
    def __init__(self, labels): self.rows=np.arange(len(labels),dtype=np.int64)
    def sample(self, epoch, batch, global_step): return self.rows


def mock_execution(s,cs,monkeypatch):
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:os.environ['SLURM_JOB_ID'])
    monkeypatch.setattr(worker,'metadata',lambda _:(dict(content_hash=s['anchor']['evidence']['prepared'],summaries=s['reuse']['summaries']),None,None))
    monkeypatch.setattr(worker,'data',lambda s,r,co,**kw:cs[r][co])
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:{'train_'+stage:str(201+i) for i,stage in enumerate(c.STAGES)})
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))


def test_exact_registration(tmp_path):
    s,_=fixture(tmp_path)
    expected=dict(UE100=['OFFLINE','M0HLT'],DE066=['OFFLINE','UE100','M0HLT'],
        DE033=['OFFLINE','UE100','DE066','M0HLT'],DE000=['OFFLINE','UE100','DE066','DE033'])
    assert list(c.STAGES)==list(expected)
    for stage,ts in expected.items():
        r=c.recipe(stage)
        assert r['teacher_order']==ts
        assert r['weights']=={t:10/len(ts) for t in ts}
        assert r['configuration']=='k2p5_e10_rel'
        assert r['logit_weights']=={t:2.5/len(ts) for t in ts}
        assert r['logit_total']==sum(r['logit_weights'].values())==2.5
        assert r['auxiliary_total']==sum(r['weights'].values())==10.
        assert r['ce_coefficient']==.5 and r['epochs']==r['base']['maximum_passes']==20
        assert r['base']['warmup_passes']==2 and r['batch_size']==128
        assert r['relation']['anchors']==2048 and not r['relation']['temperature_squared']
        assert r['relational'] and not r['warm_start'] and not r['attention']
        assert s['nodes'][stage]['deployable']==(stage=='DE000')
    assert s['new_scientific_fits']==4 and s['preflight_jobs']==s['setup_jobs']==0
    assert c.COUNTS['train']==3000000 and c.COUNTS['validation']==500000 and c.COUNTS['final_test']==1500000
    assert s['final_test_sealed'] and not s['final_test_accessed']
    assert set(s['external_refs'])==set(s['imported_banks'])=={'OFFLINE','M0HLT'}
    assert "worker.completed(s,'DE000')" in source.REPLAY
    assert source.DONOR=='420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6'
    compile(source.REPLAY,'replay','exec')


@pytest.mark.parametrize('stage',c.STAGES)
def test_losses_and_detached_targets(stage):
    x=torch.randn(5,11,requires_grad=True); y=torch.arange(5); ts=c.teachers(stage)
    aux={t:torch.randn(5,11,requires_grad=True) for t in ts}
    p={t:torch.randn(5,128,requires_grad=True) for t in ts}
    q={t:torch.softmax(torch.randn(5,11),-1).requires_grad_(True) for t in ts}
    z={t:torch.randn(5,128,requires_grad=True) for t in ts}
    a={t:torch.randn(17,128,requires_grad=True) for t in ts}
    mask=torch.ones(5,17,dtype=torch.bool); mask[:,0]=False
    loss,ce,kd,emb,rel=training.objective(x,y,aux,p,q,z,stage=stage,beta=.7,anchors=a,valid=mask)
    expected=.5*ce+2.5/len(ts)*sum(kd.values())+10/len(ts)*sum(emb.values())+.7*rel
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(rel,sum(training.reference_kl(p[t],z[t],a[t],mask) for t in ts)/len(ts))
    main=torch.autograd.grad(.5*ce,x,retain_graph=True)[0]
    # Halve only the old K5 term, including its gradient; keep CE, embedding
    # and relation exactly unchanged for each of the four registered stages.
    old_loss=.5*ce+5/len(ts)*sum(kd.values())+10/len(ts)*sum(emb.values())+.7*rel
    old_aux=torch.autograd.grad(old_loss,tuple(aux.values()),retain_graph=True)
    old_prediction=torch.autograd.grad(old_loss,tuple(p.values()),retain_graph=True)
    torch.testing.assert_close(old_loss-loss,2.5/len(ts)*sum(kd.values()))
    loss.backward(); torch.testing.assert_close(x.grad,main,rtol=0,atol=0)
    for value,old_gradient in zip(aux.values(),old_aux):
        torch.testing.assert_close(value.grad,old_gradient/2,rtol=0,atol=0)
    for value,old_gradient in zip(p.values(),old_prediction):
        torch.testing.assert_close(value.grad,old_gradient,rtol=0,atol=0)
    assert all(v.grad is None for d in (q,z,a) for v in d.values())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for d in (aux,p) for v in d.values())
    for beta in (-1,float('nan'),2):
        with pytest.raises(ValueError): training.objective(x,y,aux,p,q,z,stage=stage,beta=beta)


def test_real_balanced_anchor_rng_and_relation_math():
    labels=np.repeat(np.arange(11,dtype=np.int64),190)
    sampler=training.AnchorSampler(labels)
    np.random.seed(5); before=np.random.get_state(); torch_before=torch.get_rng_state()
    for step in range(11):
        ids=sampler.sample(1,step,step)
        assert len(ids)==len(np.unique(ids))==2048
        counts=np.bincount(labels[ids]); expected=np.full(11,186); expected[[(2*step)%11,(2*step+1)%11]]=187
        np.testing.assert_array_equal(counts,expected)
        np.testing.assert_array_equal(ids,sampler.sample(1,step,step))
    np.testing.assert_array_equal(before[1],np.random.get_state()[1]); assert torch.equal(torch_before,torch.get_rng_state())
    with pytest.raises(ValueError): training.AnchorSampler(labels[:100])
    target=torch.randn(3,128,requires_grad=True); anchors=torch.randn(2048,128,requires_grad=True)
    pred=torch.randn(3,128,requires_grad=True); valid=torch.ones(3,2048,dtype=torch.bool); valid[:,0]=False
    loss=training.reference_kl(pred,target,anchors,valid)
    norm=torch.nn.functional.normalize
    with torch.no_grad():
        q=torch.softmax((norm(target,dim=-1)@norm(anchors,dim=-1).T/.2).masked_fill(~valid,-torch.inf),-1)
    logp=torch.log_softmax((norm(pred,dim=-1)@norm(anchors.detach(),dim=-1).T/.2).masked_fill(~valid,-torch.inf),-1)
    ref=(q*(torch.where(valid,q.clamp_min(1e-30).log(),0)-torch.where(valid,logp,0))).sum(-1).mean()
    torch.testing.assert_close(loss,ref)
    perm=torch.randperm(2048)
    torch.testing.assert_close(loss,training.reference_kl(pred,target,anchors[perm],valid[:,perm]))
    assert abs(float(training.reference_kl(target,target,anchors,valid)))<1e-6
    loss.backward(); assert pred.grad is not None and target.grad is anchors.grad is None
    with pytest.raises(ValueError): training.reference_kl(pred,target,anchors,torch.zeros_like(valid))


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed Weaver CUDA')
def test_native_parity_and_all_auxiliary_gradients(tmp_path,bf16):
    s,cs=fixture(tmp_path)
    for stage in c.STAGES:
        n=s['nodes'][stage]; torch.manual_seed(n['initialization_seed'])
        native=DelphesParticleTransformer().cuda().eval(); rng=torch.get_rng_state().clone()
        student=model.new_model(n).cuda().eval(); student.capture=True
        assert torch.equal(rng,torch.get_rng_state())
        raw=cs['train'][c.STAGES[stage]].batch(np.arange(4))
        with parity_backend('cuda'):
            out=_forward(student,raw,'cuda',bf16=bf16)
            with torch.no_grad(): other=_forward(native,raw,'cuda',bf16=bf16)
            torch.testing.assert_close(out,other,rtol=0,atol=0)
            q={t:torch.softmax(torch.randn(4,11,device='cuda'),-1) for t in c.teachers(stage)}
            z={t:torch.randn(4,128,device='cuda') for t in q}
            a={t:torch.randn(19,128,device='cuda') for t in q}
            loss,ce,kd,emb,rel=training.objective(out,torch.from_numpy(raw['labels']).cuda(),student.auxiliary_logits,
                student.predictions,q,z,stage=stage,beta=1.,anchors=a,valid=torch.ones(4,19,device='cuda',dtype=torch.bool))
            head=tuple(student.mod.fc.parameters()); expected=torch.autograd.grad(.5*ce,head,retain_graph=True)
            encoder=[p for n,p in student.named_parameters() if n.startswith('mod.') and not n.startswith('mod.fc.')]
            for term in (sum(kd.values()),sum(emb.values()),rel):
                gs=torch.autograd.grad(term,encoder,retain_graph=True,allow_unused=True)
                assert any(g is not None and g.abs().sum()>0 for g in gs)
            loss.backward()
            for p,g in zip(head,expected): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        assert not any(k.startswith(('predictors.','logit_heads.')) for k in student.native_state())


def test_external_join_and_no_old_ue100(tmp_path,monkeypatch):
    tiny(monkeypatch); s,cs=fixture(tmp_path,True)
    tr=cs['train']['U100']; bank=banks.publish(s,'UE100',tr)
    assert set(bank.targets)==set(bank.probabilities)=={'OFFLINE','M0HLT'}
    assert load_json(s['imported_banks']['M0HLT']['kd']['path'])['contract']=='LUKA_FULLSTAT_SPLITHEAD_KD_BANK/v1'
    for role in ('validation','final_test'):
        bad=deepcopy(tr); bad.role=role
        with pytest.raises(PermissionError): banks.Bundle(s,'UE100',bad)
    bad=deepcopy(tr); bad.identities=bad.identities[::-1]
    with pytest.raises(ValueError): banks.Bundle(s,'UE100',bad)
    # Any old same-named report must be rejected by the new contract validator.
    refs=dict(s['external_refs']['OFFLINE'])
    with pytest.raises(ValueError): banks.check_selected(DelphesParticleTransformer(),refs,'UE100')
    with pytest.raises(FileNotFoundError): banks.manifest_ref(s,'UE100')
    damaged=Path(s['imported_banks']['M0HLT']['kd']['path']).parent/'probabilities.npz'
    damaged.write_bytes(damaged.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): banks.Bundle(s,'UE100',tr)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real installed-Weaver GPU workers')
def test_four_full_architecture_fixed20_workers(tmp_path,monkeypatch):
    s,cs=fixture(tmp_path,True); mock_execution(s,cs,monkeypatch)
    monkeypatch.setattr(training,'AnchorSampler',SmallSampler)
    before={t:{k:file_ref(v['path']) for k,v in kinds.items()} for t,kinds in s['imported_banks'].items()}
    for i,stage in enumerate(c.STAGES):
        monkeypatch.setenv('SLURM_JOB_ID',str(201+i))
        assert worker.completed(s,stage) is None
        receipt=worker.run(s,'train_'+stage); assert receipt==worker.completed(s,stage)
        directory=Path(s['root'])/('train_'+stage)
        report=load_json(directory/'training_report.json')
        assert len(report['validation_history'])==20
        bank=load_json(Path(s['root'])/'bundles'/(stage+'.json'))
        worker.report_valid(json.loads(json.dumps(report,sort_keys=True)),s,stage,bank['content_hash'])
        assert report['validation_history'][0]['relation_beta']==.5
        assert report['validation_history'][1]['relation_beta']==1.
        for key in ('weighted_relation_loss','total_loss','learning_rate'):
            bad=deepcopy(report); bad['validation_history'][0][key]+=.1
            with pytest.raises(ValueError): worker.report_valid(with_content_hash(bad),s,stage,bank['content_hash'])
        state=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
        native=DelphesParticleTransformer().cuda(); native.load_state_dict(state,strict=True)
        if stage!='DE000':
            a=load_npz_arrays(Path(s['root'])/'logit_targets'/stage/'probabilities.npz')
            np.testing.assert_array_equal(a['probabilities'],predict(native,cs['train'][c.STAGES[stage]],device='cuda',temperature=2.,batch_size=128))
        del native
        summary=worker.results(s); assert summary['pending']==list(c.STAGES)[i+1:]
        with pytest.raises(FileExistsError): worker.run(s,'train_'+stage)
    assert not (Path(s['root'])/'targets/DE000').exists()
    assert (Path(s['root'])/'targets/UE100/manifest.json').is_file()
    assert before=={t:{k:file_ref(v['path']) for k,v in kinds.items()} for t,kinds in s['imported_banks'].items()}
    assert summary['rows'][-1]['input']=='native_HLT'
    bad=Path(s['root'])/'targets/UE100/targets.npz'; bad.write_bytes(bad.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): worker.completed(s,'DE000')


def test_four_job_review_authorization_and_duplicate_guard(tmp_path,monkeypatch):
    s,_=fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s); assert plan['jobs']==plan['scientific_fits']==4
    assert plan['preflight_jobs']==plan['setup_jobs']==0
    with pytest.raises(PermissionError): submission.submit(s,execute=True,plan_hash=plan['content_hash'])
    calls=[]
    def fake(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout=str(200+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)=={'train_'+stage:str(201+i) for i,stage in enumerate(c.STAGES)}
    assert '--dependency=afterok:201:202:203' in calls[3]
    assert all('--no-requeue' in a and '--kill-on-invalid-dep=yes' in a for a in calls)
    with pytest.raises(FileExistsError): submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==4


def test_timing_and_resource_fail_closed(tmp_path):
    s,_=fixture(tmp_path)
    good=worker.timing_record(s,'UE100',.1,.01); worker.validate_timing(s,'UE100',good)
    for value in (0,float('nan'),1000000):
        with pytest.raises(ValueError): worker.timing_record(s,'UE100',value,.1)
    bad=dict(good,conservative_minutes=1)
    with pytest.raises(ValueError): worker.validate_timing(s,'UE100',bad)
    with pytest.raises(MemoryError): worker.limits(s,dict(peak_rss_bytes=180000*1024**2,gpu_peak_bytes=1))


def test_source_replay_guards(tmp_path,monkeypatch):
    original=tmp_path/'old'; project=tmp_path/'new'; root=tmp_path/'parent'; root.mkdir()
    for d in (original,project):
        (d/'src').mkdir(parents=True); (d/'src/kernel.py').write_text('# native\n')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),source_commit=source.DONOR,
        counts=c.COUNTS,contract=source.PARENT_CONTRACTS[1],runtime_prefix=source.RUNTIME,reuse=dict(environment={})))
    monkeypatch.setattr(source,'runtime',lambda _:None)
    def replay(args,**kw):
        if args[0]=='git': return SimpleNamespace(stdout='src/kernel.py\n')
        assert kw['cwd']==original and kw['env']['PYTHONPATH']==str(original/'src') and args[-2]==source.REPLAY
        return SimpleNamespace(returncode=0,stdout='RELATIONAL_REUSE={}\n',stderr='')
    monkeypatch.setattr(subprocess,'run',replay)
    c.validate(source.authenticate(root/'campaign_spec.json',project),'REUSE')
    (project/'src/kernel.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='Changed donor'): source.authenticate(root/'campaign_spec.json',project)
    assert source.compatible_changes(source.ADDITIONS,source.ADDITIONS)
    assert not source.compatible_changes({'src/old.py'},set())


def test_no_fit_without_completed_ancestor(tmp_path,monkeypatch):
    s,_=fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'202')
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:{'train_DE066':'202'})
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('Do not fit'))
    with pytest.raises(ValueError,match='incomplete'): worker.run(s,'train_DE066')
    assert not (Path(s['root'])/'train_DE066_attempt').exists()


def test_shell_cli_and_inventory():
    project=Path(__file__).resolve().parents[1]; bash='C:/Program Files/Git/bin/bash.exe'
    for p in source.ENTRYPOINTS:
        if p.endswith('.sh'): subprocess.run([bash,'-n',p],cwd=project,check=True,capture_output=True)
    assert c.AUTHORIZATION in (project/'scripts/queue_luka_fullstat_relational_ladder.sh').read_text()
    subprocess.run([sys.executable,'-B','scripts/luka_fullstat_relational_ladder.py','--help'],cwd=project,check=True,capture_output=True)
    for p in source.REQUIRED:
        text=(project/p).read_text()
        assert all(line==line.rstrip() for line in text.splitlines())
        if p.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text): assert (project/p).parent.joinpath(link).is_file()
    book=(project/'docs/LUKA_FULLSTAT_RELATIONAL_LADDER_RUNBOOK.md').read_text()
    for block in re.findall(r'^```bash\n(.*?)^```',book,re.M|re.S):
        subprocess.run([bash,'-n'],input=block,check=True,text=True,capture_output=True)


@pytest.mark.parametrize('stage',c.STAGES)
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual Weaver physical128 and2048 references')
def test_real_physical128_integrated_checks_all_stages(tmp_path,monkeypatch,stage):
    from test_luka_attention import cache as sized_cache
    s,_=fixture(tmp_path)
    co=c.STAGES[stage]; tr=sized_cache('train',co,rows=2068); val=sized_cache('validation',co)
    rng=np.random.default_rng(17)
    manifest=c.artifact('BUNDLE',prepared=tr.foundation_sha256,stage=stage,role='train',
        rows=len(tr),teacher_order=c.teachers(stage),temperature=2.)
    bank=SimpleNamespace(labels=tr.labels,identities=tr.identities,manifest=manifest,
        probabilities={t:np.full((len(tr),11),1/11,np.float32) for t in c.teachers(stage)},
        targets={t:rng.normal(size=(len(tr),128)).astype(np.float32) for t in c.teachers(stage)})
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    record=worker.startup(s,stage,tr,val,bank,device='cuda')
    worker.validate_startup(s,stage,record,manifest['content_hash'])
    assert record['relation_anchors']==2048 and record['relation_at_full_weight']
    assert record['timing']['benchmark_steps']==32 and record['timing']['validation_steps']==16
    assert not record['timing']['full_epoch_measured']


def test_poor_finite_science_and_fresh_rng_reset(tmp_path,monkeypatch):
    tiny(monkeypatch); s,cs=fixture(tmp_path,True)
    monkeypatch.setattr(training,'AnchorSampler',SmallSampler)
    tr,val=cs['train']['U100'],cs['validation']['U100']; bank=banks.publish(s,'UE100',tr)
    monkeypatch.setattr(training,'predict',lambda *a,**kw:np.full((len(val),11),1/11,np.float32))
    torch.manual_seed(777); report,state=training.train(tr,val,bank=bank,node=s['nodes']['UE100'],device='cpu')
    worker.report_valid(report,s,'UE100',bank.manifest['content_hash'])
    assert report['selected_pass']==1 and all(x['validation']['macro_ovr_auc']==.5 for x in report['validation_history'])
    torch.manual_seed(888); again,other=training.train(tr,val,bank=bank,node=s['nodes']['UE100'],device='cpu')
    assert all(torch.equal(v,other[k]) for k,v in state.items())


@pytest.mark.parametrize('change',[dict(counts={}),dict(external_refs={}),dict(jobs={}),dict(imported_banks={})])
def test_wrong_reuse_fails_closed(tmp_path,change):
    s,_=fixture(tmp_path)
    with pytest.raises((KeyError,ValueError)): campaign.policy(with_content_hash(dict(s['reuse'],**change)))


@pytest.mark.parametrize('out,code',[('',1),('',0),('RELATIONAL_REUSE={}\nRELATIONAL_REUSE={}\n',0)])
def test_failed_or_ambiguous_original_replay_refused(tmp_path,monkeypatch,out,code):
    original=tmp_path/'old'; project=tmp_path/'new'; root=tmp_path/'parent'; root.mkdir()
    for d in (original,project):
        (d/'src').mkdir(parents=True); (d/'src/kernel.py').write_text('')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),source_commit=source.DONOR,
        counts=c.COUNTS,contract=source.PARENT_CONTRACTS[1],runtime_prefix=source.RUNTIME,reuse=dict(environment={})))
    monkeypatch.setattr(source,'runtime',lambda _:None)
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='src/kernel.py\n')
        if args[0]=='git' else SimpleNamespace(returncode=code,stdout=out,stderr='rejected'))
    with pytest.raises(ValueError): source.authenticate(root/'campaign_spec.json',project)


def test_exact_success_accounting(tmp_path,monkeypatch):
    s,_=fixture(tmp_path)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:{'train_UE100':'201'})
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'rlyang')
    valid='201|lukafsrl_train_UE100|rlyang|default|COMPLETED|0:0\n'
    for value in ('',valid.replace('COMPLETED','FAILED'),valid.replace('0:0','1:0'),valid.replace('rlyang','other')):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=value))
        with pytest.raises(ValueError): submission.successful(s,'train_UE100','201')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=valid))
    submission.successful(s,'train_UE100','201')


def test_ambiguous_submission_never_retried(tmp_path,monkeypatch):
    s,_=fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s); calls=[]
    def fake(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout='ambiguous\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    with pytest.raises(Exception): submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


def test_current_source_guard_and_unchanged_runtime(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    changed=subprocess.run(['git','diff','--name-only','--diff-filter=MDR',source.DONOR,source.BASELINE,
        '--','src','scripts','sbatch'],cwd=project,check=True,capture_output=True,text=True).stdout
    assert not changed
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in args else '\n'.join(source.ADDITIONS|source.ENTRYPOINTS) if 'diff' in args else ''))
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in args else 'src/old.py' if 'diff' in args else ''))
    with pytest.raises(ValueError): source.current(project,'b'*40)


def test_actual_replay_glue_waits_for_old_final_and_exports_only_original_teachers(tmp_path,monkeypatch,capsys):
    from hlt_classification.luka_fullstat_splithead import campaign as original_campaign, submission as original_submission
    tiny(monkeypatch); original,cs=old_fixture(tmp_path,True)
    old_worker.export_target(original,'M0HLT','kd',cs['train']['D000'],device='cpu')
    original['imported_banks']=deepcopy(original['imported_banks'])
    original['imported_banks']['M0HLT']['kd']=file_ref(Path(original['root'])/'logit_targets/M0HLT/manifest.json')
    oldpath=tmp_path/'previous_recovery.json'
    write_immutable_json(oldpath,dict(root=str(tmp_path/'old'),project_dir='/originalsource',
        recovery=dict(parent_spec=original['reuse']['parent_spec'])))
    original['finalization']=dict(parent_spec=file_ref(oldpath))
    original['runtime_prefix']=source.RUNTIME
    original['contract']=source.PARENT_CONTRACTS[1]
    original['retry']=dict(previous_root='/failedroot',previous_project='/failedsource')
    parent=tmp_path/'recovery_spec_fixture.json'; write_immutable_json(parent,original)
    monkeypatch.setattr(sys,'argv',['replay',str(parent)])
    monkeypatch.setattr(original_campaign,'check',lambda s:None)
    monkeypatch.setattr(old_worker,'completed',lambda s,stage:None)
    with pytest.raises(ValueError,match='Wait for'): exec(compile(source.REPLAY,'replay','exec'),{})
    for stage in ('DE033','DE000'):
        root=Path(original['root'])/('train_'+stage); root.mkdir()
        for name in ('selected.pt','training_report.json','validation_predictions.npz','task_receipt.json'):
            (root/name).write_bytes(b'synthetic metadata-only replay witness')
    monkeypatch.setattr(old_worker,'completed',lambda s,stage:dict(content_hash='c'*64))
    monkeypatch.setattr(original_submission,'accepted_jobs',lambda s:dict(finalize_DE033='102',train_DE000='103'))
    exec(compile(source.REPLAY,'replay','exec'),{})
    lines=[x.partition('=')[2] for x in capsys.readouterr().out.splitlines() if x.startswith('RELATIONAL_REUSE=')]
    assert len(lines)==1
    evidence=json.loads(lines[0]); reuse=c.artifact('REUSE',counts=c.COUNTS,**evidence)
    policy=campaign.policy(reuse)
    assert set(policy['imported_banks'])==set(policy['external_refs'])=={'OFFLINE','M0HLT'}
    assert 'OLD_SPLITHEAD_DE000' in policy['references']
    assert '/failedroot' in evidence['protected_roots']
    assert policy['new_scientific_fits']==4 and policy['nodes']['UE100']['teacher'] is None
