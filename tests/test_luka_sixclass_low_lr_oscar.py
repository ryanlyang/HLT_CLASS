"""Owned low-LR/CE-only continuation, provenance and execution tests."""
from copy import deepcopy


from contextlib import nullcontext


from pathlib import Path


from types import SimpleNamespace


import json


import subprocess


import numpy as np


import pytest


import torch


import uproot


from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash, sha256_file


from hlt_classification.luka_sixclass_low_lr_oscar import model


from hlt_classification.luka_sixclass_balanced_oscar import contrastive


from hlt_classification.luka_sixclass_balanced_oscar.sampling import BalancedSampler


from hlt_classification.luka_sixclass_low_lr_oscar import (
    contracts as c,campaign,training,banks,artifacts,worker,startup,reporting,inspect,reuse)
from hlt_classification.luka_sixclass_balanced_oscar import population,views,contracts as historical


from test_luka_fullstat_balanced_ladder import threads


def spec_fixture(root,monkeypatch=None):
    pop=historical.artifact('POPULATION',parents=dict(foundation=c.PARENT_HASH,role_population=c.ROLE_POPULATION_HASH),
        policy=c.population_policy(),counts=dict(c.COUNTS,unused=c.UNUSED),outside_counts=c.OUTSIDE)
    if monkeypatch is not None:monkeypatch.setattr(c,'POPULATION_HASH',pop['content_hash'])
    return c.artifact('SPEC',root=str(root),dataset='/dataset',project='/project',commit='f'*40,source={},
        environment_dir='/env',environment={},gpu='l40s',foundation_producer={},population=pop,reuse={},
        population_root='/historical/population',recipe=c.recipe(),nodes={s:c.node(s) for s in c.STAGES},
        counts=c.COUNTS,cpus=6,memory_mb=192000,minutes={t:2880 for t in c.tasks()},
        preparation_workers=4,candidate_workers=3,tasks=c.tasks(),
        storage=dict(declared_available_gib=8,new_results_budget_bytes=c.RESULT_GIB*2**30),
        execution_site=c.SITE,transfer_manifest_sha256=c.TRANSFER_HASH,backend=campaign.backend())


def test_sequential_dag_and_no_duplicate_submission(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path);monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(campaign,'storage',lambda *a,**k:{})
    monkeypatch.setattr(campaign,'site',lambda *a:None)
    calls=[]
    def fake(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(100+len(calls)),stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    plan=campaign.queue(s);assert plan['scientific_fits']==3 and plan['preflight_jobs']==0 and not calls
    with pytest.raises(PermissionError):campaign.queue(s,execute=True)
    campaign.queue(s,execute=True,authorization=c.AUTH)
    assert len(calls)==3
    assert not any(x.startswith('--dependency') for x in calls[0])
    assert '--dependency=afterok:101' in calls[1] and '--dependency=afterok:102' in calls[2]
    assert [row[-1] for row in calls]==['train_OFFLINE','train_DE000','train_M0HLT']
    with pytest.raises(FileExistsError):campaign.queue(s,execute=True,authorization=c.AUTH)
    assert len(calls)==3


def test_recipe_is_new_but_preserves_population_architecture_and_seeds():
    assert c.recipe()['content_hash']!=historical.recipe()['content_hash']
    assert c.POPULATION_HASH==historical.POPULATION_HASH and c.DRAWS==1658937 and c.STEPS==12961
    assert c.TEACHERS['DE000']==('OFFLINE','CONCAT','UE100','DE050')
    assert c.learning_rate(1)==5e-5 and c.learning_rate(2)==1e-4 and c.learning_rate(20)==5e-6
    for pos in (.001,1,2,5,10,19,20):
        from hlt_classification.luka_fullstat.recipe import learning_rate
        assert c.learning_rate(pos)==pytest.approx(learning_rate(pos)/3)
    for stage in c.STAGES:
        assert c.main_policy(stage)['ce_weight']==1 and c.main_policy(stage)['teacher'] is None
        assert c.node(stage)['initialization_seed']==historical.node(stage)['initialization_seed']
    with pytest.raises(ValueError):c.validate(historical.recipe(),'RECIPE')


@pytest.mark.parametrize('stage',c.STAGES)
def test_native_head_receives_exact_unscaled_ce_auxiliary_heads_still_learn(stage,monkeypatch):
    from torch.nn import functional as F
    tiny(monkeypatch);tr=cache(coordinate=c.COORDINATES[stage]);ix=np.array([0,1,1,3,5]);rng=np.random.default_rng(12)
    targets={t:SimpleNamespace(targets=rng.normal(size=(12,128)).astype(np.float32),
        probabilities=rng.dirichlet(np.ones(6),size=12).astype(np.float32)) for t in c.TEACHERS[stage]}
    before=deepcopy(targets)
    keys=F.normalize(torch.randn(12,128),dim=-1)
    match=SimpleNamespace(candidate_keys=lambda i:keys[i]) if stage=='DE000' else None
    candidates=np.tile(np.arange(1025)%12,(len(ix),1));candidates[:,0]=ix
    net=model.new_model(stage).eval()
    loss,parts=model.objective(net,tr.batch(ix),ix,targets,update=6,steps=2,device='cpu',sampler=match,candidates=candidates)
    assert 'main_kd' not in parts
    expected=parts['ce']
    for t in targets:expected=expected+2.5*parts['embedding_'+t]+.25*parts['kd_'+t]
    if match:expected=expected+.3*parts['matching']
    torch.testing.assert_close(loss,expected)
    ce_grads=torch.autograd.grad(parts['ce'],tuple(net.mod.fc.parameters()),retain_graph=True)
    loss.backward();training.finite_gradients(net)
    for p,g in zip(net.mod.fc.parameters(),ce_grads):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    for t in targets:
        assert any(p.grad.abs().sum()>0 for p in net.predictors[t].parameters())
        assert any(p.grad.abs().sum()>0 for p in net.logit_heads[t].parameters())
        assert np.array_equal(targets[t].targets,before[t].targets)
    native=net.deployed();net.capture=False
    assert not any(k.startswith(('predictors.','logit_heads.','matching.')) for k in native.state_dict())
    with torch.no_grad():torch.testing.assert_close(model.forward(net,tr.batch(ix),'cpu',bf16=False),model.forward(native,tr.batch(ix),'cpu',bf16=False),rtol=0,atol=0)


def test_full_hbb_is_pooled_not_average_and_ties_are_included():
    y=np.array([0,0,1,1,0,0,1,1]);score=np.array([.1,.2,.3,.4,.6,.7,.8,.9])
    p=np.zeros((8,6));p[:,1]=score;p[:,0]=1-score
    allrows=reporting.binary_hbb(y,p)
    assert allrows['auc']==.75
    assert (reporting.binary_hbb(y[:4],p[:4])['auc']+reporting.binary_hbb(y[4:],p[4:])['auc'])/2==1.
    reporting.validate_hbb(allrows,8)
    p[:]=1/6
    tied=reporting.binary_hbb(y,p)
    assert tied['auc']==.5 and tied['R30']['background_passing']==4 and tied['R50']['rejection']==1.
    assert reporting.hbb_selected([dict(pass_number=2,full_hbb=tied),dict(pass_number=1,full_hbb=tied)])['pass_number']==1


def test_roles_cannot_overlap_or_include_final_test():
    val=cache(role='validation');out=cache(role='validation')
    with pytest.raises(ValueError,match='overlap'):reporting.validate_roles(val,out)
    out.identities[:,20]=1;reporting.validate_roles(val,out)
    out.role='final_test'
    with pytest.raises(PermissionError):reporting.validate_roles(val,out)


def test_new_offline_required_and_no_historical_offline_fallback(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path)
    with pytest.raises(PermissionError):reuse.completed(s,'OFFLINE')
    with pytest.raises(FileNotFoundError):artifacts.completed(s,'OFFLINE')
    monkeypatch.setattr(reuse,'completed',lambda spec,stage:dict(receipt={'content_hash':stage}))
    assert artifacts.completed(s,'DE050')['receipt']['content_hash']=='DE050'


def test_timing_accounts_for_outside_validation_each_epoch():
    import math
    r=startup.estimate('DE000',1.,2.,0.,0.)
    batches=math.ceil(c.COUNTS['validation']/128)+math.ceil(c.OUTSIDE['validation']/128)
    assert r['conservative_minutes']==math.ceil((3600+1.3*(20*c.STEPS+21*batches*2))/60)
    with pytest.raises(ValueError):startup.estimate('M0HLT',1,1,1,0)
    with pytest.raises(ValueError):startup.estimate('OFFLINE',1,1,0,0)


def test_offline_worker_records_full_hbb_each_epoch_and_exports_only_train(tmp_path,monkeypatch):
    s,tr,val,_,_=worker_fixture(tmp_path,monkeypatch)
    receipt=worker.run(s,'train_OFFLINE',device='cpu');r=artifacts.completed(s,'OFFLINE')
    history=r['report']['validation_history']
    assert len(history)==20 and all(row['full_hbb']['rows']==24 for row in history)
    assert all('main_kd' not in row['losses'] for row in history)
    assert banks.Bank(r['folder']/'bank',tr,'OFFLINE',r['refs'],s).targets.shape==(120,128)
    metric=load_json(receipt['files']['metrics']['path'])
    assert metric['full_hbb']==history[r['report']['selected_pass']-1]['full_hbb']
    with pytest.raises(PermissionError):banks.Bank(r['folder']/'bank',tr,'M0HLT',r['refs'],s)


def test_three_workers_use_new_offline_bank_and_publish_hbb_recoveries(tmp_path,monkeypatch):
    s,tr,val,sampler,_=worker_fixture(tmp_path,monkeypatch)
    rng=np.random.default_rng(52);frozen={}
    for t in c.EXTERNAL:
        frozen[t]=SimpleNamespace(identities=tr.identities,labels=tr.labels,
            targets=rng.normal(size=(len(tr),128)).astype(np.float32),
            probabilities=rng.dirichlet(np.ones(6),size=len(tr)).astype(np.float32))
    def external(spec,stage):
        assert stage in c.EXTERNAL
        return dict(spec={'original':True},folder=Path('/original')/stage,refs={},
                    receipt=dict(content_hash=stage))
    monkeypatch.setattr(reuse,'completed',external)
    def load_external(folder,cache,stage,refs,spec):
        assert spec=={'original':True} and stage in c.EXTERNAL
        return frozen[stage]
    monkeypatch.setattr(worker.historical_banks,'Bank',load_external)
    received=[];original_sampler=contrastive.Sampler
    def matching(keys,ids,labels,**kw):
        new=artifacts.completed(s,'OFFLINE')
        expected=banks.Bank(new['folder']/'bank',tr,'OFFLINE',new['refs'],s)
        assert np.array_equal(keys,expected.keys());received.append(new['receipt']['content_hash'])
        return original_sampler(keys,ids,labels,**kw)
    monkeypatch.setattr(contrastive,'Sampler',matching)
    for stage in c.STAGES:
        r=worker.run(s,'train_'+stage,device='cpu')
        assert ('bank' in r['files'])==(stage=='OFFLINE')
        assert artifacts.completed(s,stage)['report']['passes']==20
    assert len(received)==1
    result=inspect.results(s)
    assert set(result['models'])==set(c.STAGES)
    assert set(result['exploratory_full_hbb_selected'])==set(c.STAGES)
    assert 'full_validation' in result['models']['DE000']['metrics']
    de=artifacts.completed(s,'DE000')
    assert de['receipt']['fit_parents']['teacher_OFFLINE']==received[0]
    assert de['receipt']['fit_parents']['teacher_DE050']=='DE050'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA integrated qualification')
@pytest.mark.parametrize('stage',c.STAGES)
def test_real_weaver_integrated_qualification(stage,tmp_path,monkeypatch):
    import time
    n=5400;tr=cache(coordinate=c.COORDINATES[stage],rows=n)
    val=cache(role='validation',coordinate=c.COORDINATES[stage],rows=12)
    outside=deepcopy(val);outside.identities[:,20]=1
    sampler=BalancedSampler(tr.labels,650+100*((np.arange(n)//6)%20))
    rng=np.random.default_rng(71)
    targets={t:SimpleNamespace(targets=rng.normal(size=(n,128)).astype(np.float32),
        probabilities=np.full((n,6),1/6,np.float32)) for t in c.TEACHERS[stage]}
    matching=None
    if stage=='DE000':
        keys=targets['OFFLINE'].targets.copy();keys/=np.linalg.norm(keys,axis=1,keepdims=True)
        matching=contrastive.Sampler(keys,tr.identities,tr.labels,device='cuda')
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    # Actual local GPU model/gradients/parity/candidates/reload. Linux host RSS
    # and full dataset throughput remain for Oscar's integrated acceptance.
    monkeypatch.setattr(startup,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    s=spec_fixture(tmp_path)
    proof=startup.run(s,stage,tr,val,targets,sampler,matching,'cuda',tmp_path,time.monotonic(),remaining_epochs=0,outside=outside)
    assert proof['classifier_objective_verified'] and proof['installed_weaver_parity']
    assert proof['integrated_two_epoch_miniature'] and proof['probe_discarded']
    assert set(proof['timing']['validation_step_seconds_by_role'])=={'in_range','outside_range'}
    assert proof['main_head']['ce_weight']==1 and proof['main_head']['kd_weight']==0


@pytest.mark.parametrize('field,value',[('counts',{}),('tasks',{}),('recipe',{}),('gpu','a40'),('memory_mb',1),('final_test_accessed',True)])
def test_spec_rejects_rehashed_scientific_changes(tmp_path,monkeypatch,field,value):
    s=spec_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(reuse,'check',lambda *a,**k:dict(population_root=s['population_root'],population=s['population'],environment=s['environment'],backend=s['backend']))
    campaign.check(s,live=False)
    changed=deepcopy(s);changed[field]=value
    with pytest.raises(ValueError):campaign.check(with_content_hash(changed),live=False)


def test_resume_keeps_published_offline_and_preserves_execution_order(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path)
    jobs=[dict(task='train_'+stage,job=str(100+i)) for i,stage in enumerate(c.STAGES)]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda s:jobs)
    monkeypatch.setattr(campaign,'terminal',lambda job:[job,'COMPLETED','0:0'])
    folder=tmp_path/'train_OFFLINE';folder.mkdir();(folder/'result.json').write_text('{}')
    monkeypatch.setattr(artifacts,'completed',lambda s,stage:dict(receipt=dict(job_id='100',content_hash='old_offline')))
    tasks,done=campaign.remaining(s,True)
    assert tasks==['train_DE000','train_M0HLT'] and done=={'OFFLINE':'old_offline'}
    rows=campaign.commands(s,tasks)
    assert rows[0]['dependencies']==[] and rows[1]['dependencies']==['train_DE000']
    def live(job):raise ValueError('not terminal')
    monkeypatch.setattr(campaign,'terminal',live)
    with pytest.raises(ValueError):campaign.remaining(s,True)


def test_reuse_authenticates_in_original_source_and_rejects_changed_receipt(tmp_path,monkeypatch):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from test_luka_sixclass_balanced_oscar import spec_fixture as old_fixture
    old=old_fixture(tmp_path/'parent',monkeypatch)
    path=Path(old['root'])/'campaign_spec.json';write_immutable_json(path,old)
    monkeypatch.setattr(c,'PARENT_SPEC',str(path));monkeypatch.setattr(c,'PARENT_SPEC_HASH',old['content_hash'])
    monkeypatch.setattr(c,'PARENT_COMMIT',old['commit'])
    teachers={}
    for stage in c.EXTERNAL:
        result=historical.artifact('RESULT',parents=dict(spec=old['content_hash']),stage=stage)
        target=Path(old['root'])/('train_'+stage)/'result.json';write_immutable_json(target,result)
        teachers[stage]=dict(result=file_ref(target),receipt_sha256=result['content_hash'],bank_sha256='a'*64)
    def replay(args,**kw):
        assert args[0]==str(Path(old['environment_dir'])/'bin/python')
        assert kw['cwd']==old['project'] and kw['env']['PYTHONPATH']==str(Path(old['project'])/'src')
        assert args[-1]==str(path) and kw['env']['PYTHONNOUSERSITE']=='1'
        return SimpleNamespace(stdout='LOW_LR_REUSE='+json.dumps(teachers)+'\n')
    monkeypatch.setattr(subprocess,'run',replay)
    result=reuse.authenticate(path)
    assert reuse.check(result,payloads=True)==old
    changed=deepcopy(result);changed['teachers']['OFFLINE']=changed['teachers']['CONCAT']
    with pytest.raises(ValueError,match='inventory'):reuse.check(with_content_hash(changed))
    target.write_bytes(b'changed')
    with pytest.raises(ValueError):reuse.check(result)


def test_original_source_replay_authenticates_bank_parent_and_payload(tmp_path,monkeypatch,capsys):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.luka_sixclass_balanced_oscar import campaign as oc,artifacts as oa,banks as ob
    old=dict(root=str(tmp_path),content_hash='s'*64,population={'content_hash':'p'*64})
    path=tmp_path/'campaign_spec.json';write_immutable_json(path,old)
    monkeypatch.setattr(oc,'check',lambda s:None)
    def complete(s,stage,memo):
        folder=tmp_path/('train_'+stage);folder.mkdir(exist_ok=True)
        payload=folder/'targets.npy'
        if not payload.exists():payload.write_bytes(b'payload')
        refs={'checkpoint':{'sha256':'c'*64},'report':{'sha256':'r'*64}}
        bank=historical.artifact('BANK',parents=ob.parents(s,stage,refs),stage=stage,role='train',rows=historical.DRAWS,
            payloads=dict(targets=file_ref(payload,root=folder)))
        manifest=folder/'manifest.json'
        if not manifest.exists():write_immutable_json(manifest,bank)
        result=historical.artifact('RESULT',files={'bank':file_ref(manifest)})
        result_path=folder/'result.json'
        if not result_path.exists():write_immutable_json(result_path,result)
        return dict(receipt=result,refs=refs)
    monkeypatch.setattr(oa,'completed',complete)
    monkeypatch.setattr(__import__('sys'),'argv',['replay',str(path)])
    exec(compile(reuse.REPLAY,'historical_replay','exec'),{})
    line=capsys.readouterr().out.strip();assert set(json.loads(line.removeprefix('LOW_LR_REUSE=')))==set(c.EXTERNAL)
    (tmp_path/'train_DE050/targets.npy').write_bytes(b'bad')
    with pytest.raises(ValueError):exec(compile(reuse.REPLAY,'historical_replay','exec'),{})


def test_de000_epoch_continuation_preserves_full_hbb_and_all_weights(tmp_path,monkeypatch):
    s,tr,val,sampler,_=worker_fixture(tmp_path,monkeypatch)
    tr.coordinate_name=val.coordinate_name='D000';outside=deepcopy(val);outside.identities[:,20]=2
    rng=np.random.default_rng(79)
    targets={t:SimpleNamespace(identities=tr.identities,labels=tr.labels,
        targets=rng.normal(size=(len(tr),128)).astype(np.float32),
        probabilities=rng.dirichlet(np.ones(6),size=len(tr)).astype(np.float32)) for t in c.TEACHERS['DE000']}
    match=contrastive.Sampler(targets['OFFLINE'].targets,tr.identities,tr.labels)
    saved=[]
    expected,r,_=training.train(tr,val,banks=targets,sampler=sampler,stage='DE000',parents={},device='cpu',
        save_epoch=lambda e,row,state:saved.append((deepcopy(row),deepcopy(state))),matching=match,outside=outside)
    from hlt_classification.jetclass2_delphes.runner import selection_key
    best=max(saved[:10],key=lambda pair:selection_key(pair[0]['validation'],pair[0]['update']))[1]
    resumed=[]
    actual,r2,_=training.train(tr,val,banks=targets,sampler=sampler,stage='DE000',parents={},device='cpu',
        save_epoch=lambda e,*a:resumed.append(e),matching=match,outside=outside,
        resume=dict(state=saved[9][1],history=[x[0] for x in saved[:10]],best=best))
    assert resumed==list(range(11,21))
    assert r['full_hbb_selected']==r2['full_hbb_selected'] and r['selected_pass']==r2['selected_pass']
    assert [x['full_hbb'] for x in r['validation_history']]==[x['full_hbb'] for x in r2['validation_history']]
    for key,value in expected.state_dict().items():torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)


def test_hbb_diagnostic_selection_cannot_override_primary_selection(tmp_path,monkeypatch):
    s,_,_,_,_=worker_fixture(tmp_path,monkeypatch)
    receipt=worker.run(s,'train_OFFLINE',device='cpu');r=load_json(receipt['files']['report']['path'])
    primary=r['selected_pass'];other=20 if primary!=20 else 19
    for row in r['validation_history']:row['full_hbb']['auc']=1. if row['pass_number']==other else .5
    r['full_hbb_selected_pass']=other;r['full_hbb_selected']=r['validation_history'][other-1]['full_hbb']
    training.validate_report(with_content_hash(r),'OFFLINE',receipt['fit_parents'],12)
    assert r['selected_pass']==primary
    r['selected_pass']=other;r['validation']=r['validation_history'][other-1]['validation']
    with pytest.raises(ValueError,match='Selected checkpoint'):training.validate_report(with_content_hash(r),'OFFLINE',receipt['fit_parents'],12)


def test_ambiguous_submission_is_not_retried(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path);monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(campaign,'storage',lambda *a,**kw:{});monkeypatch.setattr(campaign,'site',lambda *a:None)
    calls=[]
    def fake(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout='ambiguous',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    with pytest.raises(RuntimeError):campaign.queue(s,execute=True,authorization=c.AUTH)
    with pytest.raises(ValueError,match='Ambiguous'):campaign.queue(s,execute=True,authorization=c.AUTH,resume=True)
    assert len(calls)==1


def worker_fixture(tmp_path,monkeypatch):
    tiny(monkeypatch);monkeypatch.setattr(c,'DRAWS',24)
    monkeypatch.setattr(c,'COUNTS',dict(train=120,validation=12,final_test=6))
    tr=cache(coordinate='OFFLINE',rows=120);val=cache(role='validation',coordinate='OFFLINE',rows=12)
    val.identities[:,20]=1;outside=deepcopy(val);outside.identities[:,20]=2
    monkeypatch.setattr(c,'OUTSIDE',dict(c.OUTSIDE,validation=12))
    sampler=BalancedSampler(tr.labels,650+100*(np.arange(len(tr))//6))
    s=spec_fixture(tmp_path);write_immutable_json(tmp_path/'campaign_spec.json',s)
    monkeypatch.setattr(reuse,'check',lambda *a,**kw:None)
    monkeypatch.setattr(campaign,'check',lambda s,**kw:s);monkeypatch.setattr(campaign,'source',lambda *a:{})
    monkeypatch.setattr(campaign,'storage',lambda *a,**kw:{})
    accepted=[dict(task='train_'+stage,job=str(100+i)) for i,stage in enumerate(c.STAGES)]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**kw:accepted)
    monkeypatch.setattr(worker,'allocation',lambda s,t:next(r['job'] for r in reversed(accepted) if r['task']==t))
    monkeypatch.setattr(worker.lineage,'verify',lambda *a:{})
    a=dict(validation_pt=np.full(12,650.),outside_validation_pt=np.full(12,2800.))
    monkeypatch.setattr(population,'load',lambda *a,**kw:(s['population'],None,None,None,dict(validation_pt=np.full(12,650.),outside_validation_pt=np.full(12,2800.)),sampler))
    def inputs(s,stage,*args,**kwargs):
        caches=dict(train=deepcopy(tr),validation=deepcopy(val),outside_validation=deepcopy(outside))
        for x in caches.values():x.coordinate_name=c.COORDINATES[stage]
        return caches,c.artifact('VIEW_BUILD',parents=dict(spec=s['content_hash']),stage=stage,mapping=[])
    monkeypatch.setattr(views,'load',inputs)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=1024,rss_bytes=1024,gpu_peak_bytes=0))
    monkeypatch.setattr(startup,'run',lambda s,stage,*args,**kw:c.artifact('STARTUP',parents=dict(spec=s['content_hash']),stage=stage,
        installed_weaver_parity=True,finite_connected_gradients=True,classifier_objective_verified=True,main_head=c.main_policy(stage),integrated_two_epoch_miniature=True,
        native_deployment_parity=True,full_support=True,probe_discarded=True,matching_serial_parity=stage=='DE000'))
    class Match:
        def __init__(self,keys,identities,labels,**kw):self.keys=torch.from_numpy(keys);self.identities=identities;self.labels=labels
        def candidate_keys(self,ix):return self.keys[ix]
    def lookahead(match,order,epoch,*,batch_size):
        def rows():
            for _,ix in sampler.batches(order,batch_size=batch_size):
                candidates=np.tile(np.arange(1025)%len(tr),(len(ix),1));candidates[:,0]=ix
                yield ix,candidates
        return nullcontext(rows())
    monkeypatch.setattr(contrastive,'Sampler',Match)
    monkeypatch.setattr(contrastive,'CPUWorkers',lambda *a,**kw:nullcontext())
    monkeypatch.setattr(contrastive,'Lookahead',lookahead)
    return s,tr,val,sampler,accepted


def test_failed_export_recovery_performs_no_new_updates(tmp_path,monkeypatch):
    s,tr,val,sampler,accepted=worker_fixture(tmp_path,monkeypatch);original=banks.export
    def failure(*a,**kw):raise RuntimeError('injected export failure')
    monkeypatch.setattr(banks,'export',failure)
    with pytest.raises(RuntimeError,match='injected'):worker.run(s,'train_OFFLINE',device='cpu')
    assert not (tmp_path/'train_OFFLINE/result.json').exists()
    accepted.append(dict(task='train_OFFLINE',job='200'));monkeypatch.setattr(banks,'export',original)
    monkeypatch.setattr(artifacts,'save_epoch',lambda *a,**kw:pytest.fail('Must not repeat completed updates'))
    receipt=worker.run(s,'train_OFFLINE',device='cpu')
    assert receipt['job_id']=='200' and len(artifacts.history(s,'OFFLINE',receipt['fit_parents'])[0])==20


def test_interrupted_epoch_restores_exact_optimizer_and_rng(tmp_path,monkeypatch):
    s,*_=worker_fixture(tmp_path/'reference',monkeypatch)
    reference=worker.run(s,'train_OFFLINE',device='cpu')
    expected=torch.load(reference['files']['checkpoint']['path'],map_location='cpu',weights_only=True)
    s,tr,val,sampler,accepted=worker_fixture(tmp_path/'restart',monkeypatch)
    original=artifacts.save_epoch
    def interrupted(root,folder,stage,epoch,row,state,parents,job):
        if epoch==11:
            artifacts.save_state(Path(folder)/'uncommitted_state.pt',state)
            raise RuntimeError('interrupted before commit')
        return original(root,folder,stage,epoch,row,state,parents,job)
    monkeypatch.setattr(artifacts,'save_epoch',interrupted)
    with pytest.raises(RuntimeError,match='before commit'):worker.run(s,'train_OFFLINE',device='cpu')
    assert len(list((Path(s['root'])/'train_OFFLINE/epochs').glob('*.json')))==10
    accepted.append(dict(task='train_OFFLINE',job='200'));monkeypatch.setattr(artifacts,'save_epoch',original)
    receipt=worker.run(s,'train_OFFLINE',device='cpu')
    actual=torch.load(receipt['files']['checkpoint']['path'],map_location='cpu',weights_only=True)
    for key,value in expected.items():torch.testing.assert_close(value,actual[key],rtol=0,atol=0)
    a=load_json(reference['files']['report']['path']);b=load_json(receipt['files']['report']['path'])
    assert [x['validation'] for x in a['validation_history']]==[x['validation'] for x in b['validation_history']]
    assert a['selected_pass']==b['selected_pass']


def test_bank_checksum_and_training_validation_only(tmp_path,monkeypatch):
    s,tr,val,sampler,_=worker_fixture(tmp_path,monkeypatch)
    receipt=worker.run(s,'train_OFFLINE',device='cpu');r=artifacts.completed(s,'OFFLINE')
    wrong=deepcopy(tr);wrong.identities=wrong.identities[::-1]
    with pytest.raises(ValueError):banks.Bank(r['folder']/'bank',wrong,'OFFLINE',r['refs'],s)
    with pytest.raises(PermissionError):banks.Bank(r['folder']/'bank',val,'OFFLINE',r['refs'],s)
    path=r['folder']/'bank/targets.npy';path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):banks.Bank(r['folder']/'bank',tr,'OFFLINE',r['refs'],s)


def test_shell_detachment_and_absolute_worker_paths():
    root=Path(__file__).resolve().parents[1]
    launch=(root/'scripts/launch_luka_sixclass_low_lr_oscar.sh').read_text()
    worker_text=(root/'sbatch/run_luka_sixclass_low_lr_oscar.sh').read_text()
    assert 'nohup bash' in launch and '</dev/null &' in launch and 'exec tail' in launch
    assert '--authorize '+c.AUTH in launch
    assert 'PYTHONNOUSERSITE=1' in (root/'sbatch/luka_sixclass_low_lr_oscar_common.sh').read_text() and 'luka_sixclass_low_lr_oscar_common.sh' in worker_text
    assert '${PROJECT_DIR}/scripts/luka_sixclass_low_lr_oscar.py' in worker_text


def test_quota_credit_excludes_uncommitted_attempt_files(tmp_path):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    folder=tmp_path/'train_OFFLINE/runs/123';folder.mkdir(parents=True)
    payload=folder/'state.pt';payload.write_bytes(b'committed')
    (folder/'incomplete_bank.npy').write_bytes(b'x'*4096)
    assert campaign.committed_bytes(tmp_path)==0
    write_immutable_json(tmp_path/'train_OFFLINE/epochs/01.json',
                         c.artifact('EPOCH_COMMIT',state=file_ref(payload)))
    assert campaign.committed_bytes(tmp_path)==len(b'committed')
    payload.write_bytes(b'changed size')
    with pytest.raises(ValueError):campaign.committed_bytes(tmp_path)


def test_exact_allocation_and_runtime_binding(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path);s['environment_dir']=sys_prefix=__import__('sys').prefix
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('LD_LIBRARY_PATH',str(Path(sys_prefix)/'lib'))
    monkeypatch.setattr(campaign,'binding',lambda *a:None);monkeypatch.setattr(campaign,'runtime',lambda:{})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'NVIDIA L40S')
    def response(args,**kw):
        out='ClusterName = slurmctld\n' if args[-1]=='config' else (
            f'JobId=123 JobState=RUNNING Account=default QOS=norm-gpu Partition=gpu JobName=luka6lr_train_OFFLINE '
            f'Comment={s["content_hash"]}:train_OFFLINE WorkDir=/project NumCPUs=6 NumNodes=1 NumTasks=1 CPUs/Task=6 '
            'AllocTRES=cpu=6,mem=187.5G,node=1,gres/gpu=1 TimeLimit=2-00:00:00')
        return SimpleNamespace(stdout=out)
    monkeypatch.setattr(subprocess,'run',response)
    assert worker.allocation(s,'train_OFFLINE')=='123'
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'GeForce GTX 1080 Ti')
    with pytest.raises(PermissionError):worker.allocation(s,'train_OFFLINE')


def test_frozen_donor_manifest_matches_pinned_git_blobs():
    import hashlib
    project=Path(__file__).resolve().parents[1]
    manifest=load_json(project/'src/hlt_classification/luka_sixclass_low_lr_oscar/donors.json')
    for name,digest in manifest['files'].items():
        blob=subprocess.check_output(['git','-C',str(project),'show',manifest['commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==digest
        assert (project/name).read_bytes().replace(b'\r\n',b'\n')==blob


def test_writer_lock_rejects_second_writer_and_releases_after_error(tmp_path):
    with pytest.raises(RuntimeError):
        with campaign.submission_lock(tmp_path):
            with pytest.raises(OSError):
                with campaign.submission_lock(tmp_path):pytest.fail('Concurrent writer acquired lock')
            raise RuntimeError('injected launcher exit')
    with campaign.submission_lock(tmp_path):pass


def test_real_bash_detached_launch_finishes_after_tail_exits(tmp_path):
    import os,sys,time
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def sp(path):
        p=Path(path).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True);(project/'sbatch').mkdir()
    fake=tmp_path/'env';(fake/'bin').mkdir(parents=True)
    script=project/'scripts/launch_luka_sixclass_low_lr_oscar.sh'
    script.write_bytes((Path(__file__).resolve().parents[1]/'scripts'/script.name).read_bytes())
    # Stand-in runtime activation; test the actual detached shell lifecycle.
    (project/'sbatch/luka_sixclass_low_lr_oscar_common.sh').write_bytes(b':\n')
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'calls'
    for path,body in {bindir/'git':'printf "%040d\\n" 1\n',bindir/'tail':'echo TAIL_ATTACHED\n',
                      fake/'bin/python':'sleep 1\nprintf "%s\\n" "$*" >> "$FIXTURE_CALLS"\n'}.items():
        path.write_bytes(('#!/usr/bin/env bash\n'+body).encode());path.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    data=tmp_path/'data';data.mkdir();(data/'transfer_complete.json').write_text('{}');root=tmp_path/'result'
    env=dict(os.environ,BASH_ENV=sp(envfile),FIXTURE_BIN=sp(bindir),FIXTURE_CALLS=sp(calls))
    args=[bash,str(script),'detach',sp(project),sp(data),sp(root),'0'*39+'1','14',sp(fake),'--authorize',c.AUTH]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'DETACHED launcher PID' in result.stdout and 'TAIL_ATTACHED' in result.stdout
    log=Path(str(root)+'.launch')/'launcher.log';deadline=time.monotonic()+10
    while (not log.exists() or 'LAUNCHER_EXIT=0' not in log.read_text()) and time.monotonic()<deadline:time.sleep(.05)
    assert 'LAUNCHER_EXIT=0' in log.read_text()
    lines=calls.read_text().splitlines();assert len(lines)==2 and ' register ' in lines[0] and ' queue ' in lines[1]
    assert '--authorize '+c.AUTH in lines[1] and '--execute' in lines[1]
    assert subprocess.run(args,env=env,capture_output=True,timeout=5).returncode!=0


def tiny(monkeypatch):
    config = model.model_config()
    config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=2,num_layers=1,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(model, 'old_config', lambda:deepcopy(config))


def cache(role='train',coordinate='D066',rows=12):
    from hlt_classification.jetclass2_delphes.cache import RamCache,RamBlock
    rng=np.random.default_rng(41);lengths=np.arange(rows)%4+1
    offsets=np.concatenate(([0],np.cumsum(lengths))).astype(np.int64)
    f=rng.normal(size=(offsets[-1],17)).astype(np.float32)
    p=rng.normal(size=(offsets[-1],3)).astype(np.float32)
    v=np.column_stack((p,np.sqrt((p*p).sum(-1)+1))).astype(np.float32)
    ids=np.zeros((rows,32),np.uint8)
    ids[:,:2]=np.array([[i%256,i//256] for i in range(rows)],np.uint8)
    return RamCache([RamBlock(0,offsets,f,v,ids,np.arange(rows,dtype=np.int64)%6)],
                    role=role,foundation_sha256='a'*64,coordinate_name=coordinate)

