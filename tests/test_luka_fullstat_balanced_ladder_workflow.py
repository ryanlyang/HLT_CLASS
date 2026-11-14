"""Production workflow with small fixtures, real native Weaver and durable outputs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_fullstat import fullstat,durable
from test_luka_fullstat_balanced_ladder import threads
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullstat_balanced_ladder import (
    contracts as c,campaign,submission,views,training,model,banks,population,artifacts,worker,startup,source,recovery,reporting,inspect)
from hlt_classification.luka_fullstat_balanced_ladder.sampling import BalancedSampler


def spec_fixture(root):
    inputs=c.artifact('INPUTS',evidence=dict(prepared='a'*64),refs=[],historical_model_reuse=False)
    pop=c.artifact('POPULATION',parents=dict(inputs=inputs['content_hash']))
    return c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='f'*40,source=dict(files={}),
        inputs=inputs,population=pop,population_root=str(root/'population'),recipe=c.recipe(),
        nodes={s:c.node(s) for s in c.STAGES},counts=dict(c.COUNTS),eligible_train_rows=c.ELIGIBLE,
        execution_site=execution_site('oscar_l40s'),runtime_prefix=source.RUNTIME,available_result_bytes=36*2**30,
        **campaign.resources(),tasks=c.tasks(),completed={},continuations={},recovery=None)


def test_dag_roundtrip_and_exact_journals_no_double_submission(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path);campaign.check(spec,live_source=False)
    monkeypatch.setattr(source,'current',lambda *a:spec['source'])
    monkeypatch.setattr(submission,'check_site',lambda plan:None)
    spec=json.loads(json.dumps(spec,sort_keys=True))
    plan=submission.submit(spec)
    assert plan['jobs']==plan['scientific_fits']==6 and plan['preflight_jobs']==0
    assert [r['task_id'] for r in plan['commands']]==['train_'+s for s in c.STAGES]
    with pytest.raises(PermissionError):submission.submit(spec,execute=True,plan_hash=plan['content_hash'])
    calls=[]
    def fake(args,**kw):
        calls.append(args);return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(spec)=={'train_'+s:str(101+i) for i,s in enumerate(c.STAGES)}
    assert all(not any(x.startswith('--dependency=') for x in call) for call in calls[:3])
    assert '--dependency=afterok:101:103:102' in calls[3]
    assert '--dependency=afterok:101:103:104:102' in calls[4]
    assert '--dependency=afterok:101:103:104:105' in calls[5]
    assert all('--kill-on-invalid-dep=yes' in args for args in calls)
    with pytest.raises(FileExistsError):submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==6


@pytest.mark.parametrize('key,value',[('recipe',{}),('eligible_train_rows',3000000),('tasks',{}),('memory_mb',100000),('final_test_accessed',True)])
def test_rehashed_semantic_changes_rejected(tmp_path,key,value):
    s=spec_fixture(tmp_path);s[key]=value;s=meta.with_content_hash(s)
    with pytest.raises((ValueError,PermissionError)):campaign.check(s,live_source=False)


@pytest.mark.parametrize('coordinate',['D050','CONCAT'])
def test_real_root_full_support_cut_and_sealed_roles(fullstat,durable,monkeypatch,coordinate):
    import uproot
    container,_,_=fullstat;root,original=durable;metadata=deepcopy(original)
    # The fixture's original pT is 199. Change only the in-memory audited scalar column
    # to exercise new eligibility; raw-source/receipt authentication is tested separately.
    for member in metadata[2]:
        member['jet_pt']=member['jet_pt'].copy();member['jet_pt'][:]=250.
        train=np.flatnonzero(member['roles']==0);member['jet_pt'][train[::3]]=2000.
    arrays=uproot.behaviors.TTree.TTree.arrays
    def guarded(tree,expressions,*args,**kw):
        if any('part_' in x for x in expressions):
            i=next(i for i,r in enumerate(metadata[1]['files']) if Path(tree.file.file_path).as_posix().endswith(r['path']))
            assert np.isin(metadata[2][i]['roles'][kw['entry_start']:kw['entry_stop']],(0,1,255)).all()
        return arrays(tree,expressions,*args,**kw)
    monkeypatch.setattr(uproot.behaviors.TTree.TTree,'arrays',guarded)
    evidence=dict(raw_container=str(container),cache_root=str(root))
    values,records=views.constructed(metadata,evidence,coordinate,workers=1,memory_bytes=2**30*16)
    ordinary=views.ordinary(metadata,root,'D000',memory_bytes=2**30*16)
    assert len(values['train'])<264 and len(values['validation'])==66
    for role,cache in values.items():
        np.testing.assert_array_equal(cache.identities,ordinary[role].identities)
        np.testing.assert_array_equal(cache.labels,ordinary[role].labels)
        assert np.all(views.lengths(cache)==(5 if coordinate=='CONCAT' else 2))
    # Spawn workers must preserve byte/order semantics, not just counts.
    parallel,other=views.constructed(metadata,evidence,coordinate,workers=4,memory_bytes=2**30*16)
    assert records==other
    for role in values:
        for a,b in zip(values[role].blocks,parallel[role].blocks):assert views.digest(a)==views.digest(b)


def test_d050_exact_half_retains_unmatched_hlt_and_concat_full_union():
    from test_luka_u100_multiview import particles
    h,o=particles(3),particles(2,10)
    mapping=np.array([1,0,-1],np.int64)
    result=views.build_view(identity='b'*64,proxy=h,offline=o,coordinate='D050',mapping=mapping)
    assert len(result)==len(h)
    np.testing.assert_allclose(result.p4[:2],.5*h.p4[:2]+.5*o.p4[mapping[:2]],rtol=0,atol=0)
    np.testing.assert_array_equal(result.p4[2],h.p4[2])
    union=views.union(h,o)
    swapped=views.union(o.take(np.arange(len(o))[::-1]),h)
    assert len(union)==len(h)+len(o)
    np.testing.assert_array_equal(union.p4,swapped.p4)


def short_fixture(monkeypatch):
    from test_luka_embedding_aux import tiny
    from test_luka_attention import cache
    tiny(monkeypatch)
    monkeypatch.setattr(c,'DRAWS',44)
    tr=cache(coordinate='OFFLINE',rows=198);val=cache(role='validation',coordinate='OFFLINE',rows=22)
    # Disjoint validation canonical ids.
    val.identities[:,20]=1
    sampler=BalancedSampler(tr.labels,250+100*(np.arange(len(tr))//11))
    return tr,val,sampler


def test_twenty_epoch_restart_exactness_and_finalize_no_updates(tmp_path,monkeypatch):
    tr,val,sampler=short_fixture(monkeypatch);parents=dict(spec='f'*64,population='p'*64)
    saved=[]
    def save(e,row,state):
        artifacts.save_epoch(tmp_path,e,row,state,parents);saved.append(deepcopy(state))
    net,report,best=training.train(tr,val,banks={},sampler=sampler,stage='OFFLINE',parents=parents,device='cpu',save_epoch=save)
    training.validate_report(report,'OFFLINE',parents,len(val),sampler)
    refs=artifacts.epoch_refs(tmp_path,parents,'OFFLINE',len(val),sampler)
    resume=artifacts.restore(refs[:10],parents,'OFFLINE',len(val),sampler)
    replay=[]
    net2,report2,best2=training.train(tr,val,banks={},sampler=sampler,stage='OFFLINE',parents=parents,device='cpu',
        save_epoch=lambda e,r,s:replay.append((e,s)),resume=resume)
    assert [e for e,_ in replay]==list(range(11,21))
    assert all(torch.equal(v,net2.state_dict()[k]) for k,v in net.state_dict().items())
    assert report['validation']==report2['validation'] and report['selected_pass']==report2['selected_pass']
    for epoch,state in replay:
        assert all(torch.equal(v,saved[epoch-1]['student'][k]) for k,v in state['student'].items())
    finished=artifacts.restore(refs,parents,'OFFLINE',len(val),sampler)
    training.train(tr,val,banks={},sampler=sampler,stage='OFFLINE',parents=parents,device='cpu',
        save_epoch=lambda *a:pytest.fail('Publication recovery must not train'),resume=finished)
    path=Path(meta.load_json(meta.validate_file_ref(refs[-1]))['state']['path']);path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError):artifacts.restore(refs,parents,'OFFLINE',len(val),sampler)


def test_native_bank_roundtrip_unique_train_normalization_and_corruption(tmp_path,monkeypatch):
    tr,val,sampler=short_fixture(monkeypatch);net=model.new_model('OFFLINE').deployed()
    spec=spec_fixture(tmp_path);artifacts.save_state(tmp_path/'selected.pt',net.state_dict())
    meta.write_immutable_json(tmp_path/'report.json',dict(selected_epoch=1))
    refs=dict(checkpoint=meta.file_ref(tmp_path/'selected.pt'),report=meta.file_ref(tmp_path/'report.json'))
    banks.export(tmp_path/'bank',net,tr,'OFFLINE',refs,spec,'cpu')
    bank=banks.Bank(tmp_path/'bank',tr,'OFFLINE',refs,spec)
    assert bank.targets.shape==(198,128) and bank.mean.dtype==np.float64
    np.testing.assert_allclose(bank.targets.mean(0),0,atol=1e-6)
    # Native extraction, not auxiliary predictions or replacement-weighted moments.
    from hlt_classification.luka_u100_multiview.teachers import forward
    with torch.inference_mode():
        raw=np.concatenate([forward(net,tr.batch(ix),'cpu',bf16=False,capture=True)[1].numpy()
            for ix in (np.arange(128),np.arange(128,198))])
    np.testing.assert_array_equal(bank.mean,raw.mean(0,dtype=np.float64))
    with pytest.raises(PermissionError):banks.Bank(tmp_path/'bank',val,'OFFLINE',refs,spec)
    wrong=deepcopy(tr);wrong.identities=wrong.identities[::-1]
    with pytest.raises(ValueError):banks.Bank(tmp_path/'bank',wrong,'OFFLINE',refs,spec)
    path=tmp_path/'bank/targets.npy';path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError):banks.Bank(tmp_path/'bank',tr,'OFFLINE',refs,spec)


def test_report_masks_censoring_and_fresh_recovery():
    labels=np.tile(np.arange(11),3);p=np.full((33,11),.001);p[np.arange(33),labels]=.99;p/=p.sum(1,keepdims=True)
    pt=np.repeat([199.,200.,2000.],11)
    r=reporting.metrics(labels,p,pt,{'report':'a'*64})
    assert r['populations']['in_range']['summary']['rows']==11
    assert r['populations']['outside_range']['summary']['rows']==22
    assert r['populations']['200_300']['summary']['rows']==11
    assert r['populations']['300_400']['summary']['status']=='missing_class'
    for eff in ('R50','R30'):
        assert all(v['status']=='censored' and v['background_passing']==0 for v in r['populations']['natural'][eff].values())
    assert reporting.recovery(.7,.6,.8)==pytest.approx(50.)
    assert reporting.recovery(.7,.8,.8) is None and reporting.recovery(.7,.8,.6) is None


def test_timing_has_export_and_publication_recovery_cost():
    full=startup.estimate('CONCAT',.1,.05,.1,100)
    done=startup.estimate('CONCAT',.1,.05,.1,100,remaining_epochs=0)
    assert 60<done['conservative_minutes']<full['conservative_minutes']
    for bad in (0,-1,float('nan')):
        with pytest.raises(ValueError):startup.estimate('DE000',bad,.05,0,100)


def worker_fixture(tmp_path,monkeypatch):
    from contextlib import nullcontext
    tr,val,sampler=short_fixture(monkeypatch)
    monkeypatch.setattr(c,'COUNTS',dict(train=198,validation=22,final_test=11,unused=0))
    s=spec_fixture(tmp_path);meta.write_immutable_json(tmp_path/'campaign_spec.json',s)
    monkeypatch.setattr(campaign,'check',lambda s,**kw:s)
    monkeypatch.setattr(execution_module(),'validate',lambda s,t:str(100+c.STAGES.index(t.removeprefix('train_'))))
    monkeypatch.setattr(submission,'accepted_jobs',lambda s,**kw:{'train_'+t:str(100+i) for i,t in enumerate(c.STAGES)})
    monkeypatch.setattr(submission,'accounting',lambda *a,**kw:None)
    monkeypatch.setattr(source,'replay',lambda _:None)
    monkeypatch.setattr(source,'current',lambda *a:s['source'])
    values=dict(validation_pt=np.full(len(val),350.),eligible_indices=np.arange(len(tr)))
    monkeypatch.setattr(population,'load',lambda *a,**kw:(s['population'],values,sampler))
    def inputs(s,stage,*args):
        a,b=deepcopy(tr),deepcopy(val);a.coordinate_name=b.coordinate_name=c.COORDINATES[stage]
        return a,b,c.artifact('VIEW_BUILD',parents=dict(spec=s['content_hash']),stage=stage)
    monkeypatch.setattr(views,'load',inputs)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=1024,rss_bytes=1024,gpu_peak_bytes=0))
    monkeypatch.setattr(startup,'run',lambda s,stage,*a,**kw:c.artifact('STARTUP',parents=dict(spec=s['content_hash']),stage=stage,
        installed_weaver_parity=True,finite_connected_gradients=True,classifier_ce_only=True,
        native_deployment_parity=True,full_support=True,probe_discarded=True,matching_serial_parity=stage=='DE000'))
    # Candidate semantics have separate serial/optimized tests with >=4096 real candidates.
    # This small worker fixture substitutes only that candidate population, not its loss.
    class Match:
        def __init__(self,keys,identities,labels,**kw):self.keys=torch.from_numpy(keys);self.identities=identities;self.labels=labels
        def candidate_keys(self,ix):return self.keys[ix]
    def lookahead(match,order,epoch,*,batch_size):
        def rows():
            for _,ix in sampler.batches(order,batch_size=batch_size):
                candidates=np.tile(np.arange(1025)%len(tr),(len(ix),1));candidates[:,0]=ix
                yield ix,candidates
        return nullcontext(rows())
    monkeypatch.setattr(worker.contrastive,'Sampler',Match)
    monkeypatch.setattr(worker.contrastive,'CPUWorkers',lambda *a,**kw:nullcontext())
    monkeypatch.setattr(worker.contrastive,'Lookahead',lookahead)
    return s,tr,val,sampler


def execution_module():
    from hlt_classification.luka_fullstat_balanced_ladder import execution
    return execution


def test_six_production_workers_twenty_epochs_native_exports_and_fresh_anchors(tmp_path,monkeypatch):
    spec,tr,val,sampler=worker_fixture(tmp_path,monkeypatch)
    exposures=[]
    for stage in c.STAGES:
        receipt=worker.run(spec,'train_'+stage,device='cpu')
        completed=artifacts.completed(spec,stage)
        assert receipt==completed['receipt'] and completed['report']['passes']==20
        assert len(receipt['new_epochs'])==20
        exposures.append([r['sampling'] for r in completed['report']['validation_history']])
        state=torch.load(receipt['files']['checkpoint']['path'],map_location='cpu',weights_only=True)
        assert not any(k.startswith(('predictors.','logit_heads.','matching.')) for k in state)
        if stage!='DE000':
            bank=banks.Bank(completed['folder']/'bank',tr,stage,completed['refs'],spec)
            assert bank.targets.shape==(198,128)
        with pytest.raises(FileExistsError):worker.run(spec,'train_'+stage,device='cpu')
    assert all(x==exposures[0] for x in exposures)
    results=inspect.results(spec)
    assert set(results['models'])==set(c.STAGES)
    assert results['anchors']=='fresh_balanced_M0HLT_and_OFFLINE_per_population'
    path=tmp_path/'train_DE000/task_receipt.json';original=path.read_bytes()
    value=json.loads(original);value['fit_parents']['teacher_UE100']='0'*64
    path.write_text(json.dumps(meta.with_content_hash(value)),encoding='utf-8')
    with pytest.raises(ValueError,match='teacher lineage'):artifacts.completed(spec,'DE000')
    path.write_bytes(original)
    # All losses/gradients and reports are real; scheduling/resource admissions were mocked.
    assert not (tmp_path/'train_DE000/bank').exists()


def test_epoch_boundary_failure_and_explicit_recovery_twenty_epoch_finalize(tmp_path,monkeypatch):
    spec,tr,val,sampler=worker_fixture(tmp_path,monkeypatch)
    original=banks.export
    monkeypatch.setattr(banks,'export',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('synthetic publication failure')))
    with pytest.raises(RuntimeError,match='publication'):worker.run(spec,'train_OFFLINE',device='cpu')
    assert not (tmp_path/'train_OFFLINE/task_receipt.json').exists()
    continuation=recovery.history(spec,'OFFLINE')
    assert continuation['completed_epochs']==20
    newroot=tmp_path/'recovered';newroot.mkdir()
    fields={k:v for k,v in spec.items() if k not in ('content_hash','contract','schema_version','parents',*c.FLAGS)}
    fields.update(root=str(newroot),continuations={'OFFLINE':continuation})
    recovered=c.artifact('SPEC',**fields)
    monkeypatch.setattr(banks,'export',original)
    monkeypatch.setattr(artifacts,'save_epoch',lambda *a,**kw:pytest.fail('Completed optimization must not repeat'))
    receipt=worker.run(recovered,'train_OFFLINE',device='cpu')
    assert len(receipt['inherited_epochs'])==20 and receipt['new_epochs']==[]
    assert receipt['selected_pass']==meta.load_json(tmp_path/'train_OFFLINE/training_report.json')['selected_pass']


def test_population_membership_boundary_hashes_and_no_sealed_statistics(tmp_path,monkeypatch):
    labels=np.r_[np.tile(np.arange(11),18),[0,0],np.tile(np.arange(11),2),np.arange(11),[0]]
    pt=np.r_[np.repeat(np.arange(250,2000,100),11),[199.,2000.],np.full(22,2200.),np.full(12,np.nan)]
    roles=np.r_[np.zeros(200),np.ones(22),np.full(11,2),[3]].astype(np.uint8)
    raw=np.asarray([161,0,1,2,3,9,10,11,12,13,14])[labels]
    member=dict(labels=labels.astype(np.int8),jet_label=raw,jet_pt=pt,roles=roles)
    f=dict(files=[dict(path='synthetic.root',sha256='e'*64,tree_key='tree',entries=len(labels))])
    counts=dict(train=200,validation=22,final_test=11,unused=1)
    monkeypatch.setattr(population,'COUNTS',counts);monkeypatch.setattr(population,'ELIGIBLE',198)
    monkeypatch.setattr(population,'occupancy',lambda:np.ones((11,18),np.int64))
    manifest=dict(content_hash='a'*64);folder=tmp_path/'population'
    result=population.prepare(folder,(manifest,f,[member]),input_hash='b'*64)
    loaded,values,sampler=population.load(folder,input_hash='b'*64,prepared_hash='a'*64)
    assert loaded==result and len(sampler.indices)==198
    assert values['validation_pt'].tolist()==[2200.]*22
    assert not any('final_test' in k or 'unused' in k for k in values)
    with pytest.raises(ValueError):population.load(folder,input_hash='c'*64,prepared_hash='a'*64)
    path=folder/'rows.npz';path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError):population.load(folder,input_hash='b'*64,prepared_hash='a'*64)


def test_partial_scheduler_reply_keeps_journal_and_never_autoretries(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path);monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s);calls=[]
    def fake(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='101\n' if len(calls)==1 else 'ambiguous response\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    with pytest.raises(Exception):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==2 and (tmp_path/'submission/live_claim').is_dir()
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==2


def test_recovery_dag_drops_only_completed_fits(tmp_path):
    s=spec_fixture(tmp_path);s['completed']={name:{} for name in ('OFFLINE','M0HLT','CONCAT','UE100')}
    s['tasks']={'train_DE050':[],'train_DE000':['train_DE050']};s['recovery']={'parent_spec':{}}
    plan=submission.render(s)
    assert [r['task_id'] for r in plan['commands']]==['train_DE050','train_DE000']
    assert plan['commands'][1]['dependencies']==['train_DE050']
    assert '--dependency=afterok:${JOB_train_DE050}' in plan['commands'][1]['command']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real integrated objective probe requires a CUDA GPU')
@pytest.mark.parametrize('stage',['UE100','DE000'])
def test_real_full_graph_integrated_startup_on_synthetic_population(tmp_path,monkeypatch,stage):
    import time
    from test_luka_attention import cache
    from hlt_classification.luka_fullstat_balanced_ladder import contrastive
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    n=5500;tr=cache(coordinate=c.COORDINATES[stage],rows=n)
    val=cache(role='validation',coordinate=c.COORDINATES[stage],rows=22)
    sampler=BalancedSampler(tr.labels,250+100*((np.arange(n)//11)%18))
    rng=np.random.default_rng(71)
    targets={t:SimpleNamespace(targets=rng.normal(size=(n,128)).astype(np.float32),
        probabilities=np.full((n,11),1/11,np.float32)) for t in c.TEACHERS[stage]}
    matching=None
    if stage=='DE000':
        keys=targets['OFFLINE'].targets.copy();keys/=np.linalg.norm(keys,axis=1,keepdims=True)
        matching=contrastive.Sampler(keys,tr.identities,tr.labels,device='cuda')
    # Windows lacks Linux /proc/resource RSS. Keep real GPU measurements and real
    # forward/backward, serial candidate parity, 3 workers and timing pipeline.
    monkeypatch.setattr(startup,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    spec=spec_fixture(tmp_path)
    proof=startup.run(spec,stage,tr,val,targets,sampler,matching,'cuda',tmp_path,time.monotonic(),remaining_epochs=0)
    assert proof['finite_connected_gradients'] and proof['probe_discarded']
    assert proof['matching_serial_parity']==(stage=='DE000')
    assert proof['timing']['train_step']>0 and proof['memory']['gpu_peak_bytes']>0
    assert (tmp_path/'startup_diagnostics/measurements.json').exists()
