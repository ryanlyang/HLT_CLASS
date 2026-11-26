"""Two allocations, one twenty-epoch fit; immutable walltime recovery evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash,canonical_sha256
from hlt_classification.luka_sixclass_wide_b256_oscar import (
    contracts as c,campaign,artifacts,training,worker,recovery,gradient_recovery,time_recovery,
    startup,early_memory,banks,population,inspect,model)
from test_luka_sixclass_wide_b256_convolution import second_recovery
from test_luka_sixclass_wide_b256_oscar import worker_fixture
from test_luka_fullstat_balanced_ladder import threads


def third_recovery(tmp_path,monkeypatch):
    old,v2,previous=second_recovery(tmp_path,monkeypatch)
    write_immutable_json(Path(previous['root'])/'campaign_spec.json',previous)
    gradient_recovery.claim(previous)
    for key,value in [('PREVIOUS_HASH',previous['content_hash']),('PREVIOUS_ROOT',previous['root']),
                      ('PREVIOUS_COMMIT',previous['commit'])]:monkeypatch.setattr(time_recovery,key,value)
    folder=Path(previous['root'])/'train_CONCAT/runs'/time_recovery.FAILED_JOB
    write_immutable_json(folder/'failure.json',c.artifact('FAILURE',parents=dict(spec=previous['content_hash']),
        stage='CONCAT',job_id=time_recovery.FAILED_JOB,error_type='RuntimeError',message=time_recovery.FAILURE_MESSAGE))
    measured=timing(20)
    for phase in ('parity','miniature','stress','deployment','measurements'):
        value=c.artifact('STARTUP_PHASE',parents=dict(spec=previous['content_hash']),stage='CONCAT',phase=phase,
                         **(dict(timing=measured) if phase=='measurements' else {}))
        write_immutable_json(folder/'startup_diagnostics'/f'{phase}.json',value)
        if phase=='measurements':monkeypatch.setattr(time_recovery,'MEASUREMENTS_HASH',value['content_hash'])
    write_immutable_json(folder/'early_memory.json',c.artifact('EARLY_MEMORY',parents=dict(spec=previous['content_hash']),
                                                           stage='CONCAT',passed=True))
    original=campaign.accepted_jobs
    jobs=[dict(task='train_'+stage,job=str(int(time_recovery.FAILED_JOB)+i)) for i,stage in enumerate(c.STAGES[2:])]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda s,**kw:jobs if s==previous else original(s,**kw))
    monkeypatch.setattr(time_recovery,'capture_source_snapshot',lambda *a,**kw:previous['source'])
    _,record,gradient,evidence=time_recovery.authenticate(Path(previous['root'])/'campaign_spec.json',payloads=False)
    fields={k:v for k,v in previous.items() if k not in ('contract','schema_version','content_hash','parents',*c.FLAGS)}
    fields.update(root=str(tmp_path/'segmented'),project='/new/time_project',commit='d'*40,
        population_root=str(tmp_path/'segmented/population'),tasks=c.tasks(4),
        minutes={t:2880 for t in c.tasks(4)},time_recovery=evidence)
    return old,previous,c.artifact('SPEC',version=4,**fields)


def timing(epochs):
    return startup.estimate('CONCAT',.5697819293709472,.1327431604731828,.11845183279365301,
                            27682.94549097866,epochs)


def test_measured_guard_and_two_ten_epoch_allocations():
    assert timing(20)['conservative_minutes']==3105>2880
    assert timing(10)['conservative_minutes']==1842<2880
    assert timing(10)['timing_factor']==1.3 and timing(10)['fixed_margin_minutes']==60
    assert c.recipe()['content_hash']=='69df385334411bd372031c3ad8b7e42714eae5c52695270f23d5452135b47b41'
    assert c.tasks()['train_CONCAT']==[]


def test_v4_evidence_claims_and_five_job_four_fit_dag(tmp_path,monkeypatch):
    old,previous,new=third_recovery(tmp_path,monkeypatch)
    campaign.check(new,live=False)
    assert new['recipe']==old['recipe'] and new['population']==old['population']
    assert recovery.origin(new,'OFFLINE')==old and recovery.origin(new,'M0HLT')==old
    assert recovery.model_options(new,'CONCAT')==dict(pair_storage=True,pointwise_dgrad=True)
    auth=time_recovery.authenticate
    monkeypatch.setattr(time_recovery,'authenticate',lambda path:auth(path,payloads=False))
    monkeypatch.setattr(artifacts,'completed',lambda s,stage:dict(receipt={'content_hash':stage}))
    tasks,done=campaign.remaining(new,False)
    assert tasks==['train_CONCAT_part1','train_CONCAT','train_UE100','train_DE050','train_DE000']
    assert set(done)==set(recovery.BASELINES)
    rows=campaign.commands(new,tasks)
    assert rows[0]['dependencies']==[] and rows[1]['dependencies']==['train_CONCAT_part1']
    assert rows[2]['dependencies']==['train_CONCAT']
    assert len({c.task_stage(t) for t in tasks})==4
    paths=[Path(previous['root'])/'campaign_spec.json',Path(old['root'])/'concat_memory_recovery_claim_v1.json']
    before={p:p.read_bytes() for p in paths}
    time_recovery.claim(new);time_recovery.claim(new)
    with pytest.raises(FileExistsError):time_recovery.claim(with_content_hash(dict(new,root=str(tmp_path/'other'))))
    assert before=={p:p.read_bytes() for p in paths}
    with pytest.raises(ValueError,match='superseded'):campaign.remaining(previous,True)
    for key,value in [('tasks',c.tasks()),('minutes',{t:2880 for t in c.tasks()}),('recipe',dict(new['recipe'],epochs=10))]:
        with pytest.raises(ValueError):campaign.check(with_content_hash(dict(new,**{key:value})),live=False)
    changed=deepcopy(new);changed['time_recovery']['policy']['windows']=[[1,8],[9,20]]
    changed['time_recovery']=with_content_hash(changed['time_recovery'])
    with pytest.raises(ValueError):campaign.check(with_content_hash(changed),live=False)


def test_v4_rejects_live_prior_jobs_committed_progress_and_changed_measurements(tmp_path,monkeypatch):
    old,previous,new=third_recovery(tmp_path,monkeypatch);path=Path(previous['root'])/'campaign_spec.json'
    with monkeypatch.context() as m:
        m.setattr(campaign,'terminal',lambda job:(_ for _ in ()).throw(ValueError('live job')))
        with pytest.raises(ValueError,match='live job'):time_recovery.authenticate(path,payloads=False)
    write_immutable_json(Path(previous['root'])/'train_CONCAT/epochs/01.json',dict(committed=True))
    with pytest.raises(ValueError,match='committed progress'):time_recovery.authenticate(path,payloads=False)
    p=Path(new['time_recovery']['measurements']['measurements']['path'])
    r=load_json(p);r['timing']['timing_factor']=1.;p.write_text(__import__('json').dumps(with_content_hash(r)))
    with pytest.raises(ValueError,match='measurements differ'):time_recovery.validate(new)


def test_v4_registration_copies_only_authenticated_population(tmp_path,monkeypatch):
    old,previous,new=third_recovery(tmp_path,monkeypatch)
    dataset=tmp_path/'dataset';dataset.mkdir();manifest=dict(foundation_sha256=c.PARENT_HASH)
    monkeypatch.setattr(c,'TRANSFER_HASH',canonical_sha256(manifest))
    write_immutable_json(dataset/'transfer_manifest.json',manifest)
    write_immutable_json(dataset/'transfer_complete.json',dict(manifest_sha256=c.TRANSFER_HASH,all_files_verified=True))
    old=dict(old,dataset=str(dataset),environment_dir=str(Path(sys.prefix).resolve()))
    pop=Path(old['population_root']);write_immutable_json(pop/'population.json',old['population'])
    (pop/'scalar').write_bytes(b'population metadata')
    monkeypatch.setattr(time_recovery,'authenticate',lambda path:(old,new['execution_recovery'],new['gradient_recovery'],new['time_recovery']))
    for name in ('locations','source','runtime','storage'):
        monkeypatch.setattr(campaign,name,lambda *a,**kw:{})
    monkeypatch.setattr(campaign,'backend',lambda:old['backend'])
    monkeypatch.setattr(campaign.lineage,'verify',lambda *a:{})
    monkeypatch.setattr(population,'load',lambda *a:None)
    checks=[];monkeypatch.setattr(campaign,'check',lambda s,**kw:checks.append(s))
    p=Path(previous['root'])/'campaign_spec.json';before=p.read_bytes()
    result=campaign.create(dataset,tmp_path/'registered',tmp_path/'source','e'*40,sys.prefix,
                           available_gib=18,recover=p)
    assert result['schema_version']==4 and result['time_recovery']==new['time_recovery']
    assert result['tasks']==c.tasks(4) and set(result['minutes'])==set(c.tasks(4))
    assert result['execution_recovery']==new['execution_recovery'] and result['gradient_recovery']==new['gradient_recovery']
    assert result['recipe']==new['recipe'] and p.read_bytes()==before and checks==[result]
    assert not (Path(result['root'])/'train_OFFLINE').exists()
    assert (Path(result['root'])/'population/scalar').read_bytes()==b'population metadata'


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_weaver_ten_plus_ten_equals_uninterrupted_model_optimizer_rng(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA continuation parity')
    s,tr,val,sampler,_=worker_fixture(tmp_path,monkeypatch)
    tr.coordinate_name=val.coordinate_name='CONCAT';outside=deepcopy(val);outside.identities[:,20]=2
    config=model.model_config();config['pair_embed_dims']=[8,8,8]
    monkeypatch.setattr(model,'old_config',lambda:deepcopy(config))
    args=dict(banks={},sampler=sampler,stage='CONCAT',parents={},device=device,outside=outside,
              pair_storage=True,pointwise_dgrad=True)
    expected=[]
    net,report,_=training.train(tr,val,save_epoch=lambda e,r,s:expected.append((deepcopy(r),deepcopy(s))),**args)
    split=[]
    assert training.train(tr,val,stop_epoch=10,save_epoch=lambda e,r,s:split.append((deepcopy(r),deepcopy(s))),**args)==(None,None,None)
    from hlt_classification.jetclass2_delphes.runner import selection_key
    chosen=max(split,key=lambda x:selection_key(x[0]['validation'],x[0]['update']))[1]
    # Production restores fresh tensors from disk; do not alias the audit copy
    # into AdamW, whose load_state_dict may retain CPU tensor storage.
    resume=deepcopy(dict(history=[r for r,s in split],state=split[-1][1],best=chosen))
    # Deliberately consume CPU/CUDA RNG as the new allocation's probes do.
    torch.rand(87);torch.rand(91,device=device)
    actual,r2,_=training.train(tr,val,resume=resume,save_epoch=lambda e,r,s:split.append((deepcopy(r),deepcopy(s))),**args)
    assert [r['pass_number'] for r,s in split]==list(range(1,21))
    for (a,sa),(b,sb) in zip(expected,split):
        equal_tree(sa,sb)
        assert {k:v for k,v in a.items() if not k.endswith('_seconds')}=={k:v for k,v in b.items() if not k.endswith('_seconds')}
    equal_tree(net.state_dict(),actual.state_dict())
    assert report['selected_pass']==r2['selected_pass']
    torch.cuda.empty_cache()


def segmented_worker(tmp_path,monkeypatch):
    s,tr,val,sampler,accepted=worker_fixture(tmp_path,monkeypatch)
    s=with_content_hash(dict(s,schema_version=4,contract=c.PREFIX+'_SPEC/v4',tasks=c.tasks(4),minutes={t:2880 for t in c.tasks(4)}))
    accepted.insert(2,dict(task='train_CONCAT_part1',job='900'))
    original=startup.run
    def probe(spec,stage,*args,**kw):
        r=original(spec,stage,*args,**kw)
        return with_content_hash(dict(r,timing=startup.estimate(stage,.001,.001,0. if stage=='DE000' else .001,0.,kw['remaining_epochs'])))
    monkeypatch.setattr(startup,'run',probe)
    def early(spec,stage,loaded,device,folder):
        write_immutable_json(folder/'early_memory.json',c.artifact('EARLY_MEMORY',parents=dict(spec=spec['content_hash']),stage=stage,passed=True))
    monkeypatch.setattr(early_memory,'run',early)
    # Kernel acceptance is covered separately on installed-Weaver CUDA. This
    # fixture exercises real production training/commits/continuation/publication.
    monkeypatch.setattr(recovery,'model_options',lambda *a:{})
    monkeypatch.setattr(recovery,'origin',lambda spec,stage:spec)
    from hlt_classification.luka_sixclass_wide_b256_oscar import pair_storage,pair_convolution
    monkeypatch.setattr(pair_storage,'validate_storage_stats',lambda *a,**kw:None)
    monkeypatch.setattr(pair_convolution,'validate_stats',lambda *a,**kw:None)
    monkeypatch.setattr(pair_convolution,'validate_parities',lambda *a,**kw:None)
    original_probe=probe
    def proofs(spec,stage,*args,**kw):
        folder=args[6];r=original_probe(spec,stage,*args,**kw)
        p=load_json(folder/'early_memory.json');p.update(pointwise_policy=pair_convolution.POLICY,pointwise_statistics={},parity=[])
        (folder/'early_memory.json').write_text(__import__('json').dumps(with_content_hash(p)))
        return with_content_hash(dict(r,pair_storage={'calls':3},pointwise_policy=pair_convolution.POLICY,pointwise_statistics={}))
    monkeypatch.setattr(startup,'run',proofs)
    return s,accepted


def test_worker_handoff_blocks_export_and_resumes_remaining_epochs(tmp_path,monkeypatch,capsys):
    s,accepted=segmented_worker(tmp_path,monkeypatch);export=banks.export;calls=[]
    monkeypatch.setattr(banks,'export',lambda *a,**kw:(calls.append(a),export(*a,**kw))[1])
    with pytest.raises(FileNotFoundError):worker.run(s,'train_CONCAT',device='cpu')
    accepted.append(dict(task='train_CONCAT',job='901'))
    receipt=worker.run(s,'train_CONCAT_part1',device='cpu')
    assert receipt==artifacts.segment_completed(s) and calls==[]
    assert len(list((tmp_path/'train_CONCAT/epochs').glob('*.json')))==10
    assert not (tmp_path/'train_CONCAT/result.json').exists()
    assert not list(tmp_path.glob('train_CONCAT/runs/*/training_report.json'))
    with pytest.raises(FileNotFoundError):artifacts.completed(s,'CONCAT')
    # Read-only progress must accept both job bindings while only part1 exists.
    monkeypatch.setattr(inspect,'selected_metadata',lambda *a:None)
    inspect.progress(s,'CONCAT');assert '10/20 epochs committed' in capsys.readouterr().out
    before=[p.read_bytes() for p in sorted((tmp_path/'train_CONCAT/epochs').glob('*.json'))]
    final=worker.run(s,'train_CONCAT',device='cpu')
    assert len(calls)==1 and final==artifacts.completed(s,'CONCAT')['receipt']
    paths=sorted((tmp_path/'train_CONCAT/epochs').glob('*.json'))
    assert len(paths)==20 and [p.read_bytes() for p in paths[:10]]==before
    assert {load_json(p)['job_id'] for p in paths[:10]}=={'900'}
    assert {load_json(p)['job_id'] for p in paths[10:]}=={'901'}
    # A rehashed receipt cannot assign epoch11 to the first allocation.
    p=paths[10];r=load_json(p);r['job_id']='900';p.write_text(__import__('json').dumps(with_content_hash(r)))
    with pytest.raises(ValueError,match='allocation window'):artifacts.completed(s,'CONCAT')


@pytest.mark.parametrize('failure',['handoff','epoch13','export'])
def test_segment_interruption_preserves_commits_and_never_repeats_updates(tmp_path,monkeypatch,failure):
    s,accepted=segmented_worker(tmp_path,monkeypatch)
    save_segment=artifacts.save_segment;save_epoch=artifacts.save_epoch;export=banks.export
    def stop(*a,**kw):raise RuntimeError('injected interruption')
    if failure=='handoff':
        monkeypatch.setattr(artifacts,'save_segment',stop)
        with pytest.raises(RuntimeError,match='injected'):worker.run(s,'train_CONCAT_part1',device='cpu')
        assert not (tmp_path/'train_CONCAT/segments/part1.json').exists()
        accepted.append(dict(task='train_CONCAT_part1',job='903'))
        monkeypatch.setattr(artifacts,'save_segment',save_segment)
        monkeypatch.setattr(artifacts,'save_epoch',lambda *a,**kw:pytest.fail('Repeated epoch in handoff recovery'))
        worker.run(s,'train_CONCAT_part1',device='cpu')
        monkeypatch.setattr(artifacts,'save_epoch',save_epoch)
    else:
        worker.run(s,'train_CONCAT_part1',device='cpu')
        if failure=='epoch13':
            def interrupt(root,folder,stage,epoch,*args):
                if epoch==13:stop()
                return save_epoch(root,folder,stage,epoch,*args)
            monkeypatch.setattr(artifacts,'save_epoch',interrupt)
        else:monkeypatch.setattr(banks,'export',stop)
        with pytest.raises(RuntimeError,match='injected'):worker.run(s,'train_CONCAT',device='cpu')
        accepted.append(dict(task='train_CONCAT',job='904'))
    paths=sorted((tmp_path/'train_CONCAT/epochs').glob('*.json'));before={p:p.read_bytes() for p in paths}
    want={'handoff':10,'epoch13':12,'export':20}[failure]
    assert len(paths)==want
    done=[]
    def record(root,folder,stage,epoch,*args):
        done.append(epoch);return save_epoch(root,folder,stage,epoch,*args)
    monkeypatch.setattr(artifacts,'save_epoch',record);monkeypatch.setattr(banks,'export',export)
    result=worker.run(s,'train_CONCAT',device='cpu')
    assert done==list(range(want+1,21)) and {p:p.read_bytes() for p in paths}==before
    assert result==artifacts.completed(s,'CONCAT')['receipt']


def test_resume_omits_committed_first_segment_but_rejects_failed_exit(tmp_path,monkeypatch):
    old,previous,s=third_recovery(tmp_path,monkeypatch)
    auth=time_recovery.authenticate
    monkeypatch.setattr(time_recovery,'authenticate',lambda path:auth(path,payloads=False))
    monkeypatch.setattr(artifacts,'completed',lambda spec,stage:dict(receipt={'content_hash':stage}))
    saved=campaign.accepted_jobs
    jobs=[dict(job=str(1000+i),task=t) for i,t in enumerate(list(c.tasks(4))[2:])]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda spec,**kw:jobs if spec==s else saved(spec,**kw))
    prior_terminal=campaign.terminal
    monkeypatch.setattr(campaign,'terminal',lambda job:[job,'COMPLETED','0:0'] if job=='1000'
                        else [job,'CANCELLED','0:0'] if job in {j['job'] for j in jobs} else prior_terminal(job))
    marker=Path(s['root'])/'train_CONCAT/segments/part1.json';write_immutable_json(marker,{})
    monkeypatch.setattr(artifacts,'segment_completed',lambda spec:dict(job_id='1000'))
    tasks,_=campaign.remaining(s,True)
    assert tasks==['train_CONCAT','train_UE100','train_DE050','train_DE000']
    assert campaign.commands(s,tasks)[0]['dependencies']==[]
    monkeypatch.setattr(campaign,'terminal',lambda job:[job,'FAILED','1:0'])
    with pytest.raises(ValueError):campaign.remaining(s,True)
