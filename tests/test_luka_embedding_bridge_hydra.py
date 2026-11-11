"""Scientific identities, reductions, crop isolation and native update parity."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from hlt_classification.luka_embedding_bridge_hydra import registry as r, contracts as c, regions, response, losses, model


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec():
    return dict(nodes={t:dict(initialization_seed=23,sampler_seed=29,node_id=t) for t in r.FITS})


def synthetic_dataset():
    from hlt_classification.luka_embedding_bridge_hydra import inputs,storage
    class Arrays(dict):pass
    class Dataset:
        def __init__(self):
            self.spec=dict(spec(),content_hash='a'*64,counts=dict(train=33,validation=22),embedding_hash='b'*64,
                recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569))
            self.v={};self.u={};self.rows={}
            for role,n in self.spec['counts'].items():
                a=[physical(6+i%5) for i in range(n)];self.rows[role]=a
                p=dict(identities=np.array([np.frombuffer(__import__('hashlib').sha256((role+str(i)).encode()).digest(),np.uint8) for i in range(n)]),
                    labels=np.arange(n,dtype=np.int64)%11,HLT_offsets=np.r_[0,np.cumsum([len(x) for x in a])].astype(np.int64))
                p['OFFLINE_offsets']=p['HLT_offsets']
                built=[regions.encode(x) for x in a];encoded=dict(features=np.concatenate([x.features for x in built]),vectors=np.concatenate([x.vectors for x in built]))
                for coord in r.COORDINATES:self.v[role,coord]=inputs.View(encoded,p,role=role,coordinate=coord,prepared='c'*64)
                u=Arrays(identities=p['identities'],labels=p['labels'],targets=np.random.default_rng(n).normal(size=(n,128)).astype(np.float32),
                    mean=np.zeros(128,np.float64),scale=np.ones(128,np.float64))
                u.manifest=dict(role=role,content_hash='b'*64);self.u[role]=u
        def view(self,role,coord):return self.v[role,coord]
        def u100(self,role):return self.u[role]
        def condition(self,role,indices,condition):
            raw=self.view(role,'D000').batch(indices)
            raw['features']=raw['features'].copy();raw['features'][:,0]+=.01*r.CONDITIONS.index(condition)
            return raw
        def crops(self,role,indices,source,scale):
            a=[self.rows[role][i] for i in indices]
            return regions.crops(a,a,self.view(role,'D000').labels[indices],scale)
    return Dataset()


def test_producer_banks_and_all_consumers_use_real_native_fit(tmp_path,monkeypatch):
    from hlt_classification.luka_embedding_bridge_hydra import banks,training,distributed as d
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    small(monkeypatch);ds=synthetic_dataset();s=ds.spec;ctx=d.Context(0,1,torch.device('cpu'));parents={}
    # Actual two-epoch production fitting/export/reload; no invented target tensors
    # after the fixed original U100 fixture. All declared loss paths run below.
    for task in r.FITS:
        parent=parents.get(r.cell(task)['parent'])
        target=banks.Targets(s,task,ds,parent_bank=parent)
        if task in r.PRODUCERS or task=='C0':
            m,report=training.fit(tmp_path/task,s,task,ds,target,ctx,technical=True,epochs=2,val_diagnostics=False)
            checked,_=training.validate_fit(tmp_path/task,s,task,technical=True)
            assert checked==report
            if task in r.PRODUCERS:
                checkpoint=file_ref(tmp_path/task/'selected_full.pt')
                banks.export(tmp_path/task/'bank',s,task,m,ds,ctx,checkpoint,technical=True)
                parents[task]=banks.Bank(tmp_path/task/'bank',s,task,ds.view('train','D000'),checkpoint,technical=True)
                with pytest.raises(PermissionError):banks.Bank(tmp_path/task/'bank',s,task,ds.view('train','D000'),checkpoint)
        else:
            m,w,o=d.build(s,task,ctx)
            values,_=d.step(m,w,o,task,ds,target,np.arange(15),3,0,3.,ctx)
            assert np.isfinite(list(values.values())).all()
            diagnostics=training.target_diagnostics(m,task,ds,target,ctx)
            assert diagnostics['new_target']['smooth_l1']>=0
    for task in ('C0','R5','P5'):
        target=banks.Targets(s,task,ds,parent_bank=parents.get(r.cell(task)['parent']))
        _,report=training.fit(tmp_path/('full30_'+task),s,task,ds,target,ctx,technical=True,epochs=30)
        checked,_=training.validate_fit(tmp_path/('full30_'+task),s,task,technical=True)
        assert len(checked['history'])==30 and report['native_reload']


def test_storage_atomic_corruption_and_types(tmp_path):
    from hlt_classification.luka_embedding_bridge_hydra.storage import Writer,Store
    w=Writer(tmp_path/'bank',dict(x=(np.float32,(3,128))))
    w.arrays['x'][:]=1;m=w.finish('TEST',parents=dict(source='x'))
    a=Store(tmp_path/'bank','TEST',parents=dict(source='x'),digest=m['content_hash'])
    np.testing.assert_array_equal(a['x'],1)
    with pytest.raises(ValueError):Store(tmp_path/'bank','TEST',parents=dict(source='y'))
    with open(tmp_path/'bank/x.npy','r+b') as f:f.seek(-1,2);f.write(b'\xff')
    with pytest.raises(ValueError):Store(tmp_path/'bank','TEST',parents=dict(source='x'))


def execution_spec(root):
    from hlt_classification.luka_embedding_bridge_hydra import campaign
    s=synthetic_dataset().spec
    s.update(campaign.registration(),root=str(root),project_dir=str(root/'code'),reference=dict(environment={'fixture':True}))
    return s


def resource_diagnostic(s):
    from hlt_classification.luka_embedding_bridge_hydra import resources
    devices=[dict(name='GeForce GTX 1080 Ti',compute_capability=[6,1],total_memory_bytes=11*2**30) for _ in range(2)]
    profiles={}
    for t in r.FITS:
        sec=1. if t in r.PRODUCERS else 0.;rows=1000
        profiles[t]=dict(step_samples=[.2]*12,step_seconds=float(np.mean([.2]*12)),validation_seconds=20.,probe_seconds=1.,panel_seconds=2.,
            load_seconds=120.,startup_seconds=20.,checkpoint_seconds=3.,export_measured_seconds=sec,export_measured_rows=rows,
            export_estimate_seconds=sec*(s['counts']['train']+4096)/rows,
            memory=dict(host_peak_bytes=12*2**30,ranks=[dict(peak_rss_bytes=5*2**30,gpu_peak_bytes=7*2**30) for _ in range(2)]),
            technical_targets=True,miniature_complete=True,native_reload=True,largest_support_backward=True,partial_batch_backward=True,
            condition_coverage=[r.teacher_condition(3,i,regional=t=='TP') if t in ('TR','TP') else 'native' for i in range(6)],
            train_rows=s['counts']['train'],validation_rows=s['counts']['validation'])
    return c.artifact('RESOURCES',parents=dict(spec=s['content_hash']),profiles=profiles,devices=devices,status='complete',environment=s['reference']['environment'])


def test_complete_dag_and_measured_envelope(tmp_path):
    from hlt_classification.luka_embedding_bridge_hydra import submission,resources
    s=execution_spec(tmp_path);boot=submission.plan(s,'bootstrap');envelope=submission.plan(s,'envelope')
    assert len(boot['commands'])==2 and len(envelope['commands'])==31
    assert boot['commands'][1]['dependencies']==['setup'] and boot['upper_bound_commands']==envelope['commands']
    for row in envelope['commands'][:-1]:
        assert row['dependencies']==r.dependencies(row['task_id'])
        assert '--gres=gpu:geforce_gtx_1080_ti:2' in row['command'] and '--mem=48000M' in row['command']
        assert '--ntasks=1' in row['command'] and '--no-requeue' in row['command']
    report=envelope['commands'][-1];assert report['dependencies']==list(r.FITS)
    assert '--cpus-per-task=2' in report['command'] and '--mem=8000M' in report['command']
    assert any(x.startswith('--dependency=afterany:') for x in report['command'])
    measured=submission.plan(s,'science',resources.accept(s,resource_diagnostic(s)))
    assert submission.within_envelope(envelope,measured)
    measured['commands'][0]['command'].append('--gres=gpu:3')
    with pytest.raises(PermissionError):submission.within_envelope(envelope,measured)


@pytest.mark.parametrize('bad',('host','gpu','time','coverage','missing','samples','export','finite'))
def test_resource_admission_is_technical_and_fail_closed(tmp_path,bad):
    from hlt_classification.luka_embedding_bridge_hydra import resources
    from hlt_classification.data.cache_contracts import with_content_hash
    s=execution_spec(tmp_path);value=resource_diagnostic(s)
    assert set(resources.accept(s,value)['minutes'])==set(r.FITS)
    p=value['profiles']['TR']
    if bad=='host':p['memory']['host_peak_bytes']=48*2**30
    if bad=='gpu':p['memory']['ranks'][1]['gpu_peak_bytes']=11*2**30
    if bad=='time':p.update(step_samples=[1000.]*12,step_seconds=1000.)
    if bad=='coverage':p['condition_coverage']=['B']*6
    if bad=='missing':del value['profiles']['P9']
    if bad=='samples':p['step_samples']=[.2]
    if bad=='export':p['export_measured_seconds']=0.
    if bad=='finite':p['load_seconds']=-1.
    value.pop('content_hash');value=with_content_hash(value)
    with pytest.raises((ValueError,MemoryError,RuntimeError)):resources.accept(s,value)


def fake_scheduler(monkeypatch):
    from hlt_classification.luka_embedding_bridge_hydra import submission,campaign
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls=[];reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_auto_release_exact_dag_and_no_duplicate_submission(tmp_path,monkeypatch):
    from hlt_classification.luka_embedding_bridge_hydra import submission,resources,worker
    from hlt_classification.data.cache_contracts import write_immutable_json
    s=execution_spec(tmp_path);(tmp_path/'setup').mkdir();a=resources.accept(s,resource_diagnostic(s))
    write_immutable_json(tmp_path/'setup/acceptance.json',a)
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(s);p=submission.plan(s,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(s,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    b=submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert b['jobs']==dict(setup='9001',release='9002')
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    result=submission.finish(s,'9002');assert len(result['jobs'])==31 and len(calls)==33
    assert submission.finish(s,'9002')==result and len(calls)==33
    assert [p['stage'] for p in reviews]==['bootstrap','envelope','science']
    for task in r.FITS:
        dependencies=r.dependencies(task)
        cmd=next(cmd for cmd in calls if '--job-name=lukaeb_'+task in cmd)
        if dependencies:assert '--dependency=afterok:'+result['jobs'][dependencies[0]] in cmd


def test_failed_setup_and_ambiguous_submission_never_retry(tmp_path,monkeypatch):
    from hlt_classification.luka_embedding_bridge_hydra import submission
    s=execution_spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(s);p=submission.plan(s,'bootstrap')
    submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(s,'9002')
    assert len(calls)==2
    s=execution_spec(tmp_path/'another');(tmp_path/'another').mkdir();submission.launch(s)
    (tmp_path/'another/submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous/partial'):
        submission.launch(s,execute=True,plan_hash=submission.plan(s,'bootstrap')['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==2


def test_original_physical_preparation_and_quarter_views(tmp_path,monkeypatch):
    import hashlib
    from hlt_classification.luka_embedding_bridge_hydra import inputs
    from hlt_classification.cms_proxy_ladder.views import build_view
    from hlt_classification.luka_fullsim.stage2 import ViewMeter
    from hlt_classification.data.cache_contracts import write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
    ds=synthetic_dataset();s=ds.spec;s.update(root=str(tmp_path),anchor=dict(evidence=dict(prepared='c'*64)),transfer_hash='d'*64,native_hash='e'*64)
    counts=s['counts'];monkeypatch.setattr(inputs,'COUNTS',counts);(tmp_path/'setup').mkdir();summaries={};payload={};canonical={}
    for role in counts:
        rows=ds.rows[role];ids=ds.view(role,'D000').identities;labels=ds.view(role,'D000').labels
        off=np.r_[0,np.cumsum([len(x) for x in rows])].astype(np.int64)
        a=dict(identities=ids,labels=labels,HLT_offsets=off,OFFLINE_offsets=off,HLT_values=np.concatenate(rows),
            OFFLINE_values=np.concatenate([x.copy() for x in rows]),mapping=np.concatenate([np.arange(len(x),dtype=np.int32) for x in rows]))
        a['OFFLINE_values'][:,:4]*=1.1;a['OFFLINE_values'][:,6:8]*=1.2
        payload[role]=a;meters={coord:ViewMeter() for coord in r.COORDINATES};values={coord:[] for coord in r.COORDINATES}
        for i in range(counts[role]):
            start,end=off[i:i+2];h=regions.physical(a['HLT_values'][start:end]);o=regions.physical(a['OFFLINE_values'][start:end]);identity=bytes(ids[i]).hex()
            for coord,meter in meters.items():
                v=build_view(identity=identity,proxy=h,offline=o,coordinate=coord,mapping=a['mapping'][start:end])
                encoded=regions.encode(np.column_stack((v.p4,v.charge,v.category,v.tracking,v.valid)))
                meter.add(SimpleNamespace(identity=identity,label=int(labels[i])),encoded);values[coord].append(encoded)
        for coord in inputs.native.HISTORICAL:
            canonical[role,coord]=SimpleNamespace(identities=ids,labels=labels,offsets=off,
                features=np.concatenate([x.features for x in values[coord]]),vectors=np.concatenate([x.vectors for x in values[coord]]))
        summaries[role]={coord:meter.report() for coord,meter in meters.items()}
    transfer=SimpleNamespace(manifest=dict(summaries=summaries),shards=lambda role:iter([(dict(index=0),payload[role])]))
    monkeypatch.setattr(inputs,'transferred',lambda spec:transfer)
    monkeypatch.setattr(inputs.native,'Sources',lambda spec:SimpleNamespace(block=lambda role,coord,index,a:canonical[role,coord]))
    def quarters_only(**kw):
        assert kw['coordinate'] in ('D025','D050','D075'), 'Historical native views must never be re-encoded'
        return build_view(**kw)
    monkeypatch.setattr(inputs.kernel,'build_view',quarters_only)
    original=dict(ds.u100('train'));(tmp_path/'targets/u100').mkdir(parents=True)
    write_immutable_json(tmp_path/'targets/u100/manifest.json',dict(content_hash=s['embedding_hash']))
    atomic_publish_bytes(tmp_path/'targets/u100/targets.npz',deterministic_npz_bytes(original))
    s['bundle_root']=str(tmp_path);monkeypatch.setattr(inputs,'Bundle',lambda path:SimpleNamespace(path=lambda rel:tmp_path/rel))
    monkeypatch.setattr(inputs,'embedding_bank',lambda spec,view:None)
    inputs.prepare(s);actual=inputs.Dataset(s)
    np.testing.assert_array_equal(actual.u100('train')['targets'],original['targets'])
    h=actual.view('train','D000').batch(np.arange(11));u=actual.view('train','U100').batch(np.arange(11))
    for condition in r.CONDITIONS:
        value=actual.condition('train',np.arange(11),condition)
        if condition=='B':np.testing.assert_array_equal(value['features'],h['features'])
        if condition=='KIT':
            np.testing.assert_array_equal(value['features'],u['features']);np.testing.assert_array_equal(value['vectors'],u['vectors'])
        if condition=='IT':np.testing.assert_array_equal(value['vectors'],h['vectors'])
    for coord in ('D025','D050','D075'):
        assert actual.view('train',coord).store.manifest['proof']==summaries['train'][coord]
    with pytest.raises(PermissionError):actual.view('test','D000')


@pytest.mark.parametrize('task',('C0','R8','P7','TR','TP'))
def test_full_depth_fp32_cuda_batch_and_tail(task):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from hlt_classification.luka_embedding_bridge_hydra import distributed as d,banks
    ds=synthetic_dataset();s=ds.spec;ctx=d.Context(0,1,torch.device('cuda',0));c_=r.cell(task)
    class Target:
        def batch(self,ix):
            value=dict(u100=ds.u100('train')['targets'][ix],mean=np.zeros(128,np.float32),scale=np.ones(128,np.float32))
            if task=='R8':value['extra']=np.ones((len(ix),3,128),np.float32)
            if task=='P7':value.update(extra=np.ones((len(ix),14,128),np.float32),valid=np.ones((len(ix),14),bool),groups=[list(range(2)),list(range(2,6)),list(range(6,14))])
            return value
    m,w,o=d.build(s,task,ctx)
    assert len(m.mod.blocks)==8 and len(m.mod.cls_blocks)==2
    batch=next(i for i in range(6) if r.teacher_condition(3,i,regional=True)==('OFFLINE','fine')) if task=='TP' else 0
    for count in (128,98,11):
        values,diag=d.step(m,w,o,task,ds,Target(),np.arange(count)%33,3,batch,3.,ctx,diagnose=count==11)
        assert np.isfinite(list(values.values())).all()
        assert all(p.dtype==torch.float32 for p in m.parameters())
    assert diag['ce_norm']>0
    del m,w,o;d.release_memory(ctx)


@pytest.mark.parametrize('task,requested,allocated',(('release',1,1),('release',1,2),('report',2,2),('report',2,4)))
def test_cpu_release_and_report_accept_only_expected_rounding(tmp_path,monkeypatch,task,requested,allocated):
    from hlt_classification.luka_embedding_bridge_hydra import execution
    s=execution_spec(tmp_path);site=s['cpu_site' if task=='release' else 'report_site'];monkeypatch.setenv('SLURM_JOB_ID','8123')
    for k in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'):monkeypatch.delenv(k,raising=False)
    row=f'JobId=8123 JobState=RUNNING Partition=compute Account=default QOS=normal NumNodes=1 NumTasks=1 CPUs/Task={requested} NumCPUs={allocated} ReqTRES=cpu={requested},mem={site["memory_mb"]}M,node=1 AllocTRES=cpu={allocated},mem={site["memory_mb"]}M,node=1'
    def run(cmd,**kwargs):return SimpleNamespace(stdout='ClusterName = hydra\n' if 'config'in cmd else row)
    monkeypatch.setattr(execution.subprocess,'run',run)
    assert execution.cpu_allocation(s,task)=='8123'
    row+=' gres/gpu=1';row=row.replace('AllocTRES=','AllocTRES=gres/gpu=1,')
    with pytest.raises(PermissionError):execution.cpu_allocation(s,task)


def test_source_guard_requires_all_new_files_and_unchanged_donors(monkeypatch):
    from pathlib import Path
    from hlt_classification.luka_embedding_bridge_hydra import campaign
    root=Path(__file__).resolve().parents[1];assert all((root/p).is_file() for p in campaign.REQUIRED)
    tracked=set(campaign.REQUIRED);changes=['A\t'+p for p in campaign.RUNTIME]
    monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='s'*64))
    def git(args,**kwargs):return SimpleNamespace(stdout='\n'.join(tracked) if 'ls-files'in args else '\n'.join(changes) if 'diff'in args else '',returncode=0)
    monkeypatch.setattr(campaign.subprocess,'run',git)
    assert campaign.current(root,'a'*40)==dict(content_hash='s'*64)
    changes[:]=['M\t'+p for p in campaign.RUNTIME]
    assert campaign.current(root,'a'*40)==dict(content_hash='s'*64)
    changes[0]='D\t'+next(iter(campaign.RUNTIME))
    with pytest.raises(ValueError):campaign.current(root,'a'*40)
    changes[:]=['M\t'+p for p in campaign.RUNTIME]
    changes.append('M\tsrc/hlt_classification/luka_u100_embedding/training.py')
    with pytest.raises(ValueError):campaign.current(root,'a'*40)
    changes.pop();tracked.remove('tests/test_luka_embedding_bridge_hydra.py')
    with pytest.raises(ValueError):campaign.current(root,'a'*40)


def test_historical_baseline_never_substitutes_for_missing_fresh_control(tmp_path,monkeypatch,capsys):
    from hlt_classification.luka_embedding_bridge_hydra import reporting
    ds=synthetic_dataset();s=ds.spec;s.update(root=str(tmp_path),original50k_rows=22)
    labels=ds.view('validation','D000').labels;ids=ds.view('validation','D000').identities
    prob=np.full((22,11),.01,np.float32);prob[np.arange(22),labels]=.9
    worse=prob.copy();worse[:11]=np.roll(worse[:11],1,axis=1)
    arrays={k:dict(labels=labels,identities=ids,probabilities=p) for k,p in (('M0HLT',worse),('OFFLINE',prob),('C0_BF16_HISTORICAL',prob))}
    monkeypatch.setattr(reporting,'references',lambda s:(SimpleNamespace(path=lambda p:p),arrays))
    monkeypatch.setattr(reporting,'load_npz_arrays',lambda p:dict(indices=np.arange(22)))
    monkeypatch.setattr(reporting,'states',lambda s:{})
    result=reporting.results(s)
    assert all(row['delta_C0'] is None for row in result['rows'])
    assert result['pending']==list(r.FITS) and result['aliases']==r.ALIASES
    assert 'C0_BF16_HISTORICAL' in capsys.readouterr().out


def test_optional_selective_references_are_authenticated_and_labelled(tmp_path,monkeypatch):
    from hlt_classification.luka_embedding_bridge_hydra import campaign,reporting
    from hlt_classification.luka_selective_teacher_hydra import contracts as old_contract
    from hlt_classification.luka_capacity_hydra import reuse
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.data.cache_contracts import write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
    old=old_contract.artifact('SPEC',root=str(tmp_path),anchor={'proof':'old'},bundle_root='original')
    write_immutable_json(tmp_path/'campaign_spec.json',old)
    for task in ('S2IT','S6'):
        folder=tmp_path/task/'fit';folder.mkdir(parents=True)
        write_immutable_json(folder/'training_report.json',dict(selected_epoch=16))
        atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(probabilities=np.ones((2,11),np.float32)/11)))
    monkeypatch.setattr(reuse,'original',lambda path:old)
    value=campaign.historical_registration(tmp_path/'campaign_spec.json',parent=old)
    campaign.check_historical(value,old)
    monkeypatch.setattr(reporting,'historical',lambda s:(None,dict(C0_U100_HYDRA={})))
    _,arrays=reporting.references(dict(old,historical_selective=value))
    assert set(arrays)=={'C0_BF16_HISTORICAL','S2IT_BF16_HISTORICAL','S6_BF16_HISTORICAL'}
    changed=copy.deepcopy(value);changed['files']['S6']['training_report.json']=value['files']['S2IT']['training_report.json']
    with pytest.raises(ValueError,match='ownership'):campaign.check_historical(changed,old)
    with open(value['files']['S6']['validation_predictions.npz']['path'],'ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError):campaign.check_historical(value,old)


def test_transfer_rejects_changed_original_assignment_even_with_new_shard_hash(tmp_path):
    import hashlib
    from hlt_classification.luka_embedding_bridge_hydra import transfer
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes
    ids=synthetic_dataset().view('train','D000').identities[:2]
    a=dict(identities=ids,labels=np.arange(2,dtype=np.int64),mapping=np.tile(np.arange(12,dtype=np.int32),2),
        HLT_offsets=np.array([0,12,24],np.int64),OFFLINE_offsets=np.array([0,12,24],np.int64),
        HLT_values=np.concatenate([physical(),physical()]),OFFLINE_values=np.concatenate([physical(),physical()]))
    assignment=deterministic_npz_bytes(dict(identities=ids,labels=a['labels'],mapping=a['mapping'],offsets=a['HLT_offsets']))
    atomic_publish_bytes(tmp_path/'valid.npz',deterministic_npz_bytes(a))
    t=object.__new__(transfer.Transfer);t.root=tmp_path
    row=dict(role='train',rows=2,file=file_ref(tmp_path/'valid.npz',root=tmp_path),assignment=dict(bytes=len(assignment),sha256=hashlib.sha256(assignment).hexdigest()))
    t.manifest=dict(shards=[row]);assert len(list(t.shards('train')))==1
    a['mapping'][0]=-1;atomic_publish_bytes(tmp_path/'changed.npz',deterministic_npz_bytes(a));row['file']=file_ref(tmp_path/'changed.npz',root=tmp_path)
    with pytest.raises(ValueError,match='Original assignment'):list(t.shards('train'))
    with pytest.raises(PermissionError):list(t.shards('test'))


def test_measured_cpu_paths_cover_all_recipes_and_enforce_bounds(tmp_path,monkeypatch):
    from hlt_classification.luka_embedding_bridge_hydra import reporting,training,resources
    from hlt_classification.data.cache_contracts import with_content_hash
    ds=synthetic_dataset();s=ds.spec;s.update(execution_spec(tmp_path));s['counts']=dict(train=33,validation=22);s['original50k_rows']=22
    base=dict(identities=ds.view('validation','D000').identities,labels=ds.view('validation','D000').labels,probabilities=np.ones((22,11),np.float32)/11)
    monkeypatch.setattr(reporting,'references',lambda s:(SimpleNamespace(path=lambda p:p),{'M0HLT':base}))
    monkeypatch.setattr(reporting,'load_npz_arrays',lambda p:dict(indices=np.arange(22)))
    tasks=[];monkeypatch.setattr(training,'validate_fit',lambda folder,s,t,technical:tasks.append(t))
    measured=reporting.measure_cpu(s);assert tasks==list(r.FITS) and measured['full_metric_calls']==102
    value=c.artifact('CPU_RESOURCES',parents=dict(spec=s['content_hash']),release_seconds=60.,release_peak_bytes=2**30,report_peak_bytes=2**30,**measured)
    assert resources.check_cpu(s,value)==value
    for key,val in (('release_peak_bytes',4*2**30),('report_peak_bytes',8*2**30),('release_seconds',3600.)):
        bad=dict(value);bad.pop('content_hash');bad[key]=val
        with pytest.raises((MemoryError,RuntimeError)):resources.check_cpu(s,with_content_hash(bad))


def test_selection_uses_assistant_error_and_teacher_condition_mean():
    from hlt_classification.luka_embedding_bridge_hydra.training import selection
    def row(error,auc,step=10):return dict(u100_validation_error=error,validation=dict(macro_ovr_auc=auc),update=step,
        components=dict(a=dict(macro_ovr_auc=auc,cross_entropy=1.),b=dict(macro_ovr_auc=.8,cross_entropy=1.)))
    assert selection('LS_D050',row(.1,.6))>selection('LS_D050',row(.2,.9))
    assert selection('LS_D050',row(.1,.8,5))>selection('LS_D050',row(.1,.8,10))
    assert selection('TR',row(.1,.8))>selection('TR',row(.1,.7))


def test_scientific_fit_rejects_single_rank_and_partial_epoch_override(tmp_path):
    from hlt_classification.luka_embedding_bridge_hydra import training,distributed as d,banks
    ds=synthetic_dataset();target=banks.Targets(ds.spec,'C0',ds)
    with pytest.raises(PermissionError):training.fit(tmp_path/'wrong',ds.spec,'C0',ds,target,d.Context(0,1,torch.device('cpu')))


def _two_cuda(rank,root,init_method):
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from hlt_classification.luka_embedding_bridge_hydra import distributed as d,banks,training,resources,qualification
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    dist.init_process_group('nccl',init_method=init_method,rank=rank,world_size=2,timeout=timedelta(minutes=10))
    try:
        ds=synthetic_dataset();s=ds.spec;s['root']=root;ctx=d.Context(rank,2,torch.device('cuda',rank));folder=Path(root)/'setup'
        targets=lambda task:resources.technical_target(s,task,ds,folder)
        for task in r.PRODUCERS:
            # Registry producer order is topological within each ladder.
            m,_=training.fit(folder/('mini_'+task),s,task,ds,targets(task),ctx,technical=True,val_diagnostics=False)
            banks.export(folder/('mini_'+task)/'bank',s,task,m,ds,ctx,file_ref(folder/('mini_'+task)/'selected_full.pt'),technical=True)
            del m;d.release_memory(ctx)
        qualification.qualify(s,ds,targets,ctx)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<2 or not torch.distributed.is_nccl_available(),reason='Two CUDA GPUs with NCCL required')
def test_real_two_gpu_producer_pipeline_and_all_family_syncbn_parity(tmp_path):
    (tmp_path/'setup').mkdir()
    torch.multiprocessing.spawn(_two_cuda,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=2,join=True)


def physical(n=12):
    a = np.zeros((n,14),np.float64)
    theta = np.linspace(-2.8,2.8,n); eta = .3*np.cos(theta); phi = .3*np.sin(theta)
    pt = 1+np.arange(n)/10
    a[:,:4] = np.column_stack((pt*np.cos(phi),pt*np.sin(phi),pt*np.sinh(eta),pt*np.cosh(eta)))
    a[:,4] = 1; a[:,5] = 0; a[:,6:10] = [.01,.02,.003,.004]; a[:,10:] = 1
    return a


def small(monkeypatch):
    from hlt_classification.jetclass2_delphes import model as native
    original = native.model_config
    monkeypatch.setattr(native,'model_config',lambda:dict(original(),num_layers=1,num_cls_layers=1))


def batch(n=11):
    return {k:torch.from_numpy(v) for k,v in regions.padded([regions.encode(physical()) for _ in range(n)],
                               np.arange(n)%11).items()}


def test_registry_exact_count_order_and_roles():
    assert len(r.FITS) == len(set(r.FITS)) == 30
    assert len(r.HLT_FITS) == 22 and len(r.ASSISTANTS) == 6
    seen = set()
    for t in r.FITS:
        assert set(r.dependencies(t)) <= seen
        seen.add(t)
    assert r.cell('LS_D050')['parent'] is None
    assert r.cell('LL_D050')['parent'] == 'LL_D075'
    assert r.ALIASES == {'R0':'C0','P0':'C0'}
    assert c.validate(c.recipe(),'RECIPE')
    for group in ('R','P'):
        assert [r.cell(group+str(i))['weight'] for i in range(1,10)] == [1,1,1,.3,1,3,10,1,1]
    with pytest.raises(ValueError): r.cell('R0')


def test_report_charges_all_chain_producers_without_hiding_incomplete_costs():
    from hlt_classification.luka_embedding_bridge_hydra.reporting import ancestor_costs
    costs={t:dict(worker_seconds=3600.,gpu_hours=2.) for t in r.FITS}
    result=ancestor_costs(costs)
    assert result['LS_D000']['total_chain_gpu_hours']==4.
    assert result['LM_D000']['total_chain_gpu_hours']==6.
    assert result['LL_D000']['total_chain_gpu_hours']==8.
    assert result['R5']['total_chain_gpu_hours']==4.
    assert result['P5']['total_chain_gpu_hours']==4.
    assert result['C0']['total_chain_gpu_hours']==2.
    del costs['LL_D075']
    assert ancestor_costs(costs)['LL_D000']['total_chain_gpu_hours'] is None


def test_condition_cycles_are_balanced_and_deterministic():
    for regional in (False,True):
        for epoch in range(1,31):
            a = [r.teacher_condition(epoch,i,regional=regional) for i in range(12)]
            assert len(set(a[:6])) == 6 and a[:6] == a[6:]


def test_response_algebra_and_scales():
    a = np.random.default_rng(12).normal(size=(15,6,128)).astype(np.float32)
    d = response.differences(a)
    np.testing.assert_array_equal(d['J'],d['IT']-d['I']-d['T'])
    scale = response.scales([a[:5],a[5:]])
    y = response.targets(a,scale,['I','T','J'])
    np.testing.assert_allclose(y[:,2],d['J']/scale['IT'])
    absolute = response.targets(a,scale,['I'],absolute=True)
    np.testing.assert_allclose(absolute[:,0],a[:,2]/scale['I'])
    same = np.repeat(a[:,:1],6,axis=1)
    assert not response.targets(same,response.scales([same]),['K','I','T','IT','J']).any()
    assert response.scales([same]) == dict(K=.1,I=.1,T=.1,IT=.1)
    mean,sd,n = response.moments([a[:5],a[5:]])
    np.testing.assert_allclose(mean,a.astype(float).mean((0,1)),atol=1e-14)
    np.testing.assert_allclose(sd,a.astype(float).std((0,1)),atol=1e-14)
    assert n == 90


def test_partitions_cover_nest_permute_and_crop_has_no_outside_features():
    h = physical(); source = physical(31); p = regions.partitions(h,source)
    assert np.array_equal(p['fine']//2,p['medium'])
    assert np.array_equal(p['medium']//2,p['coarse'])
    order = np.arange(len(h))[::-1]; perm = np.arange(len(source))[::-1]
    assert regions.frame(h) == regions.frame(h[order])
    for scale in r.SCALES:
        np.testing.assert_array_equal(regions.partitions(h[order],source[perm])[scale],p[scale][perm])
    query = p['fine'][0]; selected = source[p['fine'] == query]
    a = regions.encode(selected)
    changed = source.copy(); changed[p['fine'] != query,6:10] *= 20
    b = regions.encode(changed[p['fine'] == query])
    np.testing.assert_array_equal(a.features,b.features)
    assert regions.descriptors(r.SCALES)[0].shape == (14,9)
    assert regions.frame(np.empty((0,14),np.float64)) == (0.,0.,.05)


def test_regional_equal_scale_jet_reduction_and_empty_targets():
    pred = torch.zeros(2,14,128,requires_grad=True)
    target = torch.zeros_like(pred); target.data[:,2:6] = 1; target.data[:,6:] = 2
    valid = torch.ones(2,14,dtype=torch.bool); valid[1] = False
    value = losses.regional(pred,target,valid,[list(range(2)),list(range(2,6)),list(range(6,14))])
    assert float(value) == pytest.approx((0+.5+1.5)/3/2)
    value.backward(); assert not pred.grad[1].any()


@pytest.mark.parametrize('task',['C0','LS_D050','LM_D033','LS_D000','R5','R9','P5','P8','P9','TR','TP'])
def test_native_model_targets_and_inference(monkeypatch,task):
    small(monkeypatch); s=spec(); m=model.new_model(s,task); raw=batch(11); y=raw.pop('labels')
    m.eval(); out=m(**raw)
    native=model.Native().eval(); native.load_state_dict(m.native_state(),strict=True)
    torch.testing.assert_close(out['logits'],native(**raw),rtol=0,atol=0)
    target=dict(u100=torch.randn(11,128))
    if 'extra' in out:
        target['extra']=torch.randn_like(out['extra'])
    if r.cell(task)['role']=='assistant' and r.cell(task)['parent']:
        target['extra']=torch.randn(11,128)
    if task.startswith('P'):
        q=len(out['extra'][0]);target.update(valid=torch.ones(11,q,dtype=torch.bool),
            groups=[list(range(2)),list(range(2,6)),list(range(6,14))] if q==14 else [[0]])
    if task=='TR':target.update(condition='KIT',mean=torch.zeros(128),scale=torch.ones(128))
    if task=='TP':
        target['valid']=torch.ones(11,2,dtype=torch.bool)
        out=m(**raw,crops=[raw,raw],valid=target['valid'])
    total,parts=losses.objective(task,out,y,target,3.)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.shared_parameters(m))
    assert float(total)==pytest.approx(float(parts['ce']+parts['weighted_embedding']+parts['weighted_extra']))


def test_c0_is_exact_original_u100_model_update(monkeypatch):
    from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
    from hlt_classification.luka_u100_contrastive.training import optimizer
    from hlt_classification.luka_u100_embedding.training import objective
    small(monkeypatch); raw=batch();y=raw.pop('labels');z=torch.randn(11,128)
    torch.manual_seed(23);old=AuxiliaryParT();new=model.new_model(spec(),'C0')
    model.disable_dropout(old);model.disable_dropout(new)
    old.train();old.capture=True;new.train();oo=optimizer(old);no=optimizer(new)
    loss,_,_=objective(old(**raw),y,old.prediction,z,coefficient=5.)
    loss.backward();oo.step()
    value,_=losses.objective('C0',new(**raw),y,dict(u100=z),1.)
    value.backward();no.step()
    assert torch.equal(loss,value)
    for k,v in old.state_dict().items(): torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
    for a,b in zip(oo.state.values(),no.state.values()):
        for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def test_assistant_anchor_weight_and_detached_parent():
    y=torch.arange(11);p=torch.randn(11,128,requires_grad=True);z=torch.randn_like(p);prev=torch.randn_like(p,requires_grad=True)
    out=dict(logits=torch.randn(11,11,requires_grad=True),u100=p)
    total,parts=losses.objective('LM_D033',out,y,dict(u100=z,extra=prev),3.)
    assert torch.equal(parts['weighted_embedding'],2.5*parts['embedding'])
    assert torch.equal(parts['weighted_extra'],2.5*parts['extra'])
    total.backward();assert prev.grad is None
