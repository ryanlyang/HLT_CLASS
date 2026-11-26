"""New-data Oscar main-KD ladder: science, recovery, site and publication tests."""
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
from hlt_classification.luka_sixclass_wide_b256_oscar import model
from hlt_classification.luka_sixclass_wide_b256_oscar import contrastive
from hlt_classification.luka_sixclass_wide_b256_oscar.sampling import BalancedSampler
from hlt_classification.luka_sixclass_wide_b256_oscar import (
    contracts as c,population,views,campaign,training,banks,artifacts,worker,startup,reporting,inspect)
from test_luka_fullstat_balanced_ladder import threads


def spec_fixture(root,monkeypatch=None):
    pop=c.artifact('POPULATION',parents=dict(foundation=c.PARENT_HASH,role_population=c.ROLE_POPULATION_HASH),policy=c.population_policy(),
        counts=dict(c.COUNTS,unused=c.UNUSED),outside_counts=c.OUTSIDE)
    if monkeypatch is not None:monkeypatch.setattr(c,'POPULATION_HASH',pop['content_hash'])
    return c.artifact('SPEC',root=str(root),dataset='/dataset',project='/project',commit='f'*40,source={},
        environment_dir='/env',environment={},gpu='l40s',foundation_producer={},population=pop,
        population_root=str(root/'population'),recipe=c.recipe(),nodes={s:c.node(s) for s in c.STAGES},
        counts=c.COUNTS,cpus=6,memory_mb=192000,minutes={t:2880 for t in c.tasks()},
        preparation_workers=4,candidate_workers=3,tasks=c.tasks(),
        storage=dict(declared_available_gib=18,new_results_budget_bytes=c.RESULT_GIB*2**30),
        execution_site=c.SITE,transfer_manifest_sha256=c.TRANSFER_HASH,backend=campaign.backend())


@pytest.mark.parametrize('coordinate',['OFFLINE','D000','CONCAT','U100','D050'])
def test_root_reader_has_full_support_and_never_crosses_test_or_unused(tmp_path,monkeypatch,coordinate):
    from test_luka_fullstat import write_root
    path=tmp_path/'raw/s.root';write_root(path,[161]*8)
    r=dict(path='raw/s.root',sha256=sha256_file(path),entries=8,tree_key='tree',category='signal')
    # These operational codes are produced from role masks, not parent split codes.
    codes=np.array([0,1,255,255,2,255,0,1],np.uint8)
    a=dict(labels=np.zeros(8,np.int8),jet_nparticles=np.full(8,3),hlt_jet_nparticles=np.full(8,2))
    original=uproot.behaviors.TBranch.HasBranches.arrays;reads=[]
    def guarded(tree,expressions,*args,**kwargs):
        lo,hi=kwargs['entry_start'],kwargs['entry_stop'];assert np.all(codes[lo:hi]!=255)
        if coordinate=='D000':assert all(k.startswith('hlt_') for k in expressions)
        reads.append((lo,hi));return original(tree,expressions,*args,**kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'arrays',guarded)
    blocks,mapping=views.work((tmp_path,r,0,0,a,codes,coordinate,'b'*64))
    assert reads and [len(blocks[k].labels) for k in ('train','validation','outside_validation')]==[2,2,1]
    for block in blocks.values():assert np.all(np.diff(block.offsets)==(5 if coordinate=='CONCAT' else 3 if coordinate=='OFFLINE' else 2))
    if coordinate in ('U100','D050'):
        _,other=views.work((tmp_path,r,0,0,a,codes,'D050' if coordinate=='U100' else 'U100','b'*64))
        assert mapping==other


def test_retained_metadata_reader_preserves_exact_arrays_and_closes(tmp_path,monkeypatch):
    from test_luka_fullstat import write_root
    from hlt_classification.luka_fullstat_fine_tail.views import digest
    path=tmp_path/'raw/s.root';write_root(path,[161]*4)
    r=dict(path='raw/s.root',sha256=sha256_file(path),entries=4,tree_key='tree',category='signal')
    a=dict(labels=np.zeros(4,np.int8),jet_nparticles=np.full(4,3),hlt_jet_nparticles=np.full(4,2))
    arg=(tmp_path,r,0,0,a,np.array([0,1,255,2],np.uint8),'D050','b'*64)
    reference,mapping=views.work(arg)
    original=uproot.open;calls=[]
    def opened(*a,**kw):calls.append(a);assert kw['array_cache'] is None;return original(*a,**kw)
    monkeypatch.setattr(uproot,'open',opened)
    try:
        for _ in range(2):
            actual,proof=views.work(arg,retain_reader=True)
            assert proof==mapping
            assert {k:digest(v) for k,v in actual.items()}=={k:digest(v) for k,v in reference.items()}
        assert len(calls)==1
    finally:views.close_reader()
    assert views._OPEN is None


def test_oscar_dag_journals_and_no_double_submission(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path,monkeypatch);campaign.check(s,live=False)
    monkeypatch.setattr(campaign,'locations',lambda *a:None)
    monkeypatch.setattr(campaign,'runtime',lambda:{})
    s['environment_dir']=__import__('sys').prefix;s=with_content_hash(s)
    monkeypatch.setattr(campaign,'source',lambda *a:{})
    monkeypatch.setattr(campaign,'storage',lambda *a,**kw:{})
    monkeypatch.setattr(campaign,'site',lambda *a:None)
    calls=[]
    def fake(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(100+len(calls)),stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    plan=campaign.queue(s);assert plan['scientific_fits']==6 and plan['preflight_jobs']==0 and calls==[]
    campaign.queue(s,execute=True,authorization=c.AUTH)
    assert len(campaign.accepted_jobs(s))==6
    assert all(not any(x.startswith('--dependency') for x in row) for row in calls[:3])
    assert '--dependency=afterok:101:103:102' in calls[3]
    assert '--dependency=afterok:101:103:104:102' in calls[4]
    assert '--dependency=afterok:101:103:104:105' in calls[5]
    assert all('--partition=gpu' in row and '--gres=gpu:l40s:1' in row for row in calls)
    with pytest.raises(FileExistsError):campaign.queue(s,execute=True,authorization=c.AUTH)
    assert len(calls)==6


def test_ambiguous_submission_is_not_retried(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path);monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(campaign,'storage',lambda *a,**kw:{});monkeypatch.setattr(campaign,'site',lambda *a:None)
    calls=[]
    def fake(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout='ambiguous',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    with pytest.raises(RuntimeError):campaign.queue(s,execute=True,authorization=c.AUTH)
    with pytest.raises(ValueError,match='Ambiguous'):campaign.queue(s,execute=True,authorization=c.AUTH,resume=True)
    assert len(calls)==1


@pytest.mark.parametrize('field,value',[('counts',{}),('tasks',{}),('recipe',{}),('gpu','1080'),('memory_mb',1),('final_test_accessed',True)])
def test_rehashed_semantic_changes_rejected(tmp_path,monkeypatch,field,value):
    s=spec_fixture(tmp_path,monkeypatch);campaign.check(s,live=False);s[field]=value
    with pytest.raises(ValueError):campaign.check(with_content_hash(s),live=False)


def test_primary_and_outside_metrics_do_not_reselect():
    y=np.tile(np.arange(6),2);p=np.full((12,6),.001);p[np.arange(12),y]=.99;p/=p.sum(1,keepdims=True)
    data=SimpleNamespace(labels=y,__len__=lambda s:12)
    class Cache:
        labels=y
        def __len__(self):return 12
    a=dict(validation_pt=np.full(12,600.),outside_validation_pt=np.full(12,2600.))
    v=cache(role='validation');o=deepcopy(v);o.identities[:,20]=2
    m=reporting.metrics(v,p,o,np.roll(p,1,axis=1),a,dict(report='a'*64))
    assert m['selection_population']=='in_range' and m['checkpoint_reselection'] is False
    assert m['populations']['in_range']['summary']['accuracy']==1
    assert m['populations']['outside_range']['summary']['accuracy']==0
    assert all(x['status']=='censored' for x in m['populations']['in_range']['R50'].values())
    assert reporting.recovery(.7,.6,.8)==pytest.approx(50)
    a['validation_pt'][0]=2600
    with pytest.raises(ValueError):reporting.metrics(Cache(),p,Cache(),p,a,{})


def worker_fixture(tmp_path,monkeypatch):
    tiny(monkeypatch);monkeypatch.setattr(c,'DRAWS',24)
    monkeypatch.setattr(c,'COUNTS',dict(train=144,validation=12,final_test=6))
    monkeypatch.setattr(c,'OUTSIDE',dict(c.OUTSIDE,validation=12))
    tr=cache(coordinate='OFFLINE',rows=144);val=cache(role='validation',coordinate='OFFLINE',rows=12)
    val.identities[:,20]=1;outside=deepcopy(val);outside.identities[:,20]=2
    sampler=BalancedSampler(tr.labels,250+100*(np.arange(len(tr))//6))
    s=spec_fixture(tmp_path);write_immutable_json(tmp_path/'campaign_spec.json',s)
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


def test_six_worker_publications_banks_and_fresh_anchor_results(tmp_path,monkeypatch):
    s,tr,val,sampler,accepted=worker_fixture(tmp_path,monkeypatch);exposures=[]
    for stage in c.STAGES:
        receipt=worker.run(s,'train_'+stage,device='cpu');complete=artifacts.completed(s,stage)
        assert receipt==complete['receipt'] and complete['report']['passes']==20
        exposures.append([r['sampling'] for r in complete['report']['validation_history']])
        state=torch.load(receipt['files']['checkpoint']['path'],map_location='cpu',weights_only=True)
        assert not any(k.startswith(('predictors.','logit_heads.','matching.')) for k in state)
        if stage!='DE000':assert banks.Bank(complete['folder']/'bank',tr,stage,complete['refs'],s).targets.shape==(144,128)
    assert all(e==exposures[0] for e in exposures)
    result=inspect.results(s);assert set(result['models'])==set(c.STAGES)
    assert result['anchors']=='new_balanced_M0HLT_OFFLINE_on_identical_subsets'
    path=tmp_path/'train_DE000/result.json';r=load_json(path);r['fit_parents']['teacher_UE100']='x'
    path.write_text(json.dumps(with_content_hash(r)))
    with pytest.raises(ValueError,match='Teacher lineage'):artifacts.completed(s,'DE000')


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


def test_resume_refuses_live_jobs_and_queues_only_unpublished(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path)
    jobs=[dict(task='train_'+stage,job=str(100+i)) for i,stage in enumerate(c.STAGES)]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda s:jobs)
    def live(job):raise ValueError('not confirmed terminal')
    monkeypatch.setattr(campaign,'terminal',live)
    with pytest.raises(ValueError):campaign.remaining(s,True)
    monkeypatch.setattr(campaign,'terminal',lambda job:[job,'COMPLETED','0:0'])
    for stage in c.STAGES[:4]:
        folder=tmp_path/('train_'+stage);folder.mkdir();(folder/'result.json').write_text('{}')
    monkeypatch.setattr(artifacts,'completed',lambda s,stage:dict(receipt=dict(job_id=str(100+c.STAGES.index(stage)),content_hash=stage)))
    tasks,done=campaign.remaining(s,True)
    assert tasks==['train_DE050','train_DE000']
    rows=campaign.commands(s,tasks);assert rows[0]['dependencies']==[] and rows[1]['dependencies']==['train_DE050']


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
    launch=(root/'scripts/launch_luka_sixclass_wide_b256_oscar.sh').read_text()
    worker_text=(root/'sbatch/run_luka_sixclass_wide_b256_oscar.sh').read_text()
    assert 'nohup bash' in launch and '</dev/null &' in launch and 'exec tail' in launch
    assert '--authorize '+c.AUTH in launch
    assert 'PYTHONNOUSERSITE=1' in (root/'sbatch/luka_sixclass_wide_b256_oscar_common.sh').read_text() and 'luka_sixclass_wide_b256_oscar_common.sh' in worker_text
    assert '${PROJECT_DIR}/scripts/luka_sixclass_wide_b256_oscar.py' in worker_text


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


@pytest.mark.parametrize('task,minutes,limit',[('train_OFFLINE',2880,'2-00:00:00'),('train_CONCAT',4320,'3-00:00:00')])
def test_exact_allocation_and_runtime_binding(tmp_path,monkeypatch,task,minutes,limit):
    s=spec_fixture(tmp_path);s['environment_dir']=sys_prefix=__import__('sys').prefix
    s['minutes'][task]=minutes
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('LD_LIBRARY_PATH',str(Path(sys_prefix)/'lib'))
    monkeypatch.setattr(campaign,'binding',lambda *a:None);monkeypatch.setattr(campaign,'runtime',lambda:{})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'NVIDIA L40S')
    def response(args,**kw):
        out='ClusterName = slurmctld\n' if args[-1]=='config' else (
            f'JobId=123 JobState=RUNNING Account=default QOS=norm-gpu Partition=gpu JobName=luka6b256_{task} '
            f'Comment={s["content_hash"]}:{task} WorkDir=/project NumCPUs=6 NumNodes=1 NumTasks=1 CPUs/Task=6 '
            f'AllocTRES=cpu=6,mem=187.5G,node=1,gres/gpu=1 TimeLimit={limit}')
        return SimpleNamespace(stdout=out)
    monkeypatch.setattr(subprocess,'run',response)
    assert worker.allocation(s,task)=='123'
    s['minutes'][task]=minutes-1
    with pytest.raises(PermissionError,match='Walltime'):worker.allocation(s,task)
    s['minutes'][task]=minutes
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'GeForce GTX 1080 Ti')
    with pytest.raises(PermissionError):worker.allocation(s,task)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA integrated qualification')
@pytest.mark.parametrize('stage',['OFFLINE','M0HLT','CONCAT','UE100','DE050','DE000'])
def test_real_weaver_integrated_qualification(stage,tmp_path,monkeypatch):
    import time
    n=5400;tr=cache(coordinate=c.COORDINATES[stage],rows=n)
    val=cache(role='validation',coordinate=c.COORDINATES[stage],rows=12)
    sampler=BalancedSampler(tr.labels,250+100*((np.arange(n)//6)%24))
    rng=np.random.default_rng(71)
    targets={t:SimpleNamespace(targets=rng.normal(size=(n,128)).astype(np.float32),
        probabilities=np.full((n,6),1/6,np.float32)) for t in c.TEACHERS[stage]}
    matching=None
    if stage=='DE000':
        keys=targets['OFFLINE'].targets.copy();keys/=np.linalg.norm(keys,axis=1,keepdims=True)
        matching=contrastive.Sampler(keys,tr.identities,tr.labels,device='cuda')
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    # Windows lacks Linux RSS. Actual GPU measurements, forwards, gradients,
    # installed-Weaver parity, candidate workers and reload remain real.
    monkeypatch.setattr(startup,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    s=spec_fixture(tmp_path)
    proof=startup.run(s,stage,tr,val,targets,sampler,matching,'cuda',tmp_path,time.monotonic(),remaining_epochs=0,outside=val)
    assert proof['installed_weaver_parity'] and proof['integrated_two_epoch_miniature'] and proof['probe_discarded']
    mini=load_json(tmp_path/'startup_diagnostics/miniature.json')
    assert len(mini['history'])==2 and mini['checkpoint_reload_verified']
    assert proof['main_head']==c.main_policy(stage) and proof['classifier_objective_verified']


def test_six_class_recipe_is_distinct_and_roles_are_preserved():
    from hlt_classification.luka_new_balanced_hydra import population as oldpop
    assert population.extend is oldpop.extend
    assert c.ROLE_POPULATION_HASH == c.donor.POPULATION_HASH
    assert c.POPULATION_HASH != c.ROLE_POPULATION_HASH
    assert c.DRAWS == 2488148 and c.STEPS == 9720 and c.DRAWS % c.BATCH == 84
    assert c.recipe()['logit_head'] == [128,6] and c.recipe()['native_outputs'] == 6
    assert c.tasks() == c.donor.tasks()
    assert c.recipe()['batch_size']==256 and c.recipe()['export_batch_size']==128
    assert c.recipe()['peak_lr']==1e-4 and c.recipe()['floor_lr']==5e-6
    assert c.learning_rate(1)==5e-5 and c.learning_rate(2)==1e-4 and c.learning_rate(20)==5e-6
    assert c.COUNTS['validation']+c.OUTSIDE['validation']==566032
    assert c.COUNTS['final_test']+c.OUTSIDE['final_test']==1432866
    assert not c.MAIN_TEACHERS and c.recipe()['ce_weight']==1
    for stage in c.STAGES:
        assert c.node(stage)['initialization_seed'] == c.donor.node(stage)['initialization_seed']
        assert c.main_policy(stage)['ce_weight']==1 and c.main_policy(stage)['kd_weight']==0
    with pytest.raises(ValueError): c.validate(c.donor.recipe(), 'RECIPE')


@pytest.mark.parametrize('stage',c.STAGES)
def test_exact_main_classifier_gradients_and_native_deployment(stage,monkeypatch):
    from torch.nn import functional as F
    from hlt_classification.luka_sixclass_wide_b256_oscar.model import forward
    tiny(monkeypatch)
    tr=cache(coordinate=c.COORDINATES[stage],rows=12);ix=np.array([1,1,2,3,4]);rng=np.random.default_rng(47)
    targets={t:SimpleNamespace(targets=rng.normal(size=(12,128)).astype(np.float32),
             probabilities=rng.dirichlet(np.full(6,.3),size=12).astype(np.float32)) for t in c.TEACHERS[stage]}
    keys=F.normalize(torch.randn(12,128),dim=-1)
    match=SimpleNamespace(candidate_keys=lambda indices:keys[indices]) if stage=='DE000' else None
    candidates=np.tile(np.arange(1025)%12,(len(ix),1));candidates[:,0]=ix
    net=model.new_model(stage).eval()
    loss,parts=model.objective(net,tr.batch(ix),ix,targets,update=6,steps=2,device='cpu',sampler=match,candidates=candidates)
    # Independent forward-KL expression, with distinct teacher probabilities.
    logits=forward(net,tr.batch(ix),'cpu',bf16=False)
    ce=F.cross_entropy(logits,torch.tensor(tr.labels[ix]))
    expected=ce
    if stage in c.MAIN_TEACHERS:
        q=torch.tensor(targets[c.MAIN_TEACHERS[stage]].probabilities[ix])
        kd=4*(q*(q.log()-F.log_softmax(logits/2,dim=-1))).sum(-1).mean()
        torch.testing.assert_close(parts['main_kd'],kd)
        expected=.25*ce+.75*kd
    else:assert 'main_kd' not in parts
    for t in c.TEACHERS[stage]:expected+=10/len(targets)*parts['embedding_'+t]+1/len(targets)*parts['kd_'+t]
    if match is not None:expected=expected+.3*parts['matching']
    torch.testing.assert_close(loss,expected)
    # Loss coefficients must reach the deployed FC, not just auxiliary heads.
    expected_grads=torch.autograd.grad(model.main_loss(parts,stage),tuple(net.mod.fc.parameters()),retain_graph=True)
    loss.backward()
    for p,g in zip(net.mod.fc.parameters(),expected_grads):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert all(p.grad is not None for p in net.parameters())
    native=net.deployed();net.capture=False
    assert not any(k.startswith(('predictors.','logit_heads.','matching.')) for k in native.state_dict())
    with torch.no_grad():torch.testing.assert_close(forward(net,tr.batch(ix),'cpu',bf16=False),forward(native,tr.batch(ix),'cpu',bf16=False),rtol=0,atol=0)
    if targets:
        with pytest.raises(ValueError):model.objective(net,tr.batch(ix),ix,{},update=6,steps=2,device='cpu',sampler=match,candidates=candidates)


def test_storage_has_no_hydra_baseline_requirement(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=28*2**30))
    result=campaign.storage(tmp_path,18)
    assert result['required_free_bytes']==18*2**30 and 'natural_baseline' not in result
    with pytest.raises(OSError):campaign.storage(tmp_path,17)
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=12*2**30))
    with pytest.raises(OSError):campaign.storage(tmp_path,28)


def test_frozen_donor_manifest_matches_pinned_git_blobs():
    import hashlib
    project=Path(__file__).resolve().parents[1]
    manifest=load_json(project/'src/hlt_classification/luka_sixclass_wide_b256_oscar/donors.json')
    for name,digest in manifest['files'].items():
        blob=subprocess.check_output(['git','-C',str(project),'show',manifest['commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==digest
        assert (project/name).read_bytes().replace(b'\r\n',b'\n')==blob


@pytest.mark.parametrize('stage',['UE100','DE050','DE000'])
def test_student_continuation_preserves_all_heads_optimizer_rng(stage,tmp_path,monkeypatch):
    s,tr,val,sampler,_=worker_fixture(tmp_path,monkeypatch)
    tr.coordinate_name=val.coordinate_name=c.COORDINATES[stage]
    outside=deepcopy(val);outside.identities[:,20]=2
    rng=np.random.default_rng(79)
    targets={t:SimpleNamespace(identities=tr.identities,labels=tr.labels,
                targets=rng.normal(size=(len(tr),128)).astype(np.float32),
                probabilities=rng.dirichlet(np.ones(6),size=len(tr)).astype(np.float32)) for t in c.TEACHERS[stage]}
    matching=contrastive.Sampler(targets['OFFLINE'].targets,tr.identities,tr.labels) if stage=='DE000' else None
    saved=[]
    def save(epoch,row,state):saved.append((deepcopy(row),deepcopy(state)))
    expected,report,best=training.train(tr,val,banks=targets,sampler=sampler,stage=stage,parents={},device='cpu',save_epoch=save,matching=matching,outside=outside)
    from hlt_classification.jetclass2_delphes.runner import selection_key
    chosen=max(saved[:10],key=lambda pair:selection_key(pair[0]['validation'],pair[0]['update']))[1]
    resume=dict(state=saved[9][1],history=[x[0] for x in saved[:10]],best=chosen)
    resumed=[]
    actual,r2,b2=training.train(tr,val,banks=targets,sampler=sampler,stage=stage,parents={},device='cpu',
                               save_epoch=lambda epoch,*args:resumed.append(epoch),matching=matching,resume=resume,outside=outside)
    assert resumed==list(range(11,21))
    assert r2['selected_pass']==report['selected_pass']
    assert [x['validation'] for x in r2['validation_history']]==[x['validation'] for x in report['validation_history']]
    for key,value in expected.state_dict().items():torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)
    wrong=deepcopy(report['validation_history'][9]);wrong['main_head']['teacher']='CONCAT'
    with pytest.raises(ValueError,match='main-head'):training.check_row(wrong,10,stage,len(val),sampler)


def test_writer_lock_rejects_second_writer_and_releases_after_error(tmp_path):
    with pytest.raises(RuntimeError):
        with campaign.submission_lock(tmp_path):
            with pytest.raises(OSError):
                with campaign.submission_lock(tmp_path):pytest.fail('Concurrent writer acquired lock')
            raise RuntimeError('injected launcher exit')
    with campaign.submission_lock(tmp_path):pass


def test_partial_results_and_fast_progress_do_not_require_unfinished_ladder(tmp_path,monkeypatch,capsys):
    s,tr,val,sampler,_=worker_fixture(tmp_path,monkeypatch)
    worker.run(s,'train_OFFLINE',device='cpu')
    first=inspect.results(s)
    assert list(first['models'])==['OFFLINE'] and len(first['unpublished_stages'])==5
    assert first['models']['OFFLINE']['recovery_percent']['in_range'] is None
    worker.run(s,'train_M0HLT',device='cpu')
    # Progress is metadata-only; completed() would read checkpoint/prediction payloads.
    monkeypatch.setattr(artifacts,'completed',lambda *a,**kw:pytest.fail('Progress must not load payloads'))
    inspect.progress(s,'M0HLT')
    inspect.print_results(inspect.results(s))
    text=capsys.readouterr().out
    assert '20/20 epochs committed' in text and 'AUC rec.%' in text and 'Acc rec.%' in text
    assert 'Full AUC' in text and 'FULL VALIDATION' in text and 'QCD R30' in text


def test_walltime_accounts_for_batch256_all_validation_and_batch128_teacher_exports():
    import math
    val=math.ceil(c.COUNTS['validation']/256)+math.ceil(c.OUTSIDE['validation']/256)
    for stage in c.STAGES:
        export=0 if stage=='DE000' else 3.
        expected=math.ceil((10+3600+1.3*(20*9720*1.+21*val*2.+math.ceil(2488148/128)*export))/60)
        assert startup.estimate(stage,1.,2.,export,10)['conservative_minutes']==expected
    with pytest.raises(ValueError):startup.estimate('DE000',1.,2.,3.,0.)
    with pytest.raises(ValueError):startup.estimate('UE100',1.,2.,0.,0.)


@pytest.mark.parametrize('key,value',[('batch_size',128),('peak_lr',3e-4),('draws_per_epoch',3000000),
                                     ('export_batch_size',256),('main_head_supervision','direct_KD')])
def test_rehashed_recipe_tampering_fails_closed(tmp_path,monkeypatch,key,value):
    s=spec_fixture(tmp_path,monkeypatch);s['recipe'][key]=value
    s['recipe']=with_content_hash(s['recipe'])
    with pytest.raises(ValueError):campaign.check(with_content_hash(s),live=False)


def test_full_validation_epochs_survive_restart_and_cannot_change_primary_selection(tmp_path,monkeypatch):
    s,*_=worker_fixture(tmp_path,monkeypatch)
    result=worker.run(s,'train_OFFLINE',device='cpu')
    report=load_json(result['files']['report']['path']);history=report['validation_history']
    from hlt_classification.jetclass2_delphes.runner import selection_key
    chosen=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    assert report['selected_pass']==chosen['pass_number']
    assert all(r['full_validation']['rows']==24 and r['validation']['rows']==12 for r in history)
    # A better full-range diagnostic cannot alter the registered selection.
    other=1 if chosen['pass_number']!=1 else 2
    history[other-1]['full_validation']['macro_ovr_auc']=1.
    training.validate_report(with_content_hash(report),'OFFLINE',result['fit_parents'],12)
    bad=deepcopy(history[0]);bad['full_validation']['rows']=12
    with pytest.raises(ValueError,match='Full validation'):training.check_row(bad,1,'OFFLINE',12)


def test_oscar_test_only_precedes_submission_and_rejects_other_cluster(monkeypatch,tmp_path):
    s=spec_fixture(tmp_path);rows=campaign.commands(s,list(c.tasks()));calls=[]
    def run(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='ClusterName = slurmctld\n' if args[0]=='scontrol' else '',stderr='')
    monkeypatch.setattr(subprocess,'run',run);campaign.site(rows)
    assert len(calls)==7
    assert all('--test-only' in x and '--gres=gpu:l40s:1' in x and '--partition=gpu' in x and '--qos=norm-gpu' in x for x in calls[1:])
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='ClusterName = hydra\n'))
    with pytest.raises(PermissionError):campaign.site(rows)


def test_real_bash_detached_launch_finishes_after_tail_exits(tmp_path):
    import os,sys,time
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def sp(path):
        p=Path(path).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True);(project/'sbatch').mkdir()
    fake=tmp_path/'env';(fake/'bin').mkdir(parents=True)
    script=project/'scripts/launch_luka_sixclass_wide_b256_oscar.sh'
    script.write_bytes((Path(__file__).resolve().parents[1]/'scripts'/script.name).read_bytes())
    # Stand-in runtime activation; test the actual detached shell lifecycle.
    (project/'sbatch/luka_sixclass_wide_b256_oscar_common.sh').write_bytes(b':\n')
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'calls'
    for path,body in {bindir/'git':'printf "%040d\\n" 1\n',bindir/'tail':'echo TAIL_ATTACHED\n',
                      fake/'bin/python':'sleep 1\nprintf "%s\\n" "$*" >> "$FIXTURE_CALLS"\n'}.items():
        path.write_bytes(('#!/usr/bin/env bash\n'+body).encode());path.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    data=tmp_path/'data';data.mkdir();(data/'transfer_complete.json').write_text('{}');root=tmp_path/'result'
    env=dict(os.environ,BASH_ENV=sp(envfile),FIXTURE_BIN=sp(bindir),FIXTURE_CALLS=sp(calls))
    args=[bash,str(script),'detach',sp(project),sp(data),sp(root),'0'*39+'1','18',sp(fake),'--authorize',c.AUTH]
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


def test_six_class_cell_sampling_boundaries_and_last_batch():
    y=np.repeat(np.arange(6),24).astype(np.int64);pt=np.tile(np.arange(250,2600,100),6).astype(float)
    sampler=BalancedSampler(y,pt)
    assert sampler.counts.shape==(6,24) and np.all(sampler.counts==1)
    order=sampler.draw(1,draws=144000)
    assert np.array_equal(order,sampler.draw(1,draws=144000))
    assert not np.array_equal(order,sampler.draw(2,draws=144000))
    hist=np.asarray(sampler.evidence(order)['class_bin_counts'])
    assert np.all(np.abs(hist-1000)<160)
    full=sampler.draw(1)
    assert len(full)==2488148
    batches=list(sampler.batches(full));assert len(batches)==9720 and len(batches[-1][1])==84
    for role in ('validation','final_test'):
        with pytest.raises(ValueError):BalancedSampler(y,pt,role=role)
    for bad in (np.r_[6,y[1:]],np.r_[-1,y[1:]]):
        with pytest.raises(ValueError):BalancedSampler(bad,pt)
    # Upper bound excluded, lower bound included; a removed cell fails closed.
    lower=pt.copy();lower[0]=200;assert len(BalancedSampler(y,lower).indices)==144
    upper=pt.copy();upper[-1]=2600
    with pytest.raises(ValueError,match='Empty'):BalancedSampler(y,upper)


def test_filtered_inventory_preserves_extended_validation_and_excludes_test():
    roles=np.array([0,1,2,3,1,1,0,1,255,1],np.uint8)
    a=dict(roles=np.array([0,3,2,3,1,3,0,1,255,1],np.uint8),
           labels=np.array([0,1,2,3,4,5,6,10,-1,0],np.int64),
           jet_pt=np.array([600,700,800,900,199.9,2600,900,900,900,2199.9]),
           jet_nparticles=np.full(10,3),hlt_jet_nparticles=np.full(10,2))
    before=roles.copy();m=population.masks(a,roles)
    assert np.flatnonzero(m['train']).tolist()==[0]
    assert np.flatnonzero(m['validation']).tolist()==[1,9]
    assert np.flatnonzero(m['outside_validation']).tolist()==[4,5]
    # Includes formerly unused validation at both in/outside ranges; never test.
    f=dict(files=[dict(path='raw/x.root',sha256='a'*64,tree_key='tree;1')])
    counts,out,_,_,arrays,byclass=population.inventory(f,[a],[roles])
    assert counts==dict(train=1,validation=2,final_test=1,unused=1)
    assert out['validation']==2 and byclass['validation']==[1,1,0,0,0,0]
    assert not any('test' in key for key in arrays)
    assert np.array_equal(roles,before)
    assert set(arrays['validation_labels'])=={0,1}


def test_population_publication_and_tamper_replay(tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import deterministic_npz_bytes
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    y=np.tile(np.arange(6),24).astype(np.int64);pt=np.repeat(np.arange(250,2600,100),6)
    a=dict(labels=np.r_[y,y,y],jet_pt=np.r_[pt,pt,np.full(144,100)],
           roles=np.repeat(np.arange(3,dtype=np.uint8),144),
           jet_nparticles=np.full(432,3),hlt_jet_nparticles=np.full(432,2))
    # Third block is extended validation originally unused, outside range.
    a['roles'][288:]=3;rs=np.repeat(np.array([0,1,1],np.uint8),144)
    f=dict(content_hash='a'*64,files=[dict(path='raw/x.root',sha256='b'*64,tree_key='tree;1')])
    parent=c.donor.artifact('POPULATION',parents=dict(foundation=f['content_hash']))
    def build(*args,**kw):write_immutable_json(Path(args[1])/'population.json',parent)
    monkeypatch.setattr(population.parent,'build',build)
    monkeypatch.setattr(population.parent,'load',lambda *a,**kw:(parent,f,[globals_a],[rs],{},None))
    globals_a=a
    folder=tmp_path/'population';p=population.build(tmp_path/'data',folder,production=False)
    loaded=population.load(tmp_path/'data',folder,production=False)
    assert loaded[0]==p and p['counts']['train']==144 and p['outside_counts']['validation']==144
    assert loaded[-1].counts.shape==(6,24)
    assert set(loaded[4])=={r+'_'+k for r in ('train','validation','outside_validation') for k in ('identities','labels','pt')}
    # Rehashing a malicious row array and manifest still cannot alter replay.
    data={k:v.copy() for k,v in loaded[4].items()};data['train_labels'][0]=5
    (folder/'rows.npz').write_bytes(deterministic_npz_bytes(data))
    p['rows']=file_ref(folder/'rows.npz',root=folder)
    (folder/'population.json').write_text(json.dumps(with_content_hash(p)))
    with pytest.raises(ValueError,match='Canonical'):population.load(tmp_path/'data',folder,production=False)


def test_six_class_contrastive_serial_parallel_ties_and_occurrences():
    n=5400;tr=cache(rows=n);keys=np.zeros((n,128),np.float32);keys[:,0]=1
    opt=contrastive.Sampler(keys,tr.identities,tr.labels)
    ref=contrastive.SerialReference(keys,tr.identities,tr.labels,root_seed=c.NEGATIVE_SEED)
    anchors=np.array([0,0,1,5399]);positions=np.array([0,1,2,c.DRAWS-1])
    expected,proof=ref.occurrences(anchors,2,positions)
    actual,fastproof=opt.sample(anchors,2,positions)
    assert np.array_equal(actual,expected) and proof==fastproof
    assert not np.array_equal(actual[0],actual[1])
    with contrastive.CPUWorkers(opt,workers=2):
        parallel,_=opt.sample(anchors,2,positions)
    assert np.array_equal(actual,parallel)
    for anchor,row,item in zip(anchors,actual,proof):
        assert row[0]==anchor and len(np.unique(row))==1025
        assert all(len(item['components'][k])==6 for k in ('H','S','G'))
        assert item['components']['H'][tr.labels[anchor]]==0
        assert sum(item['components']['S'])==256 and item['components']['S'][tr.labels[anchor]]==256
        assert max(item['components']['G'])-min(item['components']['G'])<=1
    with pytest.raises(ValueError):contrastive.Sampler(keys,tr.identities,tr.labels,role='validation')
    with pytest.raises(ValueError,match='820'):contrastive.Sampler(keys[:4800],tr.identities[:4800],tr.labels[:4800])


def test_six_class_rejection_ties_and_bad_outputs():
    y=np.tile(np.arange(6),4);p=np.full((24,6),1/6)
    for efficiency in (.3,.5):
        result=reporting.rejection(y,p,efficiency)
        assert list(result)==list(c.CLASS_NAMES[1:])
        assert all(v['background_passing']==4 and v['signal_passing']==4 and v['rejection']==1 for v in result.values())
    for bad in (np.full((24,11),1/11),np.full((24,6),np.nan)):
        with pytest.raises(ValueError):reporting.rejection(y,bad,.5)
        with pytest.raises(ValueError):reporting.evaluate_probabilities(y,bad)


@pytest.mark.parametrize('existing',[False,True])
def test_runtime_reuses_current_quota_and_never_queues_old_run(tmp_path,monkeypatch,existing):
    from hlt_classification.luka_sixclass_wide_b256_oscar import runtime
    r=runtime.repair;target=tmp_path/'runtime';records=Path(str(target)+'.restore')
    monkeypatch.setattr(r,'TARGET',target)
    spec=tmp_path/'spec.json';expected=dict(content_hash='a'*64)
    spec.write_text(json.dumps(r.hashed(dict(environment=expected))))
    calls=[]
    def restored(*args):
        calls.append('restore');target.mkdir();records.mkdir()
        plan=r.hashed(dict(expected_environment=expected));r.publish(records/'plan.json',plan)
        r.publish(records/'complete.json',dict(contract='LUKA_OSCAR_RUNTIME_RESTORED/v1',
            environment_dir=str(target),parents=dict(plan=plan['content_hash'],environment='a'*64),
            installed_bytes=8*2**30,declared_remaining_result_gib=999))
    if existing:restored();calls.clear()
    monkeypatch.setattr(r,'restore',restored)
    monkeypatch.setattr(r,'verify_files',lambda *a:calls.append('files'))
    monkeypatch.setattr(r,'verify_runtime',lambda *a:calls.append('runtime'))
    assert runtime.ensure(spec,tmp_path,'/conda',28)==(28 if existing else 18)
    assert calls==(['files','runtime'] if existing else ['restore'])
    # Existing partial state is never silently overwritten or reused.
    (records/'complete.json').unlink()
    with pytest.raises(FileNotFoundError):runtime.ensure(spec,tmp_path,'/conda',28)


def test_new_bootstrap_cannot_launch_old_campaign():
    root=Path(__file__).resolve().parents[1]
    text=(root/'scripts/bootstrap_luka_sixclass_wide_b256_oscar.sh').read_text()
    assert 'nohup bash' in text and '</dev/null &' in text and 'exec tail' in text
    assert 'launch_luka_sixclass_wide_b256_oscar.sh' in text
    assert 'recover_luka_new_balanced_oscar.sh' not in text and 'launch_luka_new_balanced_oscar.sh' not in text
    assert 'QUOTA >= 32' in text and 'QUOTA >= 18' in text
    assert 'previous_terminal' in text and '--authorize '+c.AUTH in text


def test_real_bootstrap_detaches_then_invokes_only_new_registration(tmp_path):
    import os,sys,time
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    shell_tmp=subprocess.check_output([bash,'-c','cd "$1" && pwd','fixture',str(tmp_path)],text=True).strip()
    def sp(path):
        return shell_tmp+'/'+Path(path).relative_to(tmp_path).as_posix()
    base=tmp_path/'base';data=tmp_path/'data';data.mkdir()
    (data/'transfer_complete.json').write_text('{}')
    source=Path(__file__).resolve().parents[1];commit='0'*39+'1'
    project=base/'source/HLT_Classification_sixclass_wide_b256_oscar_00000000'
    (project/'scripts').mkdir(parents=True);(project/'sbatch').mkdir()
    script=project/'scripts/bootstrap_luka_sixclass_wide_b256_oscar.sh'
    text=(source/'scripts'/script.name).read_text().replace('/oscar/home/rlyang/hlt_classification',sp(base)).replace(
        '/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1',sp(data))
    script.write_bytes(text.encode())
    launch='launch_luka_sixclass_wide_b256_oscar.sh'
    (project/'scripts'/launch).write_bytes((source/'scripts'/launch).read_bytes())
    (project/'sbatch/luka_sixclass_wide_b256_oscar_common.sh').write_bytes(b':\n')
    old=base/'checkpoints/luka_new_balanced_oscar_f9dfc80b_r1';old.mkdir(parents=True)
    (old/'campaign_spec.json').write_text('{}')
    runtime=base/'environments/new_balanced_oscar_v1';(runtime/'bin').mkdir(parents=True)
    receipt=Path(str(runtime)+'.restore');receipt.mkdir();(receipt/'complete.json').write_text('{}')
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'calls'
    fake_python='''sleep 0.2
printf '%s\\n' "$*" >> "$FIXTURE_CALLS"
if [[ "$*" == *'available_result_gib'* ]]; then printf '18\\n'; fi
if [[ "$*" == *' - '* ]]; then cat >/dev/null; fi
'''
    for p,body in {bindir/'git':'if [[ "$*" == *"rev-parse HEAD"* ]]; then printf "%040d\\n" 1; fi\n',
                   bindir/'tail':'echo TAIL_ATTACHED\n',runtime/'bin/python':fake_python}.items():
        p.write_bytes(('#!/usr/bin/env bash\n'+body).encode());p.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    env=dict(os.environ,BASH_ENV=sp(envfile),FIXTURE_BIN=sp(bindir),FIXTURE_CALLS=sp(calls))
    args=[bash,str(script),'detach',commit,'18','','--authorize',c.AUTH]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'DETACHED six-class launcher PID' in result.stdout and 'TAIL_ATTACHED' in result.stdout
    log=base/'checkpoints/luka_sixclass_wide_b256_oscar_00000000_r1.bootstrap/launcher.log'
    deadline=time.monotonic()+12
    while (not log.exists() or 'BOOTSTRAP_EXIT=0' not in log.read_text()) and time.monotonic()<deadline:time.sleep(.05)
    assert 'BOOTSTRAP_EXIT=0' in log.read_text(),log.read_text()
    lines=calls.read_text().splitlines()
    fits=[x for x in lines if ' register ' in x or ' queue ' in x]
    assert len(fits)==2 and all('luka_sixclass_wide_b256_oscar.py' in x for x in fits)
    assert '--execute' in fits[1] and '--authorize '+c.AUTH in fits[1]
    assert subprocess.run(args,env=env,capture_output=True,timeout=5).returncode!=0
