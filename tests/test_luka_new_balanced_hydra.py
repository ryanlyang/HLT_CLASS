"""New split/evaluation semantics, bounded reads, six-stage recovery and Hydra DAG."""
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
from hlt_classification.luka_fullstat_balanced_ladder import model, contrastive
from hlt_classification.luka_fullstat_balanced_ladder.sampling import BalancedSampler
from hlt_classification.luka_new_balanced_hydra import (
    contracts as c,population,views,campaign,training,banks,artifacts,worker,startup,reporting,inspect)
from test_luka_fullstat_balanced_ladder import threads


def spec_fixture(root):
    pop=c.artifact('POPULATION',parents=dict(foundation=c.PARENT_HASH),policy=c.population_policy(),
        counts=dict(c.COUNTS,unused=689),outside_counts=c.OUTSIDE)
    return c.artifact('SPEC',root=str(root),dataset='/dataset',project='/project',commit='f'*40,source={},
        environment_dir='/env',environment={},gpu='l40s',foundation_producer={},population=pop,
        population_root=str(root/'population'),recipe=c.recipe(),nodes={s:c.node(s) for s in c.STAGES},
        counts=c.COUNTS,cpus=6,memory_mb=192000,minutes={t:2880 for t in c.tasks()},
        preparation_workers=4,candidate_workers=3,tasks=c.tasks(),
        storage=dict(declared_available_gib=29,natural_baseline=dict(spec_path='/baseline/spec.json')))


def test_recipe_and_population_are_distinct_from_oscar():
    from hlt_classification.luka_fullstat_balanced_ladder import contracts as old
    assert c.recipe()['selection_population']=='new_in_range_500000_validation'
    assert c.recipe()['final_test_primary_rows']==1200000
    assert c.STEPS==18950 and c.DRAWS%128==64
    assert c.recipe()['teacher_order']==old.recipe()['teacher_order']
    assert c.recipe()['embedding_total_weight']==10 and c.recipe()['auxiliary_kd_total_weight']==1
    assert c.recipe()['matching_weight']==.3 and c.recipe()['content_hash']!=old.recipe()['content_hash']


def scalar_fixture():
    labels=np.tile(np.arange(11),18*5+3).astype(np.int8)
    pt=np.r_[np.repeat(np.arange(250,2000,100),11*5),np.full(33,250.)]
    rs=np.tile(np.repeat(np.arange(5),11),18).astype(np.uint8)
    rs[rs==4]=3;rs=np.r_[rs,np.array([0]*11+[1]*11+[2]*11,np.uint8)]
    # The final three groups are all outside range and remain in their parent roles.
    pt[-33:]=[199.]*11+[2000.]*11+[2500.]*11
    rs=np.r_[rs,[255]].astype(np.uint8);labels=np.r_[labels,np.int8(-1)];pt=np.r_[pt,300.]
    member=dict(roles=rs,labels=labels,jet_pt=pt,jet_nparticles=np.full(len(rs),3),hlt_jet_nparticles=np.full(len(rs),2))
    record=dict(path='raw/s.root',sha256='a'*64,category='signal',tree_key='tree',entries=len(rs))
    return record,member


def test_unused_extension_preserves_parent_roles_and_endpoints():
    r,a=scalar_fixture();add=dict(train=80,validation=60,final_test=50)
    roles,q=population.extend([r],[a],add)
    again,q2=population.extend([r],[a],add)
    np.testing.assert_array_equal(roles[0],again[0]);assert q==q2
    changed=roles[0]!=a['roles']
    assert np.all(a['roles'][changed]==3)
    assert np.all((a['jet_pt'][changed]>=200)&(a['jet_pt'][changed]<2000))
    assert roles[0][-1]==255
    for code,name in enumerate(population.ROLES):assert ((roles[0]==code)&changed).sum()==add[name]
    masks=population.masks(a,roles[0])
    assert not np.any(masks['train']&masks['validation'])
    assert masks['outside_validation'].sum()==11
    assert not any(np.any(m&(roles[0]==2)) for m in masks.values())
    with pytest.raises(ValueError):population.extend([r],[a],dict(train=100000,validation=0,final_test=0))


def test_population_artifact_replay_and_corruption(tmp_path,monkeypatch):
    r,a=scalar_fixture();f=dict(files=[r],content_hash='b'*64)
    monkeypatch.setattr(population.original,'load',lambda *args,**kw:(f,[a]))
    monkeypatch.setattr(c,'ADDITIONS',dict(train=80,validation=60,final_test=50))
    p=population.build(tmp_path/'data',tmp_path/'population',production=False)
    loaded=population.load(tmp_path/'data',tmp_path/'population',production=False)
    assert loaded[0]==p and loaded[-1].counts.shape==(11,18)
    assert set(loaded[-2])=={r+'_'+k for r in ('train','validation','outside_validation') for k in ('identities','labels','pt')}
    path=tmp_path/'population/roles/00000.npz';path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError):population.load(tmp_path/'data',tmp_path/'population',production=False)


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


def test_hydra_dag_journals_and_no_double_submission(tmp_path,monkeypatch):
    s=spec_fixture(tmp_path);campaign.check(s,live=False)
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
    assert all('--partition=gpus' in row and '--gres=gpu:nvidia_l40s:1' in row for row in calls)
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
def test_rehashed_semantic_changes_rejected(tmp_path,field,value):
    s=spec_fixture(tmp_path);s[field]=value
    with pytest.raises(ValueError):campaign.check(with_content_hash(s),live=False)


def test_primary_and_outside_metrics_do_not_reselect():
    y=np.tile(np.arange(11),2);p=np.full((22,11),.001);p[np.arange(22),y]=.99;p/=p.sum(1,keepdims=True)
    data=SimpleNamespace(labels=y,__len__=lambda s:22)
    class Cache:
        labels=y
        def __len__(self):return 22
    a=dict(validation_pt=np.full(22,200.),outside_validation_pt=np.full(22,2000.))
    m=reporting.metrics(Cache(),p,Cache(),np.roll(p,1,axis=1),a,dict(report='a'*64))
    assert m['selection_population']=='in_range' and m['checkpoint_reselection'] is False
    assert m['populations']['in_range']['summary']['accuracy']==1
    assert m['populations']['outside_range']['summary']['accuracy']==0
    assert all(x['status']=='censored' for x in m['populations']['in_range']['R50'].values())
    assert reporting.recovery(.7,.6,.8)==pytest.approx(50)
    a['validation_pt'][0]=2000
    with pytest.raises(ValueError):reporting.metrics(Cache(),p,Cache(),p,a,{})


def worker_fixture(tmp_path,monkeypatch):
    from test_luka_embedding_aux import tiny
    from test_luka_attention import cache
    tiny(monkeypatch);monkeypatch.setattr(c,'DRAWS',44)
    monkeypatch.setattr(c,'COUNTS',dict(train=198,validation=22,final_test=11))
    tr=cache(coordinate='OFFLINE',rows=198);val=cache(role='validation',coordinate='OFFLINE',rows=22)
    val.identities[:,20]=1;outside=deepcopy(val);outside.identities[:,20]=2
    sampler=BalancedSampler(tr.labels,250+100*(np.arange(len(tr))//11))
    s=spec_fixture(tmp_path);write_immutable_json(tmp_path/'campaign_spec.json',s)
    monkeypatch.setattr(campaign,'check',lambda s,**kw:s);monkeypatch.setattr(campaign,'source',lambda *a:{})
    monkeypatch.setattr(campaign,'storage',lambda *a,**kw:{})
    accepted=[dict(task='train_'+stage,job=str(100+i)) for i,stage in enumerate(c.STAGES)]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**kw:accepted)
    monkeypatch.setattr(worker,'allocation',lambda s,t:next(r['job'] for r in reversed(accepted) if r['task']==t))
    monkeypatch.setattr(worker.lineage,'verify',lambda *a:{})
    a=dict(validation_pt=np.full(22,350.),outside_validation_pt=np.full(22,2200.))
    monkeypatch.setattr(population,'load',lambda *a,**kw:(s['population'],None,None,None,dict(validation_pt=np.full(22,350.),outside_validation_pt=np.full(22,2200.)),sampler))
    def inputs(s,stage,*args,**kwargs):
        caches=dict(train=deepcopy(tr),validation=deepcopy(val),outside_validation=deepcopy(outside))
        for x in caches.values():x.coordinate_name=c.COORDINATES[stage]
        return caches,c.artifact('VIEW_BUILD',parents=dict(spec=s['content_hash']),stage=stage,mapping=[])
    monkeypatch.setattr(views,'load',inputs)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=1024,rss_bytes=1024,gpu_peak_bytes=0))
    monkeypatch.setattr(startup,'run',lambda s,stage,*args,**kw:c.artifact('STARTUP',parents=dict(spec=s['content_hash']),stage=stage,
        installed_weaver_parity=True,finite_connected_gradients=True,classifier_ce_only=True,integrated_two_epoch_miniature=True,
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
        if stage!='DE000':assert banks.Bank(complete['folder']/'bank',tr,stage,complete['refs'],s).targets.shape==(198,128)
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
    launch=(root/'scripts/launch_luka_new_balanced_hydra.sh').read_text()
    worker_text=(root/'sbatch/run_luka_new_balanced_hydra.sh').read_text()
    assert 'nohup bash' in launch and '</dev/null &' in launch and 'exec tail' in launch
    assert '--authorize '+c.AUTH in launch
    assert 'PYTHONNOUSERSITE=1' in worker_text and '${ENV_DIR}/lib' in worker_text
    assert '${PROJECT_DIR}/scripts/luka_new_balanced_hydra.py' in worker_text


def test_real_bash_detached_launch_finishes_after_tail_exits(tmp_path):
    import os
    import sys
    import time
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def shellpath(path):
        p=Path(path).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True)
    fake=tmp_path/'environment';(fake/'bin').mkdir(parents=True)
    script=project/'scripts/launch_luka_new_balanced_hydra.sh'
    text=(Path(__file__).resolve().parents[1]/'scripts'/script.name).read_text()
    text=text.replace('ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv',
                      'ENV_DIR="'+shellpath(fake)+'"')
    script.write_bytes(text.encode())
    binpath=tmp_path/'bin';binpath.mkdir()
    bodies={binpath/'git':'printf "%040d\\n" 1\n',
            binpath/'tail':'printf "TAIL_ATTACHED\\n"\n',
            fake/'bin/python':'sleep 1\nprintf "%s\\n" "$*" >> "$FIXTURE_CALLS"\n'}
    for path,body in bodies.items():
        path.write_bytes(('#!/usr/bin/env bash\n'+body).encode());path.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    data=tmp_path/'data';data.mkdir();(data/'transfer_complete.json').write_text('{}')
    parent=tmp_path/'parent.json';parent.write_text('{}');root=tmp_path/'output';calls=tmp_path/'calls'
    env=dict(os.environ,BASH_ENV=shellpath(envfile),FIXTURE_BIN=shellpath(binpath),FIXTURE_CALLS=shellpath(calls))
    args=[bash,str(script),'detach',shellpath(project),shellpath(data),shellpath(root),
          '0'*39+'1','l40s','29',shellpath(parent)]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'DETACHED launcher PID' in result.stdout and 'TAIL_ATTACHED' in result.stdout
    log=Path(str(root)+'.launch')/'launcher.log';deadline=time.monotonic()+10
    while (not log.exists() or 'LAUNCHER_EXIT=0' not in log.read_text()) and time.monotonic()<deadline:time.sleep(.05)
    assert 'LAUNCHER_EXIT=0' in log.read_text()
    lines=calls.read_text().splitlines()
    assert len(lines)==2 and ' register ' in lines[0] and ' queue ' in lines[1]
    assert '--authorize '+c.AUTH in lines[1] and '--execute' in lines[1]
    assert subprocess.run(args,env=env,capture_output=True,timeout=5).returncode!=0
    # Real Bash/nohup and closed launcher streams; not a remote SSH HUP test.


def test_storage_reserves_remaining_natural_baseline_without_new_cache(tmp_path,monkeypatch):
    from hlt_classification.luka_new_fullstat_hydra import contracts as original
    dataset=tmp_path/'dataset';baseline=tmp_path/'natural';cache=tmp_path/'cache';baseline.mkdir();cache.mkdir()
    s=original.artifact('SPEC',dataset=str(dataset),foundation_sha256=c.PARENT_HASH,root=str(baseline),cache=str(cache))
    path=baseline/'campaign_spec.json';write_immutable_json(path,s)
    monkeypatch.setattr(campaign,'tree_bytes',lambda p:4*2**30 if Path(p)==cache else 0)
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=30*2**30))
    r=campaign.storage(dataset,tmp_path/'new',path,30)
    assert r['required_free_bytes']==27*2**30  # 14 new plus 17-4 reserved.
    with pytest.raises(OSError):campaign.storage(dataset,tmp_path/'new',path,26)
    assert not (tmp_path/'new').exists()


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
    monkeypatch.setattr(campaign,'binding',lambda *a:None);monkeypatch.setattr(campaign,'runtime',lambda:{})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'NVIDIA L40S')
    def response(args,**kw):
        out='ClusterName = hydra\n' if args[-1]=='config' else (
            f'JobId=123 JobState=RUNNING Account=default QOS=normal Partition=gpus JobName=lukanbal_train_OFFLINE '
            f'Comment={s["content_hash"]}:train_OFFLINE WorkDir=/project NumCPUs=6 NumNodes=1 NumTasks=1 CPUs/Task=6 '
            'AllocTRES=cpu=6,mem=187.5G,node=1,gres/gpu=1 TimeLimit=2-00:00:00')
        return SimpleNamespace(stdout=out)
    monkeypatch.setattr(subprocess,'run',response)
    assert worker.allocation(s,'train_OFFLINE')=='123'
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'GeForce GTX 1080 Ti')
    with pytest.raises(PermissionError):worker.allocation(s,'train_OFFLINE')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA integrated qualification')
@pytest.mark.parametrize('stage',['UE100','DE000'])
def test_real_weaver_integrated_qualification(stage,tmp_path,monkeypatch):
    import time
    from test_luka_attention import cache
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
