"""Memory-policy parity, authorized witness reads and preserved baseline lineage."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_sixclass_wide_b256_oscar import (
    contracts as c, campaign, artifacts, recovery, pair_storage, model, early_memory,
    views, population, worker, startup, inspect, banks)
from test_luka_sixclass_wide_b256_oscar import spec_fixture, cache, worker_fixture, tiny
from test_luka_fullstat_balanced_ladder import threads


def registered(tmp_path,monkeypatch,**overrides):
    """Real immutable metadata and six submission bindings, no scientific payload stand-ins."""
    root=tmp_path/'original';root.mkdir()
    s=spec_fixture(root,monkeypatch);s['commit']=recovery.PARENT_COMMIT;s.update(overrides);s=with_content_hash(s)
    monkeypatch.setattr(recovery,'PARENT_ROOT',str(root));monkeypatch.setattr(recovery,'PARENT_HASH',s['content_hash'])
    write_immutable_json(root/'campaign_spec.json',s)
    monkeypatch.setattr(recovery,'capture_source_snapshot',lambda *a,**kw:s['source'])
    jobs=[dict(task='train_'+t,job=str(9544070+i)) for i,t in enumerate(c.STAGES)]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**kw:jobs)
    monkeypatch.setattr(campaign,'terminal',lambda job:[job,'COMPLETED' if job in recovery.BASELINE_JOBS.values() else 'FAILED',
                                                     '0:0' if job in recovery.BASELINE_JOBS.values() else '1:0'])
    for stage in recovery.BASELINES:
        write_immutable_json(root/('train_'+stage)/'result.json',c.artifact('RESULT',
            parents=dict(spec=s['content_hash']),stage=stage,job_id=recovery.BASELINE_JOBS[stage]))
    write_immutable_json(root/'train_CONCAT/runs'/recovery.FAILED_JOB/'failure.json',
        c.artifact('FAILURE',parents=dict(spec=s['content_hash']),stage='CONCAT',job_id=recovery.FAILED_JOB,
                   error_type='OutOfMemoryError',message='CUDA out of memory'))
    _,record=recovery.authenticate(root/'campaign_spec.json',payloads=False)
    fields={k:v for k,v in s.items() if k not in ('contract','schema_version','content_hash','parents',*c.FLAGS)}
    fields.update(root=str(tmp_path/'repair'),project='/new/project',population_root=str(tmp_path/'repair/population'),
                  commit='a'*40,execution_recovery=record)
    new=c.artifact('SPEC',version=2,**fields)
    return s,new,jobs


def test_recovery_version_and_science_tampering(tmp_path,monkeypatch):
    old,new,_=registered(tmp_path,monkeypatch)
    campaign.check(new,live=False)
    assert recovery.origin(new,'OFFLINE')==old and recovery.origin(new,'CONCAT') is new
    assert recovery.enabled(new,'CONCAT') and not recovery.enabled(old,'CONCAT')
    for key,value in [('recipe',dict(new['recipe'],batch_size=128)),('environment',{'python':'changed'}),
                      ('population',dict(new['population'],counts={})),('backend',{})]:
        bad=with_content_hash(dict(new,**{key:value}))
        with pytest.raises((ValueError,KeyError)):campaign.check(bad,live=False)
    bad=deepcopy(new);bad['execution_recovery']['pair_storage']['recompute']=True
    bad['execution_recovery']=with_content_hash(bad['execution_recovery'])
    with pytest.raises(ValueError):campaign.check(with_content_hash(bad),live=False)
    bad=dict(new,schema_version=1,contract=c.PREFIX+'_SPEC/v1')
    with pytest.raises(ValueError,match='SPEC/v2'):campaign.check(with_content_hash(bad),live=False)


def test_parent_missing_bytes_live_jobs_and_committed_progress_stop_recovery(tmp_path,monkeypatch):
    old,new,jobs=registered(tmp_path,monkeypatch);path=Path(old['root'])/'campaign_spec.json'
    with monkeypatch.context() as m:
        m.setattr(campaign,'terminal',lambda job:(_ for _ in ()).throw(ValueError('live')))
        with pytest.raises(ValueError,match='live'):recovery.authenticate(path,payloads=False)
    marker=Path(old['root'])/'train_CONCAT/epochs/01.json'
    write_immutable_json(marker,{'committed':True})
    with pytest.raises(ValueError,match='committed progress'):recovery.authenticate(path,payloads=False)
    ref=new['execution_recovery']['imported']['OFFLINE'];Path(ref['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='bytes differ'):recovery.validate(new)


def test_claim_cannot_fork_twice_and_original_records_stay_unchanged(tmp_path,monkeypatch):
    old,new,_=registered(tmp_path,monkeypatch)
    paths=[Path(old['root'])/'campaign_spec.json',*[Path(r['path']) for r in new['execution_recovery']['imported'].values()]]
    before={p:p.read_bytes() for p in paths}
    recovery.claim(new);recovery.claim(new)
    other=with_content_hash(dict(new,root=str(tmp_path/'other')))
    with pytest.raises(FileExistsError,match='already claimed'):recovery.claim(other)
    assert {p:p.read_bytes() for p in paths}==before


def test_remaining_and_scheduler_dependencies_skip_both_baselines(tmp_path,monkeypatch):
    old,new,parent_jobs=registered(tmp_path,monkeypatch)
    record=new['execution_recovery']
    monkeypatch.setattr(recovery,'authenticate',lambda *a,**kw:(old,record))
    monkeypatch.setattr(campaign,'accepted_jobs',lambda s,**kw:[] if s==new else parent_jobs)
    monkeypatch.setattr(artifacts,'completed',lambda s,stage:dict(receipt={'content_hash':stage}))
    tasks,done=campaign.remaining(new,False)
    assert tasks==['train_'+s for s in c.STAGES[2:]] and set(done)==set(recovery.BASELINES)
    commands=campaign.commands(new,tasks)
    assert commands[0]['dependencies']==[]
    assert commands[1]['dependencies']==['train_CONCAT']
    assert 'train_UE100' in commands[2]['dependencies'] and 'train_DE050' in commands[3]['dependencies']
    assert all('--gres=gpu:l40s:1' in r['command'] for r in commands)


def test_scheduler_review_checks_actual_recovery_worker(monkeypatch,tmp_path):
    import subprocess
    s=spec_fixture(tmp_path);rows=campaign.commands(s,['train_CONCAT']);calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='ClusterName = slurmctld\n' if args[0]=='scontrol' else '',stderr='')
    monkeypatch.setattr(subprocess,'run',run);campaign.site(rows)
    assert calls[1][-4:]==rows[0]['command'][-4:]
    assert '--test-only' in calls[1] and '--wrap=true' not in calls[1]


def test_registration_copies_only_population_and_binds_original_results(tmp_path,monkeypatch):
    import sys
    from hlt_classification.data.cache_contracts import canonical_sha256
    dataset=tmp_path/'data';dataset.mkdir()
    manifest=dict(foundation_sha256=c.PARENT_HASH)
    monkeypatch.setattr(c,'TRANSFER_HASH',canonical_sha256(manifest))
    write_immutable_json(dataset/'transfer_manifest.json',manifest)
    write_immutable_json(dataset/'transfer_complete.json',dict(manifest_sha256=c.TRANSFER_HASH,all_files_verified=True))
    old,new,_=registered(tmp_path,monkeypatch,environment_dir=str(Path(sys.prefix).resolve()),dataset=str(dataset.resolve()))
    oldroot=Path(old['root']);pop=Path(old['population_root'])
    write_immutable_json(pop/'population.json',old['population'])
    (pop/'role_parent').mkdir();(pop/'role_parent/scalar_fixture').write_bytes(b'only scalar metadata')
    auth=recovery.authenticate
    monkeypatch.setattr(recovery,'authenticate',lambda path:auth(path,payloads=False))
    monkeypatch.setattr(campaign,'locations',lambda *a:None)
    monkeypatch.setattr(campaign,'source',lambda *a:{})
    monkeypatch.setattr(campaign,'runtime',lambda:{})
    monkeypatch.setattr(campaign.lineage,'verify',lambda *a:{})
    monkeypatch.setattr(campaign,'storage',lambda *a,**kw:old['storage'])
    calls=[];monkeypatch.setattr(population,'load',lambda *a,**kw:calls.append(a))
    root=tmp_path/'new_registered'
    result=campaign.create(dataset,root,tmp_path/'new_project','a'*40,sys.prefix,available_gib=18,
                           recover=oldroot/'campaign_spec.json')
    assert result['schema_version']==2 and calls==[(dataset.resolve(),old['population_root'])]
    assert result['recipe']==old['recipe'] and result['population']==old['population']
    assert (root/'population/role_parent/scalar_fixture').read_bytes()==b'only scalar metadata'
    assert not (root/'train_OFFLINE').exists() and not (root/'train_M0HLT').exists()
    campaign.check(result,live=False)
    assert recovery.origin(result,'M0HLT')==old
    with pytest.raises(FileExistsError):campaign.create(dataset,root,tmp_path/'new_project','a'*40,sys.prefix,
                                                      available_gib=18,recover=oldroot/'campaign_spec.json')


def test_longest_witnesses_read_only_three_authorized_real_rows(tmp_path,monkeypatch):
    from test_luka_fullstat import write_root
    import uproot
    path=tmp_path/'raw/s.root';write_root(path,[161]*8)
    record=dict(path='raw/s.root',sha256=sha256_file(path),entries=8,tree_key='tree',category='signal')
    a=dict(labels=np.zeros(8,np.int8),jet_nparticles=np.full(8,3),hlt_jet_nparticles=np.full(8,2),
           jet_pt=np.array([250,250,2800,250,250,250,250,250.]))
    roles=np.array([0,1,1,2,3,2,3,3],np.uint8)
    p={'content_hash':'b'*64,'maximum_support':{r:dict(concat=5,offline=3,hlt=2)
                                             for r in ('train','validation','outside_validation')}}
    original=uproot.behaviors.TBranch.HasBranches.arrays;reads=[]
    def guarded(tree,expressions,*args,**kw):
        lo,hi=kw['entry_start'],kw['entry_stop']
        assert hi-lo==1 and lo in (0,1,2)
        reads.append(lo);return original(tree,expressions,*args,**kw)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'arrays',guarded)
    caches,proof=early_memory.witnesses({'dataset':str(tmp_path)},'CONCAT',(p,{'files':[record]},[a],[roles],None,None))
    assert reads==[0,1,2] and all(len(x)==1 for x in caches.values())
    assert all(r['particles']==5 for r in proof.values())


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed Weaver CUDA storage parity')
def test_native_six_output_pair_storage_cuda_parity(bf16):
    raw=cache(coordinate='CONCAT',rows=12).batch(np.arange(6))
    # Production width/layers, short fixture; no local L40S full-support claim.
    result=pair_storage.parity(raw,'cuda',bf16=bf16)
    assert result['passed'] and result['statistics']['restored_cuda_tensors']>0
    torch.cuda.empty_cache()


def test_native_checkpoint_keys_and_policy_scope():
    native=model.new_model('CONCAT');stored=model.new_model('CONCAT',pair_storage=True)
    assert native.state_dict().keys()==stored.state_dict().keys()
    for k,v in native.state_dict().items():torch.testing.assert_close(v,stored.state_dict()[k],rtol=0,atol=0)
    deployed=stored.deployed()
    assert deployed.state_dict().keys()==native.state_dict().keys()
    with pytest.raises(ValueError):model.new_model('DE000',pair_storage=True)


def test_early_oom_prevents_full_view_build(tmp_path,monkeypatch):
    s,*_=worker_fixture(tmp_path,monkeypatch)
    s=dict(s,schema_version=2)
    monkeypatch.setattr(early_memory,'run',lambda *a:(_ for _ in ()).throw(torch.OutOfMemoryError('test OOM')))
    monkeypatch.setattr(views,'load',lambda *a:pytest.fail('Full particle construction must not start'))
    with pytest.raises(torch.OutOfMemoryError):worker.run(s,'train_CONCAT',device='cpu')
    failure=load_json(tmp_path/'train_CONCAT/runs/102/failure.json')
    assert failure['error_type']=='OutOfMemoryError'
    assert not list((tmp_path/'train_CONCAT/epochs').glob('*.json'))


def test_imported_baselines_used_by_real_tail_workers_and_results(tmp_path,monkeypatch):
    old,tr,val,sampler,jobs=worker_fixture(tmp_path/'original',monkeypatch)
    for stage in recovery.BASELINES:worker.run(old,'train_'+stage,device='cpu')
    before={p:p.read_bytes() for p in Path(old['root']).rglob('*.json')}
    fields={k:v for k,v in old.items() if k not in ('contract','schema_version','content_hash','parents',*c.FLAGS)}
    fields.update(root=str(tmp_path/'new'),project='/new/project')
    new=c.artifact('SPEC',version=2,**fields)
    monkeypatch.setattr(recovery,'origin',lambda s,stage:old if s==new and stage in recovery.BASELINES else s)
    def early(s,stage,loaded,device,folder):
        write_immutable_json(folder/'early_memory.json',c.artifact('EARLY_MEMORY',parents=dict(spec=s['content_hash']),
                                                               stage=stage,passed=True))
    monkeypatch.setattr(early_memory,'run',early)
    prior=startup.run
    def qualification(s,stage,*a,**kw):
        out=prior(s,stage,*a,**kw)
        if stage=='CONCAT':out=with_content_hash(dict(out,pair_storage=dict(calls=3,saved_cuda_tensors=3,
                                                                         saved_cuda_bytes=32,restored_cuda_tensors=3)))
        return out
    monkeypatch.setattr(startup,'run',qualification)
    for stage in c.STAGES[2:]:
        receipt=worker.run(new,'train_'+stage,device='cpu')
        assert artifacts.completed(new,stage)['receipt']==receipt
    result=inspect.results(new)
    inspect.progress(new,'DE000')
    inspect.progress(new,'OFFLINE')
    assert set(result['models'])==set(c.STAGES)
    assert artifacts.completed(new,'OFFLINE')['spec']==old
    assert {p:p.read_bytes() for p in before}==before
    # New banks must bind the new spec, reused banks keep their original parents.
    bank=load_json(Path(new['root'])/'train_CONCAT/runs/102/bank/manifest.json')
    assert bank['parents']['spec']==new['content_hash']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real early GPU check on short synthetic support')
def test_early_memory_cuda_production_batch_and_storage(tmp_path,monkeypatch):
    tr=cache(coordinate='CONCAT',rows=12)
    val=cache(role='validation',coordinate='CONCAT',rows=12)
    s=spec_fixture(tmp_path);s['schema_version']=2
    monkeypatch.setattr(early_memory,'witnesses',lambda *a:({'train':tr,'validation':val,'outside_validation':val},{}))
    monkeypatch.setattr(early_memory,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
                                                                gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    proof=early_memory.run(s,'CONCAT',None,'cuda',tmp_path)
    assert proof['passed'] and proof['batch_size']==256 and len(proof['parity'])==2
    assert proof['pair_storage']['calls']==3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real optimized worker qualification')
def test_real_optimized_integrated_qualification(tmp_path,monkeypatch):
    from test_luka_sixclass_wide_b256_oscar import test_real_weaver_integrated_qualification
    original=startup.run;proofs=[]
    def stored(*args,**kwargs):
        proof=original(*args,**kwargs,pair_storage=True);proofs.append(proof);return proof
    monkeypatch.setattr(startup,'run',stored)
    test_real_weaver_integrated_qualification('CONCAT',tmp_path,monkeypatch)
    assert proofs[0]['pair_storage']['restored_cuda_tensors']>0


def test_real_detached_recovery_shell_preserves_parent_and_registers_fork(tmp_path):
    import os,sys,subprocess,time
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def sp(p):
        p=Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    base=tmp_path/'base';project=base/'source/new';(project/'scripts').mkdir(parents=True)
    (project/'sbatch').mkdir()
    env_dir=base/'environments/new_balanced_oscar_v1';(env_dir/'bin').mkdir(parents=True)
    old=base/'checkpoints/luka_sixclass_wide_b256_oscar_d534fcfe_r1'
    old.mkdir(parents=True);parent=old/'campaign_spec.json';parent.write_bytes(b'old immutable spec')
    root=base/'checkpoints/new_recovery'
    source=Path(__file__).resolve().parents[1]/'scripts/recover_luka_sixclass_wide_b256_oscar.sh'
    script=project/'scripts'/source.name
    script.write_bytes(source.read_text().replace('/oscar/home/rlyang/hlt_classification',sp(base)).encode())
    (project/'sbatch/luka_sixclass_wide_b256_oscar_common.sh').write_bytes(b':\n')
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'calls'
    for path,body in {bindir/'git':'printf "%040d\\n" 1\n',bindir/'tail':'echo TAIL_ATTACHED\n',
                      env_dir/'bin/python':'sleep 0.2\nprintf "%s\\n" "$*" >> "$FIXTURE_CALLS"\n'}.items():
        path.write_bytes(('#!/usr/bin/env bash\n'+body).encode());path.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    env=dict(os.environ,BASH_ENV=sp(envfile),FIXTURE_BIN=sp(bindir),FIXTURE_CALLS=sp(calls))
    args=[bash,str(script),'detach',sp(project),sp(root),'0'*39+'1','18','--authorize',c.AUTH]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert 'DETACHED recovery' in result.stdout and 'TAIL_ATTACHED' in result.stdout
    log=Path(str(root)+'.launch/launcher.log');deadline=time.monotonic()+15
    while (not log.exists() or 'LAUNCHER_EXIT=0' not in log.read_text()) and time.monotonic()<deadline:time.sleep(.05)
    assert 'LAUNCHER_EXIT=0' in log.read_text()
    commands=calls.read_text().splitlines()
    assert len(commands)==2 and '--recover '+sp(parent) in commands[0]
    assert ' register ' in commands[0] and ' queue ' in commands[1] and '--execute' in commands[1]
    assert parent.read_bytes()==b'old immutable spec'
    assert subprocess.run(args,env=env,capture_output=True,timeout=10).returncode!=0
