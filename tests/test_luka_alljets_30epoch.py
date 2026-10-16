"""Fixed-budget execution and strict reuse, not manufactured remote admission."""
from copy import deepcopy
from pathlib import Path
import ast
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.cms2jc2_response.contracts import artifact as allocation_artifact
from hlt_classification.jetclass2_delphes import runner, batch128_30e_recipe as schedule
from hlt_classification.luka_fullsim import alljets_30epoch as thirty
from hlt_classification.luka_fullsim import alljets_report_recovery as recovery, alljets_report_replay as replay
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_preflight as pre
from hlt_classification.luka_fullsim import alljets_fusion_submission as sub
from test_luka_alljets_batch128 import CountingCache, TinyModel, gate128 as original_gate128
from test_luka_batch128_report_recovery import finished, ACCOUNT
from test_luka_fusion_isolated_preflight import local_gate
from test_luka_fullsim_stage2 import dataset, prepared, config
from test_luka_fullsim_alljets import all_prepared


@pytest.fixture
def gate128(original_gate128,monkeypatch):
    original = pre.train_kernel
    def slow(*args,**kwargs):
        report,state = original(*args,**kwargs)
        return with_content_hash(dict(report,runtime_seconds=1300.)),state
    monkeypatch.setattr(pre,'train_kernel',slow)
    return original_gate128


def test_recipe_schedule_and_unchanged_non_schedule_fields():
    old,new = c.registration(4)['training_recipe'],schedule.recipe()
    changed = {'contract','content_hash','parents','minimum_passes','maximum_passes','hold_through',
               'decay_through','patience','patience_clock','schedule','early_stopping'}
    assert {k:v for k,v in new.items() if k not in changed}=={k:v for k,v in old.items() if k not in changed}
    assert new['parents']==dict(original_batch128_recipe=old['content_hash'])
    assert new['minimum_passes']==new['maximum_passes']==30
    assert new['batch_size']==new['inference_batch_size']==128 and new['accumulation']==1
    assert new['early_stopping'] is False
    assert schedule.learning_rate(3)==3e-4
    assert schedule.learning_rate(30)==pytest.approx(1.5e-5)
    assert schedule.learning_rate(16.5)==pytest.approx((3e-4+1.5e-5)/2)
    for x in np.linspace(.01,3,20):
        assert schedule.learning_rate(x)==runner.learning_rate(x)
    for x in (0,-1,30.1,float('nan'),float('inf')):
        with pytest.raises(ValueError): schedule.learning_rate(x)
    assert c.registration(5)['training_recipe']==old
    assert c.registration(6)['nodes']==c.registration(4)['nodes']
    assert c.registration(6)['tasks']==c.registration(4)['tasks']
    assert c.registration(6)['training_recipe']==new
    assert c.registration(6)['jobs']==30


@pytest.mark.parametrize('kd',[False,True])
def test_real_kernel_exact30_complete_passes_and_all_validation_epochs(kd,monkeypatch):
    train,val = CountingCache('train'),CountingCache('validation')
    node = dict(c.old.nodes()[0],teacher='teacher' if kd else None)
    rates = []
    lr = schedule.learning_rate
    monkeypatch.setattr(schedule,'learning_rate',lambda pos:(rates.append(pos),lr(pos))[1])
    q = np.full((len(train),11),1/11,np.float32) if kd else None
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        report,_ = runner.train_kernel(TinyModel(),train,val,node=node,device='cpu',training_recipe=schedule.recipe(),
            teacher_probabilities=q,teacher_identities=train.identities if kd else None)
    finally:
        torch.set_num_threads(threads)
    assert report['passes']==30 and report['scientific_fit'] and report['selected_weights_restored']
    assert len(report['validation_history'])==30 and report['validation_history'][-1]['update']==90
    assert rates[-1]==30 and len(rates)==90
    assert [len(ix) for ix in train.calls]==[128,128,3]*30
    for start in range(0,len(train.calls),3):
        np.testing.assert_array_equal(np.sort(np.concatenate(train.calls[start:start+3])),np.arange(len(train)))
    assert [len(ix) for ix in val.calls]==[128,128,3]*30
    for field,bad in [('maximum_passes',20),('batch_size',256),('warmup_passes',2),('early_stopping',True)]:
        with pytest.raises(ValueError,match='exact registered'):
            runner.train_kernel(TinyModel(),train,val,node=node,device='cpu',
                training_recipe=with_content_hash(dict(schedule.recipe(),**{field:bad})))


def test_budget_counts_cache_once_validation_in_pass_and_readback():
    m = dict(cache_seconds=120,training=dict(runtime_seconds=1300),inference_seconds=60)
    assert thirty.projections([m])==(math.ceil(1.75*(120+30*1300+60)/60),15)
    for bad in (0,-1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError): thirty.projections([{**m,'inference_seconds':bad}])


def test_old_budget_stays_failed_but_new_budget_admits_all_technical_checks(finished):
    old,_,_ = finished
    data = replay.replay(old,recovery.parse_accounting(ACCOUNT))
    original = deepcopy(data)
    assert data['admission_error'] and not data['candidate']['fits_debug_24h']
    assert data['candidate']['projected_train_minutes']>1440
    assert max(thirty.assess(data,old))<1440
    assert data==original
    with pytest.raises(ValueError): pre.validate_evidence(data['candidate'],old,data['prepared'])
    for change in ('parity','memory','time','flag','payload','stress','pid'):
        bad = deepcopy(data); r=bad['candidate']
        if change=='parity': r['parities']['FUSION_U050_probe_bf16']['passed']=False
        if change=='memory': r['peak_rss_bytes']=300*2**30
        if change=='time': r['projected_train_minutes']=100
        if change=='flag': r['fits_debug_24h']=True
        if change=='payload': r['outputs']=[]
        if change=='stress': r['measurements'][0]['stress_batch']=64
        if change=='pid': r['isolated_children'][0]['pid']+=1
        bad['candidate']=with_content_hash(r)
        with pytest.raises(ValueError): thirty.assess(bad,old)


def test_full_v6_dry_cpu_admission_and_science_flow(finished,tmp_path,monkeypatch):
    old,parent,p = finished
    account = recovery.parse_accounting(ACCOUNT)
    active = dict(content_hash='a'*64,git_commit='a'*40)
    monkeypatch.setattr(thirty,'source',lambda *a:active)
    monkeypatch.setattr(thirty,'compatible',lambda *a:dict(fixture=True))
    monkeypatch.setattr(recovery,'accounting',lambda:account)
    monkeypatch.setattr(recovery.batch128,'location',lambda *a:parent)
    monkeypatch.setattr(recovery,'replay',lambda path,acct,project:replay.replay(load_json(path),acct))
    gate = thirty.create(failed_gate=Path(old['root'])/'gate_spec.json',root=tmp_path/'thirty',
                         project=tmp_path/'newsource',commit='a'*40)
    original_validate = c.validate_gate
    monkeypatch.setattr(c,'validate_gate',lambda g:thirty.validate_gate(g) if g['schema_version']==6 else original_validate(g))
    assert gate['schema_version']==6 and not (Path(gate['root'])/'prepared').exists()
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    calls=[]
    def submit(argv,**kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv,0,str(300000+len(calls)),'')
    monkeypatch.setattr(exact.subprocess,'run',submit)
    plan = sub.submit(gate)
    assert not calls and plan['jobs']==1
    assert '--cpus-per-task=1' in plan['commands'][0]['command']
    assert '--mem=32768M' in plan['commands'][0]['command']
    for authorization in (recovery.AUTHORIZATION,c.authorization({'schema_version':4})):
        with pytest.raises(PermissionError): sub.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=authorization)
    sub.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=thirty.AUTHORIZATION)
    assert len(calls)==1 and not any(a.startswith('--gres=') for a in calls[0])
    from hlt_classification.cms2jc2_response import measurement
    allocation = allocation_artifact('ALLOCATION',job_id='99999',cpus=1,gpus=0,partition='debug',account='reu-aisocial')
    monkeypatch.setattr(measurement,'allocation',lambda **kw:allocation)
    monkeypatch.setenv('SLURM_MEM_PER_NODE','32768')
    value = thirty.run(gate)
    assert c.evidence(gate)==(parent,p,value)
    assert not value['original_candidate']['fits_debug_24h'] and value['fits_debug_24h']
    assert value['recipe']==schedule.recipe() and value['original_candidate']['recipe']!=value['recipe']
    assert value['original_gpu_accounting']['state']=='FAILED'
    monkeypatch.setattr(sub,'accepted_jobs',lambda g:dict(preflight='99999' if g['schema_version']==6 else recovery.FAILED_JOB))
    monkeypatch.setattr(c,'preflight_accounting',lambda job:dict(job_id=job,state='COMPLETED',exit_code='0:0'))
    spec = c.create_science(gate)
    assert c.validate_spec(spec)==(parent,p,value)
    calls.clear(); plan = sub.submit(spec)
    assert not calls and plan['jobs']==30
    sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=thirty.AUTHORIZATION)
    sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=thirty.AUTHORIZATION)
    assert len(calls)==30
    assert '--dependency=afterok:300004' in calls[4] # Fresh U000 reducer precedes direct.
    assert all('--gres=gpu:a100:1' in cmd for cmd in calls[:-2])
    assert not (Path(old['root'])/'preflight/preflight.json').exists()
    with pytest.raises(FileExistsError): thirty.run(gate)


def test_too_slow_fails_before_gate_creation(finished,tmp_path,monkeypatch):
    old,_,_ = finished
    monkeypatch.setattr(thirty,'source',lambda *a:{})
    monkeypatch.setattr(thirty,'compatible',lambda *a:{})
    monkeypatch.setattr(recovery.batch128,'location',lambda *a:None)
    monkeypatch.setattr(recovery,'accounting',lambda:recovery.parse_accounting(ACCOUNT))
    monkeypatch.setattr(recovery,'replay',lambda path,acct,project:replay.replay(load_json(path),acct))
    monkeypatch.setattr(thirty,'projections',lambda _: (1441,15))
    with pytest.raises(ValueError,match='still exceeds'):
        thirty.create(failed_gate=Path(old['root'])/'gate_spec.json',root=tmp_path/'blocked',project=tmp_path,commit='a'*40)
    assert not (tmp_path/'blocked').exists()


def test_source_dispatch_is_only_change_to_training_loop_and_files_exist():
    root=Path(__file__).resolve().parents[1]
    assert all((root/name).is_file() for name in thirty.REQUIRED)
    name='src/hlt_classification/jetclass2_delphes/runner.py'
    old = ast.parse(subprocess.run(['git','show',thirty.FULLSTAT_COMMIT+':'+name],cwd=root,
        capture_output=True,text=True,check=True).stdout)
    new = ast.parse((root/name).read_text())
    # Remove ONLY the exact opt-in import/branch; rest of shared runner must match.
    train = next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='train_kernel')
    train.body=[n for n in train.body if not (isinstance(n,ast.ImportFrom) and n.module=='batch128_30e_recipe')]
    branch=next(n for n in train.body if isinstance(n,ast.If) and ast.unparse(n.test)=="config.get('contract') == thirty_contract")
    assert len(branch.orelse)==1 and isinstance(branch.orelse[0],ast.If)
    train.body[train.body.index(branch)]=branch.orelse[0]
    assert ast.dump(new)==ast.dump(old)
    helper=(root/'scripts/queue_luka_alljets_30epoch.sh').read_text()
    for required in (thirty.AUTHORIZATION,'create-30epoch','create-science','PYTHONNOUSERSITE=1','${CONDA_PREFIX}/lib'):
        assert required in helper


def test_v6_source_guard_pins_concurrent_modules_and_protects_models(tmp_path,monkeypatch):
    before,after=tmp_path/'before',tmp_path/'after'
    protected='src/hlt_classification/luka_fullsim/alljets_fusion_sync.py'
    for root in (before,after):
        (root/protected).parent.mkdir(parents=True,exist_ok=True)
        (root/protected).write_text('unchanged')
    for name in thirty.ADDITIONS|thirty.FULLSTAT:
        (after/name).parent.mkdir(parents=True,exist_ok=True)
        (after/name).write_text('donor')
    monkeypatch.setattr(thirty,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(thirty,'_files',lambda root:{p.relative_to(root).as_posix() for p in root.rglob('*.py')})
    monkeypatch.setattr(thirty.subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess(a,0,b'donor',b''))
    old=dict(project_dir=str(before),source_snapshot={})
    thirty.compatible(old,{},after)
    (after/protected).write_text('changed')
    with pytest.raises(ValueError,match='model/data'): thirty.compatible(old,{},after)
    (after/protected).write_text('unchanged')
    (after/next(iter(thirty.FULLSTAT))).write_text('unregistered edit')
    with pytest.raises(ValueError,match='Concurrent FullStat'): thirty.compatible(old,{},after)


@pytest.mark.parametrize('name',['M0HLT','FUSION_U050','FUSION_D000_D000'])
def test_installed_weaver_cuda_thirty_small_epochs(name):
    pytest.importorskip('weaver')
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    program='''
import sys
from types import SimpleNamespace
import numpy as np
import torch
from hlt_classification.context_fusion.inputs import PairedCache
from hlt_classification.jetclass2_delphes.runner import train_kernel,predict
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe,learning_rate
from hlt_classification.luka_fullsim import alljets_fusion_sync as sync,alljets_fusion_campaign as c
from test_jetclass2_dzfix_fusion_offload import inputs
raw,other=inputs()
class Cache:
    foundation_sha256='f'*64; nbytes=10000
    def __init__(self,value,role,coordinate):
        self.value=value; self.role=role; self.coordinate_name=coordinate
        self.labels=np.arange(22,dtype=np.int64)%11
        self.identities=np.arange(704,dtype=np.int64).reshape(22,32).astype(np.uint8)
        self.blocks=[SimpleNamespace(offsets=np.arange(23)*value['mask'].shape[-1])]
    def __len__(self): return 22
    def batch(self,ix):
        return dict({k:v[ix%4] for k,v in self.value.items() if k!='labels'},labels=self.labels[ix])
node=next(n for n in c.scoped_nodes(dict(schema_version=6)) if n['node_id']==sys.argv[1])
def cache(role):
    a=Cache(raw,role,node['coordinate'])
    if node['context_coordinate'] is None: return a
    b=a if node['context_coordinate']==node['coordinate'] else Cache(other,role,node['context_coordinate'])
    return PairedCache(a,b)
train,val=cache('train'),cache('validation')
if node['context_coordinate'] is None:
    assert installed_parity(train,device='cuda')['passed']
q=None if node['teacher'] is None else np.full((22,11),1/11,np.float32)
model=sync.new_model(node)
r,state=train_kernel(model,train,val,node=node,device='cuda',training_recipe=recipe(),
    teacher_probabilities=q,teacher_identities=None if q is None else train.identities)
assert r['passes']==30 and len(r['validation_history'])==30
assert r['recipe_sha256']==recipe()['content_hash']
assert [h['learning_rate'] for h in r['validation_history']]==[learning_rate(i) for i in range(1,31)]
assert all(torch.isfinite(v).all() for v in state.values())
p=predict(model,val,device='cuda',batch_size=128)
assert p.shape==(22,11) and np.isfinite(p).all()
print('FIXED30_SMALL_WEAVER_CUDA_PASS',node['node_id'],torch.cuda.get_device_name(0))
'''
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'tests'),os.environ.get('PYTHONPATH','')]),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_LAUNCH_BLOCKING='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B','-c',program,name],cwd=root,env=env,
        capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'FIXED30_SMALL_WEAVER_CUDA_PASS' in result.stdout
