"""Real execution-check dispatch plus immutable retry of job7963448."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import pytest
import torch
from test_luka_embedding_aux import threads
from test_luka_fullstat_splithead_finalize import failed, exported_failure, create
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes import execution as legacy
from hlt_classification.luka_fullstat_splithead import worker, campaign, submission, training, banks, kd_banks
from hlt_classification.luka_fullstat_splithead import finalize as f, finalize_source as fs, finalize_execution as fx, finalize_retry as fr
from hlt_classification.luka_fullstat_splithead.contracts import artifact

REAL_EXECUTION = worker.execution


def allocation_fixture(tmp_path, monkeypatch, gate=None, *, real_cuda=False):
    runtime=tmp_path/'runtime'; runtime.mkdir(exist_ok=True)
    monkeypatch.setattr(fs,'RUNTIME',str(runtime))
    monkeypatch.setattr(sys,'prefix',str(runtime))
    fingerprint=dict(fixture=True)
    gpu=dict(name='FIXTURE L40S',total_memory_bytes=48*2**30,compute_capability=[8,9])
    if gate is None:
        gate=dict(contract=fr.CONTRACT,runtime_prefix=str(runtime),execution_site=legacy.execution_site('oscar_l40s'),
            cpus=6,memory_mb=180000,anchor=dict(evidence=dict(environment=fingerprint,cudnn_v8_disabled='unset')))
    assert gate['runtime_prefix']==str(runtime)
    fields=dict(Account='default',Partition='gpu',QOS='norm-gpu',NumNodes='1',NumTasks='1',NumCPUs='6')
    for key,value in dict(SLURM_JOB_ID='8200001',SLURM_CPUS_PER_TASK='6',SLURM_MEM_PER_NODE='180000',
            SLURM_CLUSTER_NAME='slurmctld',CONDA_PREFIX=str(runtime),PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8').items():
        monkeypatch.setenv(key,value)
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED',raising=False)
    monkeypatch.setattr(fx.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(fs,'installed_environment',lambda:fingerprint)
    monkeypatch.setattr(fx,'installed_environment',lambda:fingerprint)
    monkeypatch.setattr(fx,'gpu_identity',lambda:gpu)
    if not real_cuda:
        monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
        monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
        monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'FIXTURE L40S')
    def scontrol(args, **kw):
        assert args==['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']]
        return SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items()))
    monkeypatch.setattr(subprocess,'run',scontrol)
    return gate,dict(environment=fingerprint,gpu=gpu),fields


def test_reproduces_old_prefix_failure_and_real_new_worker_dispatch(tmp_path,monkeypatch):
    gate,measured,_=allocation_fixture(tmp_path,monkeypatch)
    with pytest.raises(PermissionError,match='host/Conda environment'):
        legacy.allocation(gate['execution_site'])
    # Neither worker.execution nor the replacement execution check is mocked.
    assert worker.execution(gate,measured)=='8200001'
    old=dict(gate,contract='LUKA_FULLSTAT_SPLITHEAD_RECOVERY_SPEC/v1')
    with pytest.raises(PermissionError,match='host/Conda environment'): worker.execution(old,measured)
    assert gate['execution_site']==legacy.execution_site('oscar_l40s')


@pytest.mark.parametrize('bad',['contract','registered_prefix','prefix','conda','user_site','arch','job','account',
    'partition','qos','cluster','nodes','tasks','scheduler_cpus','cpu_envelope','allocation_cpus','memory',
    'gpu_count','gpu_family','bf16','cuda','cublas','cudnn','fingerprint','measured_environment','measured_gpu'])
def test_replacement_allocator_retains_every_guard(tmp_path,monkeypatch,bad):
    gate,measured,fields=allocation_fixture(tmp_path,monkeypatch)
    env=dict(conda='CONDA_PREFIX',user_site='PYTHONNOUSERSITE',job='SLURM_JOB_ID',cluster='SLURM_CLUSTER_NAME',
        cublas='CUBLAS_WORKSPACE_CONFIG',cudnn='TORCH_CUDNN_V8_API_DISABLED')
    scheduler=dict(account='Account',partition='Partition',qos='QOS',nodes='NumNodes',tasks='NumTasks',scheduler_cpus='NumCPUs')
    if bad in env: monkeypatch.setenv(env[bad],'wrong')
    elif bad in scheduler: fields[scheduler[bad]]='wrong'
    elif bad=='contract': gate['contract']='unregistered'
    elif bad=='registered_prefix': gate['runtime_prefix']='wrong'
    elif bad=='prefix': monkeypatch.setattr(sys,'prefix','wrong')
    elif bad=='arch': monkeypatch.setattr(fx.platform,'machine',lambda:'aarch64')
    elif bad=='cpu_envelope': monkeypatch.setenv('SLURM_CPUS_PER_TASK','99')
    elif bad=='allocation_cpus': gate['cpus']=4
    elif bad=='memory': gate['memory_mb']=170000
    elif bad=='gpu_count': monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    elif bad=='gpu_family': monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'other')
    elif bad=='bf16': monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
    elif bad=='cuda': monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    elif bad=='fingerprint': monkeypatch.setattr(fs,'installed_environment',lambda:{'changed':True})
    elif bad=='measured_environment': measured['environment']={}
    else: measured['gpu']={}
    with pytest.raises((PermissionError,ValueError)): fx.execution(gate,measured)


def failed_finalizer(tmp_path,monkeypatch,old):
    previous=create(tmp_path,old)
    f.claim(previous,submission.render(previous))
    root=Path(previous['root']); (root/'submission').mkdir()
    write_immutable_json(root/'submission/command_plan.json',submission.render(previous))
    jobs=dict(finalize_DE033=fr.FAILED_JOB,train_DE000='999002')  # Read from ledger, never arithmetic.
    write_immutable_json(root/'submission/submission_ledger.json',dict(jobs=jobs))
    (root/('finalize_DE033-'+fr.FAILED_JOB+'.out')).write_text(
        "job = worker.execution(spec, spec['reuse'])\nallocation(gate['execution_site'])\n"
        'PermissionError: Worker host/Conda environment differs from the execution site\n')
    monkeypatch.setattr(fr,'replay',lambda path:(previous,jobs))
    monkeypatch.setattr(fr,'accounting',lambda *a:dict(failed=fr.FAILED_JOB,cancelled='999002'))
    return previous


def create_retry(tmp_path,old,previous):
    return f.create(tmp_path/'retry',Path(old['root'])/'campaign_spec.json',tmp_path/'retrysource',
        'e'*40,quota_gib=2,failed_finalization=Path(previous['root'])/'campaign_spec.json')


def test_authenticated_retry_two_jobs_preserves_claim_and_rejects_duplicate(failed,tmp_path,monkeypatch):
    old,_,_=failed
    previous=failed_finalizer(tmp_path,monkeypatch,old)
    preserved={p:file_ref(p) for root in (old['root'],previous['root']) for p in Path(root).rglob('*') if p.is_file()}
    s=create_retry(tmp_path,old,previous)
    assert campaign.check(s)==old and s['contract']==fr.CONTRACT
    assert s['recipes']==old['recipes'] and s['counts']==old['counts']
    assert s['retry']['jobs']['train_DE000']=='999002'
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    calls=[]
    def submit(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=str(8200000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',submit)
    plan=submission.submit(s)
    assert plan['jobs']==2 and plan['preflight_jobs']==0 and plan['scientific_fits']==1
    assert plan['contract']=='LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_PLAN/v1'
    with pytest.raises(PermissionError):
        submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=f.AUTHORIZATION)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=fr.AUTHORIZATION)
    assert len(calls)==2 and '--dependency=afterok:8200001' in calls[1]
    assert all('7963448' not in ' '.join(args) and '999002' not in ' '.join(args) for args in calls)
    f.verify_claim(s)
    assert preserved=={p:file_ref(p) for p in preserved}
    assert (Path(previous['root'])/'retry_after_7963448_v1/intent.json').is_file()
    with pytest.raises(FileExistsError): f.claim(dict(s,root=str(tmp_path/'duplicate')),plan)


@pytest.mark.parametrize('started',['finalize_DE033','train_DE033','train_DE000_attempt','targets','bundles'])
def test_retry_rejects_started_work(failed,tmp_path,monkeypatch,started):
    old,_,_=failed; previous=failed_finalizer(tmp_path,monkeypatch,old)
    (Path(previous['root'])/started).mkdir()
    with pytest.raises(ValueError,match='already started'): create_retry(tmp_path,old,previous)
    assert not (tmp_path/'retry').exists()


@pytest.mark.parametrize('bad',['none','running','started','wrong_owner','wrong_name','wrong_account','wrong_exit','missing','duplicate'])
def test_retry_requires_failed_exact_job_and_never_started_cancelled_descendant(monkeypatch,bad):
    previous=dict(execution_site=dict(account='default'))
    jobs=dict(finalize_DE033=fr.FAILED_JOB,train_DE000='999002')
    rows=[[fr.FAILED_JOB,'lukafssf_finalize_DE033','rlyang','default','FAILED','1:0','00:03:55'],
        ['999002','lukafssf_train_DE000','rlyang','default','CANCELLED','0:0','00:00:00']]
    if bad=='running': rows[1][4]='RUNNING'
    if bad=='started': rows[1][6]='00:00:01'
    if bad.startswith('wrong_'): rows[0][dict(wrong_owner=2,wrong_name=1,wrong_account=3,wrong_exit=5)[bad]]='wrong'
    if bad=='missing': rows.pop()
    if bad=='duplicate': rows.append(rows[0])
    monkeypatch.setattr(fr.getpass,'getuser',lambda:'rlyang')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='\n'.join('|'.join(r) for r in rows)))
    if bad=='none': assert len(fr.accounting(previous,jobs))==2
    else:
        with pytest.raises(ValueError): fr.accounting(previous,jobs)


@pytest.mark.parametrize('bad',['none','spec','jobs','absent','duplicate','exit'])
def test_failed_finalizer_original_source_replay(tmp_path,monkeypatch,bad):
    project=tmp_path/'project';project.mkdir()
    previous=artifact('FINALIZE_SPEC',source_commit=fr.FAILED_COMMIT,root=str(tmp_path),project_dir=str(project),
        runtime_prefix=fs.RUNTIME,reuse=dict(environment={}))
    path=tmp_path/'campaign_spec.json';write_immutable_json(path,previous)
    result=dict(spec=deepcopy(previous),jobs=dict(finalize_DE033=fr.FAILED_JOB,train_DE000='999002'))
    if bad=='spec': result['spec']['source_commit']='changed'
    if bad=='jobs': result['jobs']['finalize_DE033']='wrong'
    line='FAILED_FINALIZER='+json.dumps(result)+'\n'
    monkeypatch.setattr(fs,'runtime',lambda _:None)
    def replay(args,**kw):
        assert args[0]==sys.executable and kw['cwd']==project
        assert kw['env']['PYTHONPATH']==str(project/'src')
        return SimpleNamespace(returncode=int(bad=='exit'),stderr='failure',
            stdout='' if bad=='absent' else line*2 if bad=='duplicate' else line)
    monkeypatch.setattr(subprocess,'run',replay)
    if bad=='none': assert fr.replay(path)==(previous,result['jobs'])
    else:
        with pytest.raises(ValueError): fr.replay(path)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed-Weaver GPU required')
def test_retry_full20_DE000_with_actual_execution_dispatch(tmp_path,monkeypatch):
    # The original data/source/scheduler remain synthetic. The dispatcher and
    # every allocation/runtime guard run, unlike the original missed test.
    runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(fs,'RUNTIME',str(runtime))
    old,_,_=exported_failure(tmp_path,monkeypatch,gpu=True)
    previous=failed_finalizer(tmp_path,monkeypatch,old)
    s=create_retry(tmp_path,old,previous)
    f.claim(s,submission.render(s))
    original={p:file_ref(p) for p in Path(old['root']).rglob('*') if p.is_file()}
    allocation_fixture(tmp_path,monkeypatch,s,real_cuda=True)
    monkeypatch.setattr(worker,'execution',REAL_EXECUTION)
    def forbidden(*a,**kw): pytest.fail('Finalizer must not train or regenerate targets')
    with monkeypatch.context() as patch:
        for mod,name in ((worker,'fit'),(worker,'startup'),(training,'train'),(banks,'export'),(kd_banks,'export')):
            patch.setattr(mod,name,forbidden)
        worker.run(s,'finalize_DE033')
    assert worker.completed(s,'DE033')
    monkeypatch.setenv('SLURM_JOB_ID','8200002')
    worker.run(s,'train_DE000')
    assert worker.completed(s,'DE000') and worker.results(s)['pending']==[]
    assert original=={p:file_ref(p) for p in original}
