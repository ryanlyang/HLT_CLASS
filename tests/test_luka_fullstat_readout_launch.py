"""Replacement allocation, completed U100-w5 provenance and detached submission."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import json
import shutil
import signal
import subprocess
import sys
import time
import pytest
import torch
from test_luka_fullstat_readout import fixture_spec as fixture
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.jetclass2_delphes import execution as legacy
from hlt_classification.luka_fullstat_readout import (
    contracts as c, source, campaign, execution as ex, worker, submission, launch)


def allocation(tmp_path, monkeypatch):
    runtime=tmp_path/'runtime'; runtime.mkdir()
    monkeypatch.setattr(source,'RUNTIME',str(runtime))
    monkeypatch.setattr(sys,'prefix',str(runtime))
    fingerprint={'fixture': True}; gpu={'name':'FIXTURE L40S'}
    gate=dict(contract='LUKA_FULLSTAT_READOUT_SPEC/v1',runtime_prefix=str(runtime),
        execution_site=legacy.execution_site('oscar_l40s'),cpus=6,memory_mb=180000,minutes={'train_readout':1440},
        anchor=dict(evidence=dict(environment=fingerprint,cudnn_v8_disabled='unset')))
    fields=dict(Account='default',Partition='gpu',QOS='norm-gpu',NumNodes='1',NumTasks='1',NumCPUs='6',TimeLimit='1-00:00:00')
    for key,value in dict(SLURM_JOB_ID='8300001',SLURM_CPUS_PER_TASK='6',SLURM_MEM_PER_NODE='180000',
            SLURM_CLUSTER_NAME='slurmctld',CONDA_PREFIX=str(runtime),PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8').items():
        monkeypatch.setenv(key,value)
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED',raising=False)
    monkeypatch.setattr(ex.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(source,'runtime',lambda expected: None if expected==fingerprint else (_ for _ in ()).throw(ValueError('Fingerprint differs')))
    monkeypatch.setattr(ex,'installed_environment',lambda:fingerprint)
    monkeypatch.setattr(ex,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'FIXTURE L40S')
    def scontrol(args,**kw):
        assert args==['scontrol','show','job','-o','8300001']
        return SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items()))
    monkeypatch.setattr(subprocess,'run',scontrol)
    return gate,dict(environment=fingerprint,gpu=gpu),fields


def test_worker_uses_actual_replacement_allocator_not_legacy(tmp_path,monkeypatch):
    gate,measured,_=allocation(tmp_path,monkeypatch)
    with pytest.raises(PermissionError,match='host/Conda'): legacy.allocation(gate['execution_site'])
    assert worker.execution is ex.execution
    assert worker.execution(gate,measured)=='8300001'
    assert gate['execution_site']==legacy.execution_site('oscar_l40s')


@pytest.mark.parametrize('bad',['contract','registered_prefix','prefix','conda','user_site','arch','job','account',
    'partition','qos','cluster','nodes','tasks','scheduler_cpus','cpu_envelope','allocation_cpus','memory',
    'gpu_count','gpu_family','bf16','cuda','cublas','cudnn','fingerprint','measured_environment','measured_gpu','walltime'])
def test_all_replacement_runtime_guards(tmp_path,monkeypatch,bad):
    gate,measured,fields=allocation(tmp_path,monkeypatch)
    env=dict(conda='CONDA_PREFIX',user_site='PYTHONNOUSERSITE',job='SLURM_JOB_ID',cluster='SLURM_CLUSTER_NAME',
        cublas='CUBLAS_WORKSPACE_CONFIG',cudnn='TORCH_CUDNN_V8_API_DISABLED')
    scheduler=dict(account='Account',partition='Partition',qos='QOS',nodes='NumNodes',tasks='NumTasks',scheduler_cpus='NumCPUs')
    if bad in env: monkeypatch.setenv(env[bad],'wrong')
    elif bad in scheduler: fields[scheduler[bad]]='wrong'
    elif bad=='contract': gate['contract']='LUKA_FULLSTAT_READOUT_SPEC/v0'
    elif bad=='registered_prefix': gate['runtime_prefix']='wrong'
    elif bad=='prefix': monkeypatch.setattr(sys,'prefix','wrong')
    elif bad=='arch': monkeypatch.setattr(ex.platform,'machine',lambda:'aarch64')
    elif bad=='cpu_envelope': monkeypatch.setenv('SLURM_CPUS_PER_TASK','99')
    elif bad=='allocation_cpus': gate['cpus']=4
    elif bad=='memory': gate['memory_mb']=170000
    elif bad=='gpu_count': monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    elif bad=='gpu_family': monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'other')
    elif bad=='bf16': monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
    elif bad=='cuda': monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    elif bad=='fingerprint': gate['anchor']['evidence']['environment']={'changed':True}
    elif bad=='walltime': fields['TimeLimit']='12:00:00'
    elif bad=='measured_environment': measured['environment']={}
    else: measured['gpu']={}
    with pytest.raises((PermissionError,ValueError)): worker.execution(gate,measured)


def setup_launch(tmp_path,monkeypatch):
    s=fixture(tmp_path)
    parent=tmp_path/'parent.json';write_immutable_json(parent,{})
    s['reuse']['parent_spec']=file_ref(parent)
    root=tmp_path/'new_campaign'; directory=Path(str(root)+'.launch'); directory.mkdir()
    s=with_content_hash(dict(s,root=str(root)))
    def create(*args,**kwargs):
        root.mkdir()
        write_immutable_json(root/'campaign_spec.json',s)
        return s
    monkeypatch.setattr(campaign,'create',create)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    checks=[]
    monkeypatch.setattr(submission,'check_site',lambda plan:checks.append(plan))
    calls=[]
    def submit(args,**kw):
        assert args[0]=='sbatch'
        assert (directory/'approval.json').is_file()
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=str(8300000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',submit)
    params=dict(root=root,previous_spec=s['reuse']['parent_spec']['path'],project='/project',commit='f'*40,
        quota_gib=4,authorization=c.LAUNCH_AUTHORIZATION)
    return s,params,calls,checks,directory


def test_auto_dry_review_and_exact_one_job_no_prompt(tmp_path,monkeypatch):
    s,p,calls,checks,directory=setup_launch(tmp_path,monkeypatch)
    result=launch.run(**p)
    assert len(calls)==1 and len(checks)==2
    assert not any(v.startswith('--dependency=') for v in calls[0])
    assert set(result['jobs'])=={'train_readout'}
    assert load_json(directory/'complete.json')==result
    assert load_json(directory/'intent.json')['parent_spec']==s['reuse']['parent_spec']
    assert load_json(directory/'approval.json')['parents']['plan']==submission.render(s)['content_hash']
    assert not (directory/'failure.json').exists()
    with pytest.raises(FileExistsError): launch.run(**p)
    assert len(calls)==1


@pytest.mark.parametrize('failure',['auth','incomplete_parent','bad_plan','ambiguous_sbatch'])
def test_auto_launch_failures_preserve_evidence_without_retry(tmp_path,monkeypatch,failure):
    s,p,calls,_,directory=setup_launch(tmp_path,monkeypatch)
    if failure=='auth':
        p['authorization']='wrong'
        with pytest.raises(PermissionError): launch.run(**p)
        assert not (directory/'process_claim').exists()
        return
    if failure=='incomplete_parent':
        def fail(*a,**kw): raise ValueError('Wait for completed parent')
        monkeypatch.setattr(campaign,'create',fail)
    elif failure=='bad_plan':
        original=submission.submit
        def fail(spec,**kw):
            value=original(spec,**kw); value['commands'][0]['command'].append('--extra-job')
            return value
        monkeypatch.setattr(submission,'submit',fail)
    else:
        def fail(args,**kw):
            calls.append(args)
            return SimpleNamespace(returncode=0,stdout='ambiguous\n',stderr='')
        monkeypatch.setattr(subprocess,'run',fail)
    with pytest.raises(Exception): launch.run(**p)
    assert len(calls)==(1 if failure=='ambiguous_sbatch' else 0)
    assert (directory/'failure.json').is_file()
    assert not (directory/'complete.json').exists()
    with pytest.raises(FileExistsError): launch.run(**p)
    assert len(calls)==(1 if failure=='ambiguous_sbatch' else 0)


@pytest.mark.skipif(sys.platform!='linux' or not shutil.which('setsid'),reason='Real HUP/setsid requires Linux')
def test_hangup_of_log_view_does_not_kill_detached_submission(tmp_path):
    # Production launcher, real nohup/setsid/HUP. Git/runtime/submission are
    # synthetic: this never connects to a cluster or invokes sbatch.
    repo=tmp_path/'project';(repo/'scripts').mkdir(parents=True);(repo/'sbatch').mkdir()
    original=Path(__file__).resolve().parents[1]/'scripts/launch_luka_fullstat_readout.sh'
    script=repo/'scripts'/original.name;script.write_bytes(original.read_bytes())
    (repo/'sbatch/luka_splithead_finalize_common.sh').write_text('# synthetic test runtime\n')
    binpath=tmp_path/'bin';binpath.mkdir()
    git=binpath/'git';git.write_text('#!/usr/bin/env bash\ncase "$*" in\n*--show-toplevel*) printf "%s\\n" "$FIXTURE_PROJECT";;\n*"rev-parse HEAD"*) printf "%040d\\n" 1;;\nesac\n');git.chmod(0o755)
    python=binpath/'python';python.write_text('#!/usr/bin/env bash\nsleep 1\nprintf "DETACHED_FINISHED\\n"\nprintf done > "$FIXTURE_DONE"\n');python.chmod(0o755)
    parent=tmp_path/'parent.json';parent.write_text('{}')
    root=tmp_path/'output';done=tmp_path/'done'
    env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],FIXTURE_PROJECT=str(repo),FIXTURE_DONE=str(done))
    child=subprocess.Popen(['bash',str(script),'0'*39+'1',str(root),str(parent),'16',
        '--authorize',c.LAUNCH_AUTHORIZATION],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        assert 'DETACHED: launcher PID' in child.stdout.readline()
        assert 'Reconnect with:' in ''.join(child.stdout.readline() for _ in range(4))
        os.kill(child.pid,signal.SIGHUP)
        child.wait(timeout=5)
        deadline=time.monotonic()+10
        while not done.exists() and time.monotonic()<deadline: time.sleep(.05)
        assert done.read_text()=='done'
        assert 'DETACHED_FINISHED' in Path(str(root)+'.launch/launcher.log').read_text()
        again=subprocess.run(['bash',str(script),'0'*39+'1',str(root),str(parent),'16',
            '--authorize',c.LAUNCH_AUTHORIZATION],env=env,capture_output=True,text=True,timeout=5)
        assert again.returncode!=0
    finally:
        if child.poll() is None: child.terminate();child.wait(timeout=5)


def test_shell_smoke_tailing_can_exit_while_background_finishes(tmp_path):
    """Actual Bash launch flow; setsid is stubbed on Windows (not a HUP test)."""
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def shellpath(p):
        p=Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    repo=tmp_path/'project';(repo/'scripts').mkdir(parents=True);(repo/'sbatch').mkdir()
    original=Path(__file__).resolve().parents[1]/'scripts/launch_luka_fullstat_readout.sh'
    script=repo/'scripts'/original.name;script.write_bytes(original.read_bytes())
    (repo/'sbatch/luka_splithead_finalize_common.sh').write_text('# fixture runtime\n')
    binpath=tmp_path/'bin';binpath.mkdir()
    bodies={
        'git':'case "$*" in\n*--show-toplevel*) printf "%s\\n" "$FIXTURE_PROJECT";;\n*"rev-parse HEAD"*) printf "%040d\\n" 1;;\nesac\n',
        'setsid':'exec "$@"\n',
        'tail':'printf "TAIL_ATTACHED\\n"\n',
        'python':'sleep 1\nprintf "DETACHED_FINISHED\\n"\nprintf done > "$FIXTURE_DONE"\n'}
    for name,body in bodies.items():
        p=binpath/name;p.write_bytes(('#!/usr/bin/env bash\n'+body).encode());p.chmod(0o755)
    parent=tmp_path/'parent.json';parent.write_text('{}')
    root=tmp_path/'output';done=tmp_path/'done'
    env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],FIXTURE_PROJECT=shellpath(repo),FIXTURE_DONE=shellpath(done))
    # Git Bash prepends its own binary directories on Windows startup. Keep
    # synthetic command fixtures first in each child shell, not system git.
    startup=tmp_path/'fixture_env.sh';startup.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    env.update(BASH_ENV=shellpath(startup),FIXTURE_BIN=shellpath(binpath))
    args=[bash,str(script),'0'*39+'1',shellpath(root),shellpath(parent),'16','--authorize',c.LAUNCH_AUTHORIZATION]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'TAIL_ATTACHED' in result.stdout and 'DETACHED: launcher PID' in result.stdout
    deadline=time.monotonic()+10
    while not done.exists() and time.monotonic()<deadline: time.sleep(.05)
    log=Path(str(root)+'.launch/launcher.log')
    assert done.read_text()=='done',log.read_text()
    assert 'DETACHED_FINISHED' in log.read_text()
    again=subprocess.run(args,env=env,capture_output=True,text=True,timeout=5)
    assert again.returncode!=0

def test_parent_needs_exact_completed_source_and_runtime(tmp_path,monkeypatch):
    contract='LUKA_FULLSTAT_U100_EMBEDDING_SPEC/v1'
    old=tmp_path/'old'; new=tmp_path/'new'; root=tmp_path/'parent';root.mkdir()
    for p in (old,new):
        (p/'src').mkdir(parents=True);(p/'src/native.py').write_text('# same native runtime\n')
    s=dict(project_dir=str(old),root=str(root),source_commit=source.DONOR,counts=c.COUNTS,
        contract=contract,runtime_prefix=source.RUNTIME,reuse=dict(environment={'accepted':True}))
    path=root/'campaign_spec.json';write_immutable_json(path,s)
    runtime=[];monkeypatch.setattr(source,'runtime',lambda e:runtime.append(e))
    def replay(args,**kw):
        if args[0]=='git': return SimpleNamespace(stdout='src/native.py\n' if 'ls-files' in args else '')
        assert kw['cwd']==old and kw['env']['PYTHONPATH']==str(old/'src')
        return SimpleNamespace(returncode=0,stdout='FULLSTAT_READOUT_REUSE={}\n',stderr='')
    monkeypatch.setattr(subprocess,'run',replay)
    assert source.authenticate(path,new)['source_commit']==source.DONOR
    assert runtime==[{'accepted':True}]
    (new/'src/native.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='Changed donor'): source.authenticate(path,new)


def test_detached_shell_and_runtime_activation_contract():
    project=Path(__file__).resolve().parents[1]
    shell=(project/'scripts/launch_luka_fullstat_readout.sh').read_text()
    # Detachment and redirects precede every slow operation; foreground is only tail.
    assert 'nohup setsid bash' in shell and '>>"${LOG}" 2>&1 </dev/null &' in shell
    assert shell.index(': >"${LOG}"') < shell.index("trap '' HUP") < shell.index('nohup setsid bash')
    assert shell.index('trap - HUP') < shell.index('exec tail')
    assert 'exec tail -n +1 -f' in shell
    assert shell.index('exec tail') < shell.index('status --porcelain') < shell.index('source "${PROJECT_DIR}/sbatch')
    assert 'mkdir -- "${LAUNCH}"' in shell and 'read -' not in shell
    assert 'rm ' not in shell and 'scancel' not in shell
    for relative in ('scripts/queue_luka_fullstat_readout.sh',
            'scripts/launch_luka_fullstat_readout.sh','sbatch/run_luka_fullstat_readout.sh'):
        assert 'sbatch/luka_splithead_finalize_common.sh' in (project/relative).read_text()
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    subprocess.run([bash,'-n',
        'scripts/launch_luka_fullstat_readout.sh'],cwd=project,check=True,capture_output=True)


def test_exact_additive_source_and_required_files(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    assert all((project/p).is_file() for p in source.REQUIRED)
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    changed=set(source.ADDITIONS|source.ENTRYPOINTS)
    def git(args,**kw):
        lines=source.REQUIRED if 'ls-files' in args else changed if 'diff' in args else []
        return SimpleNamespace(stdout='\n'.join(sorted(lines)))
    monkeypatch.setattr(subprocess,'run',git)
    assert source.current(project,'f'*40)==dict(content_hash='a'*64)
    changed.add('src/hlt_classification/luka_fullstat_u100_embedding/training.py')
    with pytest.raises(ValueError,match='additive'): source.current(project,'f'*40)
    assert not source.compatible_changes(set(source.ADDITIONS),set())


@pytest.mark.parametrize('field,value',[('counts',{}),('source_commit','b'*40),
    ('contract','LUKA_U100_MULTIVIEW_SPEC/v1')])
def test_wrong_parent_rejected_before_replay(tmp_path,monkeypatch,field,value):
    root=tmp_path/'parent';root.mkdir()
    s=dict(project_dir='/old',root=str(root),source_commit=source.DONOR,counts=c.COUNTS,
           contract='LUKA_FULLSTAT_U100_EMBEDDING_SPEC/v1')
    s[field]=value; path=root/'campaign_spec.json';write_immutable_json(path,s)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:pytest.fail('Invalid parent reached replay'))
    with pytest.raises(ValueError,match='original FullStat'): source.authenticate(path,tmp_path)


def test_failed_final_publication_preserves_all_training(tmp_path,monkeypatch):
    from test_luka_fullstat_readout import worker_fixture
    from test_luka_embedding_aux import tiny
    tiny(monkeypatch); s,*_=worker_fixture(tmp_path,monkeypatch,'cpu')
    monkeypatch.setattr(worker,'installed_parity',lambda *a,**kw:None)
    def fail(*a): raise RuntimeError('Simulated final report failure')
    monkeypatch.setattr(worker,'validate_fit',fail)
    with pytest.raises(RuntimeError,match='Simulated'): worker.run(s,'train_readout',device='cpu')
    folder=Path(s['root'])/'train_readout'
    assert len(list((folder/'epochs').glob('*/state.pt')))==20
    assert (folder/'selected.pt').is_file() and (folder/'selected_training_state.pt').is_file()
    assert load_json(folder/'completed_training.json')['passes']==20
    assert not (folder/'task_receipt.json').exists()
    with pytest.raises(FileExistsError): worker.run(s,'train_readout',device='cpu')
