"""Replacement allocation, completed-finalizer provenance and detached submission."""
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
from test_luka_fullstat_relational_ladder import fixture
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.jetclass2_delphes import execution as legacy
from hlt_classification.luka_fullstat_relational_ladder import (
    contracts as c, source, campaign, execution as ex, worker, submission, launch)


def allocation(tmp_path, monkeypatch):
    runtime=tmp_path/'runtime'; runtime.mkdir()
    monkeypatch.setattr(source,'RUNTIME',str(runtime))
    monkeypatch.setattr(sys,'prefix',str(runtime))
    fingerprint={'fixture': True}; gpu={'name':'FIXTURE L40S'}
    gate=dict(contract='LUKA_FULLSTAT_RELATIONAL_LADDER_SPEC/v3',runtime_prefix=str(runtime),
        execution_site=legacy.execution_site('oscar_l40s'),cpus=6,memory_mb=180000,
        anchor=dict(evidence=dict(environment=fingerprint,cudnn_v8_disabled='unset')))
    fields=dict(Account='default',Partition='gpu',QOS='norm-gpu',NumNodes='1',NumTasks='1',NumCPUs='6')
    for key,value in dict(SLURM_JOB_ID='8300001',SLURM_CPUS_PER_TASK='6',SLURM_MEM_PER_NODE='180000',
            SLURM_CLUSTER_NAME='slurmctld',CONDA_PREFIX=str(runtime),PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8').items():
        monkeypatch.setenv(key,value)
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED',raising=False)
    monkeypatch.setattr(ex.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(source,'installed_environment',lambda:fingerprint)
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
    'gpu_count','gpu_family','bf16','cuda','cublas','cudnn','fingerprint','measured_environment','measured_gpu'])
def test_all_replacement_runtime_guards(tmp_path,monkeypatch,bad):
    gate,measured,fields=allocation(tmp_path,monkeypatch)
    env=dict(conda='CONDA_PREFIX',user_site='PYTHONNOUSERSITE',job='SLURM_JOB_ID',cluster='SLURM_CLUSTER_NAME',
        cublas='CUBLAS_WORKSPACE_CONFIG',cudnn='TORCH_CUDNN_V8_API_DISABLED')
    scheduler=dict(account='Account',partition='Partition',qos='QOS',nodes='NumNodes',tasks='NumTasks',scheduler_cpus='NumCPUs')
    if bad in env: monkeypatch.setenv(env[bad],'wrong')
    elif bad in scheduler: fields[scheduler[bad]]='wrong'
    elif bad=='contract': gate['contract']='LUKA_FULLSTAT_RELATIONAL_LADDER_SPEC/v2'
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
    elif bad=='fingerprint': monkeypatch.setattr(source,'installed_environment',lambda:{'changed':True})
    elif bad=='measured_environment': measured['environment']={}
    else: measured['gpu']={}
    with pytest.raises((PermissionError,ValueError)): worker.execution(gate,measured)


def setup_launch(tmp_path,monkeypatch):
    s,_=fixture(tmp_path)
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
        quota_gib=16,authorization=c.LAUNCH_AUTHORIZATION)
    return s,params,calls,checks,directory


def test_auto_dry_review_and_exact_four_jobs_no_prompt(tmp_path,monkeypatch):
    s,p,calls,checks,directory=setup_launch(tmp_path,monkeypatch)
    result=launch.run(**p)
    assert len(calls)==4 and len(checks)==2
    assert '--dependency=afterok:8300001:8300002:8300003' in calls[-1]
    assert set(result['jobs'])=={'train_'+stage for stage in c.STAGES}
    assert load_json(directory/'complete.json')==result
    assert load_json(directory/'intent.json')['parent_spec']==s['reuse']['parent_spec']
    assert load_json(directory/'approval.json')['parents']['plan']==submission.render(s)['content_hash']
    assert not (directory/'failure.json').exists()
    with pytest.raises(FileExistsError): launch.run(**p)
    assert len(calls)==4


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


@pytest.mark.parametrize('contract',source.PARENT_CONTRACTS)
def test_parent_contracts_need_exact_completed_source_and_runtime(tmp_path,monkeypatch,contract):
    old=tmp_path/'old'; new=tmp_path/'new'; root=tmp_path/'parent';root.mkdir()
    for p in (old,new):
        (p/'src').mkdir(parents=True);(p/'src/native.py').write_text('# same native runtime\n')
    s=dict(project_dir=str(old),root=str(root),source_commit='a'*40,counts=c.COUNTS,
        contract=contract,runtime_prefix=source.RUNTIME,reuse=dict(environment={'accepted':True}))
    path=root/'campaign_spec.json';write_immutable_json(path,s)
    runtime=[];monkeypatch.setattr(source,'runtime',lambda e:runtime.append(e))
    def replay(args,**kw):
        if args[0]=='git': return SimpleNamespace(stdout='src/native.py\n' if 'ls-files' in args else '')
        assert kw['cwd']==old and kw['env']['PYTHONPATH']==str(old/'src')
        return SimpleNamespace(returncode=0,stdout='RELATIONAL_REUSE={}\n',stderr='')
    monkeypatch.setattr(subprocess,'run',replay)
    assert source.authenticate(path,new)['source_commit']=='a'*40
    assert runtime==[{'accepted':True}]
    (new/'src/native.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='Changed donor'): source.authenticate(path,new)


def test_execution_contract_version_and_source_boundary(tmp_path):
    s,_=fixture(tmp_path)
    assert s['contract'].endswith('/v3') and s['reuse']['contract'].endswith('/v2')
    assert c.recipe('UE100')['contract'].endswith('/v2')
    assert submission.render(s)['contract'].endswith('/v3')
    assert c.target_recipe('OFFLINE')['contract'].endswith('/v1')
    assert c.kd_recipe('OFFLINE')['contract'].endswith('/v1')
    assert submission.render(s)['authorization']==c.AUTHORIZATION
    assert source.compatible_changes(source.RECOVERY_MODIFIED|source.RECOVERY_ADDED,source.RECOVERY_ADDED)
    assert not source.compatible_changes({'src/hlt_classification/luka_fullstat/recipe.py'},set())
    assert not source.compatible_changes(source.ADDITIONS,set())
    for bad in ('/wrong/runtime',None):
        with pytest.raises(ValueError): campaign.policy(with_content_hash(dict(s['reuse'],runtime_prefix=bad)))


@pytest.mark.parametrize('kind,old_version',[('SPEC',1),('SPEC',2),('PLAN',2),('RECIPE',1),('SUBMISSION_CLAIM',2)])
def test_old_k5_registrations_are_not_silently_upgraded(kind,old_version):
    old=with_content_hash(dict(c.artifact(kind),
        contract=f'LUKA_FULLSTAT_RELATIONAL_LADDER_{kind}/v{old_version}',schema_version=old_version))
    with pytest.raises(ValueError): c.validate(old,kind)


@pytest.mark.parametrize('old_token',['AUTHORIZE_FULLSTAT_RELATIONAL_FOUR_FITS_AUTOSUBMIT_V1',
    'AUTHORIZE_FULLSTAT_RELATIONAL_LADDER_V2'])
def test_old_k5_authorization_cannot_launch_new_science(tmp_path,monkeypatch,old_token):
    _,params,calls,_,directory=setup_launch(tmp_path,monkeypatch)
    params['authorization']=old_token
    with pytest.raises(PermissionError): launch.run(**params)
    assert not calls and not (directory/'process_claim').exists()


def test_k5_recipe_cannot_pass_new_automatic_intent(tmp_path,monkeypatch):
    s,params,calls,_,directory=setup_launch(tmp_path,monkeypatch)
    # Even a rehashed current-version spec cannot use K5 under the new intent.
    r=s['recipes']['UE100']
    s['recipes']['UE100']=with_content_hash(dict(r,logit_total=5.,
        logit_weights={t:5/len(c.teachers('UE100')) for t in c.teachers('UE100')}))
    s.update(with_content_hash(s))
    with pytest.raises(ValueError,match='explicitly authorized'): launch.run(**params)
    assert not calls and (directory/'failure.json').is_file()


def test_manual_k5_authorization_cannot_submit_k2p5(tmp_path,monkeypatch):
    s,_,calls,_,_=setup_launch(tmp_path,monkeypatch)
    Path(s['root']).mkdir()
    plan=submission.submit(s)
    with pytest.raises(PermissionError):
        submission.submit(s,execute=True,plan_hash=plan['content_hash'],
            authorization='AUTHORIZE_FULLSTAT_RELATIONAL_LADDER_V2')
    assert not calls and not (Path(s['root'])/'submission/live_claim').exists()


@pytest.mark.parametrize('stage',c.STAGES)
def test_rehashed_k5_recipe_rejected_by_campaign_policy(tmp_path,monkeypatch,stage):
    s,_=fixture(tmp_path)
    s=with_content_hash(dict(s,counts=dict(c.COUNTS)))
    monkeypatch.setattr(source,'current',lambda *a:s['source_snapshot'])
    monkeypatch.setattr(source,'authenticate',lambda *a:s['reuse'])
    monkeypatch.setattr(campaign,'load_json',lambda *a:s)
    monkeypatch.setattr(campaign,'protected',lambda *a:None)
    assert campaign.check(s)==s['reuse']
    r=s['recipes'][stage]
    s['recipes'][stage]=with_content_hash(dict(r,logit_total=5.,
        logit_weights={t:5/len(c.teachers(stage)) for t in c.teachers(stage)}))
    s.update(with_content_hash(s))
    with pytest.raises(ValueError,match='registration changed'): campaign.check(s)


def test_detached_shell_and_runtime_activation_contract():
    project=Path(__file__).resolve().parents[1]
    shell=(project/'scripts/launch_luka_fullstat_relational_ladder.sh').read_text()
    # Detachment and redirects precede every slow operation; foreground is only tail.
    assert 'nohup setsid bash' in shell and '>>"${LOG}" 2>&1 </dev/null &' in shell
    assert shell.index(': >"${LOG}"') < shell.index("trap '' HUP") < shell.index('nohup setsid bash')
    assert shell.index('trap - HUP') < shell.index('exec tail')
    assert 'exec tail -n +1 -f' in shell
    assert shell.index('exec tail') < shell.index('status --porcelain') < shell.index('source "${PROJECT_DIR}/sbatch')
    assert 'mkdir -- "${LAUNCH}"' in shell and 'read -' not in shell
    assert 'rm ' not in shell and 'scancel' not in shell
    for relative in ('scripts/queue_luka_fullstat_relational_ladder.sh',
            'scripts/launch_luka_fullstat_relational_ladder.sh','sbatch/run_luka_fullstat_relational_ladder.sh'):
        assert 'sbatch/luka_splithead_finalize_common.sh' in (project/relative).read_text()
    subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n',
        'scripts/launch_luka_fullstat_relational_ladder.sh'],cwd=project,check=True,capture_output=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real installed-Weaver finalizer-to-relational readback')
def test_completed_finalizer_replay_uses_imported_fit_not_missing_local_copy(tmp_path,monkeypatch,capsys):
    from test_luka_fullstat_splithead_finalize import exported_failure,create
    from hlt_classification.luka_fullstat_splithead import finalize,worker as previous_worker,submission as previous_submission
    old,_,_=exported_failure(tmp_path,monkeypatch,gpu=True)
    parent=create(tmp_path,old)
    finalize.claim(parent,previous_submission.render(parent))
    monkeypatch.setenv('SLURM_JOB_ID','8200001')
    previous_worker.run(parent,'finalize_DE033')
    assert not (Path(parent['root'])/'train_DE033/selected.pt').exists()
    path=Path(parent['root'])/'campaign_spec.json'
    monkeypatch.setattr(sys,'argv',['replay',str(path)])
    with pytest.raises(ValueError,match='Wait for'): exec(compile(source.REPLAY,'replay','exec'),{})
    monkeypatch.setenv('SLURM_JOB_ID','8200002')
    previous_worker.run(parent,'train_DE000')
    exec(compile(source.REPLAY,'replay','exec'),{})
    lines=[x.partition('=')[2] for x in capsys.readouterr().out.splitlines() if x.startswith('RELATIONAL_REUSE=')]
    evidence=json.loads(lines[-1])
    refs=evidence['references']['OLD_SPLITHEAD_DE033']
    assert refs['checkpoint']==parent['external_refs']['DE033']['checkpoint']
    assert Path(refs['receipt']['path']).parent==Path(parent['root'])/'train_DE033'
    reused=c.artifact('REUSE',counts=c.COUNTS,**evidence)
    assert campaign.policy(reused)['runtime_prefix']==source.RUNTIME
    assert evidence['jobs']==dict(finalize_DE033='8200001',train_DE000='8200002')


@pytest.mark.skipif(sys.platform!='linux' or not shutil.which('setsid'),reason='Real HUP/setsid requires Linux')
def test_hangup_of_log_view_does_not_kill_detached_submission(tmp_path):
    # Production launcher, real nohup/setsid/HUP. Git/runtime/submission are
    # synthetic: this never connects to a cluster or invokes sbatch.
    repo=tmp_path/'project';(repo/'scripts').mkdir(parents=True);(repo/'sbatch').mkdir()
    original=Path(__file__).resolve().parents[1]/'scripts/launch_luka_fullstat_relational_ladder.sh'
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
        assert 'Detached launcher PID:' in child.stdout.readline()
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
    original=Path(__file__).resolve().parents[1]/'scripts/launch_luka_fullstat_relational_ladder.sh'
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
    assert 'TAIL_ATTACHED' in result.stdout and 'Detached launcher PID:' in result.stdout
    deadline=time.monotonic()+10
    while not done.exists() and time.monotonic()<deadline: time.sleep(.05)
    log=Path(str(root)+'.launch/launcher.log')
    assert done.read_text()=='done',log.read_text()
    assert 'DETACHED_FINISHED' in log.read_text()
    again=subprocess.run(args,env=env,capture_output=True,text=True,timeout=5)
    assert again.returncode!=0
