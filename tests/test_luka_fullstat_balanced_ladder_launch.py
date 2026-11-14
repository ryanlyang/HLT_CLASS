"""Bounded launch, allocator guards and real Bash detachment smoke."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import time
import pytest
import torch
from test_luka_fullstat_balanced_ladder_workflow import spec_fixture
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.luka_fullstat_balanced_ladder import contracts as c,launch,campaign,submission,execution,source


def test_detached_python_dry_approval_then_six_exact_submissions(tmp_path,monkeypatch):
    root=tmp_path/'study';directory=Path(str(root)+'.launch');directory.mkdir()
    parent=tmp_path/'parent.json';parent.write_text('{}')
    s=spec_fixture(root)
    def create(*a,**kw):root.mkdir();meta.write_immutable_json(root/'campaign_spec.json',s);return s
    monkeypatch.setattr(campaign,'create',create);monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None);calls=[]
    def submit(args,**kw):
        assert (directory/'approval.json').exists()
        calls.append(args);return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',submit)
    args=(root,parent,'/project','f'*40,36,c.LAUNCH_AUTHORIZATION)
    result=launch.run(*args)
    assert len(calls)==6 and result==meta.load_json(directory/'complete.json')
    with pytest.raises(FileExistsError):launch.run(*args)
    assert len(calls)==6


@pytest.mark.parametrize('failure',['authorization','registration','ambiguous'])
def test_launch_stops_preserves_evidence_never_retries(tmp_path,monkeypatch,failure):
    root=tmp_path/'study';directory=Path(str(root)+'.launch');directory.mkdir()
    parent=tmp_path/'parent.json';parent.write_text('{}');s=spec_fixture(root);calls=[]
    def create(*a,**kw):
        if failure=='registration':raise ValueError('fixture population mismatch')
        root.mkdir();return s
    monkeypatch.setattr(campaign,'create',create);monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    def sbatch(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout='unknown\n',stderr='')
    monkeypatch.setattr(subprocess,'run',sbatch)
    with pytest.raises(Exception):launch.run(root,parent,'/project','f'*40,36,'wrong' if failure=='authorization' else c.LAUNCH_AUTHORIZATION)
    assert len(calls)==(1 if failure=='ambiguous' else 0)
    if failure=='authorization':assert not (directory/'process_claim').exists()
    else:assert (directory/'failure.json').exists() and not (directory/'complete.json').exists()


@pytest.mark.parametrize('wrong',[None,'job','comment','name','memory','walltime','gpu','backend'])
def test_exact_actual_allocation_guards(tmp_path,monkeypatch,wrong):
    s=spec_fixture(tmp_path);s['inputs']['evidence'].update(environment={},cudnn_v8_disabled='unset')
    task='train_OFFLINE';job='100'
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:{task:job})
    monkeypatch.setattr(source,'runtime',lambda _:None)
    monkeypatch.setattr(execution.platform,'machine',lambda:'x86_64')
    for key,value in dict(SLURM_JOB_ID=job,SLURM_CPUS_PER_TASK='6',SLURM_MEM_PER_NODE='192000',
        SLURM_CLUSTER_NAME='slurmctld',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8').items():monkeypatch.setenv(key,value)
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED',raising=False)
    for name in ('is_available','is_bf16_supported'):monkeypatch.setattr(torch.cuda,name,lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'L40S')
    fields=dict(Account='default',Partition='gpu',QOS='norm-gpu',NumNodes='1',NumTasks='1',NumCPUs='6',
        JobName='lukafsbal_'+task,Comment='lukafsbal:'+s['content_hash']+':'+task,TimeLimit='2-00:00:00')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items())))
    if wrong=='job':monkeypatch.setenv('SLURM_JOB_ID','101')
    elif wrong=='comment':fields['Comment']='other campaign'
    elif wrong=='name':fields['JobName']='other_task'
    elif wrong=='memory':monkeypatch.setenv('SLURM_MEM_PER_NODE','180000')
    elif wrong=='walltime':fields['TimeLimit']='1-00:00:00'
    elif wrong=='gpu':monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'A40')
    elif wrong=='backend':monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','wrong')
    if wrong:
        with pytest.raises((ValueError,PermissionError)):execution.validate(s,task)
    else:assert execution.validate(s,task)==job


def test_shell_syntax_and_detached_tail_exits_before_background(tmp_path):
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    project=Path(__file__).resolve().parents[1]
    for path in ('scripts/launch_luka_fullstat_balanced_ladder.sh','scripts/queue_luka_fullstat_balanced_ladder.sh',
                 'sbatch/run_luka_fullstat_balanced_ladder.sh'):
        subprocess.run([bash,'-n',path],cwd=project,check=True,capture_output=True)
    def shellpath(p):
        p=Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    repo=tmp_path/'project';(repo/'scripts').mkdir(parents=True);(repo/'sbatch').mkdir()
    script=repo/'scripts/launch_luka_fullstat_balanced_ladder.sh'
    script.write_bytes((project/'scripts'/script.name).read_bytes())
    (repo/'sbatch/luka_splithead_finalize_common.sh').write_bytes(b'# fixture only\n')
    binpath=tmp_path/'bin';binpath.mkdir()
    bodies=dict(git='case "$*" in\n*--show-toplevel*) printf "%s\\n" "$FIXTURE_PROJECT";;\n*"rev-parse HEAD"*) printf "%040d\\n" 1;;\nesac\n',
        setsid='exec "$@"\n',tail='printf "TAIL_ATTACHED\\n"\n',
        python='if [[ "$*" == *" -c "* ]]; then exit 0; fi\nsleep 1\nprintf "DETACHED_FINISHED\\n"\nprintf done > "$FIXTURE_DONE"\n')
    for name,body in bodies.items():
        p=binpath/name;p.write_bytes(('#!/usr/bin/env bash\n'+body).encode());p.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    parent=tmp_path/'parent.json';parent.write_text('{}');root=tmp_path/'output';done=tmp_path/'done'
    env=dict(os.environ,BASH_ENV=shellpath(envfile),FIXTURE_BIN=shellpath(binpath),
        FIXTURE_PROJECT=shellpath(repo),FIXTURE_DONE=shellpath(done))
    args=[bash,str(script),'0'*39+'1',shellpath(root),shellpath(parent),'36','--authorize',c.LAUNCH_AUTHORIZATION]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'TAIL_ATTACHED' in result.stdout and 'DETACHED: balanced launcher PID' in result.stdout
    deadline=time.monotonic()+10
    while not done.exists() and time.monotonic()<deadline:time.sleep(.05)
    assert done.read_text()=='done'
    again=subprocess.run(args,env=env,capture_output=True,text=True,timeout=5)
    assert again.returncode!=0
    # Windows uses a setsid fixture: this demonstrates detached control flow,
    # not an actual Linux HUP test or Oscar process-lifetime guarantee.


def test_required_source_and_docs_exist():
    project=Path(__file__).resolve().parents[1]
    assert all((project/p).is_file() for p in source.REQUIRED)
    script=(project/'scripts/launch_luka_fullstat_balanced_ladder.sh').read_text()
    assert 'nohup setsid bash' in script and '>>"${LOG}" 2>&1 </dev/null &' in script
    assert script.index('exec tail')<script.index('status --porcelain')<script.index('source "${PROJECT_DIR}/sbatch')
    assert 'scancel' not in script and 'rm ' not in script
