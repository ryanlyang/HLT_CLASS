"""Exact pre-claim recovery; unchanged donor execution and isolated new results."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import subprocess

import numpy as np
import pytest
from hlt_classification import luka_r8_recovery as r
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.luka_next_supervision_hydra import reconstruction_followup
from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
from test_luka_direct_reconstruction import fixture, compact_native


def old_spec(root):
    return dict(reconstruction_followup.registration(),root=str(root),project_dir='/original',
        source_commit=r.DONOR,content_hash='a'*64,environment_dir='/env',bundle_root='/bundle',raw_root='/raw',delta_root='/delta',
        reference=dict(environment={}),qualification={'acceptance':{'gpu':{'name':'fixture'}}},reused={})


@pytest.fixture
def admission(tmp_path,monkeypatch):
    root=tmp_path/'old'; root.mkdir(); old=old_spec(root)
    original=root/'campaign_spec.json'; write_immutable_json(original,old)
    for name in ('submission_science','setup'): (root/name).mkdir()
    write_immutable_json(root/'submission_science/submission_ledger.json',{'jobs':r.FAILED})
    write_immutable_json(root/'setup/receipt.json',{'setup':'preserved'})
    for task,job in r.FAILED.items():
        (root/f'{task}-{job}.out').write_text(f"subprocess.CalledProcessError: Command {['scontrol','show','job','-o',job]!r} returned non-zero exit status 1.")
    acceptance=dict(content_hash='b'*64,minutes={'C3':100,'R14':180})
    monkeypatch.setattr(r,'original',lambda path:old)
    monkeypatch.setattr(r,'source',lambda *a:dict(content_hash='c'*64))
    monkeypatch.setattr(r.worker,'completed_setup',lambda s:acceptance)
    monkeypatch.setattr(r.submission,'accepted_jobs',lambda s,stage:r.BOOTSTRAP if stage=='bootstrap' else dict(r.FAILED,report='168742'))
    monkeypatch.setattr(r.submission,'accounting',lambda job:['FAILED','1:0'] if job in r.FAILED.values() else ['COMPLETED','0:0'])
    monkeypatch.setattr(r.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    return old,original,acceptance


def register(tmp_path,admission):
    old,path,acceptance=admission
    return r.create(tmp_path/'repair',path,tmp_path/'code','f'*40,3)


def test_registration_reuses_setup_and_preserves_old_bytes(tmp_path,admission):
    old,path,a=admission
    before={str(p):p.read_bytes() for p in Path(old['root']).rglob('*') if p.is_file()}
    spec=register(tmp_path,admission)
    assert spec['tasks']==['C3','R14'] and spec['superseded_jobs']==r.FAILED
    assert spec['minutes']=={'C3':115,'R14':195} and spec['parents']['acceptance']==a['content_hash']
    assert r.check(spec)==old
    assert before=={str(p):p.read_bytes() for p in Path(old['root']).rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='fresh'): register(tmp_path,admission)


@pytest.mark.parametrize('relative',['claims/C3','claims/R14','C3','R14'])
def test_existing_original_attempt_refuses_recovery(tmp_path,admission,relative):
    old,_,_=admission; (Path(old['root'])/relative).mkdir(parents=True)
    with pytest.raises(FileExistsError): register(tmp_path,admission)
    assert not (tmp_path/'repair').exists()


@pytest.mark.parametrize('state',[['RUNNING','0:0'],['COMPLETED','0:0'],['FAILED','0:0'],None])
def test_unconfirmed_or_started_original_is_not_a_preclaim_retry(tmp_path,admission,monkeypatch,state):
    monkeypatch.setattr(r.submission,'accounting',lambda j:state if j=='168726' else (['FAILED','1:0'] if j=='168727' else ['COMPLETED','0:0']))
    with pytest.raises(ValueError): register(tmp_path,admission)


def test_corrupt_original_log_and_insufficient_quota_fail_before_root(tmp_path,admission):
    old,path,_=admission
    with pytest.raises(ValueError,match='three'): r.create(tmp_path/'repair',path,tmp_path/'code','f'*40,2)
    (Path(old['root'])/'C3-168726.out').write_text('some unrelated training failure')
    with pytest.raises(ValueError,match='controller-query'): register(tmp_path,admission)
    assert not (tmp_path/'repair').exists()


def test_exact_source_and_environment_are_bound(tmp_path,admission):
    spec=register(tmp_path,admission)
    changed=deepcopy(spec); changed['minutes']['R14']+=1
    with pytest.raises(ValueError): r.check(changed)
    Path(spec['original_logs']['C3']['path']).write_text('changed')
    with pytest.raises(ValueError): r.check(spec)


def test_read_only_rpc_retry_prints_stderr_and_preserves_eventual_value(monkeypatch,capsys):
    calls=[]; sleeps=[]
    def allocation():
        calls.append(1)
        if len(calls)<3:
            raise subprocess.CalledProcessError(1,['scontrol','show','job','-o','999'],output='',stderr='controller temporarily unavailable')
        return '999'
    monkeypatch.setattr(r.time,'sleep',sleeps.append)
    assert r.retry_controller(allocation)=='999'
    assert sleeps==[5,10] and len(calls)==3
    assert 'controller temporarily unavailable' in capsys.readouterr().out


@pytest.mark.parametrize('error',[
    PermissionError('allocation differs'),ValueError('invalid GPU'),
    subprocess.CalledProcessError(1,['sbatch','job.sh']),
    subprocess.CalledProcessError(1,['scontrol','update','JobId=1']),
])
def test_scientific_or_mutating_failures_are_never_retried(monkeypatch,error):
    calls=[]
    def bad(): calls.append(1); raise error
    monkeypatch.setattr(r.time,'sleep',lambda _:pytest.fail('unexpected retry'))
    with pytest.raises(type(error)): r.retry_controller(bad)
    assert len(calls)==1


def test_persistent_controller_failure_stops_after_three_attempts(monkeypatch):
    calls=[]
    def bad():
        calls.append(1); raise subprocess.CalledProcessError(1,['scontrol','show','config'],stderr='unavailable')
    monkeypatch.setattr(r.time,'sleep',lambda _:None)
    with pytest.raises(subprocess.CalledProcessError): r.retry_controller(bad)
    assert len(calls)==3


def test_new_exact_dag_two_fits_and_snapshot_report_idempotent(tmp_path,admission,monkeypatch):
    spec=register(tmp_path,admission); old=admission[0]; calls=[]; reviews=[]
    monkeypatch.setattr(r.reconstruction_release,'scheduler_review',lambda p:reviews.append(p))
    def sbatch(cmd,**kwargs): calls.append(cmd); return SimpleNamespace(stdout=str(9000+len(calls)),returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    r.submit(spec); plan=r.plan(spec,old)
    assert [v['task_id'] for v in plan['commands']]==['C3','R14','report']
    assert len(reviews)==1 and not calls
    ledger=r.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=r.AUTHORIZATION)
    assert ledger['jobs']=={'C3':'9001','R14':'9002','report':'9003'}
    assert all(not any(a.startswith('--dependency=') for a in c) for c in calls[:2])
    assert '--dependency=afterany:9001:9002' in calls[2] and '--partition=compute' in calls[2]
    assert '--gres=gpu:nvidia_rtx_a6000:1' in calls[0]
    assert r.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=r.AUTHORIZATION)==ledger
    assert len(calls)==3 and r.jobs(spec,old)==ledger['jobs']


def test_partial_submission_is_not_blindly_resumed(tmp_path,admission,monkeypatch):
    spec=register(tmp_path,admission); calls=[]
    monkeypatch.setattr(r.reconstruction_release,'scheduler_review',lambda p:None)
    def sbatch(cmd,**kwargs):
        calls.append(cmd)
        if len(calls)==2: raise subprocess.CalledProcessError(1,cmd,stderr='transport error')
        return SimpleNamespace(stdout='9001',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    r.submit(spec); value=r.plan(spec,admission[0])
    args=dict(execute=True,plan_hash=value['content_hash'],authorization=r.AUTHORIZATION)
    with pytest.raises(subprocess.CalledProcessError): r.submit(spec,**args)
    with pytest.raises(RuntimeError,match='Ambiguous'): r.submit(spec,**args)
    assert len(calls)==2 and r.jobs(spec,admission[0])=={'C3':'9001'}


@pytest.mark.parametrize('task',['C3','R14'])
def test_real_thirty_epoch_recovered_fit_native_export_and_receipt(tmp_path,monkeypatch,task):
    original=tmp_path/'original'; original.mkdir()
    old,tr,val,bank=fixture(original,monkeypatch,rows=12)
    old.update(reconstruction_followup.registration(),counts=dict(train=12,validation=12))
    repair=tmp_path/'repair'; repair.mkdir()
    spec=dict(root=str(repair),content_hash='e'*64)
    before={str(p):p.read_bytes() for p in original.rglob('*') if p.is_file()}
    monkeypatch.setattr(r,'check',lambda *a,**k:old)
    monkeypatch.setattr(r,'jobs',lambda *a:{task:'9001'})
    monkeypatch.setattr(r,'allocation',lambda _: '9001')
    monkeypatch.setattr(r.campaign,'check',lambda *a,**k:{})
    monkeypatch.setattr(r,'require_backend',lambda:None)
    monkeypatch.setattr(r,'installed_environment',lambda:{})
    monkeypatch.setattr(r,'gpu_identity',lambda:old['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(r.inputs,'hlt',lambda s,role:tr if role=='train' else val)
    monkeypatch.setattr(r.inputs,'embedding_bank',lambda *a:bank)
    actual=r.fit
    monkeypatch.setattr(r,'fit',lambda s,o,t:actual(s,o,t,device='cpu'))
    receipt=r.run(spec,task)
    report,arrays,verified=r.recovered(spec,old,task)
    assert receipt==verified and report['epochs']==30
    assert report['settings']==reconstruction_followup.setting(task)
    assert report['deployable_hlt_only'] and np.isfinite(arrays['probabilities']).all()
    assert receipt['supersedes_job']==r.FAILED[task]
    assert before=={str(p):p.read_bytes() for p in original.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError): r.run(spec,task)


def test_progress_uses_recovered_seed3_control_and_original_epoch_parent(tmp_path,monkeypatch,capsys):
    old=old_spec(tmp_path/'old'); spec=dict(root=str(tmp_path),content_hash='b'*64)
    monkeypatch.setattr(r,'check',lambda *a,**kw:old)
    monkeypatch.setattr(r,'jobs',lambda *a:{'C3':'9001','R14':'9002'})
    monkeypatch.setattr(r.inputs,'Bundle',lambda *a:SimpleNamespace(json=lambda *a:dict(recovery_anchors={'M0HLT':.95,'OFFLINE':.97})))
    for task,auc in (('C3',.960),('R14',.961)):
        write_immutable_json(tmp_path/task/'fit/epoch_01.json',r.contracts.artifact('EPOCH',parents=dict(spec=old['content_hash']),task=task,pass_number=1,validation={'macro_ovr_auc':auc}))
    value=r.progress(spec); out=capsys.readouterr().out
    assert value['epochs']=={'C3':1,'R14':1} and 'delta_selected_C3=+0.00100000' in out


def test_cli_loads_scientific_imports_from_original_worktree(tmp_path):
    import sys
    new=tmp_path/'new'; old=tmp_path/'old'
    (new/'scripts').mkdir(parents=True); (new/'src/hlt_classification').mkdir(parents=True)
    (old/'src/hlt_classification').mkdir(parents=True)
    for root,value in ((old,'old science'),(new,'wrong science')):
        (root/'src/hlt_classification/__init__.py').write_text(f'SCIENCE={value!r}\n')
    driver=new/'src/hlt_classification/luka_r8_recovery.py'
    driver.write_text('from hlt_classification import SCIENCE\ndef results(spec): return {"loaded":SCIENCE}\n')
    script=new/'scripts/luka_r8_recovery.py'
    script.write_bytes((Path(__file__).parents[1]/'scripts/luka_r8_recovery.py').read_bytes())
    original=old/'campaign_spec.json'; original.write_text(json.dumps({'project_dir':str(old)}))
    spec=tmp_path/'recovery_spec.json'; spec.write_text(json.dumps({'original_spec':{'path':str(original)}}))
    result=subprocess.run([sys.executable,'-B','-s',str(script),'results','--spec',str(spec)],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)=={'loaded':'old science'}


def test_combined_report_replaces_missing_rows_and_recomputes_seed3_delta(tmp_path,monkeypatch,capsys):
    from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES
    classes=len(CLASS_NAMES)
    old=old_spec(tmp_path/'old'); spec=dict(root=str(tmp_path/'repair'),content_hash='b'*64)
    monkeypatch.setattr(r,'check',lambda *a,**k:old)
    monkeypatch.setattr(r,'jobs',lambda *a:{'C3':'9001','R14':'9002','report':'9003'})
    base=[]
    for role in ('validation','original50k'):
        for task,auc in (('M0HLT',.4),('OFFLINE',.9),('C2',.6),('R13',.7)):
            base.append(dict(model=task,role=role,metrics={'macro_ovr_auc':auc},epoch=1))
    monkeypatch.setattr(r.reporting,'results',lambda s:dict(rows=deepcopy(base),incomplete=['C3','R14','R15'],states={},content_hash='c'*64))
    reference=dict(identities=np.arange(2*classes),labels=np.tile(np.arange(classes),2))
    monkeypatch.setattr(r.inputs,'Bundle',lambda *a:SimpleNamespace(path=lambda name:name))
    monkeypatch.setattr(r,'load_npz_arrays',lambda name:dict(indices=np.arange(classes)) if name.endswith('subset.npz') else reference)
    arrays={}
    for seed,task in enumerate(r.FAILED):
        p=np.random.default_rng(seed).random((2*classes,classes)); p/=p.sum(1,keepdims=True)
        arrays[task]=dict(reference,probabilities=p)
        folder=Path(spec['root'])/task; folder.mkdir(parents=True)
        (folder/'receipt.json').write_text('{}')
    monkeypatch.setattr(r,'recovered',lambda s,o,t:(dict(selected_pass=3),arrays[t],dict(content_hash=('d' if t=='C3' else 'e')*64,job_id='9001' if t=='C3' else '9002')))
    value=r.results(spec)
    assert value['incomplete']==['R15']
    selected={x['model']:x for x in value['rows'] if x['role']=='validation'}
    delta=selected['R14']['metrics']['macro_ovr_auc']-selected['C3']['metrics']['macro_ovr_auc']
    assert selected['R14']['delta_matched']==delta and selected['C3']['delta_matched']==0
    assert selected['R14']['supersedes_job']=='168727' and selected['R14']['recovered_job']=='9002'
    assert selected['R13']['delta_matched']==pytest.approx(.1)
    assert '[recovered]' in capsys.readouterr().out


def test_original_reader_rejects_wrong_commit_and_import_location(tmp_path,monkeypatch):
    old=old_spec(tmp_path/'old'); path=tmp_path/'campaign_spec.json'
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError,match='runtime'): r.original(path)
    old['project_dir']=str(Path(r.campaign.__file__).resolve().parents[3])
    path.write_text(json.dumps(old))
    observed=[]; monkeypatch.setattr(r.campaign,'check',lambda s,**kw:observed.append(s))
    assert r.original(path)==old and observed==[old]
    old['source_commit']='0'*40; path.write_text(json.dumps(old))
    with pytest.raises(ValueError,match='58d0c0ed'): r.original(path)
